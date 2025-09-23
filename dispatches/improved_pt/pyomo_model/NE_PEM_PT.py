import os
import pickle
import json
import pandas as pd
import pyomo.environ as pyo
from tensorflow import keras
import omlt
from omlt.linear_tree import LinearTreeGDPFormulation
from omlt.neuralnet import ReluComplementarityFormulation
from omlt.io import load_keras_sequential
import idaes.logger as idaeslog
from idaes.apps.grid_integration import DesignModel, OperationModel
from idaes.apps.grid_integration import PriceTakerModel

H2_PROD_RATE = 20  # Rate of hydrogen production from the electrolyzer

def build_gen_design_model(m, params):
    """
    build NPP design models. 
    """
    if "capex_npp" not in list(params.keys()):
        params["capex_npp"] = 0
    if "fom_npp" not in list(params.keys()):
        params["fom_npp"] = 0
    
    m.gen_capacity = pyo.Param(
        initialize=params['npp_cap'],
        mutable=True,
        doc="Maxium capacity of the generator [in MW]",
    )

    m.capex = pyo.Expression(
        expr=m.gen_capacity * params["capex_npp"]
        )
    
    m.fom = pyo.Expression(
        expr=m.gen_capacity * params["fom_npp"]
    )

    return


def build_pem_design_model(m, params):
    """
    build PEM design models. 
    """
    if "capex_pem" not in list(params.keys()):
        params["capex_pem"] = 0
    if "fom_pem" not in list(params.keys()):
        params["fom_pem"] = 0
    
    m.pem_capacity = pyo.Var(
        within=pyo.NonNegativeReals,
        initialize=params['pem_cap'],
        doc="Maxium capacity of the PEM electrolyzer [in MW]",
    )

    m.pem_cap_ratio = pyo.Expression(
        expr=m.pem_capacity / params['npp_cap']
    )

    m.pem_bid = pyo.Var(
        within=pyo.NonNegativeReals,
        initialize=params['pem_bid'],
        doc="Bidding price for the PEM electrolyzer [in $/MWh]",
    )

    if params['fix_pem_cap']:
        m.pem_capacity.fix(params['pem_cap'])  # Fix the PEM capacity for this analysis.

    m.capex = pyo.Expression(
        expr=m.pem_capacity * params["capex_pem"]
        )
    
    m.fom = pyo.Expression(
        expr=m.pem_capacity * params["fom_pem"]
    )
    
    # ========================== Embedding the regression models ===========================
    # Load the regression models for long-term price changes
    lt_increase_path = params["lt_increas_path"]
    lt_decrease_path = params["lt_decreas_path"]

    with open(lt_increase_path, 'rb') as f:
        increase_regr = pickle.load(f)

    with open(lt_decrease_path, 'rb') as f:
        decrease_regr = pickle.load(f)

    # embed the regression models into the design model
    m.incr_lt = omlt.OmltBlock()
    # use the GDP formulation with a big-M, transformation
    formulation1_lt = LinearTreeGDPFormulation(increase_regr, transformation="bigm")
    m.incr_lt.build_formulation(formulation1_lt)
    
    m.decr_lt = omlt.OmltBlock()
    # use the GDP formulation with a big-M, transformation
    formulation2_lt = LinearTreeGDPFormulation(decrease_regr, transformation="bigm")
    m.decr_lt.build_formulation(formulation2_lt)
    
    inputs = [m.pem_cap_ratio, m.pem_bid]

    @m.Constraint(range(len(inputs)))
    def connect_inputs_incr(mdl, i):
        return inputs[i] == mdl.incr_lt.inputs[i]

    @m.Constraint(range(len(inputs)))
    def connect_inputs_decr(mdl, i):
        return inputs[i] == mdl.decr_lt.inputs[i]


    @m.Constraint()
    def connect_outputs_incr(mdl):
        return mdl.incr_lmp == mdl.incr_lt.outputs[0]

    @m.Constraint()
    def connect_outputs_decr(mdl):
        return mdl.decr_lmp == mdl.decr_lt.outputs[0]
    
    return


def build_npp_operation_model(m, design_blk, params):
    """
    Build operation model for the NPP system.
    """
    # the power output at each time period
    m.npp_power = pyo.Var(
        within=pyo.NonNegativeReals,
        doc="Net power produced by NPP at time t [in MW]",
        bounds=(0, design_blk.gen_capacity.value),
    )

    m.npp_power.fix(400)  # Assume the NPP is fixed at 400 MW.

    m.vom = pyo.Expression(expr=params['vom_npp'] * m.npp_power)
    
    return 


def build_pem_operation_model(m, design_blk, params):
    """
    Build operation model for the PEM electrolyzer.
    """
    m.pem_power = pyo.Var(
        within=pyo.NonNegativeReals,
        doc="Power consumed by PEM electrolyzer at time t [in MW]",
        bounds=(0, design_blk.pem_capacity.value),
    )

    m.h2_prod = pyo.Var(
        within=pyo.NonNegativeReals, 
        doc="Hydrogen production rate (kg/hr)"
    )

    m.h2_prod_constraint = pyo.Constraint(
        expr=m.h2_prod == m.pem_power / H2_PROD_RATE
    )

    m.vom = pyo.Expression(
        expr=params['vom_pem'] * m.pem_power
    )

    return


def build_ne_pem_flowsheet(m, npp_design_blk, pem_design_blk, npp_params, pem_params, ann_path, scaling_params_path):
    """
    Build the flowsheet for the NE_PEM system.
    """
    m.npp = OperationModel(
        model_func=build_npp_operation_model,
        model_args={"design_blk": npp_design_blk, "params": npp_params},
    )
    m.pem = OperationModel(
        model_func=build_pem_operation_model,
        model_args={"design_blk": pem_design_blk, "params": pem_params},
    )

    m.power_to_grid = pyo.Var(
        within=pyo.NonNegativeReals,
        doc="Power sent to the grid at time t [in MW]",
    )

    m.power_to_grid_constraint = pyo.Constraint(
        expr=m.power_to_grid == m.npp.npp_power - m.pem.pem_power
    )

    # This is for the standard price-taker revenue calculation
    # m.elec_revenue = pyo.Expression(expr=m.npp.LMP * m.power_to_grid)

    # ========================== Building the ANN ===========================
    # scale the inputs
    inputs = [m.npp.load, m.npp.regen, m.power_to_grid/m.npp.load, m.npp.LMP]
    with open(scaling_params_path, 'r') as f:
        scaling_params = json.load(f)
    
    input_bounds = {
        i: (scaling_params['xmin'][i], scaling_params['xmax'][i])
        for i in range(len(scaling_params['xmin']))
    }

    scaling_object = omlt.OffsetScaling(
        offset_inputs=scaling_params["xmin"],
        factor_inputs=scaling_params["xstd"],
        offset_outputs=scaling_params["y_mean"],
        factor_outputs=scaling_params["y_std"],
    )

    ann_clf = keras.models.load_model(ann_path)
    ann_defn = load_keras_sequential(ann_clf, scaling_object, input_bounds)

    m.lmp_ann_clf = omlt.OmltBlock()
    relu_ann_comp = ReluComplementarityFormulation(ann_defn)
    m.lmp_ann_clf.build_formulation(relu_ann_comp)

    @m.Constraint(range(len(inputs)))
    def connect_inputs_ann(mdl, i):
        return inputs[i] == mdl.lmp_ann_clf.inputs[i]
    
    # get the one-hot encoded outputs
    m.k_set = pyo.Set(initialize=[0,1,2])
    m.ann_out = pyo.Var(m.k_set, within=pyo.nonnegativeReals)

    @m.Constraint(m.k_set)
    def connect_outputs_ann(mdl, k):
        return mdl.ann_out[k] == mdl.lmp_ann_clf.outputs[k]

    m.delta_lmp_level = pyo.Var(m.k_set, within=pyo.NonNegativeReals)
    m.delta_lmp_level_c1 = pyo.Constraint(
        expr=m.delta_lmp_level[0] == m.pem_design.decr_lmp
    )
    m.delta_lmp_level_c2 = pyo.Constraint(
        expr=m.delta_lmp_level[1] == 0
    )
    m.delta_lmp_level_c3 = pyo.Constraint(
        expr=m.delta_lmp_level[2] == m.pem_design.incr_lmp
    )

    # --- Argmax selection variables ---
    m.y_argmax = pyo.Var(m.k_set, domain=pyo.Binary)
    m.zmax = pyo.Var(within=pyo.NonNegativeReals)
    m.delta_lmp = pyo.Var(within=pyo.NonNegativeReals)

    # Exactly one class active
    m.one_class = pyo.Constraint(expr=sum(m.y_argmax[k] for k in m.k_set) == 1)
    # Link delta_lmp to the selected class
    m.delta_lmp_def = pyo.Constraint(expr=m.delta_lmp == sum(m.delta_lmp_level[k] * m.y_argmax[k] for k in m.k_set))
    
    # ================== Revenue calculation ==========================
    # Revenue based on the ANN-predicted LMP
    m.elec_revenue = pyo.Expression(expr=(m.npp.LMP + m.delta_lmp) * m.power_to_grid)

    return


def npp_pem_npv(lmp_data, npp_params, pem_params):
    """
    Builds and returns an instance of the NE_PEM price-taker model.
    """
    m = PriceTakerModel()

    # Appending the data to the model
    m.append_lmp_data(lmp_data=lmp_data)

    # Build design models and fix the capacity
    m.npp_design = DesignModel(
        model_func=build_gen_design_model,
        model_args={"params": npp_params},
    )

    m.pem_design = DesignModel(
        model_func=build_pem_design_model,
        model_args={"params": pem_params},
    )

    m.build_multiperiod_model(
        flowsheet_func=build_ne_pem_flowsheet, 
        flowsheet_options={
            "npp_design_blk": m.npp_design,
            "pem_design_blk": m.pem_design,
            "npp_params": npp_params,
            "pem_params": pem_params,
        }, 
    )
    
    m.add_hourly_cashflows(
        revenue_streams=["elec_revenue"],
        operational_costs=[
            "vom",
            "fom"
        ],
    )

    m.add_overall_cashflows(corporate_tax_rate=0)
    m.add_objective_function(objective_type="net_profit")

    return m


npp_params = {
    'npp_cap': 400,  # Max capacity of the NPP in MW
    'capex_npp': 0,  # Capital cost of the NPP in $/MW
    'fom_npp': 120*1000,  # Fixed O&M cost of the NPP in $/MW-year
    'vom_npp': 2.3,  # Variable O&M cost of the NPP in $/MWh
}

pem_params = {
    'npp_cap': 400,  # Reference NPP capacity for PEM sizing
    'pem_cap': 200,  # Max capacity of the PEM electrolyzer in MW
    'capex_pem': 1630*1000,  # Capital cost of the PEM electrolyzer in $/MW
    'fom_pem': 47.9*1000,  # Fixed O&M cost of the PEM electrolyzer in $/MW-year
    'vom_pem': 0,  # Variable O&M cost of the PEM electrolyzer in $/MWh
    'fix_pem_cap': True,  # Whether to fix the PEM capacity during optimization
}

# Load LMP data
base_pcm_path = os.path.join(os.getcwd(), "..", "result_analysis", "Sweep_summary", "Summarized_sweep_base_case.csv")
df_base = pd.read_csv(base_pcm_path)
lmp_data = df_base['LMP DA'].to_numpy()

# Build the model
m = npp_pem_npv(lmp_data=lmp_data, npp_params=npp_params, pem_params=pem_params)