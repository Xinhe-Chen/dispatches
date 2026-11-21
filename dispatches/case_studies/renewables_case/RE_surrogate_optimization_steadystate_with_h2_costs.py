#################################################################################
# DISPATCHES was produced under the DOE Design Integration and Synthesis Platform
# to Advance Tightly Coupled Hybrid Energy Systems program (DISPATCHES), and is
# copyright (c) 2020-2023 by the software owners: The Regents of the University
# of California, through Lawrence Berkeley National Laboratory, National
# Technology & Engineering Solutions of Sandia, LLC, Alliance for Sustainable
# Energy, LLC, Battelle Energy Alliance, LLC, University of Notre Dame du Lac, et
# al. All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. Both files are also available online at the URL:
# "https://github.com/gmlc-dispatches/dispatches".
#################################################################################

# conceptual_design_problem_dynamic formation 2, only use timeseries clustering to cluster dispatch data
# NN is based on dispatch_shuffled_data_0.csv, 32 clusters including capacity factor 0/1 days.
# use omlt v1.0 in this file.

#the rankine cycle is a directory above this one, so modify path
import json
from functools import partial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import pyomo.environ as pyo
from pyomo.environ import ConcreteModel, SolverFactory, units, Var, \
    TransformationFactory, value, Block, Expression, Constraint, Param, \
    Objective, NonNegativeReals, ConstraintList, Set, units as pyunits, RangeSet
from pyomo.util.infeasible import log_infeasible_constraints, log_infeasible_bounds, log_close_to_bounds
import tensorflow as tf
from tensorflow import keras
import omlt
from omlt.neuralnet import NetworkDefinition, FullSpaceNNFormulation
from omlt.io import load_keras_sequential
import idaes.logger as idaeslog
from idaes.core.solvers.get_solver import get_solver
from idaes.apps.grid_integration.multiperiod.multiperiod import MultiPeriodModel

from dispatches.case_studies.renewables_case.load_parameters import pem_bar, pem_temp, wind_cap_cost, \
                                wind_op_cost, pem_op_cost, pem_var_cost, pem_cap_cost, PA, h2_mols_per_kg, \
                                discount_rate, N
from dispatches.case_studies.renewables_case.wind_battery_PEM_LMP import wind_battery_pem_variable_pairs, \
                                wind_battery_pem_periodic_variable_pairs, wind_battery_pem_om_costs, \
                                initialize_mp, wind_battery_pem_model, wind_battery_pem_mp_block


# RT market only or Both RT and DA markets
rt_market_only = True
include_wind_capital_cost = False
shortfall = 1000

# ===========================================================================
# Hydrogen conditioning / delivery and PEM stack replacement
# ---------------------------------------------------------------------------
# This script sweeps a FIXED (PEM bid, PEM size) grid, so the degrees of
# freedom are unchanged: both are still fixed via .fix() in
# conceptual_design_dynamic_RE(). The costs below only change the economics
# evaluated at each candidate, and the bid <-> hydrogen price mapping.
#
#   chi^del     DELIVERY_COST    all-in delivery charge [$/kg delivered]
#   eps^del     DELIVERY_LOSS    loss fraction not covered by chi^del [-]
#   kappa^del   DELIVERY_FIXED   fixed annual delivery service expense [$/yr]
#   theta^cond  COND_ENERGY      behind-meter conditioning energy [kWh/kg]
#   Lambda^st   STACK_LIFE       stack life [operating h]; 0 disables
#   sigma^st    STACK_REPL_FRAC  replacement cost / installed PEM capital [-]
#
# Defaults: LIQUID PATH WITH BEHIND-THE-METER LIQUEFACTION. The plant owns and
# runs the liquefier, so it pays in electricity (theta^cond), not in a service
# fee, and chi^del covers only the stages it still buys. Splitting Argonne
# IN025 slide 18 (liquefaction 2.69 + terminal 0.21 + truck 0.29 = 3.19 $/kg):
#
#     chi^del = 0.21 + 0.29 = 0.50 $/kg      (terminal + road transport)
#     theta^cond = 11 kWh/kg                 (liquefaction; NREL 88818 gives 9-13)
#
# IMPORTANT: do NOT also set chi^del = 3.2. That figure is the all-in external
# service, which already includes the liquefier and its electricity; charging
# it alongside theta^cond > 0 double-counts liquefaction.
#
# eps^del = 0.005 is the 0.5% mass loss implied by the 99.5% liquefaction mass
# efficiency reported in NREL/TP-5400-88818.
#
# Plus DOE current-status stack replacement (Program Record 24005: 11% of
# installed capital every 40,000 operating hours).
#
# Setting all six to zero reproduces the original script exactly.
# ===========================================================================
DELIVERY_COST = 0.50        # $/kg delivered  (terminal 0.21 + truck 0.29)
DELIVERY_LOSS = 0.005       # -               (NREL 88818 liquefaction mass eff 99.5%)
DELIVERY_FIXED = 0.0        # $/yr
COND_ENERGY = 11.0          # kWh/kg          (liquefaction, NREL 88818 9-13)
STACK_LIFE = 40000.0        # operating hours
STACK_REPL_FRAC = 0.11      # fraction of installed PEM capital

# psi used for the bid <-> hydrogen price mapping [kg/MWh].
# This is the flowsheet's actual yield, 0.00275984*3600/500*1000 = 19.8708
# (RE_flowsheet.py, m.fs.pem.electricity_to_mol), and it matches psi^pem in
# the manuscript. It makes the identity "net hydrogen contribution ==
# pem_bid * PEM energy" exact and keeps this script consistent with
# run_pricetaker_wind_PEM_with_h2_costs.py.
# (The original script hardcoded `m.pem_bid * 0.05`, i.e. psi = 20, which was
# 0.65% off the flowsheet.)
PSI_BID = 19.8708

# --- derived ---------------------------------------------------------------
# psi^eff: effective yield after behind-the-meter conditioning
PSI_EFF = PSI_BID / (1.0 + COND_ENERGY * PSI_BID / 1000.0)
YIELD_DERATE = PSI_EFF / PSI_BID                 # 1.0 when COND_ENERGY = 0
# phi^rep: levelized stack replacement reserve [$/kWh]
PHI_REP = (STACK_REPL_FRAC * pem_cap_cost / STACK_LIFE) if STACK_LIFE > 0 else 0.0

# tag so sweeps with different assumptions do not overwrite each other
_has_costs = any(abs(x) > 0.0 for x in (DELIVERY_COST, DELIVERY_LOSS,
                                        DELIVERY_FIXED, COND_ENERGY, PHI_REP))
RESULT_TAG = ""
if _has_costs:
    RESULT_TAG = (f"_cd{DELIVERY_COST:g}_loss{DELIVERY_LOSS:g}"
                  f"_cond{COND_ENERGY:g}_rep{PHI_REP*1000:g}")

# path for folder that has surrogate models
re_nn_dir = Path(__file__).parent / "data" / "steady_state_surrogate"

def load_surrogate_model(re_nn_dir):
    # load scaling and bounds for NN surrogates (rev and # of startups)
    with open(re_nn_dir / "dispatch_frequency" / "static_clustering_wind_pmax.pkl", 'rb') as f:
        model = pickle.load(f)
    centers = model.cluster_centers_
    dispatch_clusters_mean = centers[:, 0]
    pem_clusters_mean = centers[:, 1]
    resource_clusters_mean = centers[:, 2]

    # load keras neural networks
    # Input variables are PEM bid price, PEM MW, Reserve Factor and Load Shed Price
    with open(re_nn_dir / "dispatch_frequency" / "ss_surrogate_param_wind_pmax.json", 'r') as f:
        dispatch_data = json.load(f)
    nn_dispatch = keras.models.load_model(re_nn_dir / "dispatch_frequency" / "ss_surrogate_model_wind_pmax")

    if rt_market_only:
        rev_data_f = re_nn_dir / "rt_revenue" / "RE_RT_revenue_params_2_25.json"
        nn_rev = keras.models.load_model(re_nn_dir / "rt_revenue" / "RE_RT_revenue_2_25")
    else:
        rev_data_f = re_nn_dir / "revenue" / "RE_revenue_params_2_25.json"
        nn_rev = keras.models.load_model(re_nn_dir / "revenue" / "RE_revenue_2_25")

    with open(rev_data_f, 'rb') as f:
        rev_data = json.load(f)

    # load keras models and create OMLT NetworkDefinition objects
    #Revenue model definition
    input_bounds_rev = {i:(rev_data['xmin'][i],rev_data['xmax'][i]) for i in range(len(rev_data['xmin']))}
    scaling_object_rev = omlt.OffsetScaling(offset_inputs=rev_data['xm_inputs'],
                                                factor_inputs=rev_data['xstd_inputs'],
                                                offset_outputs=[rev_data['y_mean']],
                                                factor_outputs=[rev_data['y_std']])
    net_rev_defn = load_keras_sequential(nn_rev,scaling_object_rev,input_bounds_rev)

    # the dispatch frequency surrogate
    input_bounds_dispatch = {i:(dispatch_data['xmin'][i],dispatch_data['xmax'][i]) for i in range(len(dispatch_data['xmin']))}
    scaling_object_dispatch = omlt.OffsetScaling(offset_inputs=dispatch_data['xm_inputs'],
                                                factor_inputs=dispatch_data['xstd_inputs'],
                                                offset_outputs=dispatch_data['ws_mean'],
                                                factor_outputs=dispatch_data['ws_std'])
    net_frequency_defn = load_keras_sequential(nn_dispatch,scaling_object_dispatch,input_bounds_dispatch)
    return net_rev_defn, net_frequency_defn, dispatch_clusters_mean, pem_clusters_mean, resource_clusters_mean

def conceptual_design_dynamic_RE(input_params, PEM_bid=None, PEM_MW=None, verbose=False):

    net_rev_defn, net_frequency_defn, dispatch_clusters_mean, pem_clusters_mean, resource_clusters_mean = load_surrogate_model(re_nn_dir)

    num_rep_days = len(dispatch_clusters_mean) 

    m = ConcreteModel(name = 'RE_Conceptual_Design_full_surrogates')

    # add surrogate input to the model
    m.wind_system_capacity = Var(domain=NonNegativeReals, bounds=(100 * 1e3, 1000 * 1e3), initialize=input_params['wind_mw'] * 1e3)
    
    m.pem_system_capacity = Var(domain=NonNegativeReals, bounds=(127.05 * 1e3, 423.5 * 1e3), initialize=input_params['pem_mw'] * 1e3, units=pyunits.kW)
    m.pem_bid = Var(within=NonNegativeReals, bounds=(15, 45), initialize=15)                    # Energy Bid $/MWh
    m.reserve_percent = Param(within=NonNegativeReals, initialize=15)   # Reserves Fraction on Grid
    m.shortfall_price = Param(within=NonNegativeReals, initialize=shortfall)     # Energy price during load shed

    inputs = [m.pem_bid, m.pem_system_capacity * 1e-3 / 847 * m.wind_system_capacity * 1e-3, m.reserve_percent, m.shortfall_price]

    # extant_wind means if it is a built plant. If true, the captial cost is 0.
    if input_params['extant_wind']:
        m.wind_system_capacity.fix()
    if PEM_bid is not None:
        m.pem_bid.fix(PEM_bid)
    if PEM_MW is not None:
        m.pem_system_capacity.fix(PEM_MW * 1e3)

    # Affine bid <-> delivered-price mapping (manuscript Eq. 13d with costs):
    #   B = psi^eff (1 - eps^del)(rho^H2 - chi^del) - (phi_2^pem + phi^rep)
    # inverted for the delivered hydrogen price. With all costs zero this
    # reduces to m.pem_bid / PSI_BID, i.e. the original `m.pem_bid * 0.05`.
    _var_cost_per_mwh = 1000.0 * (input_params['pem_var_cost'] + PHI_REP)
    m.h2_price = Expression(
        expr=(m.pem_bid + _var_cost_per_mwh) / (PSI_EFF * (1.0 - DELIVERY_LOSS))
             + DELIVERY_COST)
    # add NN surrogates to the model using omlt
    ##############################
    # revenue surrogate
    ##############################
    m.rev_surrogate = Var()
    m.nn_rev = omlt.OmltBlock()
    formulation_rev = FullSpaceNNFormulation(net_rev_defn)
    m.nn_rev.build_formulation(formulation_rev)
    
    m.constraint_list_rev = ConstraintList()
    for i in range(len(inputs)):
        m.constraint_list_rev.add(inputs[i] == m.nn_rev.inputs[i])
    m.constraint_list_rev.add(m.rev_surrogate == m.nn_rev.outputs[0])

    # make rev non-negative, MM$
    # m.rev = Expression(expr=0.5 * pyo.sqrt(m.rev_surrogate**2 + 0.001**2) + 0.5 * m.rev_surrogate)
    m.rev = Expression(expr=m.rev_surrogate)

    ##############################
    # dispatch frequency surrogate
    ##############################
    m.dis_set = RangeSet(0, num_rep_days - 1)
    m.frequency_surrogate = Var(m.dis_set, initialize=[(x + 1) / (x + 1) / num_rep_days for x in range(num_rep_days)])
    m.nn_frequency = omlt.OmltBlock()
    formulation_frequency = FullSpaceNNFormulation(net_frequency_defn)
    m.nn_frequency.build_formulation(formulation_frequency)

    m.constraint_list_dispatch = ConstraintList()
    for i in range(len(inputs)):
        m.constraint_list_dispatch.add(inputs[i] == m.nn_frequency.inputs[i])

    m.frequency_surrogate_nonneg = Var(m.dis_set, initialize=[(x + 1) / (x + 1) / num_rep_days for x in range(num_rep_days)])
    for i in range(num_rep_days):
        m.constraint_list_dispatch.add(m.frequency_surrogate_nonneg[i] == 0.5 * pyo.sqrt(m.nn_frequency.outputs[i]**2 + 0.001**2) + 0.5 * m.nn_frequency.outputs[i])
    
    # # sum of outputs of dispatch surrogate may not be 1, scale them. 
    # for i in range(num_rep_days):
    #     m.constraint_list_dispatch.add(m.frequency_surrogate[i] == m.frequency_surrogate_nonneg[i] / sum(value(m.frequency_surrogate_nonneg[j]) for j in range(num_rep_days)))

    # Normalise the dispatch frequencies so they sum to one.
    # NOTE: the denominator must stay symbolic. Using value() here freezes it at
    # its construction-time value (exactly 1.0 given the initialisation above),
    # which silently disables the normalisation.
    m.frequency_sum = Var(domain=NonNegativeReals, bounds=(0.5, 2.0), initialize=1.0)
    m.frequency_sum_def = Constraint(
        expr=m.frequency_sum == sum(m.frequency_surrogate_nonneg[j] for j in m.dis_set))
    for i in range(num_rep_days):
        m.constraint_list_dispatch.add(
            m.frequency_surrogate[i] * m.frequency_sum == m.frequency_surrogate_nonneg[i])


    # dispatch frequency flowsheet, each scenario is a model. 
    scenario_models = []
    for i in range(num_rep_days):
        # set the capacity factor from the clustering results.
        # For each scenario, use the same wind_speed profile at this moment. 
        dispatch_capacity_factor = dispatch_clusters_mean[i]
        pem_capacity_factor = pem_clusters_mean[i]
        wind_capacity_factor = resource_clusters_mean[i]
        
        input_params['wind_resource'] = {0: {'wind_resource_config': {
                                                'capacity_factor': 
                                                    wind_capacity_factor}}}
                                                    
        scenario = MultiPeriodModel(
            n_time_points=1,
            process_model_func=partial(wind_battery_pem_mp_block, input_params=input_params, verbose=verbose),
            linking_variable_func=wind_battery_pem_variable_pairs,
            periodic_variable_func=wind_battery_pem_periodic_variable_pairs,
            )
        
        scenario.build_multi_period_model(input_params['wind_resource'])
        scenario_model = scenario.pyomo_model
        blks = scenario.get_active_process_blocks()

        # unfix wind for design, PEM is not fixed, leave battery fixed at 0 since no battery
        for blk in blks:
            if not input_params['extant_wind']:
                blk.fs.windpower.system_capacity.unfix()

        '''
        Differ from the wind_battery_LMP.py, in our problem, the wind farm 
        capacity is a design variable which we want to optimize. So we have every scenario's 
        wind_system_capacity is equal to the pmax*1e3, the unit is kW.
        '''

        scenario_model.wind_max_p = Constraint(scenario_model.TIME, 
            rule=lambda b, t: blks[t].fs.windpower.system_capacity <= m.wind_system_capacity)
        scenario_model.pem_max_p = Constraint(scenario_model.TIME, 
            rule=lambda b, t: blks[t].fs.pem.electricity[0] <= m.pem_system_capacity)

        scenario_model.dispatch_frequency = Expression(expr=m.frequency_surrogate[i])
        scenario_model.output_const = Constraint(scenario_model.TIME, 
            rule=lambda b, t: blks[t].fs.splitter.grid_elec[0] == m.wind_system_capacity * dispatch_capacity_factor)
        scenario_model.pem_const = Constraint(scenario_model.TIME, 
            rule=lambda b, t: blks[t].fs.splitter.pem_elec[0] <= m.wind_system_capacity * pem_capacity_factor)

        scenario_model.elec_grid = Expression(
            expr=scenario_model.dispatch_frequency * 8784 * blks[0].fs.splitter.grid_elec[0])
        scenario_model.elec_pem = Expression(
            expr=scenario_model.dispatch_frequency * 8784 * blks[0].fs.splitter.pem_elec[0])
        # HPROD: physical hydrogen produced. YIELD_DERATE accounts for the
        # share of the diverted power consumed by behind-meter conditioning
        # (= 1.0 when COND_ENERGY = 0).
        scenario_model.hydrogen_produced = Expression(scenario_model.TIME,
            rule=lambda b, t: scenario_model.dispatch_frequency * 8784 * blks[t].fs.pem.outlet.flow_mol[0] / h2_mols_per_kg * 3600 * YIELD_DERATE)
        # HDEL: mass that reaches the customer after delivery losses
        scenario_model.hydrogen_delivered = Expression(
            expr=(1.0 - DELIVERY_LOSS) * scenario_model.hydrogen_produced[0])
        scenario_model.hydrogen_revenue = Expression(
            # expr=scenario_model.hydrogen_produced[0] * 3)
            expr=scenario_model.hydrogen_delivered * m.h2_price)
        # DCOST (variable part): conditioning, terminal and road transport
        scenario_model.hydrogen_delivery_cost = Expression(
            expr=scenario_model.hydrogen_delivered * DELIVERY_COST)
        # PEM stack replacement reserve enters as a variable cost [$/kWh]
        scenario_model.op_var_cost = Expression(
            expr=(input_params['pem_var_cost'] + PHI_REP) * blks[0].fs.pem.electricity[0])
        scenario_model.var_total_cost = Expression(expr=scenario_model.dispatch_frequency * 8784 * scenario_model.op_var_cost)

        setattr(m, 'scenario_model_{}'.format(i), scenario_model)
        scenario_models.append(scenario_model)

    m.plant_cap_cost = Expression(
        expr=input_params['wind_cap_cost'] * m.wind_system_capacity * int(include_wind_capital_cost) + input_params['pem_cap_cost'] * m.pem_system_capacity)
    m.annual_fixed_cost = pyo.Expression(
        expr=m.wind_system_capacity * input_params["wind_op_cost"] + m.pem_system_capacity * input_params["pem_op_cost"])
    m.plant_operation_cost = Expression(
        expr=sum(scenario_models[i].var_total_cost for i in range(num_rep_days)))
    m.hydrogen_rev = Expression(
        expr=sum(scenario_models[i].hydrogen_revenue for i in range(num_rep_days)))
    m.hydrogen_produced_total = Expression(
        expr=sum(scenario_models[i].hydrogen_produced[0] for i in range(num_rep_days)))
    m.hydrogen_delivered_total = Expression(
        expr=sum(scenario_models[i].hydrogen_delivered for i in range(num_rep_days)))
    # DCOST = HDEL * chi^del + kappa^del
    m.delivery_cost = Expression(
        expr=sum(scenario_models[i].hydrogen_delivery_cost for i in range(num_rep_days))
             + DELIVERY_FIXED)
    # annual PEM energy [kWh/yr] and the levelized replacement reserve [$/yr]
    m.pem_energy_total = Expression(
        expr=sum(scenario_models[i].elec_pem for i in range(num_rep_days)))
    m.replacement_cost = Expression(expr=PHI_REP * m.pem_energy_total)

    m.NPV = Expression(expr=-m.plant_cap_cost + PA * (m.rev + m.hydrogen_rev - m.delivery_cost - m.plant_operation_cost - m.annual_fixed_cost))
    m.NPV_ann = Expression(expr=-m.plant_cap_cost / PA + (m.rev + m.hydrogen_rev - m.delivery_cost - m.plant_operation_cost - m.annual_fixed_cost))
    m.obj = Objective(expr=-m.NPV * 1e-7)
    
    return m, num_rep_days


def record_result(m, num_rep_days):
    wind_to_grid = []
    wind_to_pem = []
    wind_gen = []

    for i in range(num_rep_days):
        scenario_model = getattr(m, 'scenario_model_{}'.format(i))

        # _wind_gen = [value(scenario_model.blocks[j].process.fs.windpower.electricity[0]) for j in range(1)]
        # wind_gen.append(sum(_wind_gen))

        # _wind_to_grid = [value(scenario_model.blocks[j].process.fs.splitter.grid_elec[0]) for j in range(1)]
        _wind_to_grid = [value(scenario_model.elec_grid)]
        wind_to_grid.append(sum(_wind_to_grid))

        _wind_to_pem = [value(scenario_model.elec_pem)]
        wind_to_pem.append(sum(_wind_to_pem))

    # --- stack replacement schedule implied by the optimal dispatch ---------
    pem_kw = value(m.pem_system_capacity)
    pem_energy_kwh = value(m.pem_energy_total)
    eflh = pem_energy_kwh / pem_kw if pem_kw > 0 else 0.0
    if STACK_LIFE > 0 and eflh > 0:
        T_repl = STACK_LIFE / eflh
        repl_years = []
        _t = T_repl
        while _t < N - 1e-9:
            repl_years.append(_t)
            _t += T_repl
        kappa_disc = ((T_repl / PA) * sum((1.0 + discount_rate) ** (-y)
                                          for y in repl_years)) if repl_years else 0.0
    else:
        T_repl, repl_years, kappa_disc = float('nan'), [], 0.0

    results = {
        "wind_mw": value(m.wind_system_capacity) * 1e-3,
        "pem_mw": pem_kw * 1e-3,
        "pem_bid": value(m.pem_bid),
        "h2_price": value(m.h2_price),
        "e_revenue": value(m.rev),
        "h_revenue": value(m.hydrogen_rev),
        "h2_produced_kg": value(m.hydrogen_produced_total),
        "h2_delivered_kg": value(m.hydrogen_delivered_total),
        "delivery_cost": value(m.delivery_cost),
        "replacement_cost": value(m.replacement_cost),
        "NPV": value(m.NPV),
        "NPV_ann": value(m.NPV_ann),
        "Grid_e": sum(wind_to_grid),
        "PEM_e": sum(wind_to_pem),
        "pem_eflh": eflh,
        "pem_capacity_factor": eflh / 8760.0,
        "replacement_interval_yr": T_repl,
        "n_replacements": len(repl_years),
        "kappa_discount": kappa_disc,
        "replacement_cost_discounted": kappa_disc * value(m.replacement_cost),
        # "Wind_e": sum(wind_gen),
    }

    for day in range(num_rep_days):
        results[f'freq_day_{day}'] = value(m.frequency_surrogate[day])

    print("Wind capacity = {} MW".format(value(m.wind_system_capacity) * 1e-3))
    print("PEM capacity = {}MW".format(value(m.pem_system_capacity) * 1e-3))
    print("Plant bid = ${}".format(value(m.pem_bid)))
    print("Delivered H2 price = ${}/kg".format(value(m.h2_price)))
    print("Plant Elec Revenue Annual = ${}".format(value(m.rev)))
    print("Plant Hydrogen Revenue Annual = ${}".format(value(m.hydrogen_rev)))
    print("Hydrogen delivery cost Annual = ${}".format(value(m.delivery_cost)))
    print("PEM stack replacement Annual = ${}".format(value(m.replacement_cost)))
    print("PEM EFLH = {} h/yr ({} replacements over {} yr)".format(eflh, len(repl_years), N))
    print("Plant Total Revenue Annual = ${}".format(value(m.rev + m.hydrogen_rev)))
    print("Plant NPV = ${}".format(value(m.NPV)))
    print("Plant NPV Annualized = ${}".format(value(m.NPV_ann)))

    print('----------')
    for i in range(num_rep_days):
        print(value(m.frequency_surrogate[i]))
    print('----------')

    print('----------')
    print(sum(value(m.frequency_surrogate[j]) for j in m.dis_set))
    
    return results


def run_design(PEM_bid=None, PEM_size=None):
    model, n_rep_days = conceptual_design_dynamic_RE(default_input_params, PEM_bid=PEM_bid, PEM_MW=PEM_size, verbose=False)
    nlp_solver = SolverFactory('ipopt')
    nlp_solver.options['max_iter'] = 8000
    # nlp_solver.options['tol'] = 1e-5
    res = nlp_solver.solve(model, tee=True)
    if res.Solver.status != 'ok':
        solve_log = idaeslog.getInitLogger("infeasibility", idaeslog.INFO, tag="properties")
        log_infeasible_constraints(model, logger=solve_log, tol=1e-4, log_expression=True, log_variables=True)
        log_infeasible_bounds(model, logger=solve_log, tol=1e-4)
        return {
        "wind_mw": 0,
        "pem_mw": 0,
        "pem_bid": PEM_bid if PEM_bid is not None else 0,
        "h2_price": 0,
        "e_revenue": 0,
        "h_revenue": 0,
        "delivery_cost": 0,
        "replacement_cost": 0,
        "NPV": 0,
        "solver_failed": True,
    }
    return record_result(model, n_rep_days)

default_input_params = {
    "wind_mw": 847,
    "wind_mw_ub": 10000,
    "pem_mw": 423,
    "batt_mw": 0,
    "batt_mwh": 0,
    "pem_bar": pem_bar,
    "pem_temp": pem_temp,
    "tank_size": None,
    "tank_type": None,
    "turb_mw": None,

    "h2_price_per_kg": 3,
    "DA_LMPs": None,

    "design_opt": True,
    "extant_wind": True,        # fixed because parameter sweeps didn't change wind size

    "wind_cap_cost": wind_cap_cost if include_wind_capital_cost else 0,
    "wind_op_cost": wind_op_cost,
    "pem_cap_cost": pem_cap_cost,
    "pem_op_cost": pem_op_cost,
    "pem_var_cost": pem_var_cost
} 


if __name__ == "__main__":
    # result = run_design(PEM_bid=45, PEM_size=847*0.15)
    # result = run_design()
    # exit()

    import multiprocessing as mp
    from itertools import product

    # 7 PEM bids x 8 capacity ratios = 56 designs, the same grid the PCM
    # enumeration and the PT2 price-taker sweep use.
    pem_bid_range   = [15, 20, 25, 30, 35, 40, 45]
    pem_ratio_range = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

    bids  = np.array(pem_bid_range, dtype=float)
    sizes = np.round(np.array(pem_ratio_range, dtype=float) * 847.0, 6)
    inputs = list(product(bids, sizes))   # run_design(PEM_bid, PEM_size)

    print("=" * 72)
    print("SA sweep with H2 delivery and PEM stack replacement")
    print(f"  chi^del  = {DELIVERY_COST:8.4f} $/kg      eps^del = {DELIVERY_LOSS:g}")
    print(f"  kappa^del= {DELIVERY_FIXED:8.2f} $/yr      theta^cond = {COND_ENERGY:g} kWh/kg")
    print(f"  sigma^st = {STACK_REPL_FRAC:8.4f}          Lambda^st  = {STACK_LIFE:g} h")
    print(f"  phi^rep  = {PHI_REP:8.6f} $/kWh = {PHI_REP*1000:.2f} $/MWh")
    print(f"  psi_bid  = {PSI_BID:8.4f} kg/MWh    psi^eff = {PSI_EFF:.4f} kg/MWh")
    print(f"  delivered H2 price at bid 15 / 45 $/MWh = "
          f"{(15 + 1000*(pem_var_cost+PHI_REP))/(PSI_EFF*(1-DELIVERY_LOSS)) + DELIVERY_COST:.4f}"
          f" / {(45 + 1000*(pem_var_cost+PHI_REP))/(PSI_EFF*(1-DELIVERY_LOSS)) + DELIVERY_COST:.4f} $/kg")
    print(f"  {len(bids)} bids x {len(sizes)} sizes = {len(bids)*len(sizes)} designs")
    print("=" * 72)

    with mp.Pool(processes=24) as p:
        res = p.starmap(run_design, inputs)

    df = pd.DataFrame(res)
    df.to_csv(f"wind_PEM/surrogate_results_ss_rt_{shortfall}{RESULT_TAG}.csv")
