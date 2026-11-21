#################################################################################
# DISPATCHES was produced under the DOE Design Integration and Synthesis
# Platform to Advance Tightly Coupled Hybrid Energy Systems program (DISPATCHES),
# and is copyright (c) 2021 by the software owners: The Regents of the University
# of California, through Lawrence Berkeley National Laboratory, National
# Technology & Engineering Solutions of Sandia, LLC, Alliance for Sustainable
# Energy, LLC, Battelle Energy Alliance, LLC, University of Notre Dame du Lac, et
# al. All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. Both files are also available online at the URL:
# "https://github.com/gmlc-dispatches/dispatches".
#################################################################################

"""
Surrogate-assisted conceptual design for the nuclear (NE) case study, revised to
include behind-the-meter hydrogen conditioning (liquefaction), purchased terminal
handling and road transport, and a levelized PEM stack replacement reserve.

Every cost parameter is imported from the price-taker module
`price_taker_analysis_with_h2_costs_PEM_replacement`, so the two analyses cannot
drift apart. The surrogates themselves are unchanged: they are indexed by the PEM
bid B, and the bid is the one quantity the three methods share, so adding these
costs re-labels the price axis without retraining anything.

The hydrogen price rho^H2 is a price at the customer delivery point. For a bid B,

    rho^H2 = (B + phi_2^pem + varphi^rep) / (psi^eff (1 - eps^del)) + chi^del,

and the sweep below covers B = 15, 20, ..., 40 $/MWh, which is exactly the range
the neural networks were trained on (see `_surrogate_bid_bounds`).
"""

__author__ = "Radhakrishna Tumbalam Gooty"

# This code requires OMLT v1.0

import os
import json
import warnings
from tensorflow import keras

from pyomo.common.fileutils import this_file_dir
from pyomo.environ import (
    ConcreteModel,
    Var,
    NonNegativeReals,
    Expression,
    Constraint,
    Objective,
)

from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.solvers import get_solver

import omlt  # omlt can encode the neural networks in Pyomo
from omlt.neuralnet import FullSpaceNNFormulation
from omlt.io import load_keras_sequential

# Conditioning, delivery and stack replacement parameters are imported rather than
# restated, so that this analysis and the price-taker analysis stay consistent.
from dispatches.case_studies.nuclear_case.report.price_taker_analysis_with_h2_costs_PEM_replacement import (
    PSI_PEM,
    PSI_EFF,
    THETA_COND,
    CHI_DEL,
    EPS_DEL,
    KAPPA_DEL,
    SIGMA_STACK,
    LAMBDA_STACK,
    bid_to_price,
    price_to_bid,
    replacement_reserve,
    compute_replacement_metrics,
    case_tag,
)

PEM_CAPEX = 1200
LIFETIME = 30
TAX_RATE = 0.2                  # Corporate tax rate
DISC_RATE = 0.08                # Discount rate
r_def = 1 / (1 + DISC_RATE)
# Annualization factor for CAPEX of the system
ANN_FACTOR = (1/r_def) * ((1-r_def) / (1-r_def ** LIFETIME))

NP_CAPACITY = 400               # Capacity of the nuclear power plant (in MW)
H2_PROD_RATE = PSI_EFF          # Hydrogen yield per MWh sent to the train (in kg/MW-h)
NUM_HOURS = 8784                # Number of hours in a year

# Levelized PEM stack replacement reserve varphi^rep (in $/MWh of PEM energy).
# At PEM_CAPEX = 1200 $/kW this is 0.11 * 1200 / 40000 * 1000 = 3.30 $/MWh.
PHI_REP = replacement_reserve(PEM_CAPEX)

NPP_FOM = 13.7  # Normalized FOM = (120,000 / 8760) = $13.7 per MWh
NPP_VOM = 2.3   # Using a VOM cost of $2.3 per MWh for the nuclear power plant
PEM_FOM = 5.47  # Normalized FOM = (47,900 / 8760) = $5.47 per MWh
PEM_VOM = 0.0   # Using a VOM cost of $1.3 per MWh for the PEM electrolyzer

KW_TO_MW = 1e-3
HOURS_TO_S = 3600
MW_H2 = 2.016e-3

# path for folder that has surrogate models
surrogate_dir = os.path.join(this_file_dir(), "nn_steady_state")

# ================ Get input-output scaling factors for revenue ====================================
with open(os.path.join(surrogate_dir, "revenue_input_output_scaling_data.json"), 'rb') as f:
    revenue_data = json.load(f)

input_bounds_revenue = {
    i: (revenue_data['xmin'][i], revenue_data['xmax'][i])
    for i in range(len(revenue_data['xmin']))
}
scaling_object_revenue = omlt.OffsetScaling(
    offset_inputs=revenue_data["xm_inputs"],
    factor_inputs=revenue_data["xstd_inputs"],
    offset_outputs=revenue_data["y_mean"],
    factor_outputs=revenue_data["y_std"],
)

nn_revenue = keras.models.load_model(os.path.join(surrogate_dir, "keras_revenue"))
revenue_defn = load_keras_sequential(nn_revenue, scaling_object_revenue, input_bounds_revenue)

# ================ Get input-output scaling factors for revenue ====================================
with open(os.path.join(surrogate_dir, "cap_factor_input_output_scaling_data.json"), 'rb') as f:
    cap_factor_data = json.load(f)

input_bounds_cap_factor = {
    i: (cap_factor_data['xmin'][i], cap_factor_data['xmax'][i])
    for i in range(len(cap_factor_data['xmin']))
}
scaling_object_cap_factor = omlt.OffsetScaling(
    offset_inputs=cap_factor_data["xm_inputs"],
    factor_inputs=cap_factor_data["xstd_inputs"],
    offset_outputs=cap_factor_data["y_mean"],
    factor_outputs=cap_factor_data["y_std"],
)

nn_cap_factor = keras.models.load_model(os.path.join(surrogate_dir, "keras_cap_factor"))
cap_factor_defn = load_keras_sequential(
    nn_cap_factor,
    scaling_object_cap_factor,
    input_bounds_cap_factor,
)
# ====================================================================================================


def _surrogate_bid_bounds():
    """
    Range of the PEM bid, in $/MWh, that the surrogates were trained on.

    The first surrogate input is the threshold price, and the scaling data stores its
    bounds in standardized units, so they are mapped back here.
    """
    lo = revenue_data["xm_inputs"][0] + revenue_data["xmin"][0] * revenue_data["xstd_inputs"][0]
    hi = revenue_data["xm_inputs"][0] + revenue_data["xmax"][0] * revenue_data["xstd_inputs"][0]
    return lo, hi


def h2_price_range():
    """
    Delivered hydrogen prices [$/kg] corresponding to the surrogate's bid domain.
    """
    lo, hi = _surrogate_bid_bounds()
    return bid_to_price(lo, PHI_REP, vom_pem=PEM_VOM), bid_to_price(hi, PHI_REP, vom_pem=PEM_VOM)


def conceptual_design_ss_NE(reserve=10, max_lmp=500, H2_SELLING_PRICE=2):
    """
    Formulates the conceptual design problem for the nuclear case study

    Args:
        reserve: Percentage reserves
        max_lmp: Shortfall price
        H2_SELLING_PRICE: Hydrogen price at the customer delivery point [$/kg].
            This is no longer a plant-gate price: the plant pays chi^del per
            delivered kg for terminal handling and road transport, loses a fraction
            eps^del in transit, and liquefies its own product behind the meter.

    Returns:
        m: Optimization model embedding market surrogates
    """

    m = ConcreteModel()

    # Define a variable for the PEM capacity
    m.pem_capacity = Var(
        within=NonNegativeReals,
        doc="Capacity of the hydrogen train, i.e. electrolyzer plus liquefier [in MW]",
    )

    # Inputs to the frequency surrogates: PEM capacity/NP capacity, threshold price, reserve, max_lmp
    m.pem_np_cap_ratio = Var(
        within=NonNegativeReals,
        bounds=(0.05, 0.5),
        initialize=0.25,
        doc="Ratio of capacities of PEM and nuclear power plant",
    )
    m.threshold_price = Var(
        within=NonNegativeReals,
        initialize=20,
        doc="Threshold LMP below which selling H2 is more profitable",
    )
    m.reserve = Var(
        within=NonNegativeReals,
        doc="Percentage of reserves",
    )
    m.max_lmp = Var(
        within=NonNegativeReals,
        doc="Maximum LMP",
    )
    m.reserve.fix(reserve)
    m.max_lmp.fix(max_lmp)

    # Relation between PEM capacity and pem_np_cap_ratio
    m.pem_capacity_definition = Constraint(
        expr=m.pem_capacity == m.pem_np_cap_ratio * NP_CAPACITY
    )

    # Threshold price calculation. With conditioning, delivery and stack replacement,
    # the bid is affine in the hydrogen price rather than proportional to it:
    #     B = psi^eff (1 - eps^del) (rho^H2 - chi^del) - (phi_2^pem + varphi^rep).
    # The surrogates are indexed by this bid, so they need no retraining.
    pem_bid = price_to_bid(H2_SELLING_PRICE, PHI_REP, vom_pem=PEM_VOM)

    bid_lo, bid_hi = _surrogate_bid_bounds()
    if not (bid_lo - 1e-6 <= pem_bid <= bid_hi + 1e-6):
        warnings.warn(
            "PEM bid {0:.3f} $/MWh implied by a delivered hydrogen price of {1:.4f} $/kg "
            "lies outside the surrogate's training range [{2:.3f}, {3:.3f}] $/MWh. "
            "The networks will be extrapolating.".format(
                pem_bid, H2_SELLING_PRICE, bid_lo, bid_hi
            )
        )

    m.threshold_price.fix(pem_bid)

    # Add NN surrogate for revenue to the model
    m.nn_revenue = omlt.OmltBlock()
    m.nn_revenue.build_formulation(FullSpaceNNFormulation(revenue_defn))

    # Add NN surrogate for capacity factor to the model
    m.nn_cap_factor = omlt.OmltBlock()
    m.nn_cap_factor.build_formulation(FullSpaceNNFormulation(cap_factor_defn))

    inputs = [m.threshold_price, m.pem_np_cap_ratio, m.reserve, m.max_lmp]

    @m.Constraint(range(len(inputs)))
    def input_variable_equality_cap_factor(blk, i):
        return inputs[i] == blk.nn_cap_factor.inputs[i]

    @m.Constraint(range(len(inputs)))
    def input_variable_equality_revenue(blk, i):
        return inputs[i] == blk.nn_revenue.inputs[i]

    m.electricity_revenue = Var(within=NonNegativeReals, initialize=1e9)
    m.revenue_surrogate = Constraint(expr=m.electricity_revenue == m.nn_revenue.outputs[0])

    m.npp_capacity_factor = Var(within=NonNegativeReals, initialize=0.75)
    m.cap_factor_surrogate = Constraint(expr=m.npp_capacity_factor == m.nn_cap_factor.outputs[0])

    # Define flowsheet variables
    m.net_energy_to_pem = Var(
        within=NonNegativeReals,
        initialize=(0.25 * NP_CAPACITY * NUM_HOURS),
    )
    m.net_h2_production = Var(
        within=NonNegativeReals,
        initialize=(0.25 * NP_CAPACITY * NUM_HOURS * PSI_EFF),
    )

    # Calculate net power to PEM
    m.pem_energy_calculation = Constraint(
        expr=m.net_energy_to_pem <= (1 - m.npp_capacity_factor) * NP_CAPACITY * NUM_HOURS
    )

    # Calculate net hydrogen production. The liquefier draws theta^cond = 11 kWh/kg from
    # the same power stream, so the yield on the energy sent to the hydrogen train is
    # psi^eff = psi^pem / (1 + theta^cond psi^pem / 1000) rather than psi^pem.
    m.net_h2_prod_calculation = Constraint(
        expr=m.net_h2_production == m.net_energy_to_pem * PSI_EFF
    )

    # Mass delivered to the customer, after the delivery loss
    m.net_h2_delivered = Expression(expr=(1 - EPS_DEL) * m.net_h2_production)

    # Calculate net hydrogen revenue. rho^H2 is a delivered price, so both the revenue
    # and the delivery charge are levied on the delivered mass.
    m.h2_revenue = Expression(expr=H2_SELLING_PRICE * m.net_h2_delivered)
    m.h2_delivery_cost = Expression(expr=CHI_DEL * m.net_h2_delivered + KAPPA_DEL)

    # Levelized stack replacement reserve, a purely variable cost
    m.pem_replacement_cost = Expression(expr=PHI_REP * m.net_energy_to_pem)

    # assert degrees_of_freedom(m) == 1

    # Note: net_energy_to_pem is already an annual energy in MWh, so the PEM variable
    # costs are not multiplied by NUM_HOURS. Only the generator term is a rate per hour.
    m.total_operating_cost = Expression(
        expr=NUM_HOURS * NP_CAPACITY * NPP_VOM
        + (PEM_VOM + PHI_REP) * m.net_energy_to_pem
    )

    m.pem_cap_cost = Expression(expr=ANN_FACTOR * PEM_CAPEX * 1000 * m.pem_capacity)
    m.depreciation = Expression(expr=((PEM_CAPEX * 1000) / LIFETIME) * m.pem_capacity)
    m.pem_fom = Expression(expr=0.03 * (PEM_CAPEX * 1000) * m.pem_capacity)

    # Assuming FOM of NPP to be $120/kW
    m.npp_fom = Expression(expr=120 * 1000 * NP_CAPACITY)

    # Delivery and stack replacement are operating expenses, so both are deducted
    # before tax. Replacement already sits inside total_operating_cost.
    m.taxable_income = Expression(
        expr=m.electricity_revenue + m.h2_revenue - m.h2_delivery_cost
        - m.total_operating_cost - m.pem_fom - m.npp_fom - m.depreciation
    )

    # Nonnegative corporate tax: an unprofitable design pays no tax rather than
    # claiming a credit. The objective is minimized and increases with the tax, so
    # the tax settles at max(0, TAX_RATE * taxable_income).
    m.corporate_tax = Var(
        within=NonNegativeReals, initialize=0, doc="Corporate tax [in $]",
    )
    m.corporate_tax_calculation = Constraint(
        expr=m.corporate_tax >= TAX_RATE * m.taxable_income
    )

    m.net_profit = Expression(
        expr=m.electricity_revenue + m.h2_revenue - m.h2_delivery_cost
        - m.total_operating_cost - m.pem_fom - m.npp_fom - m.corporate_tax
    )

    # set objective functions in $
    m.obj = Objective(expr=m.pem_cap_cost - m.net_profit)

    return m


def _record_case(results, key, m, pem_capacity_mw, soln):
    """
    Stores one solved case, using the same field names as the price-taker sweep.
    """
    pem_energy_mwh = m.net_energy_to_pem.value
    h2_rev = m.h2_revenue.expr()
    delivery_cost = m.h2_delivery_cost.expr()

    results["elec_rev"][key] = m.electricity_revenue.value / 1e6
    results["h2_rev"][key] = h2_rev / 1e6
    results["h2_rev_net"][key] = (h2_rev - delivery_cost) / 1e6
    results["delivery_cost"][key] = delivery_cost / 1e6
    results["replacement_cost"][key] = m.pem_replacement_cost.expr() / 1e6
    results["h2_produced"][key] = m.net_h2_production.value
    results["h2_delivered"][key] = m.net_h2_delivered.expr()
    results["pem_energy_mwh"][key] = pem_energy_mwh
    results["corporate_tax"][key] = m.corporate_tax.value / 1e6
    results["net_npv"][key] = -m.obj.expr() / 1e6
    results["net_profit"][key] = m.net_profit.expr() / 1e6
    results["solver_stat"][key] = str(soln.solver.termination_condition)
    results["pem_cap_factor"][key] = pem_energy_mwh / (pem_capacity_mw * NUM_HOURS)
    results["opt_pem_bid"][key] = m.threshold_price.value
    results["replacement"][key] = compute_replacement_metrics(
        pem_energy_mwh=pem_energy_mwh,
        pem_capacity_mw=pem_capacity_mw,
        pem_capex=PEM_CAPEX,
        plant_life=LIFETIME,
        discount_rate=DISC_RATE,
        hours_per_year=NUM_HOURS,
    )


def run_exhaustive_enumeration(
    reserve,
    max_lmp,
    pem_bids=(15, 20, 25, 30, 35, 40),
    h2_price=None,
    output_dir=None,
):
    """
    Sweeps the PEM capacity ratio and the hydrogen price.

    The sweep is indexed by the PEM bid B [$/MWh], exactly as the price-taker sweep is,
    so the two methods are compared at the same bids. Each bid maps to a delivered
    hydrogen price through rho^H2 = (B + phi_2^pem + varphi^rep)/(psi^eff (1 - eps^del))
    + chi^del. The default bids span the surrogate's training range.

    Args:
        reserve: Percentage reserves
        max_lmp: Shortfall price
        pem_bids: PEM bids to sweep [$/MWh]
        h2_price: Delivered hydrogen prices [$/kg]; overrides `pem_bids` when given
        output_dir: Directory for the results json; defaults to a directory named after
            the market conditions and the cost assumptions. Nothing is written to the
            working directory.
    """
    if h2_price is None:
        h2_price = [
            bid_to_price(b, PHI_REP, vom_pem=PEM_VOM) for b in pem_bids
        ]
    pem_bids = [price_to_bid(p, PHI_REP, vom_pem=PEM_VOM) for p in h2_price]

    pem_cap = [i / 100 for i in range(5, 51, 5)]

    tag = case_tag(CHI_DEL, EPS_DEL, THETA_COND, KAPPA_DEL, PHI_REP)
    if output_dir is None:
        output_dir = "surrogate_ss_" + str(reserve) + "_" + str(max_lmp) + "_" + tag
    os.makedirs(output_dir, exist_ok=True)

    solver = get_solver()
    results = {
        "h2_price": h2_price, "pem_cap": pem_cap, "pem_cap_factor": {},
        "elec_rev": {}, "h2_rev": {}, "net_npv": {}, "solver_stat": {},
        "opt_pem_bid": {},
        "pem_bid": pem_bids, "net_profit": {},
        "h2_rev_net": {}, "delivery_cost": {}, "replacement_cost": {},
        "h2_produced": {}, "h2_delivered": {}, "pem_energy_mwh": {},
        "corporate_tax": {}, "replacement": {},
        "parameters": {
            "psi_pem_kg_per_mwh": PSI_PEM,
            "psi_eff_kg_per_mwh": PSI_EFF,
            "theta_cond_kwh_per_kg": THETA_COND,
            "chi_del_per_kg": CHI_DEL,
            "eps_del": EPS_DEL,
            "kappa_del_per_year": KAPPA_DEL,
            "sigma_stack": SIGMA_STACK,
            "lambda_stack_hours": LAMBDA_STACK,
            "phi_rep_per_mwh": PHI_REP,
            "pem_capex_per_kw": PEM_CAPEX,
            "npp_capacity_mw": NP_CAPACITY,
            "reserve": reserve,
            "max_lmp": max_lmp,
            "surrogate_bid_range": list(_surrogate_bid_bounds()),
        },
    }

    remaining_cases = []

    for idx1, hp in enumerate(h2_price):
        m = conceptual_design_ss_NE(reserve=reserve, max_lmp=max_lmp, H2_SELLING_PRICE=hp)

        m.pem_capacity.fix(pem_cap[0] * NP_CAPACITY)
        soln = solver.solve(m)

        for idx2, pc in enumerate(pem_cap):
            # The main purpose of this loop is to find a feasible solution
            m.pem_capacity.fix(pc * NP_CAPACITY)
            soln = solver.solve(m)

            if str(soln.solver.termination_condition) == "optimal":
                break

        unsolved_cases = []

        # m.feasibility_constraint = Constraint(
        #     expr=m.npp_capacity_factor >= 1 - m.pem_np_cap_ratio
        # )
        m.feasibility_constraint2 = Constraint(
            expr=m.net_energy_to_pem <= m.pem_capacity * NUM_HOURS
        )
        # m.feasibility_constraint.deactivate()

        for idx2, pc in enumerate(pem_cap):
            # This loop actually solves the problem and records the unsolved cases
            # m.threshold_price_definition.deactivate()
            # assert degrees_of_freedom(m) == 1

            print("Solving case: ", (idx1, idx2))
            m.pem_capacity.fix(pc * NP_CAPACITY)

            soln = solver.solve(m)

            # m.feasibility_constraint.activate()
            # soln = solver.solve(m)
            # m.feasibility_constraint.deactivate()

            if str(soln.solver.termination_condition) == "infeasible":
                unsolved_cases.append((idx2, pc))
                continue

            _record_case(results, str(idx1) + str(idx2), m, pc * NP_CAPACITY, soln)

        for idx2, pc in unsolved_cases:
            print("Solving case again: ", (idx1, idx2))
            m.pem_capacity.fix(pc * NP_CAPACITY)

            soln = solver.solve(m)

            if str(soln.solver.termination_condition) == "infeasible":
                remaining_cases.append((idx2, pc))
                continue

            _record_case(results, str(idx1) + str(idx2), m, pc * NP_CAPACITY, soln)

    print("Unsolved cases: ", remaining_cases)
    results["unsolved_cases"] = remaining_cases

    filename = os.path.join(
        output_dir,
        "ss_results_" + str(reserve) + "_" + str(max_lmp) + "_"
        + str(PEM_CAPEX) + "_" + tag + ".json",
    )
    with open(filename, "w") as fp:
        json.dump(results, fp, indent=4)

    print("Results written to " + os.path.abspath(filename))


if __name__ == '__main__':
    run_exhaustive_enumeration(reserve=15, max_lmp=500)
