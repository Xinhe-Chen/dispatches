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

"""
Price-taker analysis for the nuclear (NE) case study, revised to include

  * behind-the-meter hydrogen conditioning (liquefaction),
  * purchased terminal handling and road transport of the delivered hydrogen, and
  * a levelized PEM stack replacement reserve.

The cost boundary and every parameter value are identical to the wind (WE)
case study, so that the two case studies sit on one basis:

  psi^pem   = 19.8708 kg/MWh   beginning-of-life electrolyzer yield
  theta^cond= 11 kWh/kg        liquefaction electricity, drawn behind the meter
  psi^eff   = 16.3066 kg/MWh   effective yield, psi^pem / (1 + theta^cond psi^pem / 1000)
  chi^del   = 0.50 $/kg        terminal handling (0.21) + road transport (0.29)
  eps^del   = 0.005            delivery loss fraction
  kappa^del = 0 $/yr           fixed annual delivery service expense
  sigma^stack = 0.11           stack replacement cost as a fraction of PEM capital
  Lambda^stack = 40000 h       stack lifetime in equivalent full-load hours
  varphi^rep = sigma^stack gamma^pem / Lambda^stack = 3.30 $/MWh at gamma^pem = 1200 $/kW

Omega^pem is the capacity of the hydrogen train (electrolyzer plus liquefier), so
the liquefier draw is inside the power sent to the train and the yield applied to
that power is psi^eff rather than psi^pem.

rho^H2 is a price at the customer delivery point, not at the plant gate.  The
plant receives a netback of (1 - eps^del)(rho^H2 - chi^del) per kg produced, and
the KKT dispatch rule gives the PEM bid

    B = psi^eff (1 - eps^del) (rho^H2 - chi^del) - (phi_2^pem + varphi^rep),

which is affine in rho^H2.  The sweep below is indexed by B so that it covers the
same bids as the original cost-free analysis (15, 20, ..., 40 $/MWh), which maps
to delivered prices of 1.6279 ... 3.1687 $/kg.
"""

__author__ = "Radhakrishna Tumbalam Gooty"

# General python imports
import argparse
import json
import os
from importlib import resources
from math import floor
from pathlib import Path
import pandas as pd

# Pyomo imports
from pyomo.environ import (
    Var,
    value,
    NonNegativeReals,
    Constraint,
    Expression,
    Objective,
    Param,
    maximize,
    SolverFactory,
)

# IDAES imports
from idaes.core import FlowsheetBlock
from idaes.apps.grid_integration import MultiPeriodModel
import idaes.logger as idaeslog


# ----------------------------------------------------------------------------
# Technology, conditioning, delivery and stack replacement parameters
# ----------------------------------------------------------------------------
NPP_CAPACITY = 400                          # Capacity of 121_NUCLEAR_1 in RTS-GMLC [MW]

PSI_PEM = 0.00275984 * 3600 / 500 * 1000    # Electrolyzer yield [kg H2/MWh] = 19.870848
THETA_COND = 11.0                           # Conditioning (liquefaction) power [kWh/kg H2]
CHI_DEL = 0.50                              # Delivery charge per delivered kg [$/kg H2]
EPS_DEL = 0.005                             # Delivery loss fraction [-]
KAPPA_DEL = 0.0                             # Fixed annual delivery service expense [$/yr]

SIGMA_STACK = 0.11                          # Stack replacement cost / installed PEM capital [-]
LAMBDA_STACK = 40000.0                      # Stack lifetime [equivalent full-load hours]

# Effective hydrogen yield after the liquefier draw [kg H2/MWh sent to the train]
PSI_EFF = PSI_PEM / (1 + THETA_COND * PSI_PEM / 1000)

# Retained for backward compatibility with scripts that import this name.
H2_PROD_RATE = PSI_EFF


def replacement_reserve(pem_capex):
    """
    Levelized PEM stack replacement reserve varphi^rep [$/MWh].

    Args:
        pem_capex: Overnight capital cost of the PEM electrolyzer [$/kW]
    """
    return SIGMA_STACK * (pem_capex * 1000) / LAMBDA_STACK


def bid_to_price(bid, phi_rep, vom_pem=0.0, chi_del=CHI_DEL, eps_del=EPS_DEL):
    """
    Delivered hydrogen price rho^H2 [$/kg] implied by a PEM bid B [$/MWh].
    """
    return (bid + vom_pem + phi_rep) / (PSI_EFF * (1 - eps_del)) + chi_del


def price_to_bid(h2_price, phi_rep, vom_pem=0.0, chi_del=CHI_DEL, eps_del=EPS_DEL):
    """
    PEM bid B [$/MWh] implied by a delivered hydrogen price rho^H2 [$/kg].
    """
    return PSI_EFF * (1 - eps_del) * (h2_price - chi_del) - vom_pem - phi_rep


def get_lmp_data(m, market="DA", filename="results"):
    """
    This function reads and appends LMP data to the model depending on the price-taker variant
    the user is interested in. There are four price-taker variants:
    V1: Uses day-ahead LMPs only
    V2: Uses real-time LMPs only
    V3: Uses max(day-ahead LMP, real-time LMP) at each hour
    V4: Two-step method: Uses both day-ahead and real-time LMPs

    Args:
        m: Pyomo model
        market: Price-taker variant: V1 -> "DA", V2 -> "RT", V3 -> "Max-DA-RT", V4 -> "DA-RT"
        filename: csv file containing V1 results. This is needed for V4.

    Returns:
        None
    """
    with resources.path(
        "dispatches.case_studies.nuclear_case.report",
        "rts_gmlc_15_500.csv"
    ) as p:
        path_to_file = Path(p).resolve()

    raw_data = pd.read_csv(path_to_file)

    if market == "DA-RT":
        price_all_1 = raw_data["LMP DA"].tolist()
        price_all_2 = raw_data["LMP"].tolist()

        # Read the optimal dispatch data obtained from Step 1
        dispatch_da_df = pd.read_csv(filename + "_schedule.csv")
        dispatch_da = dispatch_da_df["np_to_grid"].tolist()

        m.DISPATCH_DA = Param(
            m.set_period,
            initialize={t + 1: lmp for (t, lmp) in enumerate(dispatch_da)},
            doc="DA dispatch [in MW]",
        )

        m.LMP_DA = Param(
            m.set_period,
            initialize={t + 1: lmp for (t, lmp) in enumerate(price_all_1)},
            doc="Locational Marginal Prices [in $/MWh]",
        )

        m.LMP_RT = Param(
            m.set_period,
            initialize={t + 1: lmp for (t, lmp) in enumerate(price_all_2)},
            doc="Locational Marginal Prices [in $/MWh]",
        )

        return

    if market == "DA":
        price_all = raw_data["LMP DA"].tolist()

    elif market == "RT":
        price_all = raw_data["LMP"].tolist()

    elif market == "Max-DA-RT":
        price_all_1 = raw_data["LMP DA"].tolist()
        price_all_2 = raw_data["LMP"].tolist()
        price_all = [max([price_all_1[i], price_all_2[i]]) for i in range(len(price_all_1))]

    m.LMP = Param(
        m.set_period,
        initialize={t + 1: lmp for (t, lmp) in enumerate(price_all)},
        doc="Locational Marginal Prices [in $/MWh]",
    )


def build_ne_flowsheet(m):
    """
    Builds the flowsheet model for a baseload nuclear power plant integrated with
    a PEM electrolyzer, a storage tank, and a hydrogen turbine. For the analysis in the report,
    the storage tank and the hydrogen turbine are not included by setting their capacities to zero.
    """
    m.fs = FlowsheetBlock(dynamic=False)

    # Declare variables
    m.fs.np_power = Var(within=NonNegativeReals, doc="Power produced by the nuclear plant (MW)")
    m.fs.np_to_grid = Var(within=NonNegativeReals, doc="Power from NP to the grid (MW)")
    m.fs.np_to_electrolyzer = Var(
        within=NonNegativeReals,
        doc="Power from NP to the hydrogen train, i.e. electrolyzer plus liquefier (MW)",
    )

    m.fs.h2_production = Var(within=NonNegativeReals, doc="Hydrogen production rate (kg/hr)")

    m.fs.tank_holdup = Var(within=NonNegativeReals, doc="Hydrogen holdup in the tank (kg)")
    m.fs.tank_holdup_previous = Var(within=NonNegativeReals, doc="Hold at the beginning of the period (kg)")
    m.fs.h2_to_pipeline = Var(within=NonNegativeReals, doc="Hydrogen flowrate to the pipeline (kg/hr)")
    m.fs.h2_to_turbine = Var(within=NonNegativeReals, doc="Hydrogen flowrate to the turbine (kg/hr)")

    m.fs.h2_turbine_power = Var(within=NonNegativeReals, doc="Power production from H2 turbine (MW)")
    m.fs.net_power = Var(within=NonNegativeReals, doc="Net power to the grid (MW)")

    # Capacity of the nuclear power plant 121_NUCLEAR_1 in RTS GMLC dataset = 400 MW
    m.fs.np_power.fix(NPP_CAPACITY)

    # Declare Constraints
    m.fs.np_power_balance = Constraint(
        expr=m.fs.np_power == m.fs.np_to_grid + m.fs.np_to_electrolyzer,
        doc="Power balance at the nuclear power plant",
    )
    # Compute the hydrogen production rate.
    # The electrolyzer produces psi^pem = 19.87 kg/MWh (NEL PEM design, ~50 kWh/kg), and the
    # liquefier consumes theta^cond = 11 kWh/kg of the same power stream, so the power sent to
    # the hydrogen train yields psi^eff = psi^pem / (1 + theta^cond psi^pem / 1000) = 16.31 kg/MWh.
    m.fs.calc_h2_production_rate = Constraint(
        expr=m.fs.h2_production == PSI_EFF * m.fs.np_to_electrolyzer,
        doc="Computes the hydrogen production rate net of the conditioning power",
    )
    # Tank holdup calculations (Assuming a delta_t of 1 hr)
    m.fs.tank_mass_balance = Constraint(
        expr=m.fs.tank_holdup - m.fs.tank_holdup_previous ==
        (m.fs.h2_production - m.fs.h2_to_pipeline - m.fs.h2_to_turbine)
    )

    # Compute the power production via h2 turbine
    # For an air_h2_ratio of 10.76, (T, P) of h2 = (300 K, 1 atm),
    # delta_p across compressor and turbine 24.1 bar, the conversion
    # factor is 0.0125 MW-hr/kg hydrogen
    m.fs.calc_turbine_power = Constraint(
        expr=m.fs.h2_turbine_power == 0.0125 * m.fs.h2_to_turbine,
        doc="Computes the power production via h2 turbine",
    )
    # Power balance at the grid
    m.fs.grid_power_balance = Constraint(
        expr=m.fs.net_power == m.fs.np_to_grid + m.fs.h2_turbine_power
    )

    return m


def get_linking_variable_pairs(t1, t2):
    return [
        (t1.fs.tank_holdup, t2.fs.tank_holdup_previous)
    ]


def build_deterministic_model(demand_type="variable", demand=None):
    """
    Formulates the multiperiod price-taker problem for the nuclear case study

    Args:
        demand_type: "fixed" or "variable" demand for hydrogen
        demand: Hydrogen demand [kg/hr]. Defaults to the largest rate the plant can
            sustain, NPP_CAPACITY * psi^eff, which leaves the bound non-binding.
    """
    if demand is None:
        demand = NPP_CAPACITY * PSI_EFF

    m = MultiPeriodModel(
        n_time_points=366 * 24,
        process_model_func=build_ne_flowsheet,
        linking_variable_func=get_linking_variable_pairs,
        use_stochastic_build=True,
        outlvl=idaeslog.WARNING,
    )

    # Declare first-stage variables
    m.pem_capacity = Var(within=NonNegativeReals, doc="Maximum capacity of the hydrogen train (in MW)")
    m.tank_capacity = Var(within=NonNegativeReals, doc="Maximum holdup of the tank (in kg)")
    m.h2_turbine_capacity = Var(within=NonNegativeReals, doc="Maximum power output from the turbine (in MW)")

    @m.Constraint(m.set_period, doc="Power input to PEM cannot exceed its capacity")
    def pem_capacity_constraint(blk, t):
        return blk.period[t].fs.np_to_electrolyzer <= blk.pem_capacity

    @m.Constraint(m.set_period, doc="Tank holdup cannot exceed its capacity")
    def tank_capacity_constraint(blk, t):
        return blk.period[t].fs.tank_holdup <= blk.tank_capacity

    @m.Constraint(m.set_period, doc="Power output from turbine cannot exceed its capacity")
    def turbine_capacity_constraint(blk, t):
        return blk.period[t].fs.h2_turbine_power <= blk.h2_turbine_capacity

    for t in m.set_period:
        if demand_type == "fixed":
            m.period[t].fs.h2_to_pipeline.fix(demand)

        elif demand_type == "variable":
            m.period[t].fs.h2_to_pipeline.setub(demand)

    return m


def append_op_costs_revenue(
    m,
    lmp,
    h2_price,
    vom_npp=2.3,
    vom_pem=1.3,
    vom_turbine=4.25,
    phi_rep=0.0,
    chi_del=CHI_DEL,
    eps_del=EPS_DEL,
    lmp_da=None,
    dispatch_da=None,
):
    """
    Appends operating costs and revenue generated at each hour.

    Args:
        lmp: Locational marginal price in this hour [$/MWh]
        h2_price: Hydrogen price at the customer delivery point [$/kg]
        vom_npp: Variable O&M cost of the nuclear generator [$/MWh]
        vom_pem: Variable O&M cost of the hydrogen train [$/MWh]
        vom_turbine: Variable O&M cost of the hydrogen turbine [$/MWh]
        phi_rep: Levelized stack replacement reserve varphi^rep [$/MWh of PEM energy]
        chi_del: Delivery charge per delivered kg [$/kg]
        eps_del: Delivery loss fraction [-]
    """
    # Stack replacement enters as a purely variable cost, because one replacement costs
    # sigma^stack gamma^pem Omega^pem and is consumed over Lambda^stack equivalent
    # full-load hours, so the design capacity cancels.
    m.fs.pem_replacement_cost = Expression(
        expr=phi_rep * m.fs.np_to_electrolyzer,
        doc="Levelized PEM stack replacement reserve in this hour [$]",
    )

    # Total variable O&M cost, including the replacement reserve
    m.fs.vom = Expression(
        expr=(vom_pem + phi_rep) * m.fs.np_to_electrolyzer
        + vom_turbine * m.fs.h2_turbine_power
        + vom_npp * m.fs.np_power
    )
    if lmp_da is None:
        m.fs.electricity_revenue = Expression(expr=lmp * m.fs.net_power)

    else:
        m.fs.electricity_revenue = Expression(
            expr=lmp_da * dispatch_da + lmp * (m.fs.net_power - dispatch_da)
        )

    # Mass delivered to the customer, after the delivery loss (e.g. boil-off in transit)
    m.fs.h2_delivered = Expression(expr=(1 - eps_del) * m.fs.h2_to_pipeline)

    # rho^H2 is a delivered price, so revenue is charged on the delivered mass, and the
    # terminal handling and road transport charge is levied on the same mass.
    m.fs.h2_revenue = Expression(expr=h2_price * m.fs.h2_delivered)
    m.fs.h2_delivery_cost = Expression(expr=chi_del * m.fs.h2_delivered)

    m.fs.net_cash_inflow = Expression(
        expr=m.fs.h2_revenue - m.fs.h2_delivery_cost + m.fs.electricity_revenue - m.fs.vom
    )


def append_npv_calculations(
    m,
    plant_life=30,
    tax_rate=0.2,
    discount_rate=0.08,
    capex_pem=1630,
    capex_tank=29,
    capex_turbine=947,
    fom_pem=47.9,
    fom_turbine=7,
    kappa_del=KAPPA_DEL,
):
    """
    Appends overall NPV calculations.
    """
    # Note: The capex and fom numbers for pem and turbine are in $/kW
    # and the capex of the tank is in $/kWh
    # Note: LHV of hydrogen is 33.3 kWh/kg
    m.pem_capex = Expression(expr=capex_pem * 1000 * m.pem_capacity)
    m.tank_capex = Expression(expr=(capex_tank * 33.3) * m.tank_capacity)
    m.turbine_capex = Expression(expr=capex_turbine * 1000 * m.h2_turbine_capacity)
    m.pem_fom = Expression(expr=1000 * fom_pem * m.pem_capacity)
    m.turbine_fom = Expression(expr=1000 * fom_turbine * m.h2_turbine_capacity)

    m.capex = Expression(
        expr=m.pem_capex + m.tank_capex + m.turbine_capex,
        doc="Total capital cost (in USD)"
    )

    m.npp_fom = Expression(expr=120 * 1000 * NPP_CAPACITY)

    m.fixed_om_cost = Expression(
        expr=m.pem_fom + m.turbine_fom + m.npp_fom,
        doc="Fixed O&M Cost (in USD)"
    )

    # Hourly net cash inflow is already net of the delivery charge and the replacement
    # reserve, both of which are operating expenses and are deducted before tax.
    m.total_cash_inflow = Expression(
        expr=sum(m.period[t].fs.net_cash_inflow for t in m.set_period),
        doc="Computes the total cash inflow (in USD)"
    )

    m.h2_delivery_cost = Expression(
        expr=sum(m.period[t].fs.h2_delivery_cost for t in m.set_period) + kappa_del,
        doc="Annual hydrogen conditioning, handling and transport cost (in USD)",
    )

    m.pem_replacement_cost = Expression(
        expr=sum(m.period[t].fs.pem_replacement_cost for t in m.set_period),
        doc="Annual levelized PEM stack replacement reserve (in USD)",
    )

    m.fixed_delivery_cost = Param(
        initialize=kappa_del, mutable=True,
        doc="Fixed annual delivery service expense kappa^del (in USD)",
    )

    m.depreciation = Expression(
        expr=m.capex / plant_life,
        doc="Straight-line depreciation with zero salvage value",
    )

    m.taxable_income = Expression(
        expr=m.total_cash_inflow - m.fixed_om_cost - m.fixed_delivery_cost - m.depreciation,
        doc="EREV + HREV - FOM - VOM - DCOST - DEP (in USD)",
    )

    # Nonnegative corporate tax: an unprofitable design pays no tax rather than
    # claiming a credit. Since a higher tax lowers the NPV, one of the two bounds
    # is active at the optimal solution, and the formulation stays linear.
    m.corporate_tax = Var(
        within=NonNegativeReals, doc="Corporate tax (in USD)",
    )

    m.corporate_tax_calculation = Constraint(
        expr=m.corporate_tax >= tax_rate * m.taxable_income,
        doc="Corporate tax is at least tax_rate times the taxable income",
    )

    m.net_profit = Expression(
        expr=m.total_cash_inflow - m.fixed_om_cost - m.fixed_delivery_cost - m.corporate_tax
    )

    # Factor for constant cash flow
    m.constant_cf_factor = (1 - (1 + discount_rate) ** (- plant_life)) / discount_rate


def append_objective_function(m):
    m.npv = Objective(
        expr=m.constant_cf_factor * m.net_profit - m.capex,
        sense=maximize
    )


def append_annualized_objective_function(m):
    m.npv = Objective(
        expr=m.net_profit - (1 / m.constant_cf_factor) * m.capex,
        sense=maximize
    )


def compute_replacement_metrics(
    pem_energy_mwh,
    pem_capacity_mw,
    pem_capex,
    plant_life=30,
    discount_rate=0.08,
    hours_per_year=366 * 24,
):
    """
    Post-processes the stack replacement schedule implied by an optimal dispatch.

    The reserve varphi^rep levelizes the replacement outlays without discounting, so it
    is an upper bound on their present cost. The discount correction reported here
    places each replacement at the year in which the stack reaches Lambda^stack
    equivalent full-load hours, discounts it, and re-annualizes it over the plant life.

    Args:
        pem_energy_mwh: Annual energy sent to the hydrogen train [MWh]
        pem_capacity_mw: Capacity of the hydrogen train [MW]
        pem_capex: Overnight capital cost of the PEM electrolyzer [$/kW]

    Returns:
        A dictionary of replacement diagnostics.
    """
    phi_rep = replacement_reserve(pem_capex)
    annual_cost = phi_rep * pem_energy_mwh

    if pem_capacity_mw <= 0 or pem_energy_mwh <= 0:
        return {
            "stack_life_hours": LAMBDA_STACK,
            "stack_repl_frac": SIGMA_STACK,
            "phi_rep_per_mwh": phi_rep,
            "pem_energy_mwh": pem_energy_mwh,
            "annual_replacement_cost": annual_cost,
            "eflh_hours": 0.0,
            "pem_capacity_factor": 0.0,
            "replacement_interval_years": None,
            "n_replacements": 0,
            "replacement_years": [],
            "kappa_discount": 0.0,
            "discounted_annual_replacement_cost": 0.0,
        }

    eflh = pem_energy_mwh / pem_capacity_mw
    interval = LAMBDA_STACK / eflh
    n_repl = int(floor(plant_life / interval))
    years = [(k + 1) * interval for k in range(n_repl)]

    one_replacement = SIGMA_STACK * (pem_capex * 1000) * pem_capacity_mw
    discounted_total = one_replacement * sum(
        (1 + discount_rate) ** (-t) for t in years
    )
    annualization_factor = (1 - (1 + discount_rate) ** (-plant_life)) / discount_rate
    discounted_annual = discounted_total / annualization_factor

    return {
        "stack_life_hours": LAMBDA_STACK,
        "stack_repl_frac": SIGMA_STACK,
        "phi_rep_per_mwh": phi_rep,
        "pem_energy_mwh": pem_energy_mwh,
        "annual_replacement_cost": annual_cost,
        "eflh_hours": eflh,
        "pem_capacity_factor": eflh / hours_per_year,
        "replacement_interval_years": interval,
        "n_replacements": n_repl,
        "replacement_years": years,
        "kappa_discount": discounted_annual / annual_cost if annual_cost else 0.0,
        "discounted_annual_replacement_cost": discounted_annual,
    }


def _write_results(m, filename="results"):
    # Create a directory to store NPV optimization results
    _filename = filename + "_schedule.csv"
    set_flowsheets = m.period[:].fs
    LMP = m.LMP if hasattr(m, "LMP") else m.LMP_DA

    results = {
        "LMP [$/MWh]": [LMP[t] for t in m.set_period],
        "np_to_grid": [value(fs.np_to_grid) for fs in set_flowsheets],
        "np_to_electrolyzer": [value(fs.np_to_electrolyzer) for fs in set_flowsheets],

        "tank_holdup_previous": [value(fs.tank_holdup_previous) for fs in set_flowsheets],
        "tank_holdup": [value(fs.tank_holdup) for fs in set_flowsheets],
        "h2_to_pipeline": [value(fs.h2_to_pipeline) for fs in set_flowsheets],
        "h2_delivered": [value(fs.h2_delivered) for fs in set_flowsheets],
        "h2_to_turbine": [value(fs.h2_to_turbine) for fs in set_flowsheets],
        "h2_turbine_power": [value(fs.h2_turbine_power) for fs in set_flowsheets],

        "h2_revenue": [value(fs.h2_revenue) for fs in set_flowsheets],
        "h2_delivery_cost": [value(fs.h2_delivery_cost) for fs in set_flowsheets],
        "pem_replacement_cost": [value(fs.pem_replacement_cost) for fs in set_flowsheets],
        "electricity_revenue": [value(fs.electricity_revenue) for fs in set_flowsheets],
        # "electricity_cost": [value(fs.electricity_cost) for fs in set_flowsheets],
        "vom": [value(fs.vom) for fs in set_flowsheets],
        "net_cash_inflow": [value(fs.net_cash_inflow) for fs in set_flowsheets],
    }

    results_df = pd.DataFrame(results)
    results_df.to_csv(_filename)


def _fmt(x):
    """Formats a float without a trailing .0, to match the WE output file names."""
    return str(int(x)) if float(x).is_integer() else str(x)


def case_tag(chi_del, eps_del, theta_cond, kappa_del, phi_rep):
    """Names a sweep after its cost assumptions, as in the wind case study."""
    return (
        "cd" + _fmt(chi_del)
        + "_loss" + _fmt(eps_del)
        + "_cond" + _fmt(theta_cond)
        + "_kfix" + _fmt(kappa_del)
        + "_rep" + _fmt(round(phi_rep, 2))
    )


# Retained for backward compatibility with the earlier private name.
_case_tag = case_tag


def run_exhaustive_enumeration(
    pem_capex=400,
    market="DA",
    pem_bids=(15, 20, 25, 30, 35, 40),
    h2_price=None,
    chi_del=CHI_DEL,
    eps_del=EPS_DEL,
    kappa_del=KAPPA_DEL,
    output_dir=None,
    step1_dir=None,
):
    """
    Performs sensitivity analysis with respect to the ratio of capacities of PEM and
    electrolyzer and the selling price of hydrogen.

    The sweep is indexed by the PEM bid B [$/MWh] so that it covers the same bids as
    the original cost-free analysis. Each bid is mapped to a delivered hydrogen price
    through rho^H2 = (B + phi_2^pem + varphi^rep) / (psi^eff (1 - eps^del)) + chi^del.
    Pass `h2_price` explicitly to sweep delivered prices directly instead.

    Args:
        pem_capex: Overnight capital cost of the PEM electrolyzer [$/kW]
        market: Price-taker variant, "DA", "RT", "Max-DA-RT" or "DA-RT"
        pem_bids: PEM bids to sweep [$/MWh]
        h2_price: Delivered hydrogen prices [$/kg]; overrides `pem_bids` when given
        chi_del: Delivery charge per delivered kg [$/kg]
        eps_del: Delivery loss fraction [-]
        kappa_del: Fixed annual delivery service expense [$/yr]
        output_dir: Directory for the hourly schedules and the summary json; defaults
            to a directory named after the market and the cost assumptions, e.g.
            "DA-RT_cd0.5_loss0.005_cond11_kfix0_rep3.3". Nothing is written to the
            working directory.
        step1_dir: Directory holding the step-1 (DA) schedules read by the "DA-RT"
            variant; defaults to the "DA" directory for the same cost assumptions, so
            the "DA" sweep must have been run first with the same arguments
    """
    vom_pem = 0
    phi_rep = replacement_reserve(pem_capex)

    if h2_price is None:
        h2_price = [
            bid_to_price(b, phi_rep, vom_pem=vom_pem, chi_del=chi_del, eps_del=eps_del)
            for b in pem_bids
        ]
    pem_bids = [
        price_to_bid(p, phi_rep, vom_pem=vom_pem, chi_del=chi_del, eps_del=eps_del)
        for p in h2_price
    ]

    pem_capacity = [i / 100 for i in range(5, 51, 5)]

    h2_demand = NPP_CAPACITY * PSI_EFF
    demand_type = "variable"

    tag = case_tag(chi_del, eps_del, THETA_COND, kappa_del, phi_rep)
    if output_dir is None:
        output_dir = market + "_" + tag
    os.makedirs(output_dir, exist_ok=True)

    # The two-step variant reads the day-ahead dispatch produced by the "DA" sweep.
    if step1_dir is None:
        step1_dir = "DA_" + tag
    if market == "DA-RT" and not os.path.isdir(step1_dir):
        raise FileNotFoundError(
            "The DA-RT variant needs the step-1 schedules in '" + step1_dir + "'. "
            "Run the sweep with market='DA' first, or pass step1_dir explicitly."
        )

    solver = SolverFactory("gurobi")
    results = {
        "h2_price": h2_price, "pem_cap": pem_capacity, "solver_stat": {},
        "elec_rev": {}, "h2_rev": {}, "net_npv": {}, "net_profit": {},
        "pem_cap_factor": {},
        "pem_bid": pem_bids,
        "h2_rev_net": {}, "delivery_cost": {}, "replacement_cost": {},
        "h2_produced": {}, "h2_delivered": {}, "pem_energy_mwh": {},
        "corporate_tax": {}, "replacement": {},
        "parameters": {
            "psi_pem_kg_per_mwh": PSI_PEM,
            "psi_eff_kg_per_mwh": PSI_EFF,
            "theta_cond_kwh_per_kg": THETA_COND,
            "chi_del_per_kg": chi_del,
            "eps_del": eps_del,
            "kappa_del_per_year": kappa_del,
            "sigma_stack": SIGMA_STACK,
            "lambda_stack_hours": LAMBDA_STACK,
            "phi_rep_per_mwh": phi_rep,
            "pem_capex_per_kw": pem_capex,
            "npp_capacity_mw": NPP_CAPACITY,
            "market": market,
        },
    }

    for idx1, hp in enumerate(h2_price):
        for idx2, pc in enumerate(pem_capacity):
            m = build_deterministic_model(demand_type=demand_type, demand=h2_demand)

            # Set tank  and turbine capacity to zero to prevent building one
            m.period[1].fs.tank_holdup_previous.fix(0)
            m.tank_capacity.fix(0)
            m.h2_turbine_capacity.fix(0)

            case_label = "case_" + str(idx1) + "_" + str(idx2)
            case_name = os.path.join(output_dir, case_label)

            get_lmp_data(
                m, market=market, filename=os.path.join(step1_dir, case_label)
            )

            if market in ["DA", "RT", "Max-DA-RT"]:
                for t in m.set_period:
                    append_op_costs_revenue(
                        m=m.period[t], lmp=m.LMP[t], h2_price=hp, vom_pem=vom_pem,
                        phi_rep=phi_rep, chi_del=chi_del, eps_del=eps_del,
                    )

            elif market == "DA-RT":
                for t in m.set_period:
                    append_op_costs_revenue(
                        m=m.period[t], lmp=m.LMP_RT[t], h2_price=hp, vom_pem=vom_pem,
                        phi_rep=phi_rep, chi_del=chi_del, eps_del=eps_del,
                        lmp_da=m.LMP_DA[t], dispatch_da=m.DISPATCH_DA[t],
                    )

            append_npv_calculations(
                m, capex_pem=pem_capex, fom_pem=0.03 * pem_capex, kappa_del=kappa_del,
            )
            append_annualized_objective_function(m)

            n_hours = len([t for t in m.period])

            print("Solving case number: ", (idx1, idx2))
            m.pem_capacity.fix(pc * NPP_CAPACITY)
            soln = solver.solve(m)

            elec_rev = sum(value(m.period[t].fs.electricity_revenue) for t in m.set_period)
            h2_rev = sum(value(m.period[t].fs.h2_revenue) for t in m.set_period)
            delivery_cost = value(m.h2_delivery_cost)
            replacement_cost = value(m.pem_replacement_cost)
            pem_energy_mwh = sum(
                value(m.period[t].fs.np_to_electrolyzer) for t in m.set_period
            )
            h2_produced = sum(value(m.period[t].fs.h2_production) for t in m.set_period)
            h2_delivered = sum(value(m.period[t].fs.h2_delivered) for t in m.set_period)
            pem_cap_factor = pem_energy_mwh / (pc * NPP_CAPACITY * n_hours)

            key = str(idx1) + str(idx2)
            results["elec_rev"][key] = elec_rev / 1e6
            results["h2_rev"][key] = h2_rev / 1e6
            results["h2_rev_net"][key] = (h2_rev - delivery_cost) / 1e6
            results["delivery_cost"][key] = delivery_cost / 1e6
            results["replacement_cost"][key] = replacement_cost / 1e6
            results["h2_produced"][key] = h2_produced
            results["h2_delivered"][key] = h2_delivered
            results["pem_energy_mwh"][key] = pem_energy_mwh
            results["corporate_tax"][key] = value(m.corporate_tax) / 1e6
            results["net_npv"][key] = value(m.npv) / 1e6
            results["net_profit"][key] = value(m.net_profit) / 1e6
            results["solver_stat"][key] = str(soln.solver.termination_condition)
            results["pem_cap_factor"][key] = pem_cap_factor
            results["replacement"][key] = compute_replacement_metrics(
                pem_energy_mwh=pem_energy_mwh,
                pem_capacity_mw=pc * NPP_CAPACITY,
                pem_capex=pem_capex,
            )

            _write_results(m, filename=case_name)

    results["capex"] = value(m.capex) / 1e6
    results["fom"] = value(m.fixed_om_cost) / 1e6

    # The summary json is written alongside the hourly schedules, in the folder named
    # after the market and the cost assumptions, so that nothing lands in the working
    # directory and the four price-taker variants never overwrite one another.
    filename = os.path.join(
        output_dir,
        "price_taker_pem_" + str(pem_capex) + "_" + market + "_" + tag + ".json",
    )
    with open(filename, "w") as fp:
        json.dump(results, fp, indent=4)

    print("Results written to " + os.path.abspath(filename))


def _parse_args(argv=None):
    """
    Command-line interface, so that a cluster job can run one market variant.

    theta^cond, sigma^stack and Lambda^stack are module constants rather than
    arguments, because `market_surrogates_with_h2_costs_PEM_replacement` imports them
    from here: edit them in one place and both analyses follow. varphi^rep is derived
    from --pem_capex, so it tracks the capital cost automatically.
    """
    parser = argparse.ArgumentParser(
        description="Price-taker sweep for the NE case study, with hydrogen "
                    "conditioning, delivery and PEM stack replacement costs.",
    )
    parser.add_argument(
        "--market", default="DA-RT", choices=["DA", "RT", "Max-DA-RT", "DA-RT"],
        help="Price-taker variant (default: DA-RT)",
    )
    parser.add_argument(
        "--pem_capex", type=float, default=1200,
        help="Overnight capital cost of the PEM electrolyzer [$/kW] (default: 1200)",
    )
    parser.add_argument(
        "--bids", type=float, nargs="+", default=[15, 20, 25, 30, 35, 40],
        help="PEM bids to sweep [$/MWh] (default: 15 20 25 30 35 40)",
    )
    parser.add_argument(
        "--h2_price", type=float, nargs="+", default=None,
        help="Delivered hydrogen prices [$/kg]; overrides --bids when given",
    )
    parser.add_argument(
        "--delivery_cost", type=float, default=CHI_DEL,
        help="chi^del, delivery charge per delivered kg [$/kg] "
             "(default: %(default)s)",
    )
    parser.add_argument(
        "--delivery_loss", type=float, default=EPS_DEL,
        help="eps^del, delivery loss fraction [-] (default: %(default)s)",
    )
    parser.add_argument(
        "--delivery_fixed", type=float, default=KAPPA_DEL,
        help="kappa^del, fixed annual delivery service expense [$/yr] "
             "(default: %(default)s)",
    )
    parser.add_argument(
        "--output_dir", default=None,
        help="Directory for the schedules and the results json "
             "(default: <market>_<cost tag>)",
    )
    parser.add_argument(
        "--step1_dir", default=None,
        help="Directory holding the step-1 day-ahead schedules read by DA-RT "
             "(default: DA_<cost tag>)",
    )
    return parser.parse_args(argv)


if __name__ == '__main__':
    args = _parse_args()

    # Keep integral capital costs out of the file names as "1200" rather than "1200.0"
    pem_capex_arg = (
        int(args.pem_capex) if float(args.pem_capex).is_integer() else args.pem_capex
    )

    run_exhaustive_enumeration(
        pem_capex=pem_capex_arg,
        market=args.market,
        pem_bids=args.bids,
        h2_price=args.h2_price,
        chi_del=args.delivery_cost,
        eps_del=args.delivery_loss,
        kappa_del=args.delivery_fixed,
        output_dir=args.output_dir,
        step1_dir=args.step1_dir,
    )
