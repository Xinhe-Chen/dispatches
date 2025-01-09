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
import numpy as np
import pandas as pd
import os
from pathlib import Path
from functools import partial
import json
from Backcaster_of_price_taker import PricetakerBackcaster, PriceBackcaster
from dispatches.case_studies.renewables_case.load_parameters import *
from wind_battery_LMP import wind_battery_variable_pairs, wind_battery_om_costs, initialize_mp, wind_battery_model, wind_battery_mp_block 
from idaes.apps.grid_integration.multiperiod.multiperiod import MultiPeriodModel
from pyomo.environ import (Constraint,
                           Var,
                           ConcreteModel,
                           Expression,
                           Param,
                           Objective,
                           maximize,
                           units as pyunits,
                           SolverFactory,
                           TransformationFactory,
                           Block,
                           NonNegativeReals,
                           Reference,
                           value)
from argparse import ArgumentParser
# from pyomo.util.infeasible import log_infeasible_constraints
# import logging

# _logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.DEBUG)
# this is for the new form of stochastic programming proposed in meeting Jan 6 2025.
# Stage 1 variables are the decisions of day 1 decisions. Stage 2 variables are (scenarios) of

usage = "Run wind-battery pricetaker optimization with uncertainty (rolling horizon)"
parser = ArgumentParser(usage)
parser.add_argument(
    "--battery_ratio",
    dest="battery_ratio",
    help="Indicate the battery ratio to the wind farm.",
    action="store",
    type=float,
    default=0.1,
)

parser.add_argument(
    "--duration",
    dest="duration",
    help="the battery duration hours",
    action="store",
    type=int,
    default=4,
)

parser.add_argument(
    "--scenario",
    dest="scenario",
    help="The number of scenarios in the stochastic optimization",
    action="store",
    type=int,
    default=3,
)

options = parser.parse_args()
battery_ratio = options.battery_ratio
duration = options.duration
scenario = options.scenario
horizon = 24
future_horizon = 48

def wind_battery_periodic_variable_pairs(m1, m2):
    """
    The final battery storage of charge must be the same as in the intial timestep. 

    Args:
        m1: final time block model
        m2: first time block model
    """
    pairs = [(m1.fs.battery.nameplate_power, m2.fs.battery.nameplate_power)]
    return pairs

def update_wind_capacity_factors(wind_capacity_factor, pointer, horizon, future_horizon, stage_1_model=True):
    """
    Update the wind capacity factor for the rolling horizion optimization
    """
    updated_wind_capacity_factors = {}
    if stage_1_model:
        for i, j in zip(range(0, horizon), range(pointer*24, pointer*24 + horizon)):
            updated_wind_capacity_factors[i] = wind_capacity_factor[j]
    else:
        if pointer*24 + horizon + future_horizon >= 366*24:
            overflow_hours = pointer*24 + horizon + future_horizon - 366*24
            for i, j in zip(range(0, future_horizon-overflow_hours), range(pointer*24 + horizon, 366*24)):
                updated_wind_capacity_factors[i] = wind_capacity_factor[j]
            for i in range(0, overflow_hours):
                updated_wind_capacity_factors[future_horizon-overflow_hours+i] = wind_capacity_factor[i]
        else:
            for i, j in zip(range(0, future_horizon), range(pointer*24 + horizon, pointer*24 + horizon + future_horizon)):
                updated_wind_capacity_factors[i] = wind_capacity_factor[j]
    
    return updated_wind_capacity_factors

def build_scenario_model(price_signals, input_params, backcaster, n_time_points=horizon, stage_1_model=True, verbose=False):
    """
    build scenario model for the wind_battery optimization problem

    Arguments:
        price_signals: list of price signals, given by the backcaster

    Return:
        scenario_model: pyomo model object
    """

    mp_wind_battery = MultiPeriodModel(
        n_time_points=n_time_points,
        process_model_func=partial(wind_battery_mp_block, input_params=input_params, verbose=verbose),
        linking_variable_func=wind_battery_variable_pairs,
        periodic_variable_func=wind_battery_periodic_variable_pairs,
    )
    stage_wind_resource = update_wind_capacity_factors(input_params["wind_resource"], backcaster.pointer, horizon=n_time_points, future_horizon=n_time_points, stage_1_model=stage_1_model)
    mp_wind_battery.build_multi_period_model(stage_wind_resource)

    scenario_model = mp_wind_battery.pyomo_model
    blks = mp_wind_battery.get_active_process_blocks()
    if stage_1_model:
        blks[0].fs.battery.initial_state_of_charge.fix(input_params["battery_soc"])
        # blks[0].fs.battery.initial_energy_throughput.fix(input_params["energy_throughput"])
    
    scenario_model.wind_system_capacity = Param(default=input_params['wind_mw'] * 1e3, units=pyunits.kW)
    scenario_model.battery_system_capacity = Param(default=input_params['batt_mw'] * 1e3, units=pyunits.kW)
    # scenario_model.battery_system_capacity.fix()
    # scenario_model.wind_system_capacity.fix()
    
    if input_params['design_opt']:
        for blk in blks:
            if not input_params['extant_wind']:
                blk.fs.windpower.system_capacity.unfix()
            blk.fs.battery.nameplate_power.unfix()
    
    scenario_model.wind_max_p = Constraint(mp_wind_battery.pyomo_model.TIME, rule=lambda b, t: blks[t].fs.windpower.system_capacity <= scenario_model.wind_system_capacity)
    scenario_model.battery_max_p = Constraint(mp_wind_battery.pyomo_model.TIME, rule=lambda b, t: blks[t].fs.battery.nameplate_power <= scenario_model.battery_system_capacity)
    
    for blk in blks:
        blk_wind = blk.fs.windpower
        blk_battery = blk.fs.battery
        
        blk_battery.op_total_cost = Expression(
            expr=scenario_model.battery_system_capacity * blk_battery.op_cost / 8784
        )

        blk.lmp_signal = Param(default=0, mutable=True)
        blk.elec_output = blk.fs.splitter.grid_elec[0] + blk_battery.elec_out[0]
        blk.revenue = (
            (blk.lmp_signal) * (blk.fs.splitter.grid_elec[0] + blk_battery.elec_out[0])
        )
        blk.profit = Expression(expr=blk.revenue 
                                         - blk_wind.op_total_cost
                                         - blk_battery.op_total_cost)
    for (i, blk) in enumerate(blks):
        blk.lmp_signal.set_value(price_signals[i]*1e-3 + 1e-3)

    scenario_model.wind_cap_cost = Param(default=wind_cap_cost, mutable=True)
    if input_params['extant_wind']:
        scenario_model.wind_cap_cost.set_value(0.0)
    
    scenario_model.batt_cap_cost_kw = Param(default=batt_cap_cost_kw, mutable=True)
    scenario_model.batt_cap_cost_kwh = Param(default=batt_cap_cost_kwh, mutable=True)

    scenario_model.total_elec_output = Expression(expr= sum([blk.elec_output for blk in blks]))
    scenario_model.elec_revenue = Expression(expr = sum([blk.revenue for blk in blks]))
    scenario_model.total_profit = Expression(expr=sum([blk.profit for blk in blks]))

    return mp_wind_battery


def build_sp_model(input_params, backcaster):
    """
    Build stochastic programming model
    """
    # build the stage 1 model.
    res_dict = {}
    price_signal = backcaster.generate_stage_1_price_signals()
    m = ConcreteModel()
    mp_wind_battery = build_scenario_model(price_signal, input_params, backcaster, n_time_points=horizon, verbose=False)
    setattr(m, "stage_1_model", mp_wind_battery.pyomo_model)

    pred_price = backcaster.generate_stage_2_price_signals()

    for i in range(scenario):
        mp_wind_battery_scenario = build_scenario_model(pred_price[i], input_params, backcaster, n_time_points=48, stage_1_model=False)
        setattr(m, f"scenario_{i}", mp_wind_battery_scenario.pyomo_model)
        
        # the inital soc of the stage 2 should equal to the end soc of stage 1
        scenario_model = getattr(m, f"scenario_{i}")
        # scenario_model = mp_wind_battery_scenario.pyomo_model
        setattr(m, f"Constraint_soc_{i}", Constraint(expr = m.stage_1_model.blocks[horizon-1].process.fs.battery.state_of_charge[0.0]==scenario_model.blocks[0].process.fs.battery.initial_state_of_charge))
        setattr(m, f"Constraint_throughput_{i}", Constraint(expr = m.stage_1_model.blocks[horizon-1].process.fs.battery.energy_throughput[0.0]==scenario_model.blocks[0].process.fs.battery.initial_energy_throughput))
        setattr(m, f"Constraint_nameplate_{i}", Constraint(expr = m.stage_1_model.blocks[horizon-1].process.fs.battery.nameplate_power==scenario_model.blocks[0].process.fs.battery.nameplate_power))
        
        # constraint of soc_init = soc_end
        setattr(m, f"Constraint_consistent_soc_{i}", Constraint(expr = m.stage_1_model.blocks[0].process.fs.battery.initial_state_of_charge==scenario_model.blocks[future_horizon-1].process.fs.battery.state_of_charge[0.0]))
    m.rev = Expression(expr=m.stage_1_model.elec_revenue + 1/(scenario) * sum(getattr(m, f"scenario_{i}").elec_revenue for i in range(scenario)))
    m.obj = Objective(expr=m.rev, sense=maximize)
    
    opt = SolverFactory("ipopt")
    soln = opt.solve(m, tee=True,  options={'tol': 1e-8})
    # infeasible = log_infeasible_constraints(m)
    
    # record results
    # res_dict['stage_1_start_soc'] = value(m.stage_1_model.blocks[0].process.fs.battery.initial_state_of_charge)
    res_dict['stage_1_end_soc'] = value(m.stage_1_model.blocks[horizon-1].process.fs.battery.state_of_charge[0.0])
    # res_dict['stage_2_start_soc'] = [value(getattr(m, f"scenario_{i}").blocks[0].process.fs.battery.initial_state_of_charge) for i in range(scenario)]
    # res_dict['stage_2_end_soc'] = [value(getattr(m, f"scenario_{i}").blocks[future_horizon-1].process.fs.battery.state_of_charge[0.0]) for i in range(scenario)]
    res_dict['stage_1_end_energy_throughput'] = value(m.stage_1_model.blocks[horizon-1].process.fs.battery.energy_throughput[0.0])
    res_dict['solver_stat'] = str(soln.solver.termination_condition)
    # res_dict['stage_1_elec_in'] = value(m.stage_1_model.blocks[0].process.fs.battery.elec_in[0.0])
    # res_dict['stage_1_elec_out'] = value(m.stage_1_model.blocks[0].process.fs.battery.elec_out[0.0])
    # res_dict['stage_1_init_et'] = value(m.stage_1_model.blocks[0].process.fs.battery.initial_energy_throughput)
    # res_dict['stage_1_step_0_et'] = value(m.stage_1_model.blocks[0].process.fs.battery.energy_throughput[0.0])
    res_dict['revenue'] = [value(m.stage_1_model.elec_revenue)]
    res_dict['wind_gen'] = [value(m.stage_1_model.blocks[i].process.fs.windpower.electricity[0]) * 1e-3 for i in range(horizon)]
    res_dict['batt_to_grid'] = [value(m.stage_1_model.blocks[i].process.fs.battery.elec_out[0]) * 1e-3 for i in range(horizon)]
    res_dict['wind_to_grid'] = [value(m.stage_1_model.blocks[i].process.fs.splitter.grid_elec[0]) * 1e-3 for i in range(horizon)]
    res_dict['wind_to_batt'] = [value(m.stage_1_model.blocks[i].process.fs.battery.elec_in[0]) * 1e-3 for i in range(horizon)]
    res_dict['total_elec_out'] = [value(m.stage_1_model.blocks[i].process.elec_output) * 1e-3 for i in range(horizon)]
    return res_dict

def build_rolling_horizon_model(input_params, backcaster, days=3):
    total_res_dict = {}
    for i in range(days):
        print(f"Solving for Day {i}...")
        backcaster.pointer = i
        res_dict = build_sp_model(input_params, backcaster)
        # update the battery soc and energy throughput
        input_params["battery_soc"] = res_dict['stage_1_end_soc']
        input_params["energy_throughput"] = res_dict['stage_1_end_energy_throughput']
        total_res_dict[i] = res_dict
    
    return total_res_dict

input_params = default_input_params.copy()
input_params["design_opt"] = False
input_params["extant_wind"] = True
input_params["wind_mw"] = 847
input_params["batt_mw"] = np.round(847*battery_ratio, 2)
input_params["tank_size"] = 0
# initial soc = 0 and energy thoughput
input_params["battery_soc"] = 0
input_params["energy_throughput"] = 0

# res_dict = build_sp_model(0, input_params)
# print(value(m.obj))
# print(res_dict)

lmps_df = pd.read_parquet(Path(__file__).parent / "data" / "303_LMPs_15_reserve_500_shortfall.parquet")
lmps = lmps_df['LMP'].values
lmps[lmps>500] = 500
signal = lmps # even we use rt lmp signals, we call it DA_LMPs to simplify the work.
pb = PriceBackcaster(signal, scenario=scenario, pointer=0, horizon=horizon, future_horizon=future_horizon)

total_res_dict = build_rolling_horizon_model(input_params, pb, days=366)
# print(total_res_dict)
# res_path = "test_wb_new_uncertainty_scenario3_24_48.json"
# with open(res_path, "w") as f:
#     json.dump(total_res_dict, f)

parent_path = "wind_battery_price_taker_uncertainty"
if not os.path.exists(parent_path):
    os.makedirs(parent_path)

scenario_path = os.path.join(parent_path, f"scenario_{scenario}")
if not os.path.exists(scenario_path):
    os.makedirs(scenario_path)

duration_path = os.path.join(scenario_path, f"duration_{duration}")
if not os.path.exists(duration_path):
    os.makedirs(duration_path)

res_path = os.path.join(duration_path, f"Wind_battery_pt_uncertainty_new_scenario_{scenario}_duration_{duration}_ratio_{battery_ratio}.json")
with open(res_path, "w") as f:
    json.dump(total_res_dict, f)