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
from numbers import Real
from pathlib import Path
from functools import partial
from Backcaster_of_price_taker import PricetakerBackcaster
from dispatches.case_studies.renewables_case.load_parameters import *
from wind_battery_LMP import wind_battery_variable_pairs, wind_battery_periodic_variable_pairs, wind_battery_om_costs, initialize_mp, wind_battery_model, wind_battery_mp_block 
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
                           NonNegativeReals,
                           Reference,
                           value)

# this is for the stochastic programming. x scenarios with 36 hours planning horizon and 24 hours are realized.
# Currently we pause this work because we have problem specifying the initial soc of the coming day in the rolling horizion optimization.

lmps_df = pd.read_parquet(Path(__file__).parent / "data" / "303_LMPs_15_reserve_500_shortfall.parquet")
lmps = lmps_df['LMP'].values
lmps[lmps>500] = 500
signal = lmps # even we use rt lmp signals, we call it DA_LMPs to simplify the work.

scenario = 3
horizon = 24
planning_horizon = 36

backcaster = PricetakerBackcaster(price_signal=signal, scenario=scenario)

def update_wind_capacity_factors(input_param, pointer, planning_horizon=planning_horizon):
    updated_wind_capacity_factors = {}
    for i, j in zip(range(0, planning_horizon), range(pointer*24, pointer*24 + planning_horizon)):
        updated_wind_capacity_factors[i] = input_param['wind_resource'][j]
    
    return updated_wind_capacity_factors

def build_scenario_model(price_signals, input_params, backcaster, n_time_points=planning_horizon, verbose=False):
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
    mp_wind_battery.build_multi_period_model(input_params["wind_resource"])

    scenario_model = mp_wind_battery.pyomo_model
    blks = mp_wind_battery.get_active_process_blocks()
    blks[0].fs.battery.initial_state_of_charge.fix(input_params["battery_soc"])
    blks[0].fs.battery.initial_energy_throughput.fix(input_params["energy_throughput"])
    
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
        
        # add operating costs, days in a year = 366
#        blk_wind.op_total_cost = Expression(
#            expr=m.wind_system_capacity * blk_wind.op_cost / 8784
#        )
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
    scenario_model.horizon_rev = Expression(expr = sum([blk.revenue for blk in blks[0:backcaster.horizon]]))

    return mp_wind_battery

def run_wind_battery_price_taker_uncertainty(input_params, backcaster):
    # rolling horizon optimization
    res_dict = {}
    for i in range(1):
        print(f"Build and solve for Day {i}")
        # update wind resource capacity factor
        backcaster.pointer = i
        input_params["wind_resource"] = update_wind_capacity_factors(default_input_params, backcaster.pointer, planning_horizon)
        
        res_dict[i] = {}
        m = ConcreteModel(name = f'wind_battery_price_taker_with_uncertainty_day_{i}')
        price_signals = backcaster.generate_price_scenarios()
        scenario_list = []
        for j in range(len(price_signals)):
            mp_wind_battery = build_scenario_model(price_signals[j], input_params, backcaster)
            scenario_list.append(mp_wind_battery)
            setattr(m, 'scenario_model_{}'.format(j), mp_wind_battery.pyomo_model)
        
        # add m.rev and other data
        m.rev = Expression(expr = 1/(len(scenario_list))*sum(scenario_list[k].pyomo_model.total_profit for k in range(len(scenario_list))))
        m.rev_24h = Expression(expr = 1/(len(scenario_list))*sum(scenario_list[k].pyomo_model.horizon_rev for k in range(len(scenario_list))))
        
        # solve m
        m.obj = Objective(expr = m.rev, sense=maximize)
        opt = SolverFactory("ipopt")
        opt.solve(m, tee=True)

        # record the hourly power output for each scenario, unit is MW.
        scenario_data = {}
        for j in range(len(scenario_list)):
            scenario_data[j] = {}
            blks = scenario_list[j].get_active_process_blocks()
            scenario_data[j]["wind_gen"] = [value(blks[m].fs.windpower.electricity[0]) * 1e-3 for m in range(backcaster.horizon)]
            scenario_data[j]["batt_to_grid"] = [value(blks[m].fs.battery.elec_out[0]) * 1e-3 for m in range(backcaster.horizon)]
            scenario_data[j]["wind_to_grid"] = [value(blks[m].fs.splitter.grid_elec[0]) * 1e-3 for m in range(backcaster.horizon)]
            scenario_data[j]["wind_to_batt"] = [value(blks[m].fs.battery.elec_in[0]) * 1e-3 for m in range(backcaster.horizon)]
        
        # record results
        res_dict[i]["wind_cf"] = input_params["wind_resource"]
        res_dict[i]["lmp"] = price_signals
        res_dict[i]["rev"] = value(m.rev_24h)
        res_dict[i]["power_data"] = scenario_data

        # update soc and energy throughput
        input_params["battery_soc"] = sum(value(scenario_list[i].pyomo_model.blocks[horizon-1].process.fs.battery.state_of_charge[0.0]) for i in range(len(scenario_list)))/len(scenario_list)
        input_params["energy_throughput"] = sum(value(scenario_list[i].pyomo_model.blocks[horizon-1].process.fs.battery.energy_throughput[0.0]) for i in range(len(scenario_list)))/len(scenario_list)

    return res_dict

input_params = default_input_params.copy()
input_params["design_opt"] = False
input_params["extant_wind"] = True
input_params["wind_mw"] = 847
input_params["batt_mw"] = 84.7
input_params["tank_size"] = 0
# initial soc = 0 and energy thoughput
input_params["battery_soc"] = 0
input_params["energy_throughput"] = 0
res_dict = run_wind_battery_price_taker_uncertainty(input_params, backcaster)
print(res_dict)

# print(input_params['wind_resource'])
# backcaster.pointer = 0
# price_signal = backcaster.generate_price_scenarios()
# build_scenario_model(price_signal[0], input_params, backcaster).pprint()