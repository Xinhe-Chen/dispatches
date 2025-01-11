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
from wind_battery_LMP import wind_battery_variable_pairs, wind_battery_mp_block 
from idaes.apps.grid_integration.multiperiod.multiperiod import MultiPeriodModel
from pyomo.environ import (Constraint,
                           Var,
                           Set,
                           ConcreteModel,
                           Expression,
                           Param,
                           Block,
                           Objective,
                           maximize,
                           units as pyunits,
                           SolverFactory,
                           TransformationFactory,
                           NonNegativeReals,
                           Reference,
                           value)
from pyomo.util.infeasible import log_infeasible_constraints
# import logging

# _logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.DEBUG)

# this is for the stochastic programming. x scenarios with 36 hours planning horizon and 24 hours are realized.
# Currently we pause this work because we have problem specifying the initial soc of the coming day in the rolling horizion optimization.

lmps_df = pd.read_parquet(Path(__file__).parent / "data" / "303_LMPs_15_reserve_500_shortfall.parquet")
lmps = lmps_df['LMP'].values
lmps[lmps>500] = 500
signal = lmps # even we use rt lmp signals, we call it DA_LMPs to simplify the work.

scenario = 3
horizon = 24
planning_horizon = 72

backcaster = PricetakerBackcaster(price_signal=signal, scenario=scenario, planning_horizon=planning_horizon)

def wind_battery_periodic_variable_pairs(m1, m2):
    """
    The final battery storage of charge must be the same as in the intial timestep. 

    Args:
        m1: final time block model
        m2: first time block model
    """
    pairs = [(m1.fs.battery.nameplate_power, m2.fs.battery.nameplate_power)]
    return pairs

def update_wind_capacity_factors(input_param, pointer, planning_horizon=planning_horizon):
    updated_wind_capacity_factors = {}
    for i, j in zip(range(0, planning_horizon), range(pointer*24, pointer*24 + planning_horizon)):
        updated_wind_capacity_factors[i] = input_param['wind_resource'][j]
    
    return updated_wind_capacity_factors

def build_scenario_model(price_signals, input_params, backcaster, n_time_points=planning_horizon, stage_1_length = 24, verbose=False):
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
    # if the scenario initial soc is full and after the charging and discharging, there will be degradation.
    # To prevent this issue, set the end soc bounds as 0.99 to 1.0 of initial soc 
    scenario_model.soc_constr_lb = Constraint(expr=mp_wind_battery.pyomo_model.blocks[planning_horizon-1].process.fs.battery.state_of_charge[0.0] >= mp_wind_battery.pyomo_model.blocks[0].process.fs.battery.initial_state_of_charge*0.99)
    scenario_model.soc_constr_ub = Constraint(expr=mp_wind_battery.pyomo_model.blocks[planning_horizon-1].process.fs.battery.state_of_charge[0.0] <= mp_wind_battery.pyomo_model.blocks[0].process.fs.battery.initial_state_of_charge)

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

def run_wind_battery_price_taker_uncertainty(input_params, backcaster, days=3):
    # rolling horizon optimization
    res_dict = {}
    for i in range(days):
        print(f"Build and solve for Day {i}")
        # update wind resource capacity factor
        backcaster.pointer = i
        input_params["wind_resource"] = update_wind_capacity_factors(default_input_params, backcaster.pointer, planning_horizon)
        
        res_dict[i] = {}
        m = ConcreteModel(name = f'wind_battery_price_taker_with_uncertainty_day_{i}')
        price_signals = backcaster.generate_price_scenarios()
        m.scenario = Set(initialize=list(range(backcaster.scenario)))
        m.scenario_blocks = Block(m.scenario)
        scenario_list = []
        for j in range(len(price_signals)):
            mp_wind_battery = build_scenario_model(price_signals[j], input_params, backcaster)
            # scenario_list.append(mp_wind_battery)
            m.scenario_blocks[j].add_component("pyomo_model", mp_wind_battery.pyomo_model)
            # m.scenario_blocks[j].pyomo_model.soc_constr = Constraint(expr = m.scenario_blocks[j].pyomo_model.blocks[0].process.fs.battery.battery.initial_state_of_charge==m.scenario_blocks[j].pyomo_model.blocks[planning_horizon-1].process.fs.state_of_charge[0])
            # m.scenario_blocks[i].pyomo_model.transfer_attributes_from(mp_wind_battery.pyomo_model)
            # setattr(m.scenario_blocks[i], 'scenario_model_{}'.format(j), mp_wind_battery.pyomo_model)
            
        # enforce the stage_1 variable to be the same
        def stage_1_consistency_rule_1(b, s, t):
            if t < 24:
                for i in range(len(price_signals)):
                    return b.scenario_blocks[s].pyomo_model.blocks[t].process.fs.battery.elec_out[0] == b.scenario_blocks[0].pyomo_model.blocks[t].process.fs.battery.elec_out[0]
            return Constraint.Skip
        
        m.stage_1_consistency_constraint_1 = Constraint(m.scenario, range(24), rule=stage_1_consistency_rule_1)

        def stage_1_consistency_rule_2(b, s, t):
            if t < 24:
                for i in range(len(price_signals)):
                    return b.scenario_blocks[s].pyomo_model.blocks[t].process.fs.battery.elec_in[0] == b.scenario_blocks[0].pyomo_model.blocks[t].process.fs.battery.elec_in[0]
            return Constraint.Skip
        
        m.stage_1_consistency_constraint_2 = Constraint(m.scenario, range(24), rule=stage_1_consistency_rule_2)

        def stage_1_consistency_rule_3(b, s, t):
            if t < 24:
                for i in range(len(price_signals)):
                    return b.scenario_blocks[s].pyomo_model.blocks[t].process.fs.splitter.grid_elec[0] == b.scenario_blocks[0].pyomo_model.blocks[t].process.fs.splitter.grid_elec[0]
            return Constraint.Skip
        
        m.stage_1_consistency_constraint_3 = Constraint(m.scenario, range(24), rule=stage_1_consistency_rule_3)

        # add m.rev and other data
        m.rev = Expression(expr = 1/backcaster.scenario * sum(m.scenario_blocks[k].pyomo_model.elec_revenue for k in range(backcaster.scenario)))
        # m.rev_24h = Expression(expr = 1/backcaster.scenario * sum(scenario_list[k].pyomo_model.horizon_rev for k in range(len(scenario_list))))
        
        # solve m
        m.obj = Objective(expr = m.rev, sense=maximize)
        opt = SolverFactory("ipopt")
        soln = opt.solve(m, tee=True)
        infeasible = log_infeasible_constraints(m)
        # record the hourly power output for each scenario, unit is MW.
        scenario_data = {}
        for j in range(backcaster.scenario):
            scenario_data[j] = {}
            # scenario_data[j]["wind_gen"] = [value(m.scenario_blocks[j].pyomo_model.blocks[k].process.fs.windpower.electricity[0]) * 1e-3 for k in range(backcaster.horizon)]
            scenario_data[j]["batt_to_grid"] = [value(m.scenario_blocks[j].pyomo_model.blocks[0].process.fs.battery.elec_out[0]) * 1e-3]
            # scenario_data[j]["wind_to_grid"] = [value(blks[m].fs.splitter.grid_elec[0]) * 1e-3 for m in range(backcaster.horizon)]
            scenario_data[j]["wind_to_batt"] = [value(m.scenario_blocks[j].pyomo_model.blocks[0].process.fs.battery.elec_in[0]) * 1e-3]
            # scenario_data[j]["soc"] = [value(m.scenario_blocks[j].pyomo_model.blocks[k].process.fs.battery.state_of_charge[0]) * 1e-3 for k in range(backcaster.horizon)]
            scenario_data[j]["soc_init"] = value(m.scenario_blocks[j].pyomo_model.blocks[0].process.fs.battery.initial_state_of_charge) * 1e-3
            # scenario_data[j]["soc_end"] = value(m.scenario_blocks[j].pyomo_model.blocks[horizon-1].process.fs.battery.state_of_charge[0]) * 1e-3
            scenario_data[j]["eng_init"] = value(m.scenario_blocks[j].pyomo_model.blocks[0].process.fs.battery.initial_energy_throughput) * 1e-3
            # scenario_data[j]["eng_0"] = value(m.scenario_blocks[j].pyomo_model.blocks[horizon-1].process.fs.battery.energy_throughput[0]) * 1e-3
            # scenario_data[j]["nameplate_eng"] = value(m.scenario_blocks[j].pyomo_model.blocks[0].process.fs.battery.nameplate_energy) * 1e-3
        # record results
        # res_dict[i]["wind_cf"] = input_params["wind_resource"]
        # res_dict[i]["lmp"] = price_signals
        # res_dict[i]["rev"] = value(m.rev_24h)
        res_dict[i]["power_data"] = scenario_data
        res_dict[i]['solver_stat'] = str(soln.solver.termination_condition)
        # print(value(m.scenario_blocks[0].pyomo_model.blocks[0].process.fs.windpower.electricity[0]))
        # update soc and energy throughput
        input_params["battery_soc"] = value(m.scenario_blocks[0].pyomo_model.blocks[horizon-1].process.fs.battery.state_of_charge[0])
        input_params["energy_throughput"] = value(m.scenario_blocks[0].pyomo_model.blocks[horizon-1].process.fs.battery.energy_throughput[0])
    # m.scenario_blocks[0].pyomo_model.blocks[0].process.fs.battery.accumulate_energy_throughput.pprint()
    # m.scenario_blocks[0].pyomo_model.blocks[0].process.fs.battery.energy_throughput.pprint()
    # m.scenario_blocks[0].pyomo_model.blocks[0].process.fs.battery.initial_energy_throughput.pprint()
    # m.scenario_blocks[0].pyomo_model.blocks[0].process.fs.battery.elec_in.pprint()
    # m.scenario_blocks[0].pyomo_model.blocks[0].process.fs.battery.elec_out.pprint()
        # m.scenario_blocks[0].pyomo_model.blocks[horizon-1].process.fs.battery.pprint()
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
# print(np.shape(price_signal))
# build_scenario_model(price_signal[0], input_params, backcaster).pprint()