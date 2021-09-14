##############################################################################
# Institute for the Design of Advanced Energy Systems Process Systems
# Engineering Framework (IDAES PSE Framework) Copyright (c) 2018-2020, by the
# software owners: The Regents of the University of California, through
# Lawrence Berkeley National Laboratory,  National Technology & Engineering
# Solutions of Sandia, LLC, Carnegie Mellon University, West Virginia
# University Research Corporation, et al. All rights reserved.
#
# Please see the files COPYRIGHT.txt and LICENSE.txt for full copyright and
# license information, respectively. Both files are also available online
# at the URL "https://github.com/IDAES/idaes-pse".
##############################################################################
"""
Simple rankine cycle model.

Boiler --> Turbine --> Condenser --> Pump --> Boiler

Note:
* Boiler and condenser are simple heater blocks
* IAPWS95 for water and steam properties
"""

__author__ = "Jaffer Ghouse"


# Import Pyomo libraries
from pyomo.environ import ConcreteModel, SolverFactory, units, Var, \
    TransformationFactory, value, Block, Expression, Constraint, Param, \
    Objective
from pyomo.network import Arc
from pyomo.util.infeasible import log_close_to_bounds

# Import IDAES components
from idaes.core import FlowsheetBlock, UnitModelBlockData

# Import heat exchanger unit model
from idaes.generic_models.unit_models import Heater, PressureChanger

from idaes.generic_models.unit_models.pressure_changer import \
    ThermodynamicAssumption
from idaes.power_generation.costing.power_plant_costing import get_PP_costing
# Import steam property package
from idaes.generic_models.properties.iapws95 import htpx, Iapws95ParameterBlock

from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util.initialization import propagate_state
from idaes.core.util import get_solver
import idaes.logger as idaeslog


def create_model(heat_recovery=False):
    m = ConcreteModel()

    m.fs = FlowsheetBlock(default={"dynamic": False})

    m.fs.steam_prop = Iapws95ParameterBlock()

    m.fs.boiler = Heater(
        default={
            "dynamic": False,
            "property_package": m.fs.steam_prop,
            "has_pressure_change": False})

    m.fs.turbine = PressureChanger(
        default={
            "property_package": m.fs.steam_prop,
            "compressor": False,
            "thermodynamic_assumption": ThermodynamicAssumption.isentropic})

    if heat_recovery:
        m.fs.pre_condenser = Heater(
            default={
                "dynamic": False,
                "property_package": m.fs.steam_prop,
                "has_pressure_change": True})

        # Spec for pre-condenser
        m.fs.pre_condenser.eq_outlet_cond = Constraint(
            expr=m.fs.pre_condenser.control_volume.
            properties_out[0].enth_mol == m.fs.pre_condenser.control_volume.
            properties_out[0].enth_mol_sat_phase["Liq"]
        )

        m.fs.feed_water_heater = Heater(
            default={
                "dynamic": False,
                "property_package": m.fs.steam_prop,
                "has_pressure_change": True})

    m.fs.condenser = Heater(
        default={
            "dynamic": False,
            "property_package": m.fs.steam_prop,
            "has_pressure_change": True})

    m.fs.bfw_pump = PressureChanger(
        default={
            "property_package": m.fs.steam_prop,
            "thermodynamic_assumption": ThermodynamicAssumption.pump})

    # create arcs
    m.fs.boiler_to_turbine = Arc(source=m.fs.boiler.outlet,
                                 destination=m.fs.turbine.inlet)

    if heat_recovery:
        m.fs.turbine_to_precondenser = Arc(
            source=m.fs.turbine.outlet,
            destination=m.fs.pre_condenser.inlet)
        m.fs.precondenser_to_condenser = Arc(
            source=m.fs.pre_condenser.outlet,
            destination=m.fs.condenser.inlet)
        m.fs.pump_to_feedwaterheater = Arc(
            source=m.fs.bfw_pump.outlet,
            destination=m.fs.feed_water_heater.inlet)
    else:
        m.fs.turbine_to_condenser = Arc(source=m.fs.turbine.outlet,
                                        destination=m.fs.condenser.inlet)

    m.fs.condenser_to_pump = Arc(source=m.fs.condenser.outlet,
                                 destination=m.fs.bfw_pump.inlet)

    # expand arcs
    TransformationFactory("network.expand_arcs").apply_to(m)

    # deactivate the flow equality
    m.fs.gross_cycle_power_output = \
        Expression(expr=(-m.fs.turbine.work_mechanical[0] -
                   m.fs.bfw_pump.work_mechanical[0]))

    # account for generator loss = 1.5% of gross power output
    m.fs.net_cycle_power_output = Expression(
        expr=0.95*m.fs.gross_cycle_power_output)

    #  cycle efficiency
    m.fs.cycle_efficiency = Expression(
        expr=m.fs.net_cycle_power_output/m.fs.boiler.heat_duty[0] * 100
    )

    m.heat_recovery = heat_recovery

    return m


def initialize_model(m, outlvl=idaeslog.INFO):

    assert degrees_of_freedom(m) == 0

    # proceed with initialization
    m.fs.boiler.initialize(outlvl=outlvl)

    propagate_state(m.fs.boiler_to_turbine)

    m.fs.turbine.initialize(outlvl=outlvl)

    if m.heat_recovery:
        propagate_state(m.fs.turbine_to_precondenser)
        m.fs.pre_condenser.initialize(outlvl=outlvl)

        propagate_state(m.fs.precondenser_to_condenser)
        m.fs.condenser.initialize()

        propagate_state(m.fs.condenser_to_pump)
        m.fs.bfw_pump.initialize(outlvl=outlvl)

        propagate_state(m.fs.pump_to_feedwaterheater)
        m.fs.feed_water_heater.initialize(outlvl=outlvl)
    else:
        propagate_state(m.fs.turbine_to_condenser)
        m.fs.condenser.initialize(outlvl=outlvl)

        propagate_state(m.fs.condenser_to_pump)
        m.fs.bfw_pump.initialize(outlvl=outlvl)

    solver = get_solver()
    solver.solve(m, tee=False)

    if m.heat_recovery:
        # Unfix feed water heater temperature
        m.fs.feed_water_heater.outlet.enth_mol[0].unfix()

        # Link precondenser heat and feed water heater
        m.fs.eq_heat_recovery = Constraint(
            expr=m.fs.pre_condenser.heat_duty[0] ==
            - m.fs.feed_water_heater.heat_duty[0]
        )

    solver.solve(m, tee=True)

    return m


def generate_report(m, unit_model_report=True):

    # Print reports
    if unit_model_report:
        for i in m.fs.component_objects(Block):
            if isinstance(i, UnitModelBlockData):
                i.report()

    print()
    print('Net power = ', value(m.fs.net_cycle_power_output)*1e-6, ' MW')
    print('Cycle efficiency = ', value(m.fs.cycle_efficiency))
    print('Boiler feed water flow = ', value(m.fs.boiler.inlet.flow_mol[0]))
    print()
    try:
        print('Capital cost = ', value(m.fs.capital_cost), '$M')
    except AttributeError:
        print("No cap cost for opex plant")
    try:
        print('Operating cost =  ', value(m.fs.operating_cost), '$/hr')
    except AttributeError:
        print("No operating cost for capex plant")


def set_inputs(m, bfw_pressure=24.23e6, bfw_flow=10000):

    # Main steam pressure
    bfw_pressure = bfw_pressure  # Pa

    # Boiler inlet
    m.fs.boiler.inlet.flow_mol[0].fix(bfw_flow)  # mol/s
    m.fs.boiler.inlet.pressure[0].fix(bfw_pressure)  # MPa
    m.fs.boiler.inlet.enth_mol[0].fix(
        htpx(T=563.6*units.K,
             P=value(m.fs.boiler.inlet.pressure[0])*units.Pa))

    # unit specifications
    m.fs.boiler.outlet.enth_mol[0].fix(
        htpx(T=866.5*units.K,
             P=value(m.fs.boiler.inlet.pressure[0])*units.Pa))

    turbine_pressure_ratio = 2e6/bfw_pressure
    m.fs.turbine.ratioP.fix(turbine_pressure_ratio)
    m.fs.turbine.efficiency_isentropic.fix(0.85)

    if m.heat_recovery:
        # precondenser
        m.fs.pre_condenser.deltaP.fix(-0.5e6)  # Pa

        # feed water heater
        m.fs.feed_water_heater.deltaP[0].fix(0)  # Pa
        m.fs.feed_water_heater.outlet.enth_mol[0].fix(
            htpx(T=563.6*units.K,
                 P=value(m.fs.condenser.outlet.pressure[0])*units.Pa))

    m.fs.condenser.outlet.pressure[0].fix(1.05e6)  # Pa
    m.fs.condenser.outlet.enth_mol[0].fix(
        htpx(T=311*units.K,
             P=value(m.fs.condenser.outlet.pressure[0])*units.Pa))

    m.fs.bfw_pump.efficiency_pump.fix(0.80)
    m.fs.bfw_pump.deltaP.fix(bfw_pressure)

    return m


def close_flowsheet_loop(m):

    # Unfix pump pressure spec
    m.fs.bfw_pump.deltaP.unfix()

    # Unfix inlet boiler enthalpy
    m.fs.boiler.inlet.enth_mol[0].unfix()

    if m.heat_recovery:
        # Constraint to link pressure
        m.fs.eq_pressure = Constraint(
            expr=m.fs.feed_water_heater.outlet.pressure[0] ==
            m.fs.boiler.inlet.pressure[0]
        )

        # Constraint to link enthalpy
        m.fs.eq_enthalpy = Constraint(
            expr=m.fs.feed_water_heater.outlet.enth_mol[0] ==
            m.fs.boiler.inlet.enth_mol[0]
        )
    else:
        # Constraint to link pressure
        m.fs.eq_pressure = Constraint(
            expr=m.fs.bfw_pump.outlet.pressure[0] ==
            m.fs.boiler.inlet.pressure[0]
        )

        # Constraint to link enthalpy
        m.fs.eq_enthalpy = Constraint(
            expr=m.fs.bfw_pump.outlet.enth_mol[0] ==
            m.fs.boiler.inlet.enth_mol[0]
        )

    return m


def add_capital_cost(m):

    m.fs.get_costing(year='2018')

    # Add boiler capital cost
    boiler_power_account = ['4.9']
    # convert flow rate of BFW from mol/s to lb/hr for costing expressions
    m.fs.bfw_lb_hr = Expression(
        expr=m.fs.boiler.inlet.flow_mol[0]*0.018*2.204*3600)
    get_PP_costing(
        m.fs.boiler, boiler_power_account, m.fs.bfw_lb_hr, 'lb/hr', 2)

    # Add turbine capital cost
    turb_power_account = ['8.1']
    # convert the turbine power from W to kW for costing expressions
    m.fs.turbine_power_mw = Expression(
        expr=-m.fs.turbine.work_mechanical[0] * 1e-3)
    get_PP_costing(
        m.fs.turbine, turb_power_account,
        m.fs.turbine_power_mw, 'kW', 2)

    # Add condenser cost
    cond_power_account = ['8.3']
    # convert the heat duty from J/s to MMBtu/hr for costing expressions
    m.fs.condenser_duty_mmbtu_h = Expression(
        expr=-m.fs.condenser.heat_duty[0] * 3.412*1e-6)
    get_PP_costing(
        m.fs.condenser, cond_power_account,
        m.fs.condenser_duty_mmbtu_h, "MMBtu/hr", 2)

    # Add feed water system costs
    # Note that though no feed water heaters were used, BFW flowrate is used
    # to cost the fed water system
    fwh_power_account = ['3.1', '3.3', '3.5']
    get_PP_costing(m.fs.bfw_pump, fwh_power_account,
                   m.fs.bfw_lb_hr, 'lb/hr', 2)

    # Add expression for total capital cost
    m.fs.capital_cost = Expression(
        expr=m.fs.boiler.costing.total_plant_cost['4.9'] +
        m.fs.turbine.costing.total_plant_cost['8.1'] +
        m.fs.condenser.costing.total_plant_cost['8.3'] +
        sum(m.fs.bfw_pump.costing.total_plant_cost[:]),
        doc="Total capital cost $ Million")

    return m


def add_operating_cost(m, include_cooling_cost=True):

    # Add condenser cooling water cost
    # temperature for the cooling water from/to cooling tower in K
    t_cw_in = 289.15
    t_cw_out = 300.15

    # compute the delta_h based on fixed temperature of cooling water
    # utility
    m.fs.enth_cw_in = Param(
        initialize=htpx(T=t_cw_in*units.K, P=101325*units.Pa),
        doc="inlet enthalpy of cooling water to condenser")
    m.fs.enth_cw_out = Param(
        initialize=htpx(T=t_cw_out*units.K, P=101325*units.Pa),
        doc="outlet enthalpy of cooling water from condenser")

    m.fs.cw_flow = Expression(
        expr=-m.fs.condenser.heat_duty[0]*0.018*0.26417*3600 /
        (m.fs.enth_cw_out-m.fs.enth_cw_in),
        doc="cooling water flow rate in gallons/hr")

    # cooling water cost in $/1000 gallons
    m.fs.cw_cost = Param(
        initialize=0.19,
        doc="cost of cooling water for condenser in $/1000 gallon")

    m.fs.cw_total_cost = Expression(
        expr=m.fs.cw_flow*m.fs.cw_cost/1000,
        doc="total cooling water cost in $/hr"
    )

    # Add coal feed costs
    # HHV value of coal (Reference - NETL baseline report rev #4)
    m.fs.coal_hhv = Param(
        initialize=27113,
        doc="Higher heating value of coal as received kJ/kg")

    # cost of coal (Reference - NETL baseline report rev #4)
    m.fs.coal_cost = Param(
        initialize=51.96,
        doc="$ per ton of Illinois no. 6 coal"
    )
    # Expression to compute coal flow rate in ton/hr using Q_boiler and
    # hhv values
    m.fs.coal_flow = Expression(
        expr=((m.fs.boiler.heat_duty[0] * 3600)/(907.18*1000*m.fs.coal_hhv)),
        doc="coal flow rate for boiler ton/hr")
    # Expression to compute total cost of coal feed in $/hr
    m.fs.total_coal_cost = Expression(
        expr=m.fs.coal_flow*m.fs.coal_cost,
        doc="total cost of coal feed in $/hr"
    )

    if include_cooling_cost:
        # Expression for total operating cost
        m.fs.operating_cost = Expression(
            expr=m.fs.total_coal_cost+m.fs.cw_total_cost,
            doc="Total operating cost in $/hr")
    else:
        # Expression for total operating cost
        m.fs.operating_cost = Expression(
            expr=m.fs.total_coal_cost,
            doc="Total operating cost in $/hr")

    return m


def square_problem(heat_recovery=None, capital_payment_years=5):
    m = ConcreteModel()

    # Create plant flowsheet
    m = create_model(heat_recovery=heat_recovery)

    # Set model inputs for the capex and opex plant
    m = set_inputs(m)

    # Initialize the capex and opex plant
    m = initialize_model(m)

    # Closing the loop in the flowsheet
    m = close_flowsheet_loop(m)

    # Unfixing the boiler inlet flowrate
    m.fs.boiler.inlet.flow_mol[0].unfix()

    # Net power constraint for the capex plant
    m.fs.eq_net_power = Constraint(
        expr=m.fs.net_cycle_power_output == 100e6
    )

    m = add_capital_cost(m)

    m = add_operating_cost(m, include_cooling_cost=True)

    # Expression for total cap and op cost - $/hr
    m.total_cost = Expression(
        expr=(m.fs.capital_cost*1e6/capital_payment_years/8760) +
        m.fs.operating_cost)

    solver = get_solver()
    solver.solve(m, tee=True)

    generate_report(m, unit_model_report=False)
    # generate_report(m.op_fs, unit_model_report=True)

    return m


def stochastic_optimization_problem(heat_recovery=False,
                                    p_lower_bound=10,
                                    p_upper_bound=500,
                                    capital_payment_years=5,
                                    plant_lifetime=20,
                                    power_demand=None, lmp=None,
                                    lmp_weights=None):

    m = ConcreteModel()

    # Create capex plant
    m.cap_fs = create_model(heat_recovery=heat_recovery)
    m.cap_fs = set_inputs(m.cap_fs)
    m.cap_fs = initialize_model(m.cap_fs)
    m.cap_fs = close_flowsheet_loop(m.cap_fs)
    m.cap_fs = add_capital_cost(m.cap_fs)

    # capital cost (M$/yr)
    cap_expr = m.cap_fs.fs.capital_cost*1e6/capital_payment_years

    # Create opex plant
    op_expr = 0
    rev_expr = 0

    for i in range(len(lmp)):

        print()
        print("Creating instance ", i)
        op_fs = create_model(heat_recovery=heat_recovery)

        # Set model inputs for the capex and opex plant
        op_fs = set_inputs(op_fs)

        # Initialize the capex and opex plant
        op_fs = initialize_model(op_fs)

        # Closing the loop in the flowsheet
        op_fs = close_flowsheet_loop(op_fs)

        op_fs = add_operating_cost(op_fs)

        op_expr += lmp_weights[i]*op_fs.fs.operating_cost
        rev_expr += lmp_weights[i]*lmp[i]*op_fs.fs.net_cycle_power_output*1e-6

        # Add inequality constraint linking net power to cap_ex
        # operating P_min <= 30% of design P_max
        op_fs.fs.eq_min_power = Constraint(
            expr=op_fs.fs.net_cycle_power_output >=
            0.3*m.cap_fs.fs.net_cycle_power_output)
        # operating P_max = design P_max
        op_fs.fs.eq_max_power = Constraint(
            expr=op_fs.fs.net_cycle_power_output <=
            m.cap_fs.fs.net_cycle_power_output)

        # only if power demand is given
        if power_demand is not None:
            op_fs.fs.eq_max_produced = Constraint(
                expr=op_fs.fs.net_cycle_power_output <=
                power_demand[i]*1e6)

        op_fs.fs.boiler.inlet.flow_mol[0].unfix()

        # Set bounds for the flow
        op_fs.fs.boiler.inlet.flow_mol[0].setlb(1)
        # op_fs.fs.boiler.inlet.flow_mol[0].setub(25000)

        setattr(m, 'scenario_{}'.format(i), op_fs)

    # Expression for total cap and op cost - $
    m.total_cost = Expression(
        expr=plant_lifetime*op_expr + capital_payment_years*cap_expr)

    # Expression for total revenue
    m.total_revenue = Expression(
        expr=plant_lifetime*rev_expr)

    # Objective $
    m.obj = Objective(
        expr=-(m.total_revenue - m.total_cost))

    # Unfixing the boiler inlet flowrate for capex plant
    m.cap_fs.fs.boiler.inlet.flow_mol[0].unfix()

    # Setting bounds for the capex plant flowrate
    m.cap_fs.fs.boiler.inlet.flow_mol[0].setlb(5)
    # m.cap_fs.fs.boiler.inlet.flow_mol[0].setub(25000)

    # Setting bounds for net cycle power output for the capex plant
    m.cap_fs.fs.eq_min_power = Constraint(
        expr=m.cap_fs.fs.net_cycle_power_output >= p_lower_bound*1e6)

    m.cap_fs.fs.eq_max_power = Constraint(
        expr=m.cap_fs.fs.net_cycle_power_output <=
        p_upper_bound*1e6)

    return m


if __name__ == "__main__":

    m = square_problem(heat_recovery=True)

    # Stochastic case P_max is equal to max power demand
    # Case 0A - power and lmp signal
    # power_demand = [50, 200]  # MW
    # price = [100, 200]  # $/MW-h

    # Case 0B - power and lmp signal
    # power_demand = [50, 200]  # MW
    # price = [50, 100]  # $/MW-h

    # Case 1A - lmp signal
    # price = [100, 200]  # $/MW-h
    # power_demand = None

    # Case 1B - lmp signal
    # price = [50, 100]  # $/MW-h
    # power_demand = None

    # Case 1C - lmp signal
    # price = [10, 100]  # $/MW-h
    # power_demand = None

    # Case 1D - lmp signal
    # price = [10, 50]  # $/MW-h
    # power_demand = None

    # ARPA-E Signal
    # import numpy as np

    # lmp_signals = np.load("nrel_scenario_12_rep_days.npy")
    # price = lmp_signals[5].tolist()
    # power_demand = None
    # m = stochastic_optimization_problem(
    #     heat_recovery=True, capital_payment_years=10,
    #     power_demand=power_demand, lmp=price)
    # solver = get_solver()
    # solver.solve(m, tee=True)
    # print("The net revenue is $", -value(m.obj))
    # print("P_max = ", value(m.cap_fs.fs.net_cycle_power_output)*1e-6, ' MW')
    # p_scenario = []
    # for i in range(len(price)):
    #     scenario = getattr(m, 'scenario_{}'.format(i))
    #     p_scenario.append(value(scenario    .fs.net_cycle_power_output)*1e-6)
    # # print("P_1 = ", value(m.scenario_0.fs.net_cycle_power_output)*1e-6, ' MW')
    # # print("BFW = ", value(m.scenario_0.fs.boiler.inlet.flow_mol[0]), ' mol/s')
    # # print("P_2 = ", value(m.scenario_1.fs.net_cycle_power_output)*1e-6, ' MW')
    # # print("BFW = ", value(m.scenario_1.fs.boiler.inlet.
    # #       flow_mol[0]), ' mol/s')
    # print(price)
    # print(p_scenario)

    # from matplotlib import pyplot as plt
    # fig, ax = plt.subplots()
    # ax.plot(price, color="red")
    # # set x-axis label
    # ax.set_xlabel("Time (h)", fontsize=14)
    # # set y-axis label
    # ax.set_ylabel("LMP", color="red", fontsize=14)

    # ax2 = ax.twinx()
    # ax2.plot(p_scenario)
    # ax2.set_ylabel("Power Produced", color="blue", fontsize=14)

    # plt.show()