import pyomo.environ as pyo
import idaes.logger as idaeslog
from pyomo.util.infeasible import log_infeasible_constraints, log_infeasible_bounds, log_close_to_bounds
from idaes.apps.multiperiod.multiperiod import MultiPeriodModel
from RE_flowsheet import *
from load_LMP import *

design_opt = False


def wind_pem_tank_variable_pairs(m1, m2):
    """
    the power output and battery state are linked between time periods

        b1: current time block
        b2: next time block
    """
    pairs = [(m1.fs.h2_tank.energy_holdup[0, 'Vap'], m2.fs.h2_tank.previous_energy_holdup[0, 'Vap']),
             (m1.fs.h2_tank.material_holdup[0, ('Vap', 'hydrogen')], m2.fs.h2_tank.previous_material_holdup[0, ('Vap', 'hydrogen')])]
    if design_opt:
        pairs += [(m1.fs.windpower.system_capacity, m2.fs.windpower.system_capacity)]
    return pairs


def wind_pem_tank_periodic_variable_pairs(m1, m2):
    """
    the final power output and battery state must be the same as the intial power output and battery state

        b1: final time block
        b2: first time block
    """
    pairs = [(m1.fs.h2_tank.energy_holdup[0, 'Vap'], m2.fs.h2_tank.previous_energy_holdup[0, 'Vap']),
             (m1.fs.h2_tank.material_holdup[0, ('Vap', 'hydrogen')], m2.fs.h2_tank.previous_material_holdup[0, ('Vap', 'hydrogen')])]
    if design_opt:
        pairs += [(m1.fs.windpower.system_capacity, m2.fs.windpower.system_capacity)]
    return pairs


def wind_pem_tank_om_costs(m):
    m.fs.windpower.op_cost = pyo.Param(
        initialize=43,
        doc="fixed cost of operating wind plant $/kW-yr")
    m.fs.windpower.op_total_cost = Expression(
        expr=m.fs.windpower.system_capacity * m.fs.windpower.op_cost / 8760,
        doc="total fixed cost of wind in $/hr"
    )
    m.fs.pem.op_cost = pyo.Param(
        initialize=47.9,
        doc="fixed cost of operating pem $/kW-yr"
    )
    m.fs.pem.var_cost = pyo.Param(
        initialize=1.3/1000,
        doc="variable cost of pem $/kW"
    )
    # m.fs.h2_tank.op_cost = Expression(
    #     expr=m.fs.pem.system_capacity * m.fs.pem.op_cost / 8760 + m.fs.pem.var_cost * m.fs.pem.electricity[0],
    #     doc="total fixed cost of pem in $/hr"
    # )


def initialize_mp(m, verbose=False):
    m.fs.windpower.initialize()

    propagate_state(m.fs.wind_to_splitter)
    m.fs.splitter.split_fraction['grid', 0].fix(.5)
    m.fs.splitter.initialize()
    m.fs.splitter.split_fraction['grid', 0].unfix()
    if verbose:
        m.fs.splitter.report(dof=True)

    propagate_state(m.fs.splitter_to_grid)
    propagate_state(m.fs.splitter_to_pem)

    m.fs.pem.initialize()
    if verbose:
        m.fs.pem.report(dof=True)

    propagate_state(m.fs.pem_to_tank)

    m.fs.h2_tank.outlet.flow_mol[0].fix(value(m.fs.h2_tank.inlet.flow_mol[0]))
    m.fs.h2_tank.initialize()
    m.fs.h2_tank.outlet.flow_mol[0].unfix()
    if verbose:
        m.fs.h2_tank.report(dof=True)

    if hasattr(m.fs, "tank_valve"):
        propagate_state(m.fs.tank_to_valve)
        # m.fs.tank_valve.outlet.flow_mol[0].fix(value(m.fs.tank_valve.inlet.flow_mol[0]))
        m.fs.tank_valve.initialize()
        # m.fs.tank_valve.outlet.flow_mol[0].unfix()
        if verbose:
            m.fs.tank_valve.report(dof=True)



def wind_pem_tank_model(wind_resource_config):
    wind_mw = 200
    pem_bar = 8
    valve_cv = 0.001
    tank_len_m = 0.5
    h2_turb_bar = 24.7
    turb_p_lower_bound = 300
    turb_p_upper_bound = 450

    m = create_model(wind_mw, pem_bar, None, valve_cv, tank_len_m, None, wind_resource_config=wind_resource_config)

    m.fs.h2_tank.previous_state[0].temperature.fix(PEM_temp)
    m.fs.h2_tank.previous_state[0].pressure.fix(pem_bar * 1e5)
    if hasattr(m.fs, "tank_valve"):
        m.fs.tank_valve.outlet.pressure[0].fix(1e5)
    # print(degrees_of_freedom(m))
    initialize_mp(m, verbose=False)
    # print(degrees_of_freedom(m))
    m.fs.h2_tank.previous_state[0].temperature.unfix()
    m.fs.h2_tank.previous_state[0].pressure.unfix()

    solve_log = idaeslog.getInitLogger("infeasibility", idaeslog.INFO,
                                       tag="properties")
    log_infeasible_constraints(m, logger=solve_log, tol=1e-7, log_expression=True, log_variables=True)
    log_infeasible_bounds(m, logger=solve_log, tol=1e-7)
    # log_close_to_bounds(m, logger=solve_log)

    wind_pem_tank_om_costs(m)

    return m


def wind_pem_tank_optimize():
    # create the multiperiod model object
    mp_wind_pem = MultiPeriodModel(n_time_points=n_time_points,
                                   process_model_func=wind_pem_tank_model,
                                   linking_variable_func=wind_pem_tank_variable_pairs,
                                   periodic_variable_func=wind_pem_tank_periodic_variable_pairs)

    mp_wind_pem.build_multi_period_model(wind_resource)

    m = mp_wind_pem.pyomo_model
    blks = mp_wind_pem.get_active_process_blocks()

    m.h2_price_per_kg = pyo.Param(default=h2_price_per_kg, mutable=True)
    m.pem_system_capacity = Var(domain=NonNegativeReals, initialize=20, units=pyunits.kW)
    m.pem_system_capacity.fix(20)
    m.contract_capacity = Var(domain=NonNegativeReals, initialize=0, units=pyunits.mol/pyunits.second)

    # add market data for each block
    for blk in blks:
        blk_wind = blk.fs.windpower
        blk_pem = blk.fs.pem
        # blk_valve = blk.fs.tank_valve
        blk_pem.max_p = Constraint(blk_pem.flowsheet().config.time,
                                 rule=lambda b, t: b.electricity[t] <= m.pem_system_capacity)
        blk_pem.op_total_cost = Expression(
            expr=m.pem_system_capacity * blk_pem.op_cost / 8760 + blk_pem.var_cost * blk_pem.electricity[0],
            doc="total fixed cost of pem in $/hr"
        )
        blk.lmp_signal = pyo.Param(default=0, mutable=True)
        blk.revenue = blk.lmp_signal*blk.fs.wind_to_grid[0]
        blk.profit = pyo.Expression(expr=blk.revenue - blk_wind.op_total_cost - blk_pem.op_total_cost)
        # blk.pem_contract = Constraint(blk_pem.flowsheet().config.time,
        #                               rule=lambda b, t: m.contract_capacity <= blk_valve.outlet.flow_mol[t])
        blk.pem_market = Constraint(blk_pem.flowsheet().config.time,
                                      rule=lambda b, t: m.contract_capacity <= blk.fs.h2_tank.outlet.flow_mol[t])

    m.wind_cap_cost = pyo.Param(default=wind_cap_cost, mutable=True)
    m.pem_cap_cost = pyo.Param(default=pem_cap_cost, mutable=True)

    n_weeks = 1
    m.hydrogen_revenue = Expression(expr=m.h2_price_per_kg * m.contract_capacity / h2_mols_per_kg
                                        * 3600 * n_time_points)
    # m.hydrogen_revenue = Expression(expr=sum([m.h2_price_per_kg * blk.fs.pem.outlet_state[0].flow_mol / h2_mols_per_kg
    #                                     * 3600 * n_time_points for blk in blks]))
    m.annual_revenue = Expression(expr=(sum([blk.profit for blk in blks]) + m.hydrogen_revenue) * 52 / n_weeks)
    m.NPV = Expression(expr=-(m.wind_cap_cost * blks[0].fs.windpower.system_capacity +
                            m.pem_cap_cost * m.pem_system_capacity) +
                          PA * m.annual_revenue)
    m.obj = pyo.Objective(expr=-m.NPV)

    opt = pyo.SolverFactory('ipopt')
    h2_prod = []
    wind_to_grid = []
    wind_to_pem = []
    wind_gen = []
    h2_tank_in = []
    h2_tank_out = []
    h2_tank_holdup = []

    for week in range(n_weeks):
        print("Solving for week: ", week)
        for (i, blk) in enumerate(blks):
            blk.lmp_signal.set_value(weekly_prices[week][i])
        opt.options['bound_push'] = 10e-10
        opt.options['max_iter'] = 5000

        print("Badly scaled variables:")
        for v, sv in iscale.badly_scaled_var_generator(m, large=1e2, small=1e-2, zero=1e-12):
            print(f"    {v} -- {sv} -- {iscale.get_scaling_factor(v)}")

        solve_log = idaeslog.getInitLogger("infeasibility", idaeslog.INFO,
                                           tag="properties")
        log_infeasible_constraints(m, logger=solve_log, tol=1e-7)
        log_infeasible_bounds(m, logger=solve_log, tol=1e-7)
        # log_close_to_bounds(m, logger=solve_log)

        print("Badly scaled variables:")
        for v, sv in iscale.badly_scaled_var_generator(m, large=1e2, small=1e-2, zero=1e-12):
            print(f"    {v} -- {sv} -- {iscale.get_scaling_factor(v)}")

        opt.solve(m, tee=True)

        solve_log = idaeslog.getInitLogger("infeasibility", idaeslog.INFO,
                                           tag="properties")
        log_infeasible_constraints(m, logger=solve_log, tol=1e-7)
        log_infeasible_bounds(m, logger=solve_log, tol=1e-7)
        # log_close_to_bounds(m, logger=solve_log)
        for (i, blk) in enumerate(blks):
            blk.fs.pem.report()
            blk.fs.h2_tank.report()
            if hasattr(blk.fs, "tank_valve"):
                blk.fs.tank_valve.report()

        h2_prod.append([pyo.value(blks[i].fs.pem.outlet_state[0].flow_mol * 3600) for i in range(n_time_points)])
        h2_tank_in.append([pyo.value(blks[i].fs.h2_tank.inlet.flow_mol[0] * 3600) for i in range(n_time_points)])
        h2_tank_out.append([pyo.value(blks[i].fs.h2_tank.outlet.flow_mol[0] * 3600) for i in range(n_time_points)])
        h2_tank_holdup.append([pyo.value(blks[i].fs.h2_tank.material_holdup[0, ('Vap', 'hydrogen')]) for i in range(n_time_points)])
        wind_gen.append([pyo.value(blks[i].fs.windpower.electricity[0]) for i in range(n_time_points)])
        wind_to_grid.append([pyo.value(blks[i].fs.wind_to_grid[0]) for i in range(n_time_points)])
        wind_to_pem.append([pyo.value(blks[i].fs.pem.electricity[0]) for i in range(n_time_points)])

    n_weeks_to_plot = 1
    hours = np.arange(n_time_points*n_weeks_to_plot)
    lmp_array = weekly_prices[0:n_weeks_to_plot].flatten()
    h2_prod = np.asarray(h2_prod[0:n_weeks_to_plot]).flatten()
    wind_to_pem = np.asarray(wind_to_pem[0:n_weeks_to_plot]).flatten()
    wind_gen = np.asarray(wind_gen[0:n_weeks_to_plot]).flatten()
    wind_out = np.asarray(wind_to_grid[0:n_weeks_to_plot]).flatten()
    h2_tank_in = np.asarray(h2_tank_in[0:n_weeks_to_plot]).flatten()
    h2_tank_out = np.asarray(h2_tank_out[0:n_weeks_to_plot]).flatten()
    h2_tank_holdup = np.asarray(h2_tank_holdup[0:n_weeks_to_plot]).flatten()

    fig, ax1 = plt.subplots(2, 1, figsize=(12, 8))

    # color = 'tab:green'
    ax1[0].set_xlabel('Hour')
    ax1[0].set_ylabel('kW', )
    ax1[0].step(hours, wind_gen, label="Wind Generation")
    ax1[0].step(hours, wind_out, label="Wind to Grid")
    ax1[0].step(hours, wind_to_pem, label="Wind to Pem")
    ax1[0].tick_params(axis='y', )
    ax1[0].legend()

    ax2 = ax1[0].twinx()
    color = 'k'
    ax2.set_ylabel('LMP [$/MWh]', color=color)
    ax2.plot(hours, lmp_array[0:len(hours)], color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    ax1[1].set_xlabel('Hour')
    ax1[1].set_ylabel('kg', )
    ax1[1].step(hours, h2_prod, label="PEM H2 production")
    ax1[1].step(hours, h2_tank_in, label="Tank inlet")
    ax1[1].step(hours, h2_tank_out, label="Tank outlet")
    ax1[1].step(hours, h2_tank_holdup, label="Tank holdup")

    ax1[1].tick_params(axis='y', )
    ax1[1].legend()

    ax2 = ax1[1].twinx()
    color = 'k'
    ax2.set_ylabel('LMP [$/MWh]', color=color)
    ax2.plot(hours, lmp_array[0:len(hours)], color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    plt.show()

    print("wind mw", value(blks[0].fs.windpower.system_capacity))
    print("pem mw", value(m.pem_system_capacity))
    print("h2 contract", value(m.contract_capacity))
    print("h2 rev", value(m.hydrogen_revenue))
    print("annual rev", value(m.annual_revenue))
    print("npv", value(m.NPV))


wind_pem_tank_optimize()

# free tank for now