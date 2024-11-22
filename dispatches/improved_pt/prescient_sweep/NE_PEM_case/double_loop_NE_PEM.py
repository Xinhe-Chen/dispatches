from prescient.simulator import Prescient
from argparse import ArgumentParser
from pathlib import Path
import pyomo.environ as pyo
from pyomo.common.fileutils import this_file_dir
from idaes.apps.grid_integration import Tracker
from idaes.apps.grid_integration.model_data import ThermalGeneratorModelData
from dispatches.workflow.coordinator import DoubleLoopCoordinator
from dispatches.case_studies.renewables_case.wind_PEM_double_loop import MultiPeriodWindPEM
# from dispatches.case_studies.renewables_case.load_parameters import *
from idaes.apps.grid_integration.bidder import PEMParametrizedBidder
from idaes.apps.grid_integration.forecaster import PerfectForecaster
from dispatches_sample_data import rts_gmlc
from dispatches.case_studies.renewables_case.double_loop_utils import read_rts_gmlc_wind_inputs
from dispatches.improved_pt.prescient_sweep.base_prescient_options import prescient_options

###
# Script to run a double loop simulation of a Nuclear + PEM flowsheet in Prescient.
# The DoubleLoopCoordinator is from dispatces.workflow.coordinator, not from idaes.apps.grid_integration.
# This version contains some modifications for generator typing and will be merged into idaes in the future.
###

ne_prescient_options = prescient_options.copy()
shortfall = 500
ne_prescient_options["price_threshold"] = shortfall
ne_generator = '121_NUCLEAR_1'

usage = "Run double loop simulation with RE model."
parser = ArgumentParser(usage)

parser.add_argument(
    "--sim_id",
    dest="sim_id",
    help="Indicate the simulation ID.",
    action="store",
    type=int,
    default=0,
)

parser.add_argument(
    "--ne_pmax",
    dest="Nuclear_pmax",
    help="Set nuclear power plant capacity in MW.",
    action="store",
    type=float,
    default=400,
)

parser.add_argument(
    "--pem_pmax",
    dest="pem_pmax",
    help="Set the PEM power capacity in MW.",
    action="store",
    type=float,
    default=200,
)

parser.add_argument(
    "--pem_bid",
    dest="pem_bid",
    help="Set the PEM bid price in $/MW.",
    action="store",
    type=float,
    default=25,
)

options = parser.parse_args()

sim_id = options.sim_id
ne_pmax = options.ne_pmax
pem_pmax = options.pem_pmax
pem_bid = options.pem_bid
p_min = 0

output_dir = Path(f"test_results/NE_double_loop_test_{sim_id}")

solver = pyo.SolverFactory('ipopt')

thermal_generator_params = {
    "gen_name": ne_generator,
    "bus": 'Attlee',
    "p_min": p_min,
    "p_max": ne_pmax,
    "min_down_time": 0,
    "min_up_time": 0,
    "ramp_up_60min": wind_pmax,
    "ramp_down_60min": wind_pmax,
    "shutdown_capacity": wind_pmax,
    "startup_capacity": wind_pmax,
    "initial_status": 1,                                        # Has been off for 1 hour before start of simulation
    "initial_p_output": 0,
    "production_cost_bid_pairs": [(p_min, 0), (wind_pmax, 0)],
    "include_default_p_cost": False,
    "startup_cost_pairs": [(0, 0)],
    "fixed_commitment": 1,                                      # Same as the plant in the parameter sweep, which was RE-type and always on
    "spinning_capacity": 0,                                     # Disable participation in some reserve services
    "non_spinning_capacity": 0,
    "supplemental_spinning_capacity": 0,
    "supplemental_non_spinning_capacity": 0,
    "agc_capable": False
}
model_data = ThermalGeneratorModelData(**thermal_generator_params)


################################################################################
################################# bidder #######################################
################################################################################

# PerfectForecaster uses Dataframe with RT and DA wind resource and LMPs. However, for the ParametrizedBidder that
# makes the bid curve based on the `pem_bid` parameter, the LMPs are not needed and are never accessed.
# For non-RE plants, the CFs are never accessed.
forecaster = PerfectForecaster(wind_df)

mp_wind_pem_bid = MultiPeriodWindPEM(
    model_data=model_data,
    wind_capacity_factors=wind_rt_cfs,
    wind_pmax_mw=wind_pmax,
    pem_pmax_mw=pem_pmax
)

# ParametrizedBidder is an alternative to the stochastic LMP problem, and serves up a bid curve per timestep
# that is a function of only the bid parameter, PEM capacity and wind resource (DA or RT)
bidder_object = PEMParametrizedBidder(
    bidding_model_object=mp_wind_pem_bid,
    day_ahead_horizon=day_ahead_horizon,
    real_time_horizon=real_time_horizon,
    solver=solver,
    forecaster=forecaster,
    renewable_mw=wind_pmax,
    pem_marginal_cost=pem_bid,
    pem_mw=pem_pmax
)

################################################################################
################################# Tracker ######################################
################################################################################

# same tracking_horizon as parameter sweep and n_tracking_hour does not need to be >1
mp_wind_pem_track = MultiPeriodWindPEM(
    model_data=model_data,
    wind_capacity_factors=wind_rt_cfs,
    wind_pmax_mw=wind_pmax,
    pem_pmax_mw=pem_pmax
)

# create a `Tracker` using`mp_wind_battery`
tracker_object = Tracker(
    tracking_model_object=mp_wind_pem_track,
    tracking_horizon=tracking_horizon,
    n_tracking_hour=n_tracking_hour,
    solver=solver,
)

mp_wind_pem_track_project = MultiPeriodWindPEM(
    model_data=model_data,
    wind_capacity_factors=wind_rt_cfs,
    wind_pmax_mw=wind_pmax,
    pem_pmax_mw=pem_pmax
)

# create a `Tracker` using`mp_wind_battery`
project_tracker_object = Tracker(
    tracking_model_object=mp_wind_pem_track_project,
    tracking_horizon=tracking_horizon,
    n_tracking_hour=n_tracking_hour,
    solver=solver,
)

################################################################################
################################# Coordinator ##################################
################################################################################

coordinator = DoubleLoopCoordinator(
    bidder=bidder_object,
    tracker=tracker_object,
    projection_tracker=project_tracker_object,
)

ne_prescient_options["output_directory"] = output_dir
ne_prescient_options["plugin"] = {
    "doubleloop": {
        "module": coordinator.prescient_plugin_module,
        "bidding_generator": ne_generator,
    }
}

Prescient().simulate(**ne_prescient_options)

# write options into the result folder
with open(output_dir / "sim_options_pem.json", "w") as f:
    f.write(str(thermal_generator_params.update(ne_prescient_options)))