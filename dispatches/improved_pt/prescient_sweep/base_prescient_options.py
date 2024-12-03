import os
from dispatches_sample_data import rts_gmlc
from dispatches.case_studies.renewables_case.double_loop_utils import read_rts_gmlc_wind_inputs


rtsgmlc_path = rts_gmlc.source_data_path
this_file_path = os.path.dirname(os.path.realpath(__file__))
# default some options
shortfall = 500
prescient_options = {
        "data_path":rtsgmlc_path,
        "reserve_factor":0.15,
        "simulate_out_of_sample":True,
        "output_directory":None,
        "monitor_all_contingencies":False,
        "input_format":"rts-gmlc",
        "start_date":"01-01-2020",
        "num_days":366,
        "sced_horizon":1,
        "ruc_mipgap":0.01,
	    "deterministic_ruc_solver": "gurobi",
	    "deterministic_ruc_solver_options" : {"threads":2, "heurstrategy":2, "cutstrategy":3, "symmetry":2, "maxnode":1000},
        "sced_solver":"gurobi",
        "sced_frequency_minutes":60,
	    "sced_solver_options" : {"threads":1},
        "ruc_horizon":36,
        "compute_market_settlements":True,
        "output_solver_logs":False,
        "price_threshold":shortfall,
        "transmission_price_threshold":shortfall/2,
        "contingency_price_threshold":None,
        "reserve_price_threshold":shortfall/10,
        "day_ahead_pricing":"aCHP",
        "enforce_sced_shutdown_ramprate":False,
        "ruc_slack_type":"ref-bus-and-branches",
        "sced_slack_type":"ref-bus-and-branches",
	"disable_stackgraphs":True,
        "symbolic_solver_labels":True,
        "output_ruc_solutions": False,
        "write_deterministic_ruc_instances": False,
        "write_sced_instances": False,
        "print_sced":False
        }

# read the wind capacity factor dataframe and make a NPP capacity factor in the same format
# this is for the perfect forecaster
wind_generator = "303_WIND_1"
NPP_df = read_rts_gmlc_wind_inputs(rts_gmlc.source_data_path, wind_generator)
NPP_df.columns = ["121_NUCLEAR_1-RTCF", "121_NUCLEAR_1-RTCF"]
NPP_df["121_NUCLEAR_1-RTCF"] = 1.0
NPP_df["121_NUCLEAR_1-DACF"] = 1.0
