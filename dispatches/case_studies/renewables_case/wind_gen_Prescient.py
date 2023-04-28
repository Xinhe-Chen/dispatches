import os
from prescient.simulator import Prescient

current_path = os.getcwd()
rtsgmlc_path = "/afs/crc.nd.edu/user/x/xchen24/GitHub/RTS-GMLC/RTS_Data/SourceData/"
output_path = os.path.join(current_path,'Benchmark_single_wind_gen_sim')
shortfall = 200
# default some options
prescient_options = {
        "data_path":rtsgmlc_path,
        "reserve_factor":0.15,
        "simulate_out_of_sample":True,
        "output_directory":output_path,
        "monitor_all_contingencies":False,
        "input_format":"rts-gmlc",
        "start_date":"01-01-2020",
        "num_days":366,
        "sced_horizon":1,
        "ruc_mipgap":0.01,
	    "deterministic_ruc_solver": "gurobi",
	    "deterministic_ruc_solver_options" : {"threads":4, "heurstrategy":2, "cutstrategy":3, "symmetry":2, "maxnode":1000},
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

Prescient().simulate(**prescient_options)