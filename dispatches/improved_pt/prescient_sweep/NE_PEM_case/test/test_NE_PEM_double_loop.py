import pandas as pd
from idaes.apps.grid_integration.bidder import PEMParametrizedBidder
from idaes.apps.grid_integration.forecaster import PerfectForecaster
from dispatches.case_studies.renewables_case.double_loop_utils import read_rts_gmlc_wind_inputs
from dispatches_sample_data import rts_gmlc

wind_generator = "303_WIND_1"
wind_df = read_rts_gmlc_wind_inputs(rts_gmlc.source_data_path, wind_generator)
wind_df.columns = ["121_NUCLEAR_1-RTCF", "121_NUCLEAR_1-RTCF"]
wind_df["121_NUCLEAR_1-RTCF"] = 1.0
wind_df["121_NUCLEAR_1-DACF"] = 1.0
# print(wind_df)