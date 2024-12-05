import pandas as pd
from idaes.apps.grid_integration.bidder import PEMParametrizedBidder
from idaes.apps.grid_integration.forecaster import PerfectForecaster
from dispatches.case_studies.renewables_case.double_loop_utils import read_rts_gmlc_wind_inputs
from dispatches_sample_data import rts_gmlc
from dispatches.improved_pt.prescient_sweep.base_prescient_options import prescient_options, NPP_df

# wind_generator = "303_WIND_1"
# NPP_df = read_rts_gmlc_wind_inputs(rts_gmlc.source_data_path, wind_generator)
# NPP_df.columns = ["121_NUCLEAR_1-RTCF", "121_NUCLEAR_1-DACF"]
# NPP_df["121_NUCLEAR_1-RTCF"] = 1.0
# NPP_df["121_NUCLEAR_1-DACF"] = 1.0

date = '2020-01-01'
hour = 12
horizon = 36
gen = '121_NUCLEAR_1'
# col=  f"{gen}-DACF"
# datetime_index = pd.to_datetime(date) + pd.Timedelta(hours=hour)
# forecast = NPP_df[NPP_df.index >= datetime_index].head(horizon)
# values = forecast[col].values
# print(NPP_df.index)


pf = PerfectForecaster(NPP_df)
pf.forecast_day_ahead_capacity_factor(date, hour, gen, horizon)