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
from Backcaster_of_price_taker import PricetakerBackcaster

lmps_df = pd.read_parquet(Path(__file__).parent / "data" / "303_LMPs_15_reserve_500_shortfall.parquet")
lmps = lmps_df['LMP'].values
lmps[lmps>500] = 500
signal = lmps # even we use rt lmp signals, we call it DA_LMPs to simplify the work.

scenario = 3
# horizon = 24

backcaster = PricetakerBackcaster(price_signal=signal, scenario=scenario)
for i in range(5):
    backcaster.pointer=i
    gen_signal = backcaster.generate_price_scenarios()
    print(gen_signal, len(gen_signal))
