import os
import sys

from parameters import update_function
from dispatches.improved_pt.prescient_sweep.base_prescient_options import prescient_options
from dispatches.improved_pt.prescient_sweep.utils import run_sweep

prescient_options["output_directory"] = 'test_ne_sweep'

# test the code, only run 3 cases.
start, stop = 0, 3

run_sweep(update_function, prescient_options, start, stop)