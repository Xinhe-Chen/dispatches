import os
import sys
from argparse import ArgumentParser
from parameters import update_function
from dispatches.improved_pt.prescient_sweep.base_prescient_options import prescient_options
from dispatches.improved_pt.prescient_sweep.utils import parameter_sweep_runner

usage = "Run PCM sweep with NE+PEM model."
parser = ArgumentParser(usage)

parser.add_argument(
    "--index",
    dest="index",
    help="Indicate the simulation index.",
    action="store",
    type=int,
    default=0,
)

parser.add_argument(
    "--pem_pmax_ratio",
    dest="pem_pmax_ratio",
    help="Set the PEM power as a ratio of the NPP capacity",
    action="store",
    type=float,
    default=0.5,
)

parser.add_argument(
    "--pem_bid",
    dest="pem_bid",
    help="Set the PEM bid price in $/MW.",
    action="store",
    type=float,
    default=15,
)
options = parser.parse_args()

index = options.index
pem_pmax_ratio = options.pem_pmax_ratio
pem_bid = options.pem_bid
prescient_options["output_directory"] = 'test_ne_sweep'

PEM_data = {"PEM_indifference_point":pem_bid, "PEM_fraction":pem_pmax_ratio}
parameter_sweep_runner(update_function, prescient_options, index)