from dispatches.improved_pt.prescient_sweep.base_prescient_options import prescient_options
from prescient.simulator import Prescient

prescient_options["output_directory"] = f"base_pcm_simulation"
Prescient().simulate(**prescient_options)
