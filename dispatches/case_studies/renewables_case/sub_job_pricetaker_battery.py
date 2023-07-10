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
import os
from prescient_options import sim_name
from load_parameters import year, scenario, duration

this_file_path = os.path.dirname(os.path.realpath(__file__))


def submit_job(
    battery_ratio,
    year,
    scenario,
    battery_duration
):

    # create a directory to save job scripts
    job_scripts_dir = os.path.join(this_file_path, "sim_job_scripts")
    if not os.path.isdir(job_scripts_dir):
        os.mkdir(job_scripts_dir)

    file_name = os.path.join(job_scripts_dir, f"{scenario}_{year}_wind_battery_rerun_size_duration_{battery_duration}_ratio_{battery_ratio}"  + ".sh")
    with open(file_name, "w") as f:
        f.write(
            "#!/bin/bash\n"
            + "#$ -M xchen24@nd.edu\n"
            + "#$ -m ae\n"
            + "#$ -q long\n"
            + "#$ -N " + f"{scenario}_{year}_wind_battery_rerun_size_duration_{battery_duration}_ratio_{battery_ratio}" + "\n"
            + "conda activate regen\n"
            + "export LD_LIBRARY_PATH=~/.conda/envs/regen/lib:$LD_LIBRARY_PATH \n"
            + "module load gurobi/9.5.1\n"
            + "module load ipopt/3.14.2 \n"
            + f"python ./run_pricetaker_battery_ratio_size.py --battery_ratio {battery_ratio}  --duration {battery_duration} --year {year}"
        )

    os.system(f"qsub {file_name}")


if __name__ == "__main__":
    
    scenario = scenario
    year = year
    battery_duration = duration
    battery_ratio = 0
#    for i in range(1 ,11, 1):
#        battery_ratio = i/10
#        submit_job(battery_ratio, year, scenario, battery_duration)
    submit_job(battery_ratio, year, scenario, battery_duration)    
