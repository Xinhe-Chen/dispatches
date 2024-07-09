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

this_file_path = os.path.dirname(os.path.realpath(__file__))

# submit price-taker jobs for wind PEM case study.  

def submit_job(
    sim_id,
    PEM_ratio=0.1,
    H2_price=0.8,
    market = 'DA'
):

    # create a directory to save job scripts
    job_scripts_dir = os.path.join(this_file_path, "sim_job_scripts")
    if not os.path.isdir(job_scripts_dir):
        os.mkdir(job_scripts_dir)

    file_name = os.path.join(job_scripts_dir, f"wind_PEM_price_taker_{market}_sim_{sim_id}.sh")
    with open(file_name, "w") as f:
        f.write(
            "#!/bin/bash\n"
            + "#$ -M xchen24@nd.edu\n"
            + "#$ -m ae\n"
            + "#$ -q long\n"
            + f"#$ -N wind_PEM_price_taker_sim_{sim_id}\n"
            + "conda activate regen\n"
            + "export LD_LIBRARY_PATH=~/.conda/envs/regen/lib:$LD_LIBRARY_PATH \n"
            + "module load gurobi/9.5.1\n"
            + "module load ipopt/3.14.2 \n"
            + f"python ./run_pricetaker_wind_PEM_new.py --PEM_ratio {PEM_ratio} --H2_price {H2_price}"
        )

    os.system(f"qsub {file_name}")


if __name__ == "__main__":
    sim_id = 0
    pem_ratio_range = [i / 100 for i in range(5, 51, 5)] 
    h2_price_range = [0.75, 1, 1.25, 1.5, 1.75, 2]
    market = 'DA'
    for i in h2_price_range:
        for j in pem_ratio_range:
            sim_id += 1
            PEM_ratio = j
            H2_price = i
            submit_job(sim_id, PEM_ratio, H2_price)