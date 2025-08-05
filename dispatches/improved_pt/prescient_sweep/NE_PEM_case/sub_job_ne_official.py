import os
import numpy as np
import json

this_file_path = os.path.dirname(os.path.realpath(__file__))


def submit_job(index, pem_pmax_ratio, pem_bid):
    # create a directory to save job scripts
    job_scripts_dir = os.path.join(this_file_path, "sim_job_scripts")
    if not os.path.isdir(job_scripts_dir):
        os.mkdir(job_scripts_dir)

    file_name = os.path.join(job_scripts_dir, f"NE_PEM_pcm_sweep_bid_{pem_bid}_{index}.sh")
    with open(file_name, "w") as f:
        f.write(
            "#!/bin/bash\n"
            + "#$ -M xchen24@nd.edu\n"
            + "#$ -m ae\n"
            + "#$ -q long\n"
            + f"#$ -N NE_PEM_pcm_sweep_bid_{pem_bid}_{index}\n"
            + "conda activate regen\n"
            + "export LD_LIBRARY_PATH=~/.conda/envs/regen/lib:$LD_LIBRARY_PATH \n"
            + "module load gurobi/9.5.1\n"
            + "module load ipopt/3.14.2 \n"
            + f"python ./nuclear_sweep_test.py --index {index} --pem_pmax_ratio {pem_pmax_ratio} --pem_bid {pem_bid}"
        )

    os.system(f"qsub {file_name}")


if __name__ == "__main__":
    
    # for the sweep, pem_pmax_ratio starts from 0.01 (4MW) to 1.0 (400MW)
    idx = 0
    sweep_record = {}
    job_scripts_dir = os.path.join(this_file_path, "sim_job_scripts")
    pem_bid = list(range(5, 50, 5))  # bid from 5 to 45 with step 5
    PEM_ratio = list(range(1, 11, 1))
    for i in PEM_ratio:
        for bid in pem_bid:
            ratio = np.round(i/10, 1)  # convert to 0.1, 0.2, ..., 1.0
            index = idx
            submit_job(index, ratio, bid)
            sweep_record[idx] = {"pem_pmax_ratio": ratio, "pem_bid": bid}
            idx += 1

    # save the sweep record
    with open(os.path.join(job_scripts_dir, "sweep_record.json"), "w") as f:
        json.dump(sweep_record, f)