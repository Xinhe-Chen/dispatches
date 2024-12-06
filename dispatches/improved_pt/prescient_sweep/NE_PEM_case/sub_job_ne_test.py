import os
import numpy as np

this_file_path = os.path.dirname(os.path.realpath(__file__))


def submit_job(index, pem_pmax_ratio, pem_bid):
    # create a directory to save job scripts
    job_scripts_dir = os.path.join(this_file_path, "sim_job_scripts")
    if not os.path.isdir(job_scripts_dir):
        os.mkdir(job_scripts_dir)

    file_name = os.path.join(job_scripts_dir, f"test_ne_pcm_sweep_bid_{pem_bid}_{index}.sh")
    with open(file_name, "w") as f:
        f.write(
            "#!/bin/bash\n"
            + "#$ -M xchen24@nd.edu\n"
            + "#$ -m ae\n"
            + "#$ -q long\n"
            + f"#$ -N test_ne_pcm_sweep_bid_{pem_bid}_{index}\n"
            + "conda activate regen\n"
            + "export LD_LIBRARY_PATH=~/.conda/envs/regen/lib:$LD_LIBRARY_PATH \n"
            + "module load gurobi/9.5.1\n"
            + "module load ipopt/3.14.2 \n"
            + f"python ./nuclear_sweep_test.py --index {index} --pem_pmax_ratio {pem_pmax_ratio} --pem_bid {pem_bid}"
        )

    os.system(f"qsub {file_name}")


if __name__ == "__main__":
    
    # for the sweep, pem_pmax_ratio starts from 0.01 (4MW) to 1.0 (400MW)
    idx = 1
    pem_bid = 15
    for i in np.linspace(0.01, 1, 100):
        pem_pmax_ratio = np.round(i,2)
        index = idx
        idx += 1    
        submit_job(index, pem_pmax_ratio, pem_bid)