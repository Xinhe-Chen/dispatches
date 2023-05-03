import os

shortfall = 200
real_time_horizon = 4
this_file_path = os.path.dirname(os.path.realpath(__file__))


def submit_job(sim_id):

    # create a directory to save job scripts
    job_scripts_dir = os.path.join(this_file_path, "sim_job_scripts")
    if not os.path.isdir(job_scripts_dir):
        os.mkdir(job_scripts_dir)

    file_name = os.path.join(job_scripts_dir, f"new_Benchmark_only_wind_rf_15_shortfall_{shortfall}_rth_{real_time_horizon}.sh")
    with open(file_name, "w") as f:
        f.write(
            "#!/bin/bash\n"
            + "#$ -M xchen24@nd.edu\n"
            + "#$ -m ae\n"
            + "#$ -q long\n"
            + f"#$ -N new_Benchmark_wind_gen_Prescient_shortfall_{shortfall}_rth_{real_time_horizon}\n"
            + "conda activate regen\n"
            + "module load gurobi/9.5.1\n"
            + f"python ./wind_gen_Prescient.py")

    os.system(f"qsub {file_name}")


sim_id = 0


submit_job(sim_id)
