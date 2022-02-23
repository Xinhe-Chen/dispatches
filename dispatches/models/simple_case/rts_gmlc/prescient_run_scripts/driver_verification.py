#NOTE: This uses the `naerm` branch of Prescient on the Sandia gitlab server
import prescient.scripts.simulator as simulator
from prescient.scripts.runner import parse_line
from pyomo.common.fileutils import this_file_dir
import os

start_date = '01-02-2020'
days = 364
base_index = 0
base_output_dir = this_file_dir()+'/../prescient_results/verification_runs/scikit_run_{}'.format(base_index)
#base_output_dir = this_file_dir()+'/../prescient_results/verification_runs/alamo_run_{}'.format(base_index)
os.makedirs(base_output_dir, exist_ok=True)

def main():
    import prescient.scripts.simulator as simulator
    from prescient.scripts.runner import parse_line

    options = [
    '--data-directory=deterministic_scenarios',
    '--model-directory=/home/jhjalvi/git/prescient_idaes/prescient/models/knueven',
    '--output-directory='+base_output_dir+'/deterministic_simulation_output_index_{}'.format(base_index),
    '--run-deterministic-ruc',
    '--start-date='+start_date,
    '--num-days={}'.format(days),
    '--sced-horizon=4',
    '--traceback',
    '--ruc-mipgap=0.001',
    '--deterministic-ruc-solver=gurobi_direct',
    '--deterministic-ruc-solver-options="threads=4"',
    '--sced-solver=gurobi_direct',
    '--sced-solver-options="threads=4"',
    '--ruc-horizon=36',
	'--simulator-plugin=/home/jhjalvi/git/dispatches/dispatches/models/simple_case/rts_gmlc/prescient_run_scripts/plugin_verify_surrogate.py',
    '--disable-stackgraphs'
    ]

    simulator.main(args=options)

if __name__ == '__main__':
    main()


