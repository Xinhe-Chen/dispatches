#################################################################################
# DISPATCHES was produced under the DOE Design Integration and Synthesis
# Platform to Advance Tightly Coupled Hybrid Energy Systems program (DISPATCHES),
# and is copyright (c) 2021 by the software owners: The Regents of the University
# of California, through Lawrence Berkeley National Laboratory, National
# Technology & Engineering Solutions of Sandia, LLC, Alliance for Sustainable
# Energy, LLC, Battelle Energy Alliance, LLC, University of Notre Dame du Lac, et
# al. All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. Both files are also available online at the URL:
# "https://github.com/gmlc-dispatches/dispatches".
#################################################################################

from dispatches.workflow.train_market_surrogates.dynamic.Simulation_Data import SimulationData
from dispatches.workflow.train_market_surrogates.dynamic.Train_NN_Surrogates import TrainNNSurrogates
from dispatches.workflow.train_market_surrogates.dynamic.Time_Series_Clustering import TimeSeriesClustering
import pathlib
from dispatches_data.api import path

# this is for training revenue/dynamic dispatch frequency surrogates

def main():
    # for NE case study
    path_to_data_package = path("dynamic_sweep")
    case_type = "NE"
    model_id = "dispatch"
    num_clusters = 30
    num_sims = 192

    dispatch_data_path = path_to_data_package / case_type / f"Dispatch_data_{case_type}_Dispatch_whole.csv"
    input_data_path = path_to_data_package / case_type / f"sweep_parameters_results_{case_type}_whole.h5"
    input_layer_node = 4
    filter_opt = True
    clustering_result_path = str(pathlib.Path.cwd().joinpath(f'{case_type}_case_study', f'{case_type}_{num_sims}years_{num_clusters}clusters_OD.json'))
    revenue_data_path = str(pathlib.Path.cwd().joinpath(f'{case_type}_case_study', "RE_H2_RT_revenue.csv"))



    # test TimeSeriesClustering
    print('Read simulation data')
    simulation_data = SimulationData(dispatch_data_path, input_data_path, num_sims, case_type)

    if model_id not in ["revenue", "dispatch", "clustering"]:
        raise TypeError("Invalid model type.")

    if model_id == "clustering":
        print('Start Time Series Clustering')
        clusteringtrainer = TimeSeriesClustering(simulation_data, num_clusters, filter_opt)
        # clustering_model = clusteringtrainer.clustering_data()
        clustering_result_path = str(pathlib.Path.cwd().joinpath(f'{case_type}_case_study', f'{case_type}_{num_sims}years_{num_clusters}clusters_OD.json'))
        # clusteringtrainer.save_clustering_model(clustering_model, fpath = clustering_result_path)
        # plot results
        clusteringtrainer.plot_results(clustering_result_path)
        # clusteringtrainer.box_plots(clustering_result_path)
    
    elif model_id == "revenue":
        # TrainNNSurrogates, revenue
        print('Start train revenue surrogate')
        hidden_nodes = 30
        hidden_layers = 2

        NN_rev_model_path = str(pathlib.Path.cwd().joinpath(f'{case_type}_case_study', 'revenue', f'{case_type}_revenue_{hidden_layers}_{hidden_nodes}'))
        NN_rev_param_path = str(pathlib.Path.cwd().joinpath(f'{case_type}_case_study', 'revenue', f'{case_type}_revenue_params_{hidden_layers}_{hidden_nodes}.json'))

        NNtrainer_rev = TrainNNSurrogates(simulation_data, revenue_data_path, filter_opt)
        # model_rev = NNtrainer_rev.train_NN_revenue([input_layer_node,hidden_nodes,hidden_nodes,1])

        # save to given path
        NNtrainer_rev.model_type = 'revenue'
        # NNtrainer_rev.save_model(model_rev, NN_rev_model_path, NN_rev_param_path)
        # NNtrainer_rev.plot_R2_results(NN_rev_model_path, NN_rev_param_path)

    # TrainNNSurrogates, dispatch frequency
    elif model_id == "dispatch":
        print('Start train dispatch frequency surrogate')
        clustering_model_path = clustering_result_path
        NNtrainer_df = TrainNNSurrogates(simulation_data, clustering_model_path, filter_opt = filter_opt)
        model_df = NNtrainer_df.train_NN_frequency([input_layer_node,75,75,75,num_clusters+2])
        NN_frequency_model_path = str(pathlib.Path.cwd().joinpath(f'{case_type}_case_study', f'{case_type}_{num_clusters}clusters_dispatch_frequency'))
        NN_frequency_param_path = str(pathlib.Path.cwd().joinpath(f'{case_type}_case_study', f'{case_type}_{num_clusters}clusters_dispatch_frequency_params.json'))
        # NNtrainer_df.save_model(model_df, NN_frequency_model_path, NN_frequency_param_path)
        NNtrainer_df.plot_R2_results(NN_frequency_model_path, NN_frequency_param_path)

if __name__ == "__main__":
    main()