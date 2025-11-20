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

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dispatches.workflow.train_market_surrogates.dynamic.static_surrogate_results.RE_case_study.clustering_dispatch_wind_pem_static import ClusteringDispatchWind
from dispatches.workflow.train_market_surrogates.dynamic.static_surrogate_results.Simulation_Data_subscenario import SimulationData
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
import os
import pickle

'''
This script trains NN frequenct surrogate using the static clustering model 

for (P_grid, P_pem, P_wind).
'''

class TrainNNSurrogates:
    
    '''
    Train neural network surrogates for the dispatch frequency

    For RE case, filter is False.
    '''
    
    def __init__(self, simulation_data, clustering_class, clustering_model_path):

        '''
        Initialization for the class

        Arguments:
            simulation data: object, composition from ReadData class

            clustering_model_path: path of the saved clustering model

        Return

            None
        '''
        self.simulation_data = simulation_data
        self.clustering_class = clustering_class
        self.clustering_model_path = clustering_model_path
        

    def _read_clustering_model(self):
        # read clustering model
        with open (self.clustering_model_path, 'rb') as f:
            model = pickle.load(f)
        
        return model


    def _generate_label_data(self):

        '''
        Calculate the labels for NN training. 

        Arguments:

            wind_data: the wind profile.

        Return:

            dispatch_frequency_dict: {run_index: [dispatch frequency]}

        '''
        train_data = self.clustering_class.transform_data()
        clustering_model = self._read_clustering_model()
        
        # for train_data, the shape is 196716*3. Each 366*24 hour data is one year
        num_sims = int(len(train_data)/366/24)
        total_year_hours = 8784     # 366*24
        
        # reshape data into (num_sims), so that we can visit the data by each sweep simulation.
        label_data_reshaped = clustering_model.labels_.reshape((num_sims, total_year_hours))

        dispatch_frequency_dict = {}

        for idx in range(num_sims):
            # year_data has shape of (8784,3)
            elements, count = np.unique(label_data_reshaped[idx], return_counts=True)
            pred_result_dict = dict(zip(elements, count))
            count_dict = {}

            for j in range(self.clustering_class.num_clusters):

                if j in pred_result_dict.keys():
                    count_dict[j] = pred_result_dict[j]/total_year_hours
                
                else:   # if the frequency of cluster x is 0 in some years
                    count_dict[j] = 0

            dispatch_frequency_dict[idx] = []

            for key, value in count_dict.items():
                dispatch_frequency_dict[idx].append(value)      

        return dispatch_frequency_dict


    def _transform_dict_to_array(self):

        '''
        transform the dictionary data to array that keras can train

        Arguments:
        
            None

        Returns:

            x: features (input)
            y: labels (dispatch frequency)
        '''

        dispatch_frequency_dict = self._generate_label_data()

        index_list = list(self.simulation_data._dispatch_dict.keys())

        x = []
        y = []

        for idx in index_list:
            x.append(self.simulation_data._input_data_dict[idx])
            y.append(dispatch_frequency_dict[idx])

        return np.array(x), np.array(y)


    def train_NN_frequency(self, NN_size, cross_val=None, return_history=False):

        '''
        train the dispatch frequency NN surrogate model.
        print the R2 results of each cluster.

        Arguments:
            
            NN_size: list, the size of neural network. (input nodes, hidden layer 1 size, ..., output nodes )
            cross_val: int or None, number of folds for cross validation. If None, simple train-test split is used.
            return_history: bool, whether to return training history along with model

        Return:

            model: the NN model
            history: training history (if return_history=True)
        '''
        x, ws = self._transform_dict_to_array()

        # the first element of the NN_size dict is the input layer size, the last element is output layer size. 
        input_layer_size = NN_size[0]
        output_layer_size = NN_size[-1]
        hidden_layers = NN_size[1:-1]  # Store hidden layers without modifying original list

        if cross_val is not None and cross_val > 1:
            print(f"Performing {cross_val}-fold cross validation...")
            return self._train_frequency_with_cross_validation(x, ws, input_layer_size, output_layer_size, hidden_layers, cross_val)
        else:
            # Original train-test split approach
            x_train, x_test, ws_train, ws_test = train_test_split(x, ws, test_size=0.2, random_state=0)

            # scale the data both x and ws
            xm = np.mean(x_train,axis = 0)
            xstd = np.std(x_train,axis = 0)
            wsm = np.mean(ws_train,axis = 0)
            wsstd = np.std(ws_train,axis = 0)
            x_train_scaled = (x_train - xm) / xstd
            ws_train_scaled = (ws_train - wsm)/ wsstd

            # train a keras MLP (multi-layer perceptron) Regressor model
            model = keras.Sequential(name='static_clustering_NN')
            model.add(layers.Input(input_layer_size))
            for layer_size in hidden_layers:
                model.add(layers.Dense(layer_size, activation='sigmoid'))
            model.add(layers.Dense(output_layer_size))
            model.compile(optimizer=Adam(), loss='mse')
            history = model.fit(x=x_train_scaled, y=ws_train_scaled, verbose=0, epochs=500, validation_split=0.1)

            print("Making NN Predictions...") 

            # normalize the data
            x_test_scaled = (x_test - xm) / xstd
            ws_test_scaled = (ws_test - wsm) / wsstd

            print("Evaluate on test data")
            evaluate_res = model.evaluate(x_test_scaled, ws_test_scaled)
            print(evaluate_res)
            print(history.history['loss'][-1])
            if 'val_loss' in history.history:
                print(history.history['val_loss'][-1])
            predict_ws = np.array(model.predict(x_test_scaled))
            predict_ws_unscaled = predict_ws*wsstd + wsm

            R2 = []

            for rd in range(self.clustering_class.num_clusters):
                # compute R2 metric
                wspredict = predict_ws_unscaled.transpose()[rd]
                SS_tot = np.sum(np.square(ws_test.transpose()[rd] - wsm[rd]))
                SS_res = np.sum(np.square(ws_test.transpose()[rd] - wspredict))
                residual = 1 - SS_res/SS_tot
                R2.append(residual)

            print('The R2 of frequency surrogate validation is:', R2)

            xmin = list(np.min(x_train_scaled, axis=0))
            xmax = list(np.max(x_train_scaled, axis=0))

            data = {"xm_inputs":list(xm),"xstd_inputs":list(xstd),"xmin":xmin,"xmax":xmax,
                "ws_mean":list(wsm),"ws_std":list(wsstd)}

            self._model_params = data

            if return_history:
                return model, history
            else:
                return model


    def _train_frequency_with_cross_validation(self, x, ws, input_layer_size, output_layer_size, hidden_layers, k_folds):
        '''
        Train frequency neural network with k-fold cross validation
        
        Arguments:
            x: input features
            ws: target values (dispatch frequencies)
            input_layer_size: size of input layer
            output_layer_size: size of output layer
            hidden_layers: list of hidden layer sizes
            k_folds: int, number of folds for cross validation
            
        Returns:
            model: best performing model from cross validation
        '''
        
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        
        fold_scores = []
        fold_models = []
        fold_params = []
        
        print(f"Starting {k_folds}-fold cross validation for frequency model...")
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(x)):
            print(f"Training fold {fold + 1}/{k_folds}")
            
            # Split data for this fold
            x_train_fold, x_val_fold = x[train_idx], x[val_idx]
            ws_train_fold, ws_val_fold = ws[train_idx], ws[val_idx]
            
            # Scale the data
            xm = np.mean(x_train_fold, axis=0)
            xstd = np.std(x_train_fold, axis=0)
            wsm = np.mean(ws_train_fold, axis=0)
            wsstd = np.std(ws_train_fold, axis=0)
            
            # Avoid division by zero
            xstd = np.where(xstd == 0, 1, xstd)
            wsstd = np.where(wsstd == 0, 1, wsstd)
            
            x_train_scaled = (x_train_fold - xm) / xstd
            ws_train_scaled = (ws_train_fold - wsm) / wsstd
            x_val_scaled = (x_val_fold - xm) / xstd
            ws_val_scaled = (ws_val_fold - wsm) / wsstd
            
            # Create and train model
            model = keras.Sequential(name=f'static_clustering_NN_fold_{fold}')
            model.add(layers.Input(input_layer_size))
            for layer_size in hidden_layers:
                model.add(layers.Dense(layer_size, activation='sigmoid'))
            model.add(layers.Dense(output_layer_size))
            model.compile(optimizer=Adam(), loss='mse')
            
            # Train the model
            history = model.fit(
                x=x_train_scaled, 
                y=ws_train_scaled, 
                verbose=0, 
                epochs=500, 
                validation_data=(x_val_scaled, ws_val_scaled)
            )
            
            # Evaluate on validation set
            val_loss = model.evaluate(x_val_scaled, ws_val_scaled, verbose=0)
            predict_ws_val = model.predict(x_val_scaled, verbose=0)
            predict_ws_val_unscaled = predict_ws_val * wsstd + wsm
            
            # Calculate R2 for each cluster
            cluster_R2 = []
            cluster_RMSE = []
            
            for rd in range(self.clustering_class.num_clusters):
                # compute R2 metric for each cluster
                wspredict = predict_ws_val_unscaled.transpose()[rd]
                SS_tot = np.sum(np.square(ws_val_fold.transpose()[rd] - wsm[rd]))
                SS_res = np.sum(np.square(ws_val_fold.transpose()[rd] - wspredict))
                R2 = 1 - SS_res / SS_tot
                
                # Calculate RMSE for this cluster
                rmse = mean_squared_error(ws_val_fold.transpose()[rd], wspredict, squared=False)
                
                cluster_R2.append(R2)
                cluster_RMSE.append(rmse)
            
            # Calculate overall metrics
            overall_R2 = np.mean(cluster_R2)
            overall_RMSE = np.mean(cluster_RMSE)
            
            print(f"Fold {fold + 1} - Overall R2: {overall_R2:.4f}, Overall RMSE: {overall_RMSE:.4f}, Val Loss: {val_loss:.4f}")
            
            fold_scores.append({
                'overall_R2': overall_R2,
                'overall_RMSE': overall_RMSE,
                'cluster_R2': cluster_R2,
                'cluster_RMSE': cluster_RMSE,
                'val_loss': val_loss,
                'fold': fold + 1
            })
            fold_models.append(model)
            
            # Store parameters for this fold
            xmin = list(np.min(x_train_scaled, axis=0))
            xmax = list(np.max(x_train_scaled, axis=0))
            params = {
                "xm_inputs": list(xm),
                "xstd_inputs": list(xstd),
                "xmin": xmin,
                "xmax": xmax, 
                "ws_mean": list(wsm),
                "ws_std": list(wsstd)
            }
            fold_params.append(params)
        
        # Calculate cross-validation statistics
        cv_r2_scores = [score['overall_R2'] for score in fold_scores]
        cv_rmse_scores = [score['overall_RMSE'] for score in fold_scores]
        cv_loss_scores = [score['val_loss'] for score in fold_scores]
        
        print("\n" + "="*50)
        print("FREQUENCY MODEL CROSS VALIDATION RESULTS")
        print("="*50)
        print(f"Overall R2 - Mean: {np.mean(cv_r2_scores):.4f} ± {np.std(cv_r2_scores):.4f}")
        print(f"Overall RMSE - Mean: {np.mean(cv_rmse_scores):.4f} ± {np.std(cv_rmse_scores):.4f}")
        print(f"Val Loss - Mean: {np.mean(cv_loss_scores):.4f} ± {np.std(cv_loss_scores):.4f}")
        
        # Print cluster-wise statistics
        print("\nCluster-wise R2 Statistics:")
        for cluster_idx in range(self.clustering_class.num_clusters):
            cluster_r2_values = [score['cluster_R2'][cluster_idx] for score in fold_scores]
            print(f"Cluster {cluster_idx} - R2: {np.mean(cluster_r2_values):.4f} ± {np.std(cluster_r2_values):.4f}")
        print("="*50)
        
        # Select best model based on highest overall R2
        best_fold_idx = np.argmax(cv_r2_scores)
        best_model = fold_models[best_fold_idx]
        best_params = fold_params[best_fold_idx]
        
        print(f"Best model from fold {best_fold_idx + 1} with overall R2: {cv_r2_scores[best_fold_idx]:.4f}")
        
        # Store the best model parameters and cross-validation results
        self._model_params = best_params
        self._cv_results = {
            'fold_scores': fold_scores,
            'mean_overall_r2': np.mean(cv_r2_scores),
            'std_overall_r2': np.std(cv_r2_scores),
            'mean_overall_rmse': np.mean(cv_rmse_scores),
            'std_overall_rmse': np.std(cv_rmse_scores),
            'best_fold': best_fold_idx + 1,
            'num_clusters': self.clustering_class.num_clusters
        }
        
        return best_model


    def get_cross_validation_results(self):
        '''
        Get cross-validation results if available
        
        Returns:
            dict: Cross-validation results or None if not available
        '''
        if hasattr(self, '_cv_results'):
            return self._cv_results
        else:
            return None


    def plot_training_history(self, history, save_path=None):
        '''
        Plot training and validation loss from Keras history object
        
        Arguments:
            history: Keras History object from model.fit()
            save_path: str, optional path to save the plot
        
        Returns:
            None
        '''
        
        # Set up the plot style
        font1 = {
            'weight': 'bold',
            'size': 18,
        }
        
        font2 = {
            'weight': 'normal',
            'size': 15,
        }
        
        # Create the plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot training & validation loss
        ax1.plot(history.history['loss'], label='Training Loss', color='blue', linewidth=2)
        if 'val_loss' in history.history:
            ax1.plot(history.history['val_loss'], label='Validation Loss', color='red', linewidth=2)
        ax1.set_title('Model Loss', fontdict=font1)
        ax1.set_xlabel('Epoch', fontdict=font2)
        ax1.set_ylabel('Loss (MSE)', fontdict=font2)
        ax1.legend(prop=font2)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(labelsize=12)
        
        # Plot learning curve (loss in log scale for better visualization)
        ax2.semilogy(history.history['loss'], label='Training Loss', color='blue', linewidth=2)
        if 'val_loss' in history.history:
            ax2.semilogy(history.history['val_loss'], label='Validation Loss', color='red', linewidth=2)
        ax2.set_title('Model Loss (Log Scale)', fontdict=font1)
        ax2.set_xlabel('Epoch', fontdict=font2)
        ax2.set_ylabel('Loss (MSE) - Log Scale', fontdict=font2)
        ax2.legend(prop=font2)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(labelsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to: {save_path}")
        
        plt.show()


    def save_model(self, model, NN_model_path, NN_param_path):

        '''
        Save the model to the path which can be specified by the user. 

        Arguments:

            model: trained model from self.train_NN()

            fpath: if fpath == None, save to default path. 

        Return:

            None
        '''
        # save the NN model
        model.save(NN_model_path)

        # Include cross-validation results if available
        params_to_save = self._model_params.copy()
        if hasattr(self, '_cv_results'):
            params_to_save['cross_validation_results'] = self._cv_results
            print('Cross-validation results included in saved parameters')

        # save scaling parameters
        with open(NN_param_path, 'w') as f:
            json.dump(params_to_save, f, indent=2)

        return


    def plot_R2_results(self, NN_model_path, NN_param_path):
        
        # set the font for the plots
        font1 = {'weight' : 'bold',
            'size' : 18}
        
        font2 = {
            'weight' : 'normal',
            'size'   : 15,
            }
        x, ws = self._transform_dict_to_array()
        x_train, x_test, ws_train, ws_test = train_test_split(x, ws, test_size=0.2, random_state=0)
        # load the NN model from the given path
        NN_model = keras.models.load_model(NN_model_path)

        with open(NN_param_path, 'r') as f:
            NN_param = json.load(f)

        # scale data
        xm = NN_param['xm_inputs']
        xstd = NN_param['xstd_inputs']
        wsm = NN_param['ws_mean']
        wsstd = NN_param['ws_std']

        x_train_scaled = (x_train - xm)/xstd
        x_test_scaled = (x_test - xm)/xstd
        pred_ws_test = NN_model.predict(x_test_scaled)
        pred_ws_train = NN_model.predict(x_train_scaled)
        pred_ws_train_unscaled = pred_ws_train*wsstd + wsm
        pred_ws_test_unscaled = pred_ws_test*wsstd + wsm

        # calculate the R2 for each representative day
        test_R2 = []

        for rd in range(self.clustering_class.num_clusters):
            # compute R2 metric
            ws_test_predict = pred_ws_test_unscaled.transpose()[rd]
            SS_tot = np.sum(np.square(ws_test.transpose()[rd] - wsm[rd]))
            SS_res = np.sum(np.square(ws_test.transpose()[rd] - ws_test_predict))
            residual = 1 - SS_res/SS_tot
            test_R2.append(residual)
        
        print(test_R2)
        
        # plot the figure
        for i in range(self.clustering_class.num_clusters):
            fig, axs = plt.subplots()
            axs.set_ylabel('Predicted dispatch frequency [%]', font = font1)
            axs.set_xlabel('True dispatch frequency [%]', font = font1)
            fig.set_size_inches(8,8)

            wst_train = ws_train.transpose()[i]
            wsp_train = pred_ws_train_unscaled.transpose()[i]
            wst_test = ws_test.transpose()[i]
            wsp_test = pred_ws_test_unscaled.transpose()[i]

            axs.scatter(wst_train*100, wsp_train*100, color = "blue",alpha = 1, label = 'train')
            axs.scatter(wst_test*100, wsp_test*100, color = "red", marker="^", alpha = 1, label = 'test')
            axs.plot([min(min(wst_train*100), min(wst_test*100)),max(max(wst_train*100), max(wst_test*100))],[min(min(wst_train*100), min(wst_test*100)),max(max(wst_train*100), max(wst_test*100))], color = "black")
            axs.set_title(f'Cluster_{i}',font = font1)
            xcoor = (max(max(wst_train*100), max(wst_test*100)) - min(min(wst_train*100), min(wst_test*100)))*0.6 + min(min(wst_train*100), min(wst_test*100))
            axs.annotate("$R^2 = {}$".format(round(test_R2[i],3)), xy=(xcoor, min(min(wst_train*100), min(wst_test*100))), font = font1)
            
            plt.legend(prop=font2)
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            plt.tick_params(direction="in",top=True, right=True)

            fig_name_ = f'static_clustering_NN_cluster_{i}.jpg'
            plt.savefig(f"R2_figures\\{fig_name_}",dpi =300)
    
    

    def check_results(self, input_data_path):
        # read the dispatch data
        dispatch_array = self.clustering_class.read_dispatch_data()
        wind_data = self.clustering_class.read_wind_data()
        pem_data = self.clustering_class.calculate_PEM_cf()

        df_input_data = pd.read_hdf(input_data_path)
        num_col = df_input_data.shape[1]
        num_row = df_input_data.shape[0]
        X = df_input_data.iloc[list(range(num_row)),list(range(1,num_col))].to_numpy()

        clustering_model = self._read_clustering_model()
        dispatch_frequency_dict = self._generate_label_data()

        check_dict = {}
        true_dict = {}
        for i in dispatch_frequency_dict:
            d = 0
            p = 0
            w = 0
            for j in range(len(clustering_model.cluster_centers_)):
                d += dispatch_frequency_dict[i][j]*clustering_model.cluster_centers_[j][0]    # dispatch
                p += dispatch_frequency_dict[i][j]*clustering_model.cluster_centers_[j][1]    # PEM
                w += dispatch_frequency_dict[i][j]*clustering_model.cluster_centers_[j][2]    # wind
            rd = sum(dispatch_array[i]/847)/8784
            rp = sum(pem_data[i])/8784
            rw = sum(wind_data/847)/8784
            check_dict[i] = [d,p,w]
            true_dict[i] = [rd,rp,rw]

        for i, j in zip(check_dict, true_dict):
            print(true_dict[j], check_dict[i])

        return

