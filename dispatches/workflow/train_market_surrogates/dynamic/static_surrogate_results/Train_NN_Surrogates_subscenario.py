#################################################################################
# DISPATCHES was produced under the DOE Design Integration and Synthesis
# Platform to Advance Tightly Coupled Hybrid Energy Systems program (DISPATCHES),
# and is copyright (c) 2022 by the software owners: The Regents of the University
# of California, through Lawrence Berkeley National Laboratory, National
# Technology & Engineering Solutions of Sandia, LLC, Alliance for Sustainable
# Energy, LLC, Battelle Energy Alliance, LLC, University of Notre Dame du Lac, et
# al. All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. Both files are also available online at the URL:
# "https://github.com/gmlc-dispatches/dispatches".
#
#################################################################################
import os
import pathlib
from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset
from sklearn.model_selection import train_test_split, KFold
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# this script supports subscenario analysis.
class TrainNNSurrogates:
    
    '''
    Train neural network surrogates for the dispatch frequency/ revenue
    '''
    
    def __init__(self, simulation_data, data_file, filter_opt = True):

        '''
        Initialization for the class

        Arguments:

            simulation data: object, composition from ReadData class

            data_file: path of the data file. If the model_type = frequency, the data_file should be the clustering_model_path, 
            if the model_type = revenue, the data_file should be the revenue.csv

            filter_opt: bool, if we are going to filter out 0/1 capacity days

        Return

            None
        '''

        self.simulation_data = simulation_data
        self.data_file = data_file
        self.filter_opt = filter_opt
        # set a class property which is the time length of a day.
        self._time_length = 24


    @property
    def simulation_data(self):

        '''
        Porperty getter of simulation_data
        
        Arguments:

            None

        Returns:

            simulation_data
        '''
        
        return self._simulation_data


    @simulation_data.setter
    def simulation_data(self, value):

        '''
        Porperty setter of simulation_data
        
        Arguments:

            value: object, composition from ReadData class

        Returns:

            None
        '''
        
        if not isinstance(value, object):
            raise TypeError(
                f"The simulation_data must be an object, but {type(value)} is given."
            )
        self._simulation_data = value


    @property
    def data_file(self):

        '''
        Porperty getter of data_file

        Arguments:

            None

        Returns:

            data_file
        '''
        
        return self._data_file


    @data_file.setter
    def data_file(self, value):

        '''
        Porperty setter of data_file
        
        Arguments:

            value: str, path of the clustering model

        Returns:

            None
        '''
        
        if not (isinstance(value, str) or isinstance(value, pathlib.WindowsPath) or isinstance(value, pathlib.PosixPath)):
            raise TypeError(
                f"The data_file must be str or object, but {type(value)} is given."
            )
        self._data_file = value


    @property
    def filter_opt(self):

        '''
        Property getter of filter_opt

        Arguments:
        
            None

        Return:
        
            bool: if want filter 0/1 days in clustering
        '''

        return self._filter_opt


    @filter_opt.setter
    def filter_opt(self, value):

        '''
        Property setter of filter_opt

        Arguments:
        
            value: bool.
        
        Returns:
        
            None
        '''

        if not isinstance(value, bool):
            raise TypeError(
                f"filter_opt must be bool, but {type(value)} is given"
            )

        self._filter_opt = value



    def _read_clustering_model(self, clustering_model_path):

        '''
        Read the time series clustering model from the given path

        Arguments:

            clustering_model_path: path of clustering model

        Returns:

            Clustering model
        '''

        clustering_model = TimeSeriesKMeans.from_json(clustering_model_path)

        # read the number of clusters from the clustering model
        self.num_clusters = clustering_model.n_clusters
        self.clustering_model = clustering_model

        return clustering_model


    def _generate_label_data(self):

        '''
        Calculate the labels for NN training. 

        Arguments:

            None

        Return:

            dispatch_frequency_dict: {run_index: [dispatch frequency]}

        '''
        # scale the dispatch data
        scaled_dispatch_dict = self.simulation_data._scale_data()
        sim_index = list(scaled_dispatch_dict.keys())
        single_day_dataset = {}
        dispatch_frequency_dict = {}
        
        # filter out 0/1 days in each simulaion data
        if self.filter_opt == True:          
            for idx in sim_index:
                sim_year_data = scaled_dispatch_dict[idx]
                single_day_dataset[idx] = []
                # calculate number of days in a simulation
                day_num = int(len(sim_year_data)/self._time_length)
                zero_day = 0
                full_day = 0
                
                for day in range(day_num):
                    # slice the annual data into days
                    sim_day_data = sim_year_data[day*self._time_length:(day+1)*self._time_length]
                    
                    if sum(sim_day_data) == 0:
                        zero_day += 1
                    
                    elif sum(sim_day_data) == 24:
                        full_day += 1
                   
                    else:
                        single_day_dataset[idx].append(sim_day_data)
            
                # frequency of 0/1 days
                ws0 = zero_day/day_num
                ws1 = full_day/day_num


                if len(single_day_dataset[idx]) == 0:
                    labels = np.array([])

                else:
                    to_pred_data = to_time_series_dataset(single_day_dataset[idx])
                    labels = self.clustering_model.predict(to_pred_data)

                # count the how many representative days and how many days in the representative days
                elements, count = np.unique(labels,return_counts=True)

                pred_result_dict = dict(zip(elements, count))
                count_dict = {}

                for j in range(self.num_clusters):
                    
                    if j in pred_result_dict.keys():
                        # if there are days in this simulation year belong to cluster i, count the frequency 
                        count_dict[j] = pred_result_dict[j]/day_num
                    
                    else:
                        # else, the frequency of this cluster is 0
                        count_dict[j] = 0

                # the first element in w is frequency of 0 cf days
                dispatch_frequency_dict[idx] = [ws0]

                for key, value in count_dict.items():
                    dispatch_frequency_dict[idx].append(value)

                # the last element in w is frequency of 1 cf days
                dispatch_frequency_dict[idx].append(ws1)
        
        # filter_opt = False then we do not filter 0/1 days
        else:
            for idx in sim_index:
                sim_year_data = scaled_dispatch_dict[idx]
                single_day_dataset[idx] = []
                # calculate number of days in a simulation
                day_num = int(len(sim_year_data)/self._time_length)
                
                for day in range(day_num):
                    sim_day_data = sim_year_data[day*self._time_length:(day+1)*self._time_length]
                    single_day_dataset[idx].append(sim_day_data)

                to_pred_data = to_time_series_dataset(single_day_dataset[idx])
                labels = self.clustering_model.predict(to_pred_data)

                elements, count = np.unique(labels,return_counts=True)
                pred_result_dict = dict(zip(elements, count))
                count_dict = {}
                
                for j in range(self.num_clusters):
                    
                    if j in pred_result_dict.keys():
                        count_dict[j] = pred_result_dict[j]/day_num
                    
                    else:
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

        index_list = list(self.simulation_data._dispatch_dict.keys())

        x = []
        y = []

        if self.model_type == 'frequency':
            y_dict = self._generate_label_data()

        if self.model_type == 'revenue':
            y_dict = self.simulation_data.read_rev_data(self.data_file)

        for idx in index_list:
            x.append(self.simulation_data._input_data_dict[idx])
            y.append(y_dict[idx])

        return np.array(x), np.array(y)


    def train_NN_frequency(self, NN_size):

        '''
        train the dispatch frequency NN surrogate model.
        print the R2 results of each cluster.

        Arguments:
            
            NN_size: list, the size of neural network. (input nodes, hidden layer 1 size, ..., output nodes )

        Return:

            model: the NN model
        '''
        # set the class property model_type
        self.model_type = 'frequency'

        # read and save the clustering model in self.clustering_model
        self._read_clustering_model(self.data_file)

        x, ws = self._transform_dict_to_array()

        # the first element of the NN_size dict is the input layer size, the last element is output layer size. 
        input_layer_size = NN_size[0]
        output_layer_size = NN_size[-1]

        # train test split
        x_train, x_test, ws_train, ws_test = train_test_split(x, ws, test_size=0.2, random_state=0)

        # scale the data both x and ws
        xm = np.mean(x_train,axis = 0)
        xstd = np.std(x_train,axis = 0)
        wsm = np.mean(ws_train,axis = 0)
        wsstd = np.std(ws_train,axis = 0)
        x_train_scaled = (x_train - xm) / xstd
        ws_train_scaled = (ws_train - wsm)/ wsstd

        # train a keras MLP (multi-layer perceptron) Regressor model
        model = keras.Sequential(name=self.model_type)
        model.add(layers.Input(input_layer_size))
        # excpt the first and last element in the list are the hidden layer size.
        for layer_size in NN_size[1:-1]:
            model.add(layers.Dense(layer_size, activation='sigmoid'))
        model.add(layers.Dense(output_layer_size))
        model.compile(optimizer=Adam(), loss='mse')
        history = model.fit(x=x_train_scaled, y=ws_train_scaled, verbose=0, epochs=500)

        print("Making NN Predictions...") 

        # normalize the data
        x_test_scaled = (x_test - xm) / xstd
        ws_test_scaled = (ws_test - wsm) / wsstd

        print("Evaluate on test data")
        evaluate_res = model.evaluate(x_test_scaled, ws_test_scaled)
        print(evaluate_res)
        predict_ws = np.array(model.predict(x_test_scaled))
        predict_ws_unscaled = predict_ws*wsstd + wsm

        if self.filter_opt == True:
            clusters = self.num_clusters + 2

        else:
            clusters = self.num_clusters

        R2 = []

        for rd in range(0,clusters):
            # compute R2 metric
            wspredict = predict_ws_unscaled.transpose()[rd]
            SS_tot = np.sum(np.square(ws_test.transpose()[rd] - wsm[rd]))
            SS_res = np.sum(np.square(ws_test.transpose()[rd] - wspredict))
            residual = 1 - SS_res/SS_tot
            R2.append(residual)

        print('The R2 of dispatch surrogate validation is', R2)

        xmin = list(np.min(x_train_scaled, axis=0))
        xmax = list(np.max(x_train_scaled, axis=0))

        data = {"xm_inputs":list(xm),"xstd_inputs":list(xstd),"xmin":xmin,"xmax":xmax,
            "ws_mean":list(wsm),"ws_std":list(wsstd)}

        self._model_params = data

        return model


    def train_NN_revenue(self, NN_size, cross_val=None, return_history=False):

        '''
        train the revenue NN surrogate model.
        print the R2 results.

        Arguments:
            
            NN_size: list, the size of neural network. (input nodes, hidden layer 1 size, ..., output nodes)
            cross_val: int or None, number of folds for cross validation. If None, simple train-test split is used.
            return_history: bool, whether to return training history along with model

        Return:

            model: the NN model
            history: training history (if return_history=True)
        '''

        self.model_type = 'revenue'
        x, y = self._transform_dict_to_array()

        # the first element of the NN_size dict is the input layer size, the last element is output layer size. 
        input_layer_size = NN_size[0]
        output_layer_size = NN_size[-1]

        if cross_val is not None and cross_val > 1:
            print(f"Performing {cross_val}-fold cross validation...")
            return self._train_revenue_with_cross_validation(x, y, NN_size, cross_val)
        else:
            # Original train-test split approach
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

            # scale the data both x and ws
            xm = np.mean(x_train,axis = 0)
            xstd = np.std(x_train,axis = 0)
            ym = np.mean(y_train,axis = 0)
            ystd = np.std(y_train,axis = 0)
            x_train_scaled = (x_train - xm) / xstd
            y_train_scaled = (y_train - ym)/ ystd

            # train a keras MLP (multi-layer perceptron) Regressor model
            model = keras.Sequential(name=self.model_type)
            model.add(layers.Input(input_layer_size))
            for layer_size in NN_size[1:-1]:
                model.add(layers.Dense(layer_size, activation='tanh'))
            model.add(layers.Dense(output_layer_size))
            model.compile(optimizer=Adam(), loss='mse')
            history = model.fit(x=x_train_scaled, y=y_train_scaled, verbose=0, epochs=500, validation_split=0.1)

            print("Making NN Predictions...") 

            # normalize the data
            x_test_scaled = (x_test - xm) / xstd
            y_test_scaled = (y_test - ym) / ystd

            print("Evaluate on test data")
            evaluate_res = model.evaluate(x_test_scaled, y_test_scaled)
            print(evaluate_res)
            print(history.history['loss'][-1])
            print(history.history['val_loss'][-1])
            predict_y = np.array(model.predict(x_test_scaled))
            predict_y_unscaled = predict_y*ystd + ym

            # calculate R2
            ypredict = predict_y_unscaled.transpose()
            SS_tot = np.sum(np.square(y_test.transpose() - ym))
            SS_res = np.sum(np.square(y_test.transpose() - ypredict))
            R2 = 1 - SS_res/SS_tot

            print('The R2 of revenue surrogate validation is ', R2)

            xmin = list(np.min(x_train_scaled,axis=0))
            xmax = list(np.max(x_train_scaled,axis=0))

            data = {"xm_inputs":list(xm),"xstd_inputs":list(xstd),"xmin":xmin,"xmax":xmax, "y_mean":ym,"y_std":ystd}

            self._model_params = data

            if return_history:
                return model, history
            else:
                return model


    def _train_revenue_with_cross_validation(self, x, y, NN_size, k_folds):
        '''
        Train revenue neural network with k-fold cross validation
        
        Arguments:
            x: input features
            y: target values  
            NN_size: list, the size of neural network
            k_folds: int, number of folds for cross validation
            
        Returns:
            model: best performing model from cross validation
        '''
        
        input_layer_size = NN_size[0]
        output_layer_size = NN_size[-1]
        
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        
        fold_scores = []
        fold_models = []
        fold_params = []
        
        print(f"Starting {k_folds}-fold cross validation for revenue model...")
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(x)):
            print(f"Training fold {fold + 1}/{k_folds}")
            
            # Split data for this fold
            x_train_fold, x_val_fold = x[train_idx], x[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            # Scale the data
            xm = np.mean(x_train_fold, axis=0)
            xstd = np.std(x_train_fold, axis=0)
            ym = np.mean(y_train_fold, axis=0)
            ystd = np.std(y_train_fold, axis=0)
            
            # Avoid division by zero
            xstd = np.where(xstd == 0, 1, xstd)
            ystd = np.where(ystd == 0, 1, ystd)
            
            x_train_scaled = (x_train_fold - xm) / xstd
            y_train_scaled = (y_train_fold - ym) / ystd
            x_val_scaled = (x_val_fold - xm) / xstd
            y_val_scaled = (y_val_fold - ym) / ystd
            
            # Create and train model
            model = keras.Sequential(name=f'{self.model_type}_fold_{fold}')
            model.add(layers.Input(input_layer_size))
            for layer_size in NN_size[1:-1]:
                model.add(layers.Dense(layer_size, activation='tanh'))
            model.add(layers.Dense(output_layer_size))
            model.compile(optimizer=Adam(), loss='mse')
            
            # Train the model
            history = model.fit(
                x=x_train_scaled, 
                y=y_train_scaled, 
                verbose=0, 
                epochs=500, 
                validation_data=(x_val_scaled, y_val_scaled)
            )
            
            # Evaluate on validation set
            val_loss = model.evaluate(x_val_scaled, y_val_scaled, verbose=0)
            predict_y_val = model.predict(x_val_scaled, verbose=0)
            predict_y_val_unscaled = predict_y_val * ystd + ym
            
            # Calculate R2 for validation data
            ypredict = predict_y_val_unscaled.transpose()
            SS_tot = np.sum(np.square(y_val_fold.transpose() - ym))
            SS_res = np.sum(np.square(y_val_fold.transpose() - ypredict))
            R2 = 1 - SS_res / SS_tot
            
            # Calculate RMSE
            rmse = mean_squared_error(y_val_fold, predict_y_val_unscaled, squared=False)
            
            print(f"Fold {fold + 1} - R2: {R2:.4f}, RMSE: {rmse:.4f}, Val Loss: {val_loss:.4f}")
            
            fold_scores.append({
                'R2': R2,
                'RMSE': rmse, 
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
                "y_mean": ym,
                "y_std": ystd
            }
            fold_params.append(params)
        
        # Calculate cross-validation statistics
        cv_r2_scores = [score['R2'] for score in fold_scores]
        cv_rmse_scores = [score['RMSE'] for score in fold_scores]
        cv_loss_scores = [score['val_loss'] for score in fold_scores]
        
        print("\n" + "="*50)
        print("REVENUE MODEL CROSS VALIDATION RESULTS")
        print("="*50)
        print(f"R2 - Mean: {np.mean(cv_r2_scores):.4f} ± {np.std(cv_r2_scores):.4f}")
        print(f"RMSE - Mean: {np.mean(cv_rmse_scores):.4f} ± {np.std(cv_rmse_scores):.4f}")
        print(f"Val Loss - Mean: {np.mean(cv_loss_scores):.4f} ± {np.std(cv_loss_scores):.4f}")
        print("="*50)
        
        # Select best model based on highest R2
        best_fold_idx = np.argmax(cv_r2_scores)
        best_model = fold_models[best_fold_idx]
        best_params = fold_params[best_fold_idx]
        
        print(f"Best model from fold {best_fold_idx + 1} with R2: {cv_r2_scores[best_fold_idx]:.4f}")
        
        # Store the best model parameters and cross-validation results
        self._model_params = best_params
        self._cv_results = {
            'fold_scores': fold_scores,
            'mean_r2': np.mean(cv_r2_scores),
            'std_r2': np.std(cv_r2_scores),
            'mean_rmse': np.mean(cv_rmse_scores),
            'std_rmse': np.std(cv_rmse_scores),
            'best_fold': best_fold_idx + 1
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


    def save_model(self, model, NN_model_path = None, NN_param_path = None):

        '''
        Save the model to the path which can be specified by the user. 

        Arguments:

            model: trained model from self.train_NN()

            fpath: if fpath == None, save to default path. 

        Return:

            None
        '''

        print('Saving model')

        this_file_path = os.getcwd()

        if self.model_type == 'frequency':
            NN_default_model_path = f'NN_models/keras_{self.simulation_data.case_type}_dispatch_frequency_sigmoid'
            NN_default_param_path = f'NN_models/keras_{self.simulation_data.case_type}_dispatch_frequency_params.json'
        else:
            NN_default_model_path = f'NN_models/keras_{self.simulation_data.case_type}_revenue_sigmoid'
            NN_default_param_path = f'NN_models/keras_{self.simulation_data.case_type}_revenue_params.json'

        # NN_model_path == none
        if NN_model_path == None:
            # save the NN model
            model_save_path = os.path.join(this_file_path, NN_default_model_path)
            model.save(model_save_path)

            if NN_param_path == None:
                # save the sacling parameters
                param_save_path = os.path.join(this_file_path, NN_default_param_path)
                with open(param_save_path, 'w') as f:
                    json.dump(self._model_params, f)
            else:
                with open(NN_param_path, 'w') as f:
                    json.dump(self._model_params, f)

        else:
            model.save(NN_model_path)
            if NN_param_path == None:
                param_save_path = os.path.join(this_file_path, NN_default_param_path)
                # Include cross-validation results if available
                params_to_save = self._model_params.copy()
                if hasattr(self, '_cv_results'):
                    params_to_save['cross_validation_results'] = self._cv_results
                    print('Cross-validation results included in saved parameters')
                with open(param_save_path, 'w') as f:
                    json.dump(params_to_save, f, indent=2)
            else:
                # Include cross-validation results if available
                params_to_save = self._model_params.copy()
                if hasattr(self, '_cv_results'):
                    params_to_save['cross_validation_results'] = self._cv_results
                    print('Cross-validation results included in saved parameters')
                with open(NN_param_path, 'w') as f:
                    json.dump(params_to_save, f, indent=2)


    def plot_R2_results(self, NN_model_path = None, NN_param_path = None, fig_name = None):

        '''
        Visualize the R2 result

        Arguments: 

            train_data: list, [x, ws] where x is the input of NN and ws is output. 

            NN_model_path: the path of saved NN model

            NN_param_path: the path of saved NN params

        '''
        this_file_path = os.getcwd()

        # set the font for the plots
        font1 = {
            'weight' : 'bold',
            'size'   : 18,
            }
        font2 = {
            'weight' : 'normal',
            'size'   : 15,
            }
        if self.model_type == 'frequency':
            
            x, ws = self._transform_dict_to_array()
            # use a different random_state from the training
            # x_train, x_test, ws_train, ws_test = train_test_split(x, ws, test_size=0.2, random_state=42)

            if NN_model_path == None:
                # load the NN model from default path
                model_save_path = os.path.join(this_file_path, f'NN_models/keras_{self.simulation_data.case_type}_dispatch_frequency_sigmoid')
                NN_model = keras.models.load_model(model_save_path)
            else:
                # load the NN model from the given path
                NN_model = keras.models.load_model(NN_model_path)

            if NN_param_path == None:
                # load the NN parameters from default path
                param_save_path = os.path.join(this_file_path, f'NN_models/keras_{self.simulation_data.case_type}_dispatch_frequency_params.json')
                with open(param_save_path) as f:
                    NN_param = json.load(f)
            else:
                # load the NN parameters from the given path
                with open(NN_param_path) as f:
                    NN_param = json.load(f)

            # scale data
            xm = NN_param['xm_inputs']
            xstd = NN_param['xstd_inputs']
            wsm = NN_param['ws_mean']
            wsstd = NN_param['ws_std']

            x_scaled = (x - xm)/xstd
            pred_ws = NN_model.predict(x_scaled)
            pred_ws_unscaled = pred_ws*wsstd + wsm

            if self.filter_opt == True:
                num_clusters = self.num_clusters + 2

            else:
                num_clusters = self.num_cluster

            # calculate the R2 for each representative day
            R2 = []

            for rd in range(num_clusters):
                # compute R2 metric
                wspredict = pred_ws_unscaled.transpose()[rd]
                SS_tot = np.sum(np.square(ws.transpose()[rd] - wsm[rd]))
                SS_res = np.sum(np.square(ws.transpose()[rd] - wspredict))
                residual = 1 - SS_res/SS_tot
                R2.append(residual)
            print(R2)

            # plot the figure
            for i in range(num_clusters):
                fig, axs = plt.subplots()
                fig.text(0.0, 0.5, 'Predicted dispatch frequency', va='center', rotation='vertical',font = font1)
                fig.text(0.4, 0.05, 'True dispatch frequency', va='center', rotation='horizontal',font = font1)
                fig.set_size_inches(10,10)

                wst = ws.transpose()[i]
                wsp = pred_ws_unscaled.transpose()[i]

                axs.scatter(wst*366,wsp*366,color = "green",alpha = 0.5)
                # plot by day instead of frequency
                axs.plot([min(wst)*366,max(wst)*366],[min(wst)*366,max(wst)*366],color = "black")
                # axs.set_xlim(-5,370)
                # axs.set_ylim(-5,370)
                axs.annotate("$R^2 = {}$".format(round(R2[i],3)),(min(wst)*366,0.75*max(wst)*366),font = font2)
                # when filter = True, we have zero/full clusters. To make the index consistent with the index in the clustering, do this step.
                if self.filter_opt == True and (i == 0 or i == num_clusters-1):
                    if i == 0:
                        name = 'zero'
                        axs.set_title(f'cluster_zero',font = font1)
                    if i == num_clusters-1:
                        axs.set_title(f'cluster_full',font = font1)
                        name = 'full'
                else:
                    axs.set_title(f'cluster_{i-1}',font = font1)
                    name = str(i-1)


                plt.xticks(fontsize=15)
                plt.yticks(fontsize=15)
                plt.tick_params(direction="in",top=True, right=True)

                if fig_name == None:
                    default_path = os.path.join(f"{self.simulation_data.case_type}_case_study","R2_figures",f"{self.simulation_data.case_type}_dispatch_cluster{name}.png")
                    plt.savefig(default_path, dpi =300)
                else:
                    fig_name_ = fig_name + f'_cluster_{name}'
                    fpath = os.path.join(f"{self.simulation_data.case_type}_case_study","R2_figures",f"{fig_name_}")
                    plt.savefig(fpath, dpi =300)


        if self.model_type == 'revenue':

            x, y = self._transform_dict_to_array()
            # use a different random_state from the training
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

            if NN_model_path == None:
                # load the NN model from default path
                model_save_path = os.path.join(this_file_path, f'NN_models/keras_{self.simulation_data.case_type}_revenue_sigmoid')
                NN_model = keras.models.load_model(model_save_path)
            else:
                NN_model = keras.models.load_model(NN_model_path)

            if NN_param_path == None:
                # load the NN parameters
                param_save_path = os.path.join(this_file_path, f'NN_models/keras_{self.simulation_data.case_type}_revenue_params.json')
                with open(param_save_path) as f:
                    NN_param = json.load(f)
            else:
                with open(NN_param_path) as f:
                    NN_param = json.load(f)

            # scale data
            xm = NN_param['xm_inputs']
            xstd = NN_param['xstd_inputs']
            ym = NN_param['y_mean']
            ystd = NN_param['y_std']

            x_train_scaled = (x_train - xm)/xstd
            x_test_scaled = (x_test - xm)/xstd
            pred_y_test = NN_model.predict(x_test_scaled)
            pred_y_train = NN_model.predict(x_train_scaled)
            pred_y_train_unscaled = pred_y_train*ystd + ym
            pred_y_test_unscaled = pred_y_test*ystd + ym

            # compute R2 over all the regression data points
            # ypredict = pred_y_unscaled.transpose()
            # SS_tot = np.sum(np.square(y.transpose() - ym))
            # SS_res = np.sum(np.square(y.transpose() - ypredict))
            # R2 = 1 - SS_res/SS_tot
            # print(R2)

            # plot results.
            fig, axs = plt.subplots()
            # fig.text(0.0, 0.5, 'Predicted revenue/M$', va='center', rotation='vertical',font = font1)
            # fig.text(0.4, 0.05, 'True revenue/M$', va='center', rotation='horizontal',font = font1)
            axs.set_xlabel('True Revenue [M$]', font = font1)
            axs.set_ylabel('Predicted Revenue [M$]', font = font1)
            fig.set_size_inches(6,6)

            yt_train = y_train.transpose()
            yt_test = y_test.transpose()
            yp_train = pred_y_train_unscaled.transpose()
            yp_test = pred_y_test_unscaled.transpose()
            test_mse = mean_squared_error(yt_test/1e6, pred_y_test_unscaled/1e6)
            test_mae = mean_absolute_error(yt_test/1e6, pred_y_test_unscaled/1e6)
            print(test_mse, test_mae)

            axs.scatter(yt_train/1e6,yp_train/1e6,color = "blue",alpha = 1,label = 'train')
            axs.scatter(yt_test/1e6,yp_test/1e6,color = "red", marker='^',alpha = 1,label = 'test')
            axs.plot([min(min(yt_train/1e6), min(yt_test/1e6)),max(max(yt_train/1e6), max(yt_test/1e6))],[min(min(yt_train/1e6), min(yt_test/1e6)),max(max(yt_train/1e6), max(yt_test/1e6))],color = "black")
            axs.set_title(f'Revenue Surrogate',font = font1)
            # axs.annotate("$R^2 = {}$".format(round(R2,3)),(min(yt)/1e6,0.85*max(yt)/1e6),fontsize = 18)
            plt.legend(prop=font2)

            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            plt.tick_params(direction="in",top=True, right=True)

            if fig_name == None:
                default_path = os.path.join(this_file_path,f"{self.simulation_data.case_type}_case_study","R2_figures",f"{self.simulation_data.case_type}_revenue.png")
                plt.savefig(default_path, dpi =300)
            
            else:
                fpath = os.path.join(this_file_path,f"{self.simulation_data.case_type}_case_study","R2_figures",f"{fig_name}")
                plt.savefig(fpath, dpi =300)
