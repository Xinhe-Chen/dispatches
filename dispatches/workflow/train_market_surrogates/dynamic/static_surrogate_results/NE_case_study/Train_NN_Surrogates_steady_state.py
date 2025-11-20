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

from sklearn.model_selection import train_test_split, KFold
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error


class TrainNNSurrogates:
    
    '''
    Train neural network surrogates for the dispatch frequency/ revenue
    '''
    
    def __init__(self, simulation_data):

        '''
        Initialization for the class

        Arguments:

            simulation data: object, composition from ReadData class

        Return

            None
        '''

        self.simulation_data = simulation_data
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


    def calculate_capacity_factors(self):
        '''
        calculate the capacity factor of the NPP
        '''
        dispatch_data_dict, input_data_dict = self.simulation_data.read_data_to_dict()

        dispatch_cf = {}
        pem_cf = {}
        # i is the index of sweep simulations
        for i in dispatch_data_dict:
            rt_dispatch = dispatch_data_dict[i]
            rt_dispatch_cf = np.sum(rt_dispatch)/(400*len(rt_dispatch))
            dispatch_cf[i] = rt_dispatch_cf
        
        return dispatch_cf


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


        y_dict = self.calculate_capacity_factors()

        for idx in index_list:
            x.append(self.simulation_data._input_data_dict[idx])
            y.append(y_dict[idx])

        return np.array(x), np.array(y)


    def train_NN_cf(self, NN_size, cross_val=None, return_history=False):

        '''
        train the NE steady state dispatch capacity factor surrogate model.
        print the R2 results.

        Arguments:
            
            NN_size: list, the size of neural network. (input nodes, hidden layer 1 size, ..., output nodes)
            cross_val: int or None, number of folds for cross validation. If None, simple train-test split is used.
            return_history: bool, whether to return training history along with model

        Return:

            model: the NN model
            history: training history (if return_history=True)
        '''

        x, y = self._transform_dict_to_array()

        # the first element of the NN_size dict is the input layer size, the last element is output layer size. 
        input_layer_size = NN_size[0]
        output_layer_size = NN_size[-1]

        if cross_val is not None and cross_val > 1:
            print(f"Performing {cross_val}-fold cross validation...")
            return self._train_with_cross_validation(x, y, NN_size, cross_val)
        else:
            # Original train-test split approach
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

            # scale the data both x and ws
            xm = np.mean(x_train, axis = 0)
            xstd = np.std(x_train, axis = 0)
            ym = np.mean(y_train, axis = 0)
            ystd = np.std(y_train, axis = 0)
            x_train_scaled = (x_train - xm) / xstd
            y_train_scaled = (y_train - ym) / ystd

            # train a keras MLP (multi-layer perceptron) Regressor model
            model = keras.Sequential(name='NE_steady_state')
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
            # print(evaluate_res)
            # print(history.history['loss'][-1])
            # print(history.history['val_loss'][-1])
            predict_y = np.array(model.predict(x_test_scaled))
            predict_y_unscaled = predict_y*ystd + ym

            # calculate R2 for test data
            ypredict = predict_y_unscaled.transpose()
            SS_tot = np.sum(np.square(y_test.transpose() - ym))
            SS_res = np.sum(np.square(y_test.transpose() - ypredict))
            R2 = 1 - SS_res/SS_tot
            rmse = mean_squared_error(y_test, predict_y_unscaled, squared=False)
            print(rmse)
            print('The R2 of revenue surrogate validation is ', R2)

            xmin = list(np.min(x_train_scaled,axis=0))
            xmax = list(np.max(x_train_scaled,axis=0))

            data = {"xm_inputs":list(xm),"xstd_inputs":list(xstd),"xmin":xmin,"xmax":xmax, "y_mean":ym,"y_std":ystd}

            self._model_params = data

            if return_history:
                return model, history
            else:
                return model


    def _train_with_cross_validation(self, x, y, NN_size, k_folds):
        '''
        Train neural network with k-fold cross validation
        
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
        
        print(f"Starting {k_folds}-fold cross validation...")
        
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
            model = keras.Sequential(name=f'NE_steady_state_fold_{fold}')
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
        print("CROSS VALIDATION RESULTS")
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


    def save_model(self, model, NN_model_path, NN_param_path):
        
        '''
        Save the model to the path which can be specified by the user. 

        Arguments:

            model: trained model from self.train_NN()

            fpath: if fpath == None, save to default path. 

        Return:

            None
        '''

        print('Saving model')

        # NN_model_path == none

        model.save(NN_model_path)

        # Include cross-validation results if available
        params_to_save = self._model_params.copy()
        if hasattr(self, '_cv_results'):
            params_to_save['cross_validation_results'] = self._cv_results
            print('Cross-validation results included in saved parameters')

        with open(NN_param_path, 'w') as f:
            json.dump(params_to_save, f, indent=2)

        return


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


    def plot_R2_results(self, NN_model_path, NN_param_path, fig_name):

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

        x, y = self._transform_dict_to_array()
        # Plot R2 for all the data
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        NN_model = keras.models.load_model(NN_model_path)

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
        # fig.text(0.0, 0.5, 'Predicted capacity factor', va='center', rotation='vertical',font = font1)
        # fig.text(0.4, 0.05, 'True capacity factor', va='center', rotation='horizontal',font = font1)
        axs.set_xlabel('True Capacity Factors [MW/MW]', font = font1)
        axs.set_ylabel('Predicted Capacity Factors [MW/MW]', font = font1)
        fig.set_size_inches(6,6)

        yt_train = y_train.transpose()
        yt_test = y_test.transpose()
        yp_train = pred_y_train_unscaled.transpose()
        yp_test = pred_y_test_unscaled.transpose()
        test_mse = mean_squared_error(yt_test, pred_y_test_unscaled)
        test_mae = mean_absolute_error(yt_test, pred_y_test_unscaled)
        print(test_mse, test_mae)

        axs.scatter(yt_train,yp_train,color = "blue",alpha = 1, label = 'train')
        axs.scatter(yt_test,yp_test,color = "red",marker="^",alpha = 1, label = 'test')
        axs.plot([min(min(yt_train), min(yt_test)),max(max(yt_train), max(yt_test))],[min(min(yt_train), min(yt_test)),max(max(yt_train), max(yt_test))],color = "black")
        axs.set_title('Frequency Surrogate', font=font1)
        plt.legend(prop=font2)
        # axs.set_title(f'NE Capacity Factor',font = font1)
        # axs.annotate("$R^2 = {}$".format(round(R2,3)),(min(yt),0.85*max(yt)),fontsize = 18)

        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.tick_params(direction="in",top=True, right=True)

        fpath = os.path.join(this_file_path,"R2_figures",f"{fig_name}")
        plt.savefig(fpath)
