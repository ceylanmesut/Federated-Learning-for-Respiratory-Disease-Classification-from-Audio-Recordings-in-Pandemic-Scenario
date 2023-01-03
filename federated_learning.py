"""Main federated learning functions."""

import flwr as fl
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import warnings
import pickle
import pandas as pd
import os
from src.utils import fl_utils
import math
from sklearn.metrics import matthews_corrcoef, confusion_matrix

from typing import Tuple, List, Optional, Dict
from flwr.common import (EvaluateRes, FitRes, Parameters, parameters_to_weights, weights_to_parameters, Scalar)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAdam


# Making TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def main():

    # Experiment parameters 
    NUM_CLIENTS = 1 # number of participating clients to federated learning 
    NUM_ROUNDS = 100 # number of communication rounds
    SAVE_RESULTS = True
    
    # Creating a client experiment
    exp_path = fl_utils.create_experiment("client")
    
    # Loading the pretrained model
    model  = tf.keras.models.load_model("pretrained.h5", custom_objects={"MatthewsCorrelationCoefficient": tfa.metrics.MatthewsCorrelationCoefficient(num_classes=4, name="mcc")})
    
    # Defining federated learning experiment parameters
    strategy=AggregateCustomMetricStrategy(
                    fraction_fit=0.1,  
                    fraction_eval=0.1, 
                    min_fit_clients=1, #minimum fit clients
                    min_eval_clients=1, #minimum evaluation clients
                min_available_clients=1, #minimum available clients
                on_fit_config_fn=fit_config, #to track comm. rounds
                initial_parameters =fl.common.weights_to_parameters(model.get_weights()), #initial model weights
                eta = 0.0001, #server side learning rate
                eta_l = 0.0001)     

    # Initiating FL simulation
    history = fl.simulation.start_simulation(client_fn=client_fn, num_clients=NUM_CLIENTS, client_resources={"num_cpus": 4},
                                    num_rounds=NUM_ROUNDS, strategy=strategy)
    # Reporting the outcome of the experiment
    fl_utils.report(history, exp_path, NUM_ROUNDS, SAVE_RESULTS, "client")
    
    
class FlowerClient(fl.client.NumPyClient):
    """Flower Client instance for each participating patient.
        
    Args:
        cid: client ID.
        model: pretrained model.
        X_train: training set.
        X_test: test set.
        y_train: groundtruth labels.
        y_test: test labels.
        clients: number of patients participating federated learning.
        
    """    

    def __init__(self, cid, model, X_train, X_test, y_train, y_test, clients) -> None:
        
        self.clients = clients
        self.cid = cid
        self.model = model
        
        # Loading feature embedding
        with open('feature_embedding.pkl','rb') as f:
            self.feature_matrix_loaded = pickle.load(f)
        # Loading data csv
        self.cough_data = pd.read_csv("main_data_wo_outliers.csv")
        
        # Partioning COVID-19 data w.r.t number of participating patients
        self.covid_index_dict, self.X_train_loc = fl_utils.partion_covid_data(X_train, y_train, self.clients)
        X_train, y_train = fl_utils.get_covid_train_data(self.covid_index_dict, self.cough_data, self.feature_matrix_loaded, self.X_train_loc, random_choice=int(self.cid))
        
        self.x_train, self.y_train = X_train, y_train
        self.x_val, self.y_val = X_test, y_test 

    def get_parameters(self):
        """Gets model weights."""
        return self.model.get_weights()        
        
    def fit(self, parameters, config):
        """Fits pretrained model with respective training set of the patient."""
        
        # Sets model weights
        self.model.set_weights(parameters)
         
        # Increments training dataset
        # e.g linear, exponential, logarithmic, full
        self.x_train, self.y_train = fl_utils.increment(self.x_train, self.y_train, config=config, cid=self.cid, inc_type="logarithmic")
         
        # Setting class weights
        class_weight = {0:6.5, 1:15, 2:50, 3:1}
        # Fitting the model and training
        self.model.fit(self.x_train, self.y_train, class_weight=class_weight, epochs=100, verbose=0)
        
        return self.model.get_weights(), len(self.x_train), {}         

    def evaluate(self, parameters, config):
        """Evalutes trained model, computes MCC and returns loss."""
        
        self.model.set_weights(parameters)
        
        # Evaluation of the model
        loss, _ = self.model.evaluate(self.x_val, self.y_val, verbose=0)
        y_pred = self.model.predict(self.x_val)
        y_pred_int = y_pred.argmax(axis=1)
        
        # Computing MCC.
        mcc = matthews_corrcoef(self.y_val, y_pred_int)
        
        return loss, len(self.x_val), {"mcc": mcc}        

class AggregateCustomMetricStrategy(FedAdam):
    """Federated Adam optimization strategy."""

    def aggregate_fit(self, rnd: int, results: List[Tuple[ClientProxy, FitRes]], 
                      failures: List[BaseException],) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        
        fedavg_parameters_aggregated, metrics_aggregated = super().aggregate_fit(rnd=rnd, results=results, failures=failures)
        
        if fedavg_parameters_aggregated is None:
            return None, {}

        fedavg_weights_aggregate = parameters_to_weights(fedavg_parameters_aggregated)
        
        # Monitoring the performance in excel file
        result_file_path = r"fl_results\\performance.xlsx"
        if not os.path.exists(result_file_path):
            col_names=["rnd","lr","perf", "loss"]
            performance=pd.DataFrame(columns=col_names)
            performance.to_excel("fl_results\\performance.xlsx", index=False)
        
        if rnd == 1:
            pass
        
        else:
                    
            performance=pd.read_excel(r"fl_results\\performance.xlsx")
            
            p = f"fl_results\\round-{rnd-1}-results.npz"
            res = np.load(p, allow_pickle=True)
            mcc = res["arr_0"][0][1].metrics["mcc"]
            loss = res["arr_0"][0][1].loss
            
            fl_round = rnd
            learning_rate = self.eta
            
            inf = [fl_round, learning_rate, mcc, loss]
            performance.loc[len(performance)] = inf
            
            performance.to_excel("fl_results\\performance.xlsx", index=False)        

        # Defining shrinkage parameter
        shrink = 0.7
        
        # Adam optimization step
        delta_t = [shrink * x - y for x, y in zip(fedavg_weights_aggregate, self.current_weights)]

        # m_t calculation
        if not self.m_t:
            self.m_t = [np.zeros_like(x) for x in delta_t]
        self.m_t = [self.beta_1 * x + (1 - self.beta_1) * y for x, y in zip(self.m_t, delta_t)]

        # v_t calculation
        if not self.v_t:
            self.v_t = [np.zeros_like(x) for x in delta_t]
            
        self.v_t = [self.beta_2 * x + (1 - self.beta_2) * np.multiply(y, y) for x, y in zip(self.v_t, delta_t)]

        # Computing updated weights
        new_weights = [x + self.eta * y / (np.sqrt(z) + self.tau) for x, y, z in zip(self.current_weights, self.m_t, self.v_t)]

        # Replacing current weights with updated weights
        self.current_weights = new_weights
        
        # Saving the updated weights
        np.savez(f"fl_results\\round-{rnd}-weights.npz", new_weights) 

        return weights_to_parameters(self.current_weights), metrics_aggregated
   
    def aggregate_evaluate(self, rnd: int, results: List[Tuple[ClientProxy, EvaluateRes]], 
                           failures: List[BaseException]) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results using weighted average."""
        
        if not results:
            return None
        
        # Saving results of each round
        np.savez(f"fl_results\\round-{rnd}-results.npz", results) 
        # Collecting MCC resukts
        mccs = [r.metrics["mcc"] * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]
        # Aggregating MCC results
        mcc_aggregated = sum(mccs) / sum(examples)
        
        print(f"Round {rnd} MCC aggregated from client results: {mcc_aggregated}")
        params, _ = super().aggregate_evaluate(rnd, results, failures)
        
        return params, {"mcc": mcc_aggregated} 


def client_fn(cid: str) -> fl.client.Client:
    """CLient instance creator function."""
    
    # Defining client ID, number of clients.
    cid = cid
    NUM_CLIENTS = 1
    clients = NUM_CLIENTS
    
    # Loading the pretrained model.
    model  = tf.keras.models.load_model("pretrained.h5", custom_objects={"MatthewsCorrelationCoefficient": tfa.metrics.MatthewsCorrelationCoefficient(num_classes=4, name="mcc")})
    
    with open('feature_embedding.pkl','rb') as f:
        feature_matrix_loaded = pickle.load(f)
    
    cough_data = pd.read_csv("main_data_wo_outliers.csv") 
    X_train, X_test, y_train, y_test  = fl_utils.load_data(cough_data, feature_matrix_loaded)

    return FlowerClient(int(cid), model, X_train, X_test, y_train, y_test, clients)

def fit_config(rnd: int):
    return {"rnd": rnd}     
      
      
if __name__ == "__main__":

    main()