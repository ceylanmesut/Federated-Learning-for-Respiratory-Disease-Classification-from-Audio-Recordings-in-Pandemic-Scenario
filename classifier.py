"""Main experiment functions for centralized and pretrained classifier."""

import pickle
import pandas as pd
from sklearn.utils import shuffle
from src.utils import classifier_utils
from collections import Counter


def run_experiment(data_version, exp_no, save, plot_performance, training_type, val_set):
    
    """Runs experiments and saves experiment results.
        
    Args:
        data_version: data version that experiment is to conduct.
        exp_no: experiment number
        save: binary parameter for saving the classifier performance.
        plot_performance: binary parameter for plotting classifier performance.
        training_type: centralized or pretrained training type.
        val_set: shuffles validation set if it exist.
    
    Returns:
        training history and trained model.
    """    
            
    if data_version == "main_data":
        # Loading final feature embedding
        with open('feature_embedding.pkl','rb') as f:     
            feature_matrix_loaded = pickle.load(f) 
            print("Feature Matrix Shape:", feature_matrix_loaded.shape)
        # Loading data csv file
        cough_data = pd.read_csv("main_data_wo_outliers.csv")        
    
    # Loading datasets
    X_train, X_test, y_train, y_test = classifier_utils.load_data(cough_data, feature_matrix_loaded)
    index_dict, X_train_loc, X_val_loc = classifier_utils.partion_data_train(X_train, y_train, clients=1, training_type=TRAINING_TYPE, validation=val_set)
    X_train, y_train, X_val, y_val = classifier_utils.get_train_data(index_dict, cough_data, feature_matrix_loaded, X_train_loc, X_val_loc, random_choice=0)
    X_train, y_train = shuffle(X_train, y_train)
    
    print(X_train.shape)
    print("Training Set:",Counter(y_train))
    print("Test Set:", Counter(y_test))
    print("Validation Set:", Counter(y_val))
        
    if val_set:
        X_val, y_val = shuffle(X_val, y_val)
        print(X_val.shape)
    
    # Model training
    history, model = classifier_utils.train_model(X_train, y_train, X_val, y_val, plot_performance, training_type)
    
    # Saving model performance
    if save:
        classifier_utils.save_clf_performance(exp_no, model, training_type, X_test, y_test)

    return(history, model)


# EXPERIMENT PARAMETERS
DATA_VERSION = "main_data"
PLOTTING_PERFORMANCE = True
TRAINING_TYPE = "centralized" # "centralized" or "pretrained" >> with COVID-19 class or without

# If model tuning then validation set set to True
# if not training on full training set without validation split
VAL_SET = True
SAVE = True
NUM_EXP = 1

for i in range(NUM_EXP):    
    
    print("EXPERIMENT:%i" % (i))
    EXP_NO = i
    
    a, b = run_experiment(data_version=DATA_VERSION, exp_no=EXP_NO, save=SAVE, plot_performance=PLOTTING_PERFORMANCE, training_type=TRAINING_TYPE, val_set=VAL_SET)
    print("EXPERIMENT:%i IS FINISHED!" % (i))

print("ALL EXPERIMENTS FINISHED!")