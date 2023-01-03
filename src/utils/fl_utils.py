"""Utility functions for federated learning, performance plotting and reporting."""

from typing import Tuple, Union, List
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from sklearn.metrics import confusion_matrix, matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
import glob
import re
import os
import math
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=15)


def load_data(cough_data, feature_matrix):
    """Loads data from cough paths and extracted feature matrix, encodes test labels
        and saves test set.
    
    Args:
        cough_data: data csv file
        feature_matrix: feature embedding of corresponding data
    
    Returns:
        Train and test datasets with 80-20% ratio.
    """
    
    cough_data_grouped = cough_data.groupby(["patient_ID"])["disease"].value_counts().reset_index(name='counts')
    cough_data_grouped_X = cough_data_grouped[["counts", "patient_ID", "disease"]]
    cough_data_grouped_y = cough_data_grouped[["disease", "patient_ID"]]
    

    # Global splitting
    X_train_main, X_test_main, y_train_main, y_test_main = train_test_split(cough_data_grouped_X, cough_data_grouped_y, 
                                                                            test_size=0.2, stratify=cough_data_grouped[["disease"]], random_state=42)

    
    patients_test = X_test_main[X_test_main.index.isin(X_test_main.index.values)]["patient_ID"].values
    cough_indices_test = cough_data[cough_data["patient_ID"].isin(patients_test)].index.values

    # Saving indices of the test data
    name_test = "test.csv"
    if not os.path.exists(name_test):
        
        test_df = pd.DataFrame(cough_indices_test)
        test_df.to_csv(name_test, index=False)
            
    X_test = feature_matrix[cough_indices_test] 
    
    y_test_main = cough_data[cough_data["patient_ID"].isin(patients_test)]["disease"].values
    y_test_main = encode_labels(cough_data).transform(y_test_main)
        
    return(X_train_main, X_test, y_train_main["disease"].values, y_test_main)


def partion_covid_data(X_train, y_train, clients):
    """Partions COVID-19 data w.r.t number of edge devices participating federated learning."""
        
    X_train_loc = X_train
    
    num_of_clients = clients
    
    X_train_loc_covid_parts = np.array_split(X_train_loc[X_train_loc["disease"]=="covid-19"].index, num_of_clients)
    
    covid_index_dict = {}

    for ind, patient_ind in enumerate(X_train_loc_covid_parts):
        
        covid_training_set = []
        covid_training_set = list(patient_ind.values)
        covid_index_dict[ind] = covid_training_set
            
    return(covid_index_dict, X_train_loc)
    
def get_covid_train_data(data_dict, cough_data, feature_matrix, X_train_loc, random_choice):
    """Gets COVID-19 training set, saves them to csvs and encode groundtruth."""    
    
    train_data_ind = data_dict[random_choice][0]   
    patients_train = X_train_loc[X_train_loc.index==(train_data_ind)]["patient_ID"].values
    cough_indices_train = cough_data[cough_data["patient_ID"].isin(list(patients_train))].index.values
    name_tr = "training_0%s.csv" % str(random_choice) 

    if not os.path.exists(name_tr):
        
        tr_df = pd.DataFrame(cough_indices_train)
        tr_df.to_csv(name_tr, index=False)
        
    X_train = feature_matrix[cough_indices_train] 
    y_train = cough_data[cough_data["patient_ID"].isin(patients_train)]["disease"].values
    
    y_train = encode_labels(cough_data).transform(y_train)

    return(X_train, y_train)

def increment(x_train, y_train, config, cid, inc_type="linear"):
    """Increments the data w.r.t incrementing type e.g linear, logarithmic, exponential, full."""
    
    if inc_type == "linear":
        
        if int(cid) >= 1:

            if int(config["rnd"]) % 5 == 0: 

                if int(config["rnd"]) < 75:
                    x_train = x_train[: int(len(x_train) * (int(config["rnd"]) / 100))]
                    y_train = y_train[: int(len(y_train) * (int(config["rnd"]) / 100))]
                else:
                    
                    x_train = x_train[:]
                    y_train = y_train[:]
                    
            else:

                if int(config["rnd"]) < 75:
                    
                    if int(config["rnd"]) < 5:

                        x_train = x_train[: int(len(x_train) * 0.05)] 
                        y_train = y_train[: int(len(y_train) * 0.05)] 
                        
                    else:
                        reminder = int(config["rnd"]) % 5
                        
                        nearest_divisor = int(config["rnd"]) - reminder 
                        
                        x_train = x_train[: int(len(x_train) * (int(nearest_divisor) / 100))]
                        y_train = y_train[: int(len(y_train) * (int(nearest_divisor) / 100))]                     
    
                else:
                    x_train = x_train[:]
                    y_train = y_train[:]
                    
    elif inc_type == "exponential":
        
        percentages = np.logspace(0.5,2, 75)/100
        
        if int(config["rnd"]) < 75:
            x_train = x_train[: int(len(x_train) * (percentages[config["rnd"]]))]
            y_train = y_train[: int(len(y_train) * (percentages[config["rnd"]]))]
        else:
            x_train = x_train[:]
            y_train = y_train[:] 
            
    elif inc_type == "logarithmic":
        
        if int(config["rnd"]) < 75:
        
            intervals = np.linspace(1, 75, 75)
            
            if intervals[config["rnd"]] == 1:
                
                eps = 0.1
                value = intervals[config["rnd"]] + eps
                percentage = math.log(value, 75) 
                
            else:
                percentage = math.log(intervals[config["rnd"]], 75) 
            
            x_train = x_train[: int(len(x_train) * (percentage))]
            y_train = y_train[: int(len(y_train) * (percentage))]
            
        else:
            x_train = x_train[:]
            y_train = y_train[:]
            
    elif inc_type == "full":
            
            x_train = x_train[:]
            y_train = y_train[:]  
                    
    return(x_train, y_train)
            
def encode_labels(cough_data):
    """Encodes labels with label encoder and returns the fitted encoder."""    
    
    main_labels = np.array(cough_data[["disease"]]).reshape((-1,))
    encoder = LabelEncoder()
    encoded_labels  = encoder.fit_transform(main_labels)
    
    return(encoder)   

def report(history, export_path, num_rounds, save, mode):
    """Reports the outcome of federated learning experiment and plots optimization performance.""" 

    with open('feature_embedding.pkl','rb') as f:
        feature_matrix_loaded = pickle.load(f)
    
    cough_data = pd.read_csv("main_data_wo_outliers.csv")    
    
    # Loading data for client accuracies
    X_train, X_test, y_train, y_test  = load_data(cough_data, feature_matrix_loaded)

    # model = tf.keras.models.load_model("finalpretrained.h5")
    model  = tf.keras.models.load_model("pretrained.h5")
    
    if mode=="client":
        # Plotting distributed evaluation
        plot_dist_eval(history, export_path, num_rounds, save, X_test, y_test, model)

    files = get_split_indices(export_path)
    file_names = get_file_names(files)
    
    # file names without test.csv
    files = files[:-1]
    file_names = file_names[:-1]

    # Sorting csv files according to the split number
    files.sort(key=lambda f: int(re.sub('\D', '', f)))
    file_names.sort(key=lambda f: int(re.sub('\D', '', f)))

    # adding test.csv back
    files.append("test.csv")
    file_names.append("test.csv")    

    # Plotting client cough distribution w.r.t experiment
    data_dict, main_train,  main_test = get_client_disease_dist(export_path, cough_data, files, file_names) 
    plot_client_cough_dist(export_path, main_train, main_test, save) 
    
    # Plotting patient distribution w.r.t experiment  
    data_dict, main_train, main_test = get_client_patient_dist(export_path, cough_data, files, file_names) 
    plot_client_patient_dist(export_path, main_train, main_test, save) 
            
    
def get_file_names(files):
    """"Gets the file names."""    
    
    names = []
    for i in files:

        split = i.split(".")
        names.append(split[0])

    return(names)

    
def create_experiment(mode):
    """"Creates experiments with experiment date and time."""
    
    if not os.path.exists("experiments"):
        os.mkdir("experiments")
    
    dt_now = datetime.now().strftime('%Y-%m-%d %H-%M')
    file_name = dt_now.split(" ")[0] + "_" + dt_now.split(" ")[1] + "_" + mode
    
    exp_path = "experiments" + "\\" + file_name
    print(exp_path)
    if not os.path.exists(exp_path):
        os.mkdir(exp_path)
        
    return(exp_path)             
        
################## VISUALIZATIONS ##################
def get_client_disease_dist(export_path, cough_data, files, file_names):
    """Extracts disease distribution of participating clients (patients)."""
    
    data_dict={}
    
    num_clients = len(files)-1
    
    for file, name in zip(files, file_names):
        
        path = export_path + "\\" + file
        file = pd.read_csv(path)
        
        coughs = cough_data[cough_data.index.isin(list(file["0"]))]["disease"]
        coughs = cough_data[cough_data.index.isin(list(file["0"]))]["disease"]
        pivot_table = coughs.value_counts().rename_axis('disease').reset_index(name='counts')
        
        cl = pivot_table["disease"].nunique()

        if name == "test.csv":
            client_no = "Test Set"
            client_col = [client_no for i in range(0, cl)]
            
        else:
            client_no = name.split("_")[-1]
            client_col = "Client" + " " + client_no # * cl
            client_col = [client_col for i in range(0, cl)]
            
            
        pivot_table["client"] = client_col
        data_dict[name] = pivot_table

    tr_data_df = pd.DataFrame()
    test_data_df = pd.DataFrame()

    for ind, file_name in enumerate(file_names[:-1]):
        
        if ind == 0: 
            main_train = pd.concat([tr_data_df, data_dict[file_name]])
        else:
            main_train = pd.concat([main_train, data_dict[file_name]]) 
            
    main_test = pd.concat([test_data_df, data_dict[file_names[-1]]])         
                
    return(data_dict, main_train, main_test)


def plot_client_cough_dist(export_path, main_train, main_test, save):    
    """Plots the distribution cough samples of participating clients (patients)."""
    
    fig, axs = plt.subplots(1, 2, figsize=(24,12))

    
    legend = ["Asthma", "COPD", "COVID-19", "Healthy"]
    
    # diseases = ["Asthma", "COPD", "COVID-19", "Healthy"]
    # colors = ["steelblue", "seagreen", "indianred", "gold"]
    # title1="Client Validation Cough Distribution"
    # clients = ["Client 0", "Client 1", "Client 2"]

    title0="Client Training Cough Distribution"
    title2= "Global Test Set Cough Distribution"
    
    axs[0] = main_train.groupby(["client", "disease"], sort=False)["counts"].sum().unstack().plot(kind='bar', stacked=True, ax=axs[0], legend=True,
                                                                             rot=0, color="firebrick")
    axs[0].set_title(title0, fontsize=17)
    axs[0].set_xlabel("")
    axs[0].set_ylabel("Number of Coughs", fontsize = 13)
    
 
    for i in range(1):
        axs[0].bar_label(axs[0].containers[i], fmt="%i", fontsize=13, label_type='center')
    axs[0].legend(loc='best', fontsize=14)

    axs[1] = main_test.groupby(["client", "disease"], sort=False)["counts"].sum().unstack().plot(kind='bar', stacked=True, ax=axs[1], legend=True, rot=0)
    axs[1].set_title(title2, fontsize=17)
    for i in range(4):
        axs[1].bar_label(axs[1].containers[i], fmt="%i", fontsize=13, label_type='center')
    axs[1].legend(loc='best', fontsize=14) 
        
    axs[1].set_xlabel("")
        
    if save:
        fig_path2 = export_path + "\\"+ "client_cough_dist.png"
        fig.savefig(fig_path2, dpi=400)        


def get_client_patient_dist(export_path, cough_data, files, file_names):
    """Gets the disease distribution of participating clients (patients)."""
    
    data_dict={}
    
    num_clients = len(files)-1
    
    for file, name in zip(files, file_names):
        
        path = export_path + "\\" + file
        file = pd.read_csv(path)        
        
        patients = cough_data[cough_data.index.isin(list(file["0"]))][["disease","patient_ID"]]
        pivot_table = patients.groupby("disease").patient_ID.nunique()
        pivot_table = pivot_table.to_frame()
        
        cl = len(pivot_table) # returns number of client labels to add

        if name == "test.csv":
            client_no = "Test Set"
            client_col = [client_no for i in range(0, cl)]
            
        else:
            client_no = name.split("_")[-1]
            client_col = "Client" + " " + client_no #client_no * cl
            client_col = [client_col for i in range(0, cl)]
        
        pivot_table["client"] = client_col
        data_dict[name] = pivot_table
        
    tr_data_df = pd.DataFrame()
    test_data_df = pd.DataFrame()    
    
    for ind, file_name in enumerate(file_names[:-1]):    
        if ind == 0:
            main_train = pd.concat([tr_data_df, data_dict[file_name]])

        else:
            main_train = pd.concat([main_train, data_dict[file_name]]) 
        
    main_test = pd.concat([test_data_df, data_dict[file_names[-1]]])                        

    return(data_dict, main_train, main_test) 


def plot_client_patient_dist(export_path, main_train, main_test, save):
    """Plots the disease distribution of participating clients (patients)."""
    
    fig, axs = plt.subplots(1, 2, figsize=(24,12)) 
    
    # clients = ["Client 0", "Client 1", "Client 2"]
    # colors = ["steelblue", "seagreen", "indianred", "gold"]
    # title1="Client Validation Patient Distribution"
    # diseases = ["Asthma", "COPD", "COVID-19", "Healthy"]
    
    legend = ["Asthma", "COPD", "COVID-19", "Healthy"]
    
    title0="Client Training Patient Distribution"
    title2= "Global Test Set Patient Distribution"
   
    axs[0] = main_train.groupby(["client", "disease"], sort=False)["patient_ID"].sum().unstack().plot(kind='bar', stacked=True, ax=axs[0], legend=True, rot=0, color="firebrick")
    axs[0].set_title(title0, fontsize=17)
    axs[0].set_xlabel("")
    axs[0].set_ylabel("Number of Patients", fontsize = 13)
    for i in range(1):
        axs[0].bar_label(axs[0].containers[i], fmt="%i", fontsize=13, label_type='center')
    axs[0].legend(loc='best', fontsize=14)
    axs[0].set_ylim([0,3])
  
    axs[1] = main_test.groupby(["client", "disease"], sort=False)["patient_ID"].sum().unstack().plot(kind='bar', stacked=True, ax=axs[1], legend=True, rot=0)
    axs[1].set_title(title2, fontsize=17)
    axs[1].set_xlabel("")   
    for i in range(4):
        axs[1].bar_label(axs[1].containers[i], fmt="%i", fontsize=13, label_type='center')
    axs[1].legend(loc="best", fontsize=14) 
    
    if save:
        fig_path2 = export_path + "\\"+ "client_patient_dist.png"
        fig.savefig(fig_path2, dpi=400)   


def plot_dist_eval(hist, export_path, num_rounds, save, X_test, y_test, model):
    """Plots distributed evaluation results for each round of federated learning."""    
    
    num_rounds=num_rounds

    number_of_rounds =  list(range(0, num_rounds))
    losses =[]
    metric = []

    # Obtaining FL loss
    for i in hist.losses_distributed: 
        losses.append(i[1])
        
    # Obtaining FL MCC results
    for i in hist.metrics_distributed["mcc"]:
        metric.append(i[1])
    
    # Getting client MCCs per disease
    asthma, copd, covid, healthy = get_client_mccs(X_test, y_test, model)

    # Plot-1: Federated Loss
    plt.figure(figsize=(14,8))
    plt.subplot(121)
    plt.plot(number_of_rounds, losses, color='tab:purple') 
    plt.ylabel("Loss", fontsize=16)
    plt.xlabel("Number of Communication Rounds", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True)

    # Plot-2: MCCs per disease type
    plt.subplot(122)
    plt.plot(number_of_rounds, metric, color='tab:purple') 
    plt.plot(number_of_rounds, asthma, color='tab:blue')
    plt.plot(number_of_rounds, copd, color='tab:green')
    plt.plot(number_of_rounds, covid, color='tab:red')
    plt.plot(number_of_rounds, healthy, color='tab:brown')
    
    plt.ylabel("Matthews Correlation Coefficient", fontsize=16)
    plt.xlabel("Number of Communication Rounds", fontsize=16)
    plt.legend(['Weighted MCC', 'Asthma', 'COPD', 'COVID-19', 'Healthy'], loc='best', frameon=False)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)    
    plt.ylim([-1.05, 1.05])

    plt.grid(True)
    
    if save:
        fig_path2 = export_path + "\\"+ "dist_eval_mcc_loss.png"
        fig_path3 = export_path + "\\"+ "dist_eval_mcc_loss.pdf"
        plt.savefig(fig_path2, dpi=200)
        plt.savefig(fig_path3, dpi=200)
    

def get_split_indices(export_path):
    """Gets the sample indicies of train and test splits."""
    
    train_files = sorted(glob.glob("training_0*.csv"), key=os.path.getmtime)
    test_files = glob.glob("test.csv")
    
    ind_files = train_files + test_files
    
    for file in ind_files:
        shutil.copy(file, export_path)
    
    return(ind_files)
         
def get_client_mccs(X_test, y_test, model):
    """Computes MCC metric of each communication round."""
    
    # Creating empty lists per disease
    asthma_mccs = []
    copd_mccs = []
    covid_mccs = []
    healthy_mccs =[] 
    
    file_pattern = "fl_results\\round-*-weights.npz"
    result_files = sorted(glob.glob(file_pattern), key=os.path.getmtime)
        
    for result in result_files:
        # Loading the weights on each round
        weights = np.load(result, allow_pickle=True)
        model.set_weights(weights["arr_0"])
        
        # Computing predictions
        y_pred = model.predict(X_test)
        y_pred_int = y_pred.argmax(axis=1)
        
        # Obtaining performance measures for MCC computation
        _, TP, FP, TN, FN = perf_measure(y_test, y_pred_int)
        
        # Computing MCCs
        MCC_Score_asthma = calculate_MCC(TP[0], FP[0], TN[0], FN[0])
        MCC_Score_copd = calculate_MCC(TP[1], FP[1], TN[1], FN[1])
        MCC_Score_covid = calculate_MCC(TP[2], FP[2], TN[2], FN[2])
        MCC_Score_healthy = calculate_MCC(TP[3], FP[3], TN[3], FN[3])
        
        # Storing MCCs
        asthma_mccs.append(MCC_Score_asthma)
        copd_mccs.append(MCC_Score_copd)
        covid_mccs.append(MCC_Score_covid)
        healthy_mccs.append(MCC_Score_healthy)
        
    return(asthma_mccs, copd_mccs, covid_mccs, healthy_mccs)

    
def perf_measure(y_actual, y_pred):
    """Computes TP, TN, FP, FN per class label."""
    
    class_id = set(y_actual).union(set(y_pred))
    TP = []
    FP = []
    TN = []
    FN = []

    for index ,_id in enumerate(class_id):
        TP.append(0)
        FP.append(0)
        TN.append(0)
        FN.append(0)
        for i in range(len(y_pred)):
            if y_actual[i] == y_pred[i] == _id:
                TP[index] += 1
            if y_pred[i] == _id and y_actual[i] != y_pred[i]:
                FP[index] += 1
            if y_actual[i] == y_pred[i] != _id:
                TN[index] += 1
            if y_pred[i] != _id and y_actual[i] != y_pred[i]:
                FN[index] += 1

    return class_id,TP, FP, TN, FN

def calculate_MCC(tp, fp, tn, fn):
    """Calculates MCC performance metric."""
    
    nominator = ((tp * tn) - (fp * fn))
    demoninator = (tp+fp) * (tp+fn) * (tn+fp) * (tn+fn)
    denominator_squared = np.sqrt(demoninator)
    
    # Handling division by zero
    if denominator_squared == 0:
        mcc = 0
    else:
        mcc = nominator / denominator_squared
    
    return(mcc)