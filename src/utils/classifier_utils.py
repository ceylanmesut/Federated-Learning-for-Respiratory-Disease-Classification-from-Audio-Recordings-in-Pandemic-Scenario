"""Utility functions for classifier training and performance reporting."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_score, matthews_corrcoef
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import tensorflow_addons as tfa
import os
import matplotlib.pyplot as plt
# use LaTeX fonts in the plot
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=15)

# plt.style.use('seaborn')


def load_data(cough_data, feature_matrix):
    """Loads data from cough paths and extracted feature matrix.

    Args:
        cough_data: data csv file
        feature_matrix: feature embedding of corresponding data
    
    Returns:
        Train and test datasets with 80-20% ratio.
    """
    
    # Grouping data csv by patient ID and disease
    # to ensure patient/subject-wise division
    cough_data_grouped = cough_data.groupby(["patient_ID"])["disease"].value_counts().reset_index(name='counts')
    cough_data_grouped_X = cough_data_grouped[["counts", "patient_ID", "disease"]]
    cough_data_grouped_y = cough_data_grouped[["disease", "patient_ID"]]
    
    # Global splitting
    X_train_main, X_test_main, y_train_main, y_test_main = train_test_split(cough_data_grouped_X, cough_data_grouped_y, 
                                                                            test_size=0.2, stratify=cough_data_grouped[["disease"]], random_state=42)

    # Determining test set patients
    patients_test = X_test_main[X_test_main.index.isin(X_test_main.index.values)]["patient_ID"].values
    cough_indices_test = cough_data[cough_data["patient_ID"].isin(patients_test)].index.values
    # Extracting corresponding feature embedding of test set patients' coughs
    X_test = feature_matrix[cough_indices_test] 
    
    y_test_main = cough_data[cough_data["patient_ID"].isin(patients_test)]["disease"].values
    # Encoding groundtruth 
    y_test_main = encode_labels(cough_data).transform(y_test_main)
        
    return(X_train_main, X_test, y_train_main["disease"].values, y_test_main)


def partion_data_train(X_train, y_train, clients, training_type="centralized", validation=False):
    """Partions training data by index acc.to training type and validation set requirement.

    Args:
        X_train: indices of training set cough samples
        y_train: training set labels
        cleints: number of clients participating the federated optimization
        training_type: training type of classifier. Centralized or pretraining.
            Centralized includes all diseases while pretraining includes only
            COVID-19 free datasets.
        validation: If validation set requested. If yes, 80-20% patient/subject-wise
            split ratio applies.
    
    Returns:
        Dictionary including indices of cough samples w.r.t training and 
        validation set.
        
    Raises:
        ValueError: If wrong training type is entered.
    """    
 
    # Local dataset splitting
    if validation == True:
        X_train_loc, X_val_loc, y_train_loc, y_val_loc = train_test_split(X_train, y_train, 
                                                                                test_size=0.2, stratify=y_train, random_state=42)
    else:
        X_train_loc = X_train
        y_train_loc = y_train

    num_of_clients = clients
    
    # Returns patient indices by disease type
    X_train_loc_asthma_parts = np.array_split(X_train_loc[X_train_loc["disease"]=="asthma"].index, num_of_clients)
    X_train_loc_copd_parts = np.array_split(X_train_loc[X_train_loc["disease"]=="copd"].index, num_of_clients)
    X_train_loc_healthy_parts = np.array_split(X_train_loc[X_train_loc["disease"]=="healthy"].index, num_of_clients)
    X_train_loc_covid_parts = np.array_split(X_train_loc[X_train_loc["disease"]=="covid-19"].index, num_of_clients)
    
    if validation == True:
        # Returns patient indices by disease 
        # if validation set requested
        X_val_loc_asthma_parts = np.array_split(X_val_loc[X_val_loc["disease"]=="asthma"].index, num_of_clients)
        X_val_loc_copd_parts = np.array_split(X_val_loc[X_val_loc["disease"]=="copd"].index, num_of_clients)
        X_val_loc_healthy_parts = np.array_split(X_val_loc[X_val_loc["disease"]=="healthy"].index, num_of_clients)
        X_val_loc_covid_parts = np.array_split(X_val_loc[X_val_loc["disease"]=="covid-19"].index, num_of_clients)
        
    
    if training_type == "centralized":
      
        # All disease classes  
        train_1 = list(X_train_loc_asthma_parts[0].values) + list(X_train_loc_copd_parts[0].values) + list(X_train_loc_covid_parts[0].values) + list(X_train_loc_healthy_parts[0].values)
        
        if validation == True:
            val_1 = list(X_val_loc_asthma_parts[0].values) + list(X_val_loc_copd_parts[0].values) + list(X_val_loc_covid_parts[0].values) + list(X_val_loc_healthy_parts[0].values)
        
    elif training_type == "pretrained":
        # COVID-19 Free Datasets
        train_1 = list(X_train_loc_asthma_parts[0].values) + list(X_train_loc_copd_parts[0].values) + list(X_train_loc_healthy_parts[0].values)
        
        if validation == True:
            val_1 = list(X_val_loc_asthma_parts[0].values) + list(X_val_loc_copd_parts[0].values) + list(X_val_loc_healthy_parts[0].values)
        
    else:
        raise ValueError("Wrong training type")
    
    # Defining data dictionary including indices of cough samples of
    # patient-wise train and val splits 
    index_dict = {}
    
    if validation == True:
        index_dict[0] = [train_1, val_1]
        return(index_dict, X_train_loc, X_val_loc)
    else:
        index_dict[0] = [train_1]
        return(index_dict, X_train_loc, None)


def get_train_data(data_dict, cough_data, feature_matrix, X_train_loc, X_val_loc, random_choice=1):
    """Forms final feature embedding of training and validation set from
        splitted and indexed data from feature. 
    Args:
        data_dict: data dictionary consisting of datasets and respective indices.
        cough_data: data csv file
        feature_matrix: final feature embedding
        X_train_loc: training set indices
        X_val_loc: validation set indices
        random_choice: random choice maker
        
    Returns:
        Training and validation sets.
    """
    
    train_data_ind = data_dict[random_choice][0]
    # Patient IDs are used as keys to determine training set patients
    patients_train = X_train_loc[X_train_loc.index.isin(train_data_ind)]["patient_ID"].values
    cough_indices_train = cough_data[cough_data["patient_ID"].isin(patients_train)].index.values
    # Obtaining feature vector of each cough 
    X_train = feature_matrix[cough_indices_train] 
    y_train = cough_data[cough_data.index.isin(cough_indices_train)]["disease"].values
    y_train = cough_data[cough_data["patient_ID"].isin(patients_train)]["disease"].values
    
    # Encoding groundtruth
    y_train = encode_labels(cough_data).transform(y_train)
        
    # If validation set is requested
    if  not isinstance(X_val_loc, type(None)):       
        
        val_data_ind = data_dict[random_choice][1]  
        patients_val = X_val_loc[X_val_loc.index.isin(val_data_ind)]["patient_ID"].values
        cough_indices_val = cough_data[cough_data["patient_ID"].isin(patients_val)].index.values
        X_val = feature_matrix[cough_indices_val]
        y_val = cough_data[cough_data.index.isin(cough_indices_val)]["disease"].values
        y_val = cough_data[cough_data["patient_ID"].isin(patients_val)]["disease"].values
        y_val = encode_labels(cough_data).transform(y_val)
        return(X_train, y_train, X_val, y_val)

    else:
        return(X_train, y_train, None, None)

def encode_labels(cough_data):
    """Encodes labels with label encoder and returns the fitted encoder."""
    
    main_labels = np.array(cough_data[["disease"]]).reshape((-1,))
    encoder = LabelEncoder()
    encoded_labels  = encoder.fit_transform(main_labels)
    
    return(encoder)   

def train_model(X_train, y_train, X_val, y_val, plot_performance, training_type):
    """Trains multi-layer perceptron algorithm hard-coded parameters.
    
    Args:
        X_train: training set
        y_train: training labels
        X_val: validation set
        y_val: validation set labels
        plot_performance: binary parameter, plots the 
            algorithm performance
        training_type: centralized or pretrained. Impacts 
            class_weight parameters.
        
    Returns:
        trained classifier with Keras history object.
    """   
    # Clearing the keras session
    tf.keras.backend.clear_session()

    # Defining the network
    model = tf.keras.models.Sequential(
        [
    
            tf.keras.layers.Dense(64, input_shape=(122,),activation="relu", kernel_initializer = 'glorot_normal'),
            tf.keras.layers.Dense(64, activation="relu", kernel_initializer = 'glorot_normal'),        
            tf.keras.layers.Dense(32, activation="relu", kernel_initializer = 'glorot_normal'),
            tf.keras.layers.Dense(4, activation="softmax")])      

    # Defining the optimizer
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001)
    
    # Defining model performance metric, MCC.
    metric = tfa.metrics.MatthewsCorrelationCoefficient(num_classes=4, name="mcc")
    
    # Compiling the optimizer, loss function and metric
    model.compile(optimizer, "sparse_categorical_crossentropy", metrics=[metric])
    # Printing model summary.
    model.summary()    
    # Including early stopping mechanism that monitors validation loss
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)

    # Setting class weights w.r.t training type
    if training_type == "centralized":
        class_weight = {0:1, 1:8.0, 2:5.0, 3:1}
    elif training_type == "pretrained":
        class_weight = {0:1, 1:6.0, 2:1, 3:1} 
    
    # Model training for 300 epochs.
    if  not isinstance(X_val, type(None)):          
        history = model.fit(X_train, y_train, batch_size=32, class_weight=class_weight, epochs=300, validation_data=(X_val, y_val), verbose=1, callbacks=[callback])
        
    else:
        history = model.fit(X_train, y_train, batch_size=32, class_weight=class_weight, epochs=300, validation_split=0.2, verbose=1, callbacks=[callback])

    # Saving the trained model.
    model_name = training_type + ".h5"
    model.save(model_name)
    
    if plot_performance:
        
        plot_model_performance(model, history, training_type) 
    
    return(history, model)


def plot_model_performance(model, history, training_type):
    """Plots model training performance and saves its graphs in pdf and png format."""
    
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    ax = ax.ravel()

    for i, metric in enumerate(["mcc", "loss"]):
        ax[i].plot(model.history.history[metric])
        ax[i].plot(model.history.history["val_" + metric])
        ax[i].set_title("Model {}".format(metric), fontsize=16)
        ax[i].set_xlabel("epochs", fontsize = 12)
        ax[i].set_ylabel(metric, fontsize = 12)
        ax[i].legend(["train", "validation"], loc='best', fontsize=14)
        ax[i].tick_params(axis='x', labelsize=12)
        ax[i].tick_params(axis='y', labelsize=12)
    
    name = training_type
    plot_name1 = "plots" + "\\" + str(name) + ".pdf"
    plot_name2 = "plots" + "\\" + str(name) + ".png"
    
    plt.savefig(plot_name1, dpi=200) 
    plt.savefig(plot_name2, dpi=200)         

def get_clf_performance(model, X_test, y_test):
    """Creates predictions from trained model and computes accuracy, MCC, specificity,
        sensitivity, precision."""    
    
    results = model.evaluate(X_test, y_test)
    
    y_pred = model.predict(X_test) 
    y_pred_int = y_pred.argmax(axis=1) 
    y_preds_bin = y_pred.argmax(axis=-1)
    
    class_names = ['Asthma', 'COPD', 'COVID-19', 'Healthy']
    clf_report = classification_report(y_test, y_pred_int, target_names=class_names, output_dict=True)
    
    accuracy = results[1]
    mcc = matthews_corrcoef(y_test, y_pred_int)
    precision = precision_score(y_test, y_pred_int, average="micro")    
    conf_matrix = confusion_matrix(y_test, y_pred_int)
    
    FP = conf_matrix.sum(axis=0) - np.diag(conf_matrix) 
    FN = conf_matrix.sum(axis=1) - np.diag(conf_matrix)
    TP = np.diag(conf_matrix)
    TN = conf_matrix.sum() - (FP + FN + TP)
    
    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)
    
    specificity_class = TN/(TN+FP)
    specificity = np.mean(specificity_class)
    
    sensitivity_class = TP/(TP+FN)
    sensitivity = np.mean(sensitivity_class)
    
    clf_performance = [accuracy, mcc, specificity, sensitivity, precision, specificity_class, sensitivity_class]
        
    return(clf_performance)

def save_clf_performance(exp_no, model, training_type, X_test, y_test):
    """Saves classifier performance metrics to clf_performance excel file if file
        exists. Otherwise, it creates excel file and saves the results."""
    
    clf_performance = r"clf_performance.xlsx"
    
    col_names = ['exp', 'training_type', 'acc', 'mcc', 'specificity', 'sensitivity', 'precision', 'specificity_class', 'sensitivity_class']
    
    # If file does not exist, creates one
    if not os.path.exists(clf_performance):
        
        df = pd.DataFrame(columns=col_names)
        df.to_excel('clf_performance.xlsx', index=False)
    else:
        
        df = pd.read_excel("clf_performance.xlsx")
    
    main_inf = []
    exp_no = exp_no
    training_type = training_type
    
    # Obtaining classifier performance.
    performances = get_clf_performance(model, X_test, y_test)
    
    main_inf.append(exp_no)
    main_inf.append(training_type)
    
    for i in performances:
        
        if type(i)==list:
            for v in i:
                main_inf.append(v)
        else:
            main_inf.append(i)
            
    series = pd.Series(main_inf, index=col_names)
    df_final = pd.concat([df, series.to_frame().T], ignore_index=True)

    df_final.to_excel("clf_performance.xlsx",index=False)
    

