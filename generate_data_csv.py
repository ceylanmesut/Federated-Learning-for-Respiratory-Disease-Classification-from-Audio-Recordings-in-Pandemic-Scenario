"""Summarizes dataset and creates a csv file with cough paths, age and gender information."""

import pandas as pd
import numpy as np
import os
import glob
from tqdm import tqdm 
from pathlib import Path
import librosa


def summarize_dataset(disease_name=None, root_path=None, audio_csv_path=None):
    
    """Summarizes dataset w.r.t gender information for each respiratory disease.
    
    Args:
        disease_name: respiratory disease
        root_path: path where data stored
        audio_csv_path: csv file path for asthmatic coughs as their csv is available.
        
    Returns:
        Summary of each dataset w.r.t gender of cough and total cough amount.
    """

    male_counter = 0
    female_counter = 0

    if disease_name.lower() == "asthma":
        configfiles = pd.read_csv(audio_csv_path) 

        for p in tqdm(list(configfiles["astma_audio_path"])):

            file =  os.path.basename(p)

            if file.startswith("cough") & file.endswith("-m.wav"):

                    if file.startswith("coughburst"):
                        continue
                    else:    
                        male_counter += 1

            elif file.startswith("cough") & file.endswith("-f.wav"):
                
                    if file.startswith("coughburst"):
                        continue
                    else:      
                        female_counter += 1                
    else:
        for _, _, files in os.walk(root_path, topdown = True):
                
            for file in files:

                if file.startswith('cough') & file.endswith("-m.wav"):

                    if file.endswith("-x.wav"):
                        continue
                    else:
                        male_counter += 1

                elif file.startswith("cough") & file.endswith("-f.wav"):
                    
                    if file.endswith("-x.wav"):
                        continue
                    else:
                        female_counter += 1    

    print("Number of male %s coughs: %i" % (disease_name, male_counter))
    print("Number of female %s coughs: %i" % (disease_name, female_counter))
    print("Number of Total %s coughs: %i\n" % (disease_name, (male_counter+female_counter)))    
    
def get_cough_paths(disease_names):
    
    """Collects paths of the stored coughs.
    
    Args:
        disease_name: respiratory disease
        
    Returns:
        Dataframe consisting of each cough path.
    """    

    cough_path_list =[]
    name = []

    for disease in disease_names: 

        if disease.lower() == "covid-19":
            path = r"data\raw\Covid\*\*\cough*.wav"
        elif disease.lower()  == "copd":    
            path = r"data\raw\Codex\LabeledData\*\*\*\cough*.wav"

        elif disease.lower() == "asthma":
            print("Asthma paths are already stored in csv file")
            continue
        else: 
            print("Faulty disease name")
            continue

        for sound_file in glob.glob(path):
            name.append(disease)
            cough_path_list.append(sound_file)

    df = pd.DataFrame(cough_path_list, columns = ["cough_path"])   
    df["disease"] = name

    # Excluding other file paths.
    df = df[~df['cough_path'].str.contains("-x.wav")]
    df = df[~df['cough_path'].str.contains("-2.wav")]

    return(df)

def get_ID_cough_gender(dataframe):
    
    """Collects patient ID and cough gender information.
    
    Args:
        dataframe: dataframe that includes cough paths.
        
    Returns:
        Dataframe with patient ID and cough gender information.
    """    
    
    patient_ID_list = []
    cough_gender_list = []

    for idx, rows in tqdm(dataframe.iterrows()):

        path = Path(dataframe["cough_path"][idx])
        disease = dataframe["disease"][idx]

        if disease.lower() == "asthma":
            patient_ID = path.parent.parent.parent.name #S1 Guiliano etc.     

        elif disease.lower() == "copd":
            patient_ID = path.parent.parent.parent.name #CODEX01 etc.     

        elif disease.lower() == "covid-19": 
            patient_ID = (path.parent.parent.name).split("_")[0] #COCO001 etc.

        cough_gender = path.stem[-1] #f or m
    
        patient_ID_list.append(patient_ID)
        cough_gender_list.append(cough_gender)

    dataframe["patient_ID"] = patient_ID_list
    dataframe["cough_gender"]= cough_gender_list        

    return(dataframe)


def get_age_gender(dataframe, asthma_s_data, covid_s_data):
    
    """Gathers patient age and gender information.
    
    Args:
        dataframe: dataframe that includes cough paths.
        asthma_s_data: study data of asthma patients.
        covid_s_data: study data of covid patients.
        
    Returns:
        Dataframe with age and gender information.
    """        

    asthma_add = asthma_s_data
    covid_add = covid_s_data
    
    asthma_add = asthma_add.rename(columns = {"ID":"patient_ID"})
    asthma_add2 = asthma_add.rename(columns = {"Sex":"gender"})
    covid_add = covid_add.rename(columns={"patient_name": "patient_ID"})
    covid_add2 = covid_add.rename(columns = {"Sex":"gender"})

    asthma_to_add = asthma_add[["Age","patient_ID"]].drop_duplicates()
    asthma_to_add2 = asthma_add2[["gender","patient_ID"]].drop_duplicates()
    covid_to_add = covid_add[["Age","patient_ID"]].drop_duplicates()
    covid_to_add2 = covid_add2[["gender","patient_ID"]].drop_duplicates()

    for idx, row in tqdm(dataframe.iterrows()):

        disease = dataframe["disease"][idx]

        if disease == "asthma":
            pt_id = dataframe["patient_ID"][idx].split(" ")[0]
            
            if pt_id in list(asthma_to_add["patient_ID"]):

                age = asthma_to_add["Age"][asthma_to_add["patient_ID"]==pt_id]
                dataframe["age"][idx] = age.values[0]

            if pt_id in list(asthma_to_add2["patient_ID"]):
                gender = asthma_to_add2["gender"][asthma_to_add2["patient_ID"]==pt_id]
                dataframe["gender"][idx] = gender.values[0]                

        elif disease == "covid-19":
            pt_id = dataframe["patient_ID"][idx]
            
            if pt_id in list(covid_to_add["patient_ID"]):

                age = covid_to_add["Age"][covid_to_add["patient_ID"]==pt_id]
                dataframe["age"][idx] = age.values[0]

            if pt_id in list(covid_to_add2["patient_ID"]):

                gender = covid_to_add2["gender"][covid_to_add2["patient_ID"]==pt_id]
                dataframe["gender"][idx] = gender.values[0]
    
        elif disease =="copd":
            if dataframe["patient_ID"][idx] == "CODEX01":
                age_C = 62
                gender = "male"
            elif dataframe["patient_ID"][idx] == "CODEX05":
                age_C = 75
                gender = "male"
            elif dataframe["patient_ID"][idx] == "CODEX06":
                age_C = 63
                gender = "male"
            elif dataframe["patient_ID"][idx] == "CODEX07":
                age_C = 64
                gender = "male"
            elif dataframe["patient_ID"][idx] == "CODEX08":
                age_C = 78
                gender = "female"
            elif dataframe["patient_ID"][idx] == "CODEX09":
                age_C = 76
                gender = "male"    
            elif dataframe["patient_ID"][idx] == "CODEX10":
                age_C = 53
                gender = "male"    
            elif dataframe["patient_ID"][idx] == "CODEX11":
                age_C = 51
                gender = "male"
            elif dataframe["patient_ID"][idx] == "CODEX12":
                age_C = 85
                gender = "male"                                
            else:
                continue
            
            dataframe["age"][idx] = age_C  
            dataframe["gender"][idx] = gender

    return(dataframe)   

def get_cough_duration(dataframe):
    
    """Extracts cough duration.
    
    Args:
        dataframe: dataframe that includes cough paths.
        
    Returns:
        Dataframe with cough duration.
    """        

    cough_durations= []

    for cough in tqdm(dataframe["cough_path"]):
        
        y, sr = librosa.load(cough)
        cough_durations.append(round((librosa.get_duration(y=y, sr=sr)), 6))
    
    dataframe["cough_duration"] = cough_durations

    return dataframe 

def add_healthy_coughs(main_data, path):
    
    """Adds information of the coughs from healthy subjects.
    
    Args:
        main_data: dataframe that includes information w.r.t all respiratory diseases.
        path: path where healthy cough signals are stored.
        
    Returns:
        Dataframe that also includes healthy cough signal information.
    """       
    
    cough_path= path

    cough_files = glob.glob(cough_path)

    # Constructing pandas df
    cough_path_list=[]
    disease_list = np.repeat("healthy", len(cough_files)).reshape(-1,1)
    cough_gender = np.repeat("unknown", len(cough_files)).reshape(-1,1)
    cough_duration= np.repeat(0.5, len(cough_files)).reshape(-1,1)
    age_list= np.repeat(99, len(cough_files)).reshape(-1,1)
    gender = np.repeat("unknown", len(cough_files)).reshape(-1,1)
    patient_ID_list=[]


    for cough in tqdm(glob.glob(cough_path)):
        
        cough_path_list.append(cough)
        
        patient_ID = cough.split("\\")[-2]
        patient_ID_list.append(patient_ID)
        

    cough_path_list = np.asarray(cough_path_list).reshape(-1,1)
    patient_ID_list = np.asarray(patient_ID_list).reshape(-1,1)
        
    data = np.hstack((cough_path_list, disease_list, patient_ID_list, 
                        cough_gender, cough_duration, age_list, gender)) 

    cols = ["cough_path","disease", "patient_ID", "cough_gender",
            "cough_duration", "age", "gender"]

    data_df = pd.DataFrame(data, columns=cols)

    main_data_concat = pd.concat([main_data, data_df], ignore_index=True)

    # Just filling unknown Gender Information and replacing female and male with respective abbrevation
    gender_column = main_data_concat["gender"]
    gender_column = gender_column.replace("unknown", "m")
    gender_column = gender_column.replace("female", "f")
    gender_column = gender_column.replace("male", "m")
    main_data_concat["gender"] = gender_column
    cough_gender = main_data_concat["cough_gender"]
    cough_gender = cough_gender.replace("unknown", "m")
    main_data_concat["cough_gender"] = cough_gender
    main_data_concat.dropna(subset=['gender'], inplace=True)
    
    return(main_data_concat)

def add_healthy_gender_age(dataframe, healthy_study_data):
    
    """Adds age and gender information of healthy subjects.
    
    Args:
        dataframe: dataframe that includes information w.r.t all respiratory diseases.
        healthy_study_data: path where healthy subject study data located.
        
    Returns:
        Dataframe that includes gender, cough gender and age of healthy subjects.
    """         
    
    healthy_study_data =  healthy_study_data
    
    for ind, row in dataframe.iterrows():
        
        if row["disease"] == "healthy":
            
            subject_id = row["patient_ID"]
            gender = healthy_study_data["gender"][healthy_study_data["patient_id"] == int(subject_id)]
            age = healthy_study_data["age"][healthy_study_data["patient_id"] == int(subject_id)]            
            
            dataframe["gender"][ind] = gender.values[0]
            dataframe["cough_gender"][ind] = gender.values[0]
            dataframe["age"][ind] = age.values[0]
             
    return(dataframe)
                
def exclude_partner_coughs(dataframe):
    """Accepts a dataframe and excludes partner coughs from it."""
    
    # Excluding partner coughs (healthy coughs) from sick coughs 
    for idx, _ in tqdm(dataframe.iterrows()):

        if dataframe["cough_gender"][idx] != dataframe["gender"][idx]:
            dataframe = dataframe.drop(idx, axis=0)
            
    return(dataframe)

def save_files(dataframe):
    """Saves created dataframe in a csv and excel formats."""
    
    dataframe.to_excel("data\raw\main_data.xlsx", index=False)
    dataframe.to_csv("data\raw\main_data.csv", index=False)
    



# run time 30 min. 
# asthma_cough_durations = get_cough_duration(only_coughs["asthma_cough_path"])

# Defining root paths to the files and reading asthdata\rawCodex\LabeledData"
root_path_copd = r"data\raw\Codex\LabeledData"
root_path_covid = r"data\raw\Covid"
root_path_healthy= r"data\raw\Healthy Coughs\Data\Samsung\close\*\cough_*.wav"
asthma_audio_csv_path= r"study_data\\asthma_audio_path.csv"

# Reading study data
healthy_study_data =  pd.read_excel("study_data\\Healthy_Subjects.xlsx")
asthma_df = pd.read_csv("study_data\\asthma_cough_paths.csv")
asthma_df["disease"] = "asthma"
asthma_study_data = pd.read_csv("study_data\\200920 - ACP_data_export_Asthma.csv")
covid_study_data = pd.read_csv("study_data\\preprocessed_data_COVID19.csv")

# Summarizing datasets
summarize_dataset(disease_name="Asthma", root_path=None, audio_csv_path=asthma_audio_csv_path)
summarize_dataset(disease_name="COPD", root_path = root_path_copd, audio_csv_path=None)
summarize_dataset(disease_name="COVID-19", root_path = root_path_covid, audio_csv_path=None)
    
# Building a complete df
copd_covid_df = get_cough_paths(["asthma", "covid-19","copd"])
# combining Asthma df and COPD&COVID-19 df
main_data = pd.concat([asthma_df, copd_covid_df], ignore_index=True)

# Obtaining gender of cough
main_data = get_ID_cough_gender(main_data)
# Obtaining cough durations
main_data = get_cough_duration(main_data)

# Adding gender and age information
main_data["age"] = None
main_data["gender"] = None
main_data = get_age_gender(main_data, asthma_study_data, covid_study_data)
main_data = add_healthy_coughs(main_data, path=root_path_healthy)

main_data = add_healthy_gender_age(main_data, healthy_study_data)
main_data = exclude_partner_coughs(main_data)
save_files(main_data)

