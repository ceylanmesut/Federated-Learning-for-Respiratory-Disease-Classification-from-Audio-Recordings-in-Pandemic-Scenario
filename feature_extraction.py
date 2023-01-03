"""Extracts audio features from cough signals and generates the final feature embedding."""

import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm 
import librosa
import librosa.display
from sklearn.preprocessing import MinMaxScaler 
import scipy.stats.mstats


class Feature_Extractor:
    """nx122-dimensional feature embedding extractor.
    
    Extracts:
        -Mel-Frequency Cepstral Coefficients their velocity and acceleration.
        -Log-Filterbanks and their velocity and acceleration.
        -Zero-Crossing Rate
        -Kurtosis
        
    Args:
        path: data csv file path where cough paths are stored.
        scaler: scaler method used to transfor the signal before feature extraction.
        
    """
    
    def __init__(self, path, scaler=None):
        self.y, self.sr = librosa.load(path, sr=44100)
        self.scaler = scaler
        if scaler != None:
            self.y = self.scaler.fit_transform(np.reshape(self.y, newshape=(-1,1)))
        self.features = None

    def get_features(self):

        self._get_mfcc()
        self._get_mfcc_delta()
        self._get_mfcc_delta_delta()
        self._get_log_banks()
        self._get_log_banks_delta()
        self._get_log_banks_delta_delta()
        self._get_zero_crossing()
        self._get_kurtosis()

    def _concat_features(self, feature):
        
        self.features = np.hstack(
            [self.features, feature]
            if self.features is not None else feature)

    def _get_mfcc(self):
        
        # 1. Mel-Frequency Cepstral Coefficients: (n X n_mfcc 20)
        self.mfcc = librosa.feature.mfcc(y=self.y, sr=self.sr, n_mfcc = 20, n_fft=2048, hop_length=112)
        mfcc_mean = np.mean((self.mfcc).T, axis=0)
        self._concat_features(mfcc_mean) 

    def _get_mfcc_delta(self):

        # 2. Mel-Frequency Cepstral Coefficients Delta: (n X n_mfcc 16)
        mfcc_delta = librosa.feature.delta(self.mfcc, mode = "constant")
        mfcc_delta_mean = np.mean(mfcc_delta.T, axis=0) 
        self._concat_features(mfcc_delta_mean) 
    
    def _get_mfcc_delta_delta(self):

        # 3. Mel-Frequency Cepstral Coefficients Delta-Delta: (n X n_mfcc 16)
        mfcc_delta_delta = librosa.feature.delta(self.mfcc, mode = "constant", order = 2)
        mfcc_delta_delta_mean = np.mean(mfcc_delta_delta.T, axis=0) 
        self._concat_features(mfcc_delta_delta_mean)         

    def _get_log_banks(self):

        # 4. Log-Filterbanks (n X n_mels 20)
        specto = librosa.feature.melspectrogram(y = self.y, sr = self.sr, n_fft=2048, hop_length=112, n_mels=20)
        # Convert an amplitude spectrogram to dB-scaled spectrogram.
        self.log_specto = librosa.core.amplitude_to_db(specto)
        log_specto_mean = np.mean(self.log_specto, axis=1) # Averaging out by samples and not mels, returns 20 mels 
        self._concat_features(log_specto_mean)
        
    def _get_log_banks_delta(self):

        # 5. Log-Filterbanks Delta (n X n_mels 20)
        log_specto_delta = librosa.feature.delta(self.log_specto, mode = "constant")
        log_specto_delta_mean = np.mean(log_specto_delta.T, axis=0) 
        self._concat_features(log_specto_delta_mean)

    def _get_log_banks_delta_delta(self):

        # 6. Log-Filterbanks Delta-Delta (n X n_mels 20)
        log_specto_delta_delta = librosa.feature.delta(self.log_specto, mode = "constant", order = 2)
        log_specto_delta_delta_mean = np.mean(log_specto_delta_delta.T, axis=0) 
        self._concat_features(log_specto_delta_delta_mean)

    def _get_zero_crossing(self):
        
        # 7. Zero-Crossing Rate (n x 1)
        zero_rate = librosa.feature.zero_crossing_rate(self.y)
        zero_rate = np.mean(zero_rate, axis=1)
        self._concat_features(zero_rate)
        
    def _get_kurtosis(self):   

        # 8. Kurtosis (n x 1)
        kurtosis = scipy.stats.kurtosis(self.y)
        self._concat_features(kurtosis)

cough_file = pd.read_csv("data\\processed\\main_data_wo_outliers.csv")
audio_features  = []

asthma = cough_file[cough_file["disease"]=="asthma"] 
copd = cough_file[cough_file["disease"]=="copd"]
covid = cough_file[cough_file["disease"]=="covid-19"]
healthy = cough_file[cough_file["disease"]=="healthy"]

subset = [asthma, copd, covid, healthy]

for sub in subset:

    # Iterating the cough paths
    for i, row in tqdm(sub.iterrows(), total = sub.shape[0]):
        cough = Feature_Extractor(row["cough_path"], scaler=MinMaxScaler(feature_range=(-1,1)))
        # cough = Feature_Extractor(row["cough_path"], scaler=None)
        cough.get_features()
        audio_features.append(cough)

# Returns n X 122 dimensional feature matrix
feature_matrix = np.vstack([cough.features for cough in audio_features])
# Saving the feature embedding
with open("data\\final\\feature_embedding.pkl", "wb") as file:
    pickle.dump(feature_matrix, file)


