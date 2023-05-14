import os
import librosa
import numpy as np
from _utils import Utils
from pyAudioAnalysis import ShortTermFeatures as aF

class Dataset:

    def __init__(self, dataset_path, window=0.02, step=0.01, sample_rate=16000, metric="std"):
        self.videos_path = os.path.join(dataset_path, "videos")
        self.audios_path = os.path.join(dataset_path, "audios")
        self.window = window
        self.step = step
        self.sample_rate = sample_rate
        self.metric = metric
        self.fn = None
        self.video_name = None
        self.X = []
        self.y = []

    def prepare_data(self):
        for self.video_name in os.listdir(self.videos_path):
            self.y.append("v" if self.video_name.startswith("V") else "n")
            self.extract_audio_features()
        self.X = np.array(self.X)
        self.y = np.array(self.y)
        return (self.X, self.y)

    def extract_audio_features(self):
        audio_path = os.path.join(self.audios_path, os.path.splitext(self.video_name)[0] + ".wav")
        s, fs = librosa.load(audio_path, sr=self.sample_rate) 
        [f, self.fn] = aF.feature_extraction(s, fs, int(fs * self.window), int(fs * self.step))
        self.X.append(Utils.represent(f, self.metric))
    
    def get_feature_names(self):
        return self.fn