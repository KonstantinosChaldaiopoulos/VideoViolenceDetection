import os
import whisper
import librosa
import numpy as np
from _utils import Utils
import moviepy.editor as mp
from transformers import pipeline, AutoTokenizer, TFAutoModel
from pyAudioAnalysis import ShortTermFeatures as aF

class Dataset:

    def __init__(self, dataset_path, window=0.02, step=0.01, sample_rate=16000, metric="std", model="facebook/wav2vec2-base-960h"):
        self.videos_path = os.path.join(dataset_path, "videos")
        self.audios_path = os.path.join(dataset_path, "audios")
        self.texts_path = os.path.join(dataset_path, "texts")
        self.window = window
        self.step = step
        self.sample_rate = sample_rate
        self.metric = metric
        self.model = model
        self.fn = None
        self.asr = pipeline("automatic-speech-recognition", model=model)
        # self.asr = pipeline("automatic-speech-recognition", model="mozilla/deepspeech-0.9.3")
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.model = TFAutoModel.from_pretrained('bert-base-uncased')
        self.video = None
        self.video_name = None
        self.Xa = []
        self.Xt = []
        self.y = []

    def prepare_data(self):
        for self.video_name in os.listdir(self.videos_path):
            video_path = os.path.join(self.videos_path, self.video_name)
            self.video = mp.VideoFileClip(video_path)
            self.y.append("v" if self.video_name.startswith("V") else "n")
            self.extract_audio_features()
            self.extract_text_features()
        self.Xa = np.array(self.Xa)
        self.Xt = self.get_bert_embeddings()
        self.y = np.array(self.y)
        return (self.Xa, self.y), (self.Xt, self.y)

    def extract_audio_features(self):
        audio_path = os.path.join(self.audios_path, os.path.splitext(self.video_name)[0] + ".wav")
        audio = self.video.audio
        audio.write_audiofile(audio_path)
        s, fs = librosa.load(audio_path, sr=self.sample_rate) 
        [f, self.fn] = aF.feature_extraction(s, fs, int(fs * self.window), int(fs * self.step))
        self.Xa.append(Utils.represent(f, self.metric))

    def extract_text_features(self):
        audio_path = os.path.join(self.audios_path, os.path.splitext(self.video_name)[0] + ".wav")
        text_path = os.path.join(self.texts_path, os.path.splitext(self.video_name)[0] + ".txt")
        text = self.asr(librosa.load(audio_path, sr=None)[0])["text"]
        with open(text_path, "w") as f:
            f.write(text)
        self.Xt.append(text)

    def get_bert_embeddings(self):
        inputs = self.tokenizer(self.Xt, return_tensors='tf', padding=True, truncation=True, max_length=None)
        outputs = self.model(inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].numpy() # [CLS] token at id=0, represents the entire input sequence (captures its overall meaning)
        return embeddings

    def get_feature_names(self):
            return self.fn