import os
import math
import glob
import torch
from PIL import Image
import librosa
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from transformers import DistilBertTokenizer, DistilBertModel
from pyAudioAnalysis import MidTermFeatures
from _utils import *
from _model import *

class MovieDataset:
    def __init__(self, device, dataset_path, sample_rate=16000, metric="std"):
        self.device = device
        self.dataset_path = dataset_path
        self.sample_rate = sample_rate
        self.metric = metric
        self.batch_size = 1
        self.mfn = None
        self.audios_path = os.path.join(dataset_path, "audios")
        self.images_path = os.path.join(dataset_path, "images")
        self.texts_path = os.path.join(dataset_path, "texts")
        self.processes = ["training", "validation", "testing"]
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(self.device)
        self.Xa, self.ya, self.Xt, self.yt = ({process: [] for process in self.processes} for _ in range(4))

    def prepare_data(self):
        return self.get_audio_data(), self.get_image_data(), self.get_text_data()

    def get_audio_data(self):
        for process in self.processes:
            for type in ["nonviolence", "violence"]:
                path = os.path.join(self.audios_path, process, type)
                for audio_name in os.listdir(path):
                    audio_path = os.path.join(path, audio_name)
                    s, fs = librosa.load(audio_path, sr=self.sample_rate)
                    [mf, sf, self.mfn] = MidTermFeatures.mid_feature_extraction(signal=s,
                                                                           sampling_rate=fs,
                                                                           mid_window=1*fs,
                                                                           mid_step=get_step(window=1*fs, overlap=0.5),
                                                                           short_window=0.05*fs,
                                                                           short_step=get_step(window=0.05*fs, overlap=0.5))
                    self.Xa[process].append(represent(mf, self.metric))
                    self.ya[process].append(0 if type == "nonviolence" else 1)
        return (to_npdict(self.Xa), to_npdict(self.ya)) # Xa[process] is a list of numpy arrays: [ numpy array 136 elements, numpy array 136 elements, ..... ]

    def get_text_data(self):
        for process in self.processes:
            texts = []
            for type in ["nonviolence", "violence"]:
                path = os.path.join(self.texts_path, process, type)
                for text_name in os.listdir(path):
                    text_path = os.path.join(path, text_name)
                    with open(text_path, "r") as f:
                        texts.append(f.read())
                    self.yt[process].append(0 if type == "violence" else 1)
            embeddings = self.get_bert_embeddings(texts).cpu().detach()
            self.Xt[process].extend(flatten2D(to_nplist(to_tensorlist(embeddings))))
        return (to_npdict(self.Xt), to_npdict(self.yt))

    def get_image_data(self):
        train_dataset = FrameDataset(os.path.join(self.images_path, "training"), ToTensor())
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=8)
        val_dataset = FrameDataset(os.path.join(self.images_path, "validation"), ToTensor())
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=8)
        test_dataset = FrameDataset(os.path.join(self.images_path, "testing"), ToTensor())
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=8)
        return (train_loader, val_loader, test_loader)
    
    def get_bert_embeddings(self, texts):
        embeddings, num_texts = [], len(texts)
        num_batches = math.ceil(num_texts / self.batch_size)
        for i in range(num_batches):
            start_idx = i * self.batch_size
            end_idx = start_idx + self.batch_size
            batch = texts[start_idx:end_idx]
            inputs = self.tokenizer(batch, return_tensors='pt', padding='max_length', truncation=True, max_length=70)
            outputs = self.model(**inputs.to(self.device))[0]
            embeddings.append(outputs)
        return torch.cat(embeddings, dim=0)
    
    def get_feature_names(self):
        return self.mfn

    
class FrameDataset(Dataset):
    def __init__(self, root_dir, transform=ToTensor()):
        self.root_dir = root_dir
        self.video_folders = glob.glob(os.path.join(root_dir, '*/*'))
        self.labels = [0 if 'nonviolence' in path else 1 for path in self.video_folders]
        self.transform = transform

    def __getitem__(self, index):
        folder_path = self.video_folders[index]
        frame_paths = sorted(glob.glob(os.path.join(folder_path, '*.png')))
        if not frame_paths:
            print(f'No frames found in {folder_path}')
        frames = [self.transform(Image.open(frame)) for frame in frame_paths]
        video = torch.stack(frames)
        label = self.labels[index]
        return video, label

    def __len__(self):
        return len(self.video_folders)


class MultiMovieDataset(Dataset):
    def __init__(self, audio_data, image_data, text_data, exists=False):
        self.audio_data = audio_data
        self.image_data = image_data
        self.text_data = text_data
        self.exists = exists
    
    def prepare_multidata(self):
        self.load_and_save(self.audio_data, "audio") if not self.exists else None
        self.load_and_save(self.text_data, "text") if not self.exists else None
        return (self.get_loader("training"), self.get_loader("validation"), self.get_loader("testing"))

    def load_and_save(self, data, type):
        training = reverse_halves(data[0]["training"])
        validation = reverse_halves(data[0]["validation"])
        testing = reverse_halves(data[0]["testing"])
        np.save('training_' + type + '.npy', training)
        np.save('validation_' + type + '.npy', validation)
        np.save('testing_' + type + '.npy', testing)

    def get_loader(self, type):
        image_path = '/content/drive/MyDrive/DATASETS/movie-dataset/images/'+ type
        audio_path = '/content/drive/MyDrive/mdml-run-env/' + type + '_audio.npy'
        text_path = '/content/drive/MyDrive/mdml-run-env/' + type + '_text.npy'
        dataset = Trintegrator(image_path, audio_path, text_path)
        return DataLoader(dataset, batch_size=8, shuffle=True, num_workers=8)


class Trintegrator(Dataset):
    def __init__(self, root_dir, audio_dir, text_dir, transform=ToTensor()):
        self.root_dir = root_dir
        self.audio_data = np.load(audio_dir)
        self.text_data = np.load(text_dir)
        self.video_folders = glob.glob(os.path.join(root_dir, '*/*'))
        self.labels = [0 if 'nonviolence' in path else 1 for path in self.video_folders]
        self.transform = transform

    def __len__(self):
        return len(self.video_folders)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        folder_path = self.video_folders[idx]
        frame_paths = sorted(glob.glob(os.path.join(folder_path, '*.png')))
        if not frame_paths:
            print(f'No frames found in {folder_path}')
        frames = [self.transform(Image.open(frame)) for frame in frame_paths]
        image_data = torch.stack(frames)

        label = self.labels[idx]

        audio_sample = torch.from_numpy(self.audio_data[idx])
        text_sample = torch.from_numpy(self.text_data[idx])

        return image_data, audio_sample, text_sample, label  
