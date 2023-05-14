from _model import SVMClassifier
from _dataset import Dataset
from google.colab import drive

drive.mount('/content/drive')

#path = "/content/drive/MyDrive/DATASETS/movie-dataset"
path = "/content/drive/MyDrive/DATASETS/tester"

ds = Dataset(path, window=0.02, step=0.01, sample_rate=16000, metric="std")
svm = SVMClassifier(*ds.prepare_data(), ds.get_feature_names(), test_size=0.2, num_features=7, display=False)
svm.run()