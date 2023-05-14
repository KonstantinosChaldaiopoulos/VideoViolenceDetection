from _model import SVMClassifier
from _dataset import Dataset
from google.colab import drive

drive.mount('/content/drive')

#path = "/content/drive/MyDrive/DATASETS/movie-dataset"
path = "/content/drive/MyDrive/DATASETS/tester"

ds = Dataset(path, window=0.02, step=0.01, sample_rate=16000, n=2, metric="std")

audio_data, test_data = ds.prepare_data()
svm = SVMClassifier(*audio_data, ds.get_feature_names(), test_size=0.2, num_features=7)
ya_pred, y_test, audio_certainty, audio_accuracy = svm.run()

#TODO: in an array (rows: this stuff and columns: modalities)

print("Selected audio features:", svm.get_feature_names)
print("Best audio parameters:", svm.get_best_parameters)
print("Predicted audio labels:", ya_pred)

print("True labels:", y_test)
print("Audio Certainty:", audio_certainty)
print("Audio Accuracy:", audio_accuracy)