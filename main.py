import random
from _model import SVMClassifier
from _dataset import Dataset
from _utils import Utils
from google.colab import drive

drive.mount('/content/drive')

random_state = random.randint(1, 100)

#path = "/content/drive/MyDrive/DATASETS/movie-dataset"
path = "/content/drive/MyDrive/DATASETS/tester"

################################################################################################################
ds = Dataset(path, window=0.02, step=0.01, sample_rate=16000, metric="std")

audio_data, text_data = ds.prepare_data()

aSVM = SVMClassifier(*audio_data, ds.get_feature_names(), random_state, test_size=0.2, num_features=7)
ya_pred, y_test, audio_certainty, audio_accuracy = aSVM.run()

tSVM = SVMClassifier(*text_data, None, random_state, test_size=0.2, num_features=7)
yt_pred, y_test, text_certainty, text_accuracy = tSVM.run()

results = ((ya_pred, yt_pred), (y_test, y_test), (audio_accuracy, text_accuracy), (audio_certainty, text_certainty))
Utils.visualize(results, ['Predicted Labels', 'True Labels', 'Accuracy', 'Certainty'])