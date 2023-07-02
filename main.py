from google.colab import drive
import warnings
import numpy as np
from _model import *
from _dataset import *
from _utils import *

drive.mount('/content/drive')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.set_printoptions(suppress=True)
warnings.filterwarnings("ignore", category=UserWarning)

path = "/content/drive/MyDrive/DATASETS/movie-dataset"

ds = MovieDataset(device, path, sample_rate=16000, metric="std")
audio_data, image_data, text_data = ds.prepare_data()

aSVM = TRClassifier(*audio_data, ds.get_feature_names(), num_features=7, classifier="svm", scaler=None)
audio_val_acc, audio_test_conf, audio_labels = aSVM.run()

tSVM = TRClassifier(*text_data, None, num_features=10, classifier="svm", scaler=None)
text_val_acc, text_test_conf, text_labels = tSVM.run()

iNN = NNClassifier(*image_data, epochs=20, lr=0.00001, wd=0.05, device=device)
image_val_acc, image_test_conf, image_labels = iNN.run()

mds = MultiMovieDataset(audio_data, image_data, text_data, exists=False)
multi_data = mds.prepare_multidata()

mdMNN = MNNClassifier(*multi_data, epochs=25, lr=0.00001, wd=0.0, device=device)
multi_test_acc = mdMNN.run()

late_fusion_acc = late_fuse((audio_val_acc, text_val_acc, image_val_acc), (audio_test_conf, text_test_conf, image_test_conf), audio_labels)
print("Late Fusion Accuracy: {:.2f} %".format(late_fusion_acc))
print("Early Fusion Accuracy: {:.2f} %".format(multi_test_acc))