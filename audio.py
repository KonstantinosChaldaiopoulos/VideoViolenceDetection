import os
import librosa
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pyAudioAnalysis import ShortTermFeatures as aF
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
import moviepy.editor as mp
from google.colab import drive

def get_statistic(features, type): # features: 2D numpy array with each row representing a feature and each column representing a frame
    if type == "means":
        return np.mean(features, axis=1)
    elif type == "std":
        return np.std(features, axis=1)
    elif type == "median":
        return np.median(features, axis=1)
    elif type == "range":
        return np.ptp(features, axis=1)

def select_features(X_train, y_train, X_test, k, names, display):  # selects the k best features based on F-score
    selector = SelectKBest(f_classif, k=k)
    X_train = selector.fit_transform(X_train, y_train)
    X_test = selector.transform(X_test)
    names = [names[i] for i, selected in enumerate(selector.get_support()) if selected]
    print(names) if display else None
    return X_train, X_test

def scale_features(X_train, X_test):
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test

def grid_search(clf, X_train, y_train):
    param_grid = {
        'kernel': ['linear', 'rbf', 'poly'],
        'C': [0.1, 0.5, 1, 2, 5, 10, 20],
        'gamma': ['scale', 'auto']}
    grid_search = GridSearchCV(clf, param_grid, cv=2) # creates a grid search object with 5-fold cross-validation
    grid_search.fit(X_train, y_train)
    print("Best parameters:", grid_search.best_params_)
    return grid_search.best_params_

drive.mount('/content/drive')

# ACTUAL DATASET
#"""
dataset_path = "/content/drive/MyDrive/DATASETS/movie-dataset"
videos_path = os.path.join(dataset_path, "videos")
audios_path = os.path.join(dataset_path, "audios")
#"""

# TESTER DATASET
"""
dataset_path = "/content/drive/MyDrive/DATASETS/tester"
videos_path = os.path.join(dataset_path, "videos")
audios_path = os.path.join(dataset_path, "audios")
"""

window = 0.02; step = 0.01  
X = []; y = []

for video_name in os.listdir(videos_path):
    if video_name.endswith(".avi"):
        vipath = os.path.join(videos_path, video_name)
        aupath = os.path.join(audios_path, os.path.splitext(video_name)[0] + ".wav")
        video = mp.VideoFileClip(vipath)
        audio = video.audio
        audio.write_audiofile(aupath)

        # number of frames = 1 + int((duration-window)/step)
        s, fs = librosa.load(aupath, sr=16000) # s: the audio signal as a 1D numpy array | fs: the sampling rate of the audio signal
        [f, fn] = aF.feature_extraction(s, fs, int(fs * window), int(fs * step)) # f: 68 rows (features) and columns (frames) | fn: 68 feature names

        value = get_statistic(f, "std")
        X.append(value)

        if video_name.startswith("V"):
            y.append("V")
        elif video_name.startswith("NV"):
            y.append("NV")

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None, shuffle=True)
X_train, X_test = scale_features(X_train, X_test)
X_train, X_test = select_features(X_train, y_train, X_test, 7, fn, False)

# Train the SVM classifier on the training set
svm = SVC(C=1.0, kernel='rbf', gamma='scale')

# NOTE: alternatively you can find the best parameters that fit the training dataset
"""
svm = SVC()
svm.set_params(**grid_search(svm, X_train, y_train))  # sets best parameters
"""

svm.fit(X_train, y_train)

# Evaluate the classifier on the testing set
y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Predicted labels: {y_pred}\nTrue labels: {y_test}\nAccuracy: {accuracy:.2f}")