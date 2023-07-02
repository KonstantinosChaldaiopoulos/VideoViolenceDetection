import torch
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler

def represent(array, metric):
    if metric == "means":
        return np.mean(array, axis=1)
    elif metric == "std":
        return np.std(array, axis=1)
    elif metric == "median":
        return np.median(array, axis=1)
    elif metric == "range":
        return np.ptp(array, axis=1)
    
def select_classifier(classifier_name):
    if classifier_name == "knn":
        return {"instance": KNeighborsClassifier(), "param_grid": {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance'], 'p': [1, 2]}}
    elif classifier_name == "rf":
        return {"instance": RandomForestClassifier(), "param_grid": {'n_estimators': [100, 200, 300], 'max_depth': [None, 5, 10], 'min_samples_split': [2, 5, 10]}}
    elif classifier_name == "dt":
        return {"instance": DecisionTreeClassifier(), "param_grid": {'max_depth': [None, 5, 10], 'min_samples_split': [2, 5, 10]}}
    elif classifier_name == "gnb":
        return {"instance": GaussianNB(), "param_grid": {}}
    elif classifier_name == "lr":
        return {"instance": LogisticRegression(), "param_grid": {'C': [0.1, 0.5, 1, 2, 5, 10]}}
    elif classifier_name == "gb":
        return {"instance": GradientBoostingClassifier(), "param_grid": {'learning_rate': [0.1, 0.5, 1], 'n_estimators': [50, 100, 200]}}
    else: # default classifier: classifier_name = "svm"
        return {"instance": SVC(probability=True), "param_grid": {'kernel': ['linear', 'rbf', 'poly'], 'C': [0.1, 0.5, 1, 2, 5, 10, 20], 'gamma': ['scale', 'auto']}}
    
def select_scaler(scaler_name):
    if scaler_name == "standard":
        return StandardScaler()
    elif scaler_name == "minmax":
        return MinMaxScaler()
    elif scaler_name == "robust":
        return RobustScaler()
    elif scaler_name == "maxabs":
        return MaxAbsScaler()
    else: # default scaler: scaler_name = None
        return None
    
def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    elif isinstance(x, list):
        return [to_numpy(i) for i in x]
    elif isinstance(x, tuple):
        return tuple(to_numpy(i) for i in x)
    elif isinstance(x, dict):
        return {k: to_numpy(v) for k, v in x.items()}
    else:
        return x

def late_fuse(val_accuracies, test_confidences, labels):
    weights = tuple(val / sum(val_accuracies) for val in val_accuracies)
    test_confidences = to_numpy(test_confidences)
    fused_confidence = sum(w * np.array(conf) for w, conf in zip(weights, test_confidences))
    prediction = fused_confidence.argmax(axis=1)
    late_fusion_accuracy = accuracy_score(labels, prediction)
    return late_fusion_accuracy

def to_npdict(dict):
    return {key: np.array(value) for key, value in dict.items()}

def to_tensorlist(tensor3D):
    list2Dtensors = [tensor.squeeze(0) for tensor in torch.split(tensor3D, 1, dim=0)]
    return list2Dtensors

def to_nplist(list2Dtensors):
    list2Darrays = [tensor.numpy() for tensor in list2Dtensors]
    return list2Darrays

def flatten2D(features):
    return [x.flatten() for x in features]

def get_step(window, overlap):
    return int(window * (1 - overlap))

def reverse_halves(array):
    midpoint = len(array) // 2
    return np.concatenate((array[midpoint:], array[:midpoint]))

def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.legend()
    plt.title('Train vs Validation Loss')
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.legend()
    plt.title('Train vs Validation Accuracy')
    plt.show()