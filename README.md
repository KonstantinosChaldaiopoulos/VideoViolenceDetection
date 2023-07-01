# Multimodal Violence Detection in Videos

## Project Description
In this project, we implemented a multimodal classification system for classifying videos into violent and nonviolent. The system uses three modalities: text, audio, and image data. The approach includes both early fusion, where all modalities are combined and fed to one neural network, and late fusion, where the outputs of individual models for each modality are combined to make a final decision.

## Overview
Our code implementation consists of three main sections:

* **Data Preparation:**
We load, extract and preprocess audio, image, and text data. For audio data, it uses the librosa library to load audio files and extract midterm features, which are then processed and stored with corresponding labels (0 for "nonviolence" and 1 for "violence") for each step (training, validation, and testing). The text data is processed using the DistilBERT model, tokenizing the text, converting it into PyTorch tensors, and obtaining embeddings. These embeddings, along with their labels, are stored for each step (training, validation, and testing). Lastly, for image data, a custom FrameDataset class is created using PyTorch's built-in classes. The code loads image frames from the directories (training, validation, and testing), converts them into tensors, and assigns labels based on the "nonviolence" or "violence" class.

* **Modalities Classification:**
We implement two different multimodal fusion approaches: Early Fusion and Late Fusion. In Early Fusion, we combine data from audio, image, and text modalities beforehand using the MultiMovieDataset class, storing the combined dataset as numpy files for training, validation, and testing. The Trintegrator class loads and processes this combined data during training, containing video frames, audio features, text embeddings, and labels. In Late Fusion, we create separate classifiers for each modality (audio, image, and text). Preprocessed audio and text features are directly used, and FrameDataset class handles image frame loading with corresponding labels.

* **Evaluate Results:**
After training the classifiers using both the Early Fusion and Late Fusion approach, the code evaluates the performance of the multimodal classification system. It calculates and displays the accuracy scores for both approaches. This evaluation helps demonstrate the effectiveness of multimodal fusion in improving classification accuracy and leveraging complementary information from different data sources.

## Running the Code
To run the code, please follow these steps:
1. Download [violence-detection.ipynb](https://github.com/KonstantinosChaldaiopoulos/VideoViolenceDetection/blob/main/violence-detection.ipynb), along with [_utils.py](https://github.com/KonstantinosChaldaiopoulos/VideoViolenceDetection/blob/main/_utils.py), [_model.py](https://github.com/KonstantinosChaldaiopoulos/VideoViolenceDetection/blob/main/_model.py), and [_dataset.py](https://github.com/KonstantinosChaldaiopoulos/VideoViolenceDetection/blob/main/_dataset.py).
2. Download the [movie-dataset](https://drive.google.com/drive/folders/1K6P0tfItrAPVkCyPw-8al3Br2le1B7JH?usp=sharing) and store it in a folder named DATASETS inside your Google Drive.
3. In your Google Drive, create an empty folder named `mdml-run-env` and place _utils.py, _model.py, and _dataset.py inside it.
4. Open Google Colab and upload the violence-detection.ipynb notebook.
5. Make sure the notebook has access to the necessary libraries and runtime environment.
6. Run the notebook cell by cell to execute the code.

## Prerequisites
The following python packages are required for the code to run:
* Python 3: https://www.python.org/downloads/
* Scikit-learn: ```pip install -U scikit-learn```
* PrettyTable: ```pip install prettytable```

**Alternatively:** you can download [requirements.txt](https://github.com/KonstantinosChaldaiopoulos/VideoViolenceDetection/blob/main/requirements.txt) and run ```pip install -r requirements.txt```, to automatically install all the packages needed to reproduce my project on your own machine.

**```>```** For this project, we used ..... You can download the dataset directly from this link: ....  Please note that in order to run the code, you should have .... in your local folder.

## Authors
Konstantinos Chaldaiopoulos & Natalia Koliou
