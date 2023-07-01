# Multimodal Violence Detection in Videos

## Project Description
In this project, we implemented a multimodal classification system for classifying videos into violent and nonviolent. The system uses three modalities: text, audio, and image data. The approach includes both early fusion, where all modalities are combined and fed to one neural network, and late fusion, where the outputs of individual models for each modality are combined to make a final decision.

## Overview
Our code implementation consists of three main sections:

* **Data Preparation:**
We load, extract and preprocess audio, image, and text data. For the audio data, the code uses the librosa library to load audio files and extract useful features like Mel-frequency cepstral coefficients (MFCC) and short-term features. These features are then processed and stored along with corresponding labels (0 for "nonviolence" and 1 for "violence") for each step of the process (training, validation, and testing). The text data is processed using the DistilBERT model, a state-of-the-art transformer-based language model. The code tokenizes the text, converts it into PyTorch tensors, and passes it through the DistilBERT model to obtain embeddings. These embeddings are stored along with their corresponding labels for each process (training, validation, and testing). Finally, for the image data, a custom FrameDataset class is created using PyTorch's built-in classes. The code loads image frames from the provided directories (training, validation, and testing), converts them into tensors, and assigns labels based on whether the video belongs to the "nonviolence" or "violence" class.

* **Modalities Classification:**
We implement two different multimodal fusion approaches: Early Fusion and Late Fusion. In the Early Fusion approach, data from all three modalities (audio, image, and text) are combined together beforehand. The code creates a MultiMovieDataset class that takes the preprocessed audio and text data as input. It then prepares the combined dataset, storing it as separate numpy files for training, validation, and testing. The Trintegrator class is responsible for loading and processing the combined data during the training process. The combined data contains video frames, audio features, text embeddings, and their corresponding labels. In the Late Fusion approach, separate classifiers are created for each modality (audio, image, and text). The code uses the preprocessed audio and text features directly, while for the image data the FrameDataset class is used to load the image frames and their corresponding labels. Classifiers are implemented for each modality, and their outputs are combined using techniques like voting or averaging.

* **Evaluate Results:**
After training the classifiers using both the Early Fusion and Late Fusion approach, the code evaluates the performance of the multimodal classification system. It calculates and displays the accuracy scores for both approaches. This evaluation helps demonstrate the effectiveness of multimodal fusion in improving classification accuracy and leveraging complementary information from different data sources.

## Prerequisites
The following python packages are required for the code to run:
* Python 3: https://www.python.org/downloads/
* Scikit-learn: ```pip install -U scikit-learn```
* PrettyTable: ```pip install prettytable```

**Alternatively:** you can download [requirements.txt](https://github.com/KonstantinosChaldaiopoulos/VideoViolenceDetection/blob/main/requirements.txt) and run ```pip install -r requirements.txt```, to automatically install all the packages needed to reproduce my project on your own machine.

**```>```** For this project, we used ..... You can download the dataset directly from this link: ....  Please note that in order to run the code, you should have .... in your local folder.

## Authors
Konstantinos Chaldaiopoulos & Natalia Koliou
