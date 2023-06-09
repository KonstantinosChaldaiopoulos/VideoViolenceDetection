# Violence Detection in Movies: A Multimodal Approach

With the rapid growth of video content, the efficient detection of violence in videos has become increasingly crucial for ensuring effective regulation. In this project, our main focus was on developing a multimodal classification system capable of classifying videos into violent and nonviolent. To achieve this, we used three modalities: text, audio, and image data. We explored both early fusion, where all modalities are combined and processed through one neural network, and late fusion, where individual models for each modality produce outputs that are combined to make a final decision.

## Overview
Our code implementation consists of three main sections:

* **Data Preparation:**
We load, extract and preprocess audio, image, and text data. For audio data, it uses the librosa library to load audio files and the pyAudioAnalysis library to extract midterm features, which are then processed and stored with corresponding labels (0 for "nonviolence" and 1 for "violence") for each step (training, validation, and testing). The text data is processed using the DistilBERT model, tokenizing the text, converting it into PyTorch tensors, and obtaining embeddings. These embeddings, along with their labels, are stored for each step (training, validation, and testing). Lastly, for image data, a custom FrameDataset class is created using PyTorch's built-in classes. The code loads image frames from the directories (training, validation, and testing), converts them into tensors, and assigns labels based on the "nonviolence" or "violence" class.

* **Modalities Classification:**
We implement two different multimodal fusion approaches: Early Fusion and Late Fusion. In Early Fusion, we combine data from audio, image, and text modalities beforehand using the MultiMovieDataset class, storing the combined dataset as numpy files for training, validation, and testing. The Trintegrator class loads and processes this combined data during training, containing video frames, audio features, text embeddings, and labels. In Late Fusion, we create separate classifiers for each modality (audio, image, and text). Preprocessed audio and text features are directly used, and FrameDataset class handles image frame loading with corresponding labels.

   > We constructed the image classification model that was also used for early and late fusion, based on the paper [Detecting Violence in Video Based on Deep Features Fusion Technique](https://arxiv.org/ftp/arxiv/papers/2204/2204.07443.pdf) by Heyam M. Bin Jahlan and Lamiaa A. Elrefaei with minor changes . As there was no existing GitHub implementation, we took the initiative to build the model from scratch ourselves, adhering to the methodologies and findings presented in the paper. Full credit goes to the authors for their work, which served as a valuable foundation for our implementation in this project.

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
* Scikit-learn: `pip install -U scikit-learn`
* PyTorch: `pip install torch`
* Transformers: `pip install transformers`
* pyAudioAnalysis: `pip install pyAudioAnalysis`
* tqdm: `pip install tqdm`
* NumPy: `pip install numpy`
* Pillow: `pip install pillow`
* Librosa: `pip install librosa`

**Alternatively:** you can download [requirements.txt](https://github.com/KonstantinosChaldaiopoulos/VideoViolenceDetection/blob/main/requirements.txt) and run ```pip install -r requirements.txt```, to automatically install all the packages needed to reproduce my project on your own machine.

**```>```** For this project, we used a custom movie-dataset. You can download the dataset directly from this [link](https://drive.google.com/drive/folders/1K6P0tfItrAPVkCyPw-8al3Br2le1B7JH?usp=sharing).

## Authors
Konstantinos Chaldaiopoulos & Natalia Koliou
