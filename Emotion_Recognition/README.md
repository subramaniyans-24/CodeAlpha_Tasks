# Emotion Recognition System

## Project Overview

This project was developed as part of the CodeAlpha Machine Learning Internship.

The objective of this project is to recognize human emotions from facial images using Deep Learning techniques. A Convolutional Neural Network (CNN) is trained on facial expression images to classify emotions.

## Dataset

FER2013 Dataset

Emotion Classes:

* Angry
* Disgust
* Fear
* Happy
* Neutral
* Sad
* Surprise

## Technologies Used

* Python
* TensorFlow
* Keras
* NumPy
* Matplotlib
* OpenCV
* Scikit-Learn

## Model Architecture

The model uses a Convolutional Neural Network (CNN) consisting of:

* Convolution Layers
* Max Pooling Layers
* Flatten Layer
* Dense Layers
* Dropout Layer
* Softmax Output Layer

## Project Structure

Emotion_Recognition/

├── dataset/

├── notebook/

├── model/

├── screenshots/

├── emotion_recognition.py

├── requirements.txt

└── README.md

## Workflow

1. Dataset Collection
2. Data Preprocessing
3. Image Augmentation
4. CNN Model Creation
5. Model Training
6. Model Evaluation
7. Accuracy Visualization
8. Model Saving

## Evaluation

The model is evaluated using:

* Training Accuracy
* Validation Accuracy
* Loss Function

## How to Run

Install dependencies:

pip install -r requirements.txt

Run the project:

python emotion_recognition.py

## Results

The model successfully classifies facial expressions into seven emotion categories using a Convolutional Neural Network.

## Author

Subramaniyan

CodeAlpha Machine Learning Internship – Task 2
