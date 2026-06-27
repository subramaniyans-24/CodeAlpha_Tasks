````markdown
# Handwritten Character Recognition

## CodeAlpha Machine Learning Internship - Task 3

### Project Overview

This project implements a Handwritten Character Recognition system using a Deep Learning Neural Network built with TensorFlow and Keras. The model is trained on the MNIST Handwritten Digits dataset to recognize handwritten digits from 0 to 9 with high accuracy.

---

## Features

- Loads the MNIST handwritten digits dataset
- Preprocesses and normalizes image data
- Builds a Deep Neural Network (DNN)
- Trains the model using TensorFlow/Keras
- Evaluates model performance
- Predicts handwritten digits
- Displays sample images and prediction results
- Saves the trained model for future use

---

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib

---

## Dataset

The project uses the MNIST Handwritten Digits Dataset provided by TensorFlow.

- Training Images: 60,000
- Testing Images: 10,000
- Image Size: 28 × 28 pixels
- Number of Classes: 10 (Digits 0–9)

The dataset is automatically downloaded using:

```python
from tensorflow.keras.datasets import mnist
````

No manual dataset download is required.

---

## Project Structure

```
Handwritten_Character_Recognition/
│
├── dataset/
│   └── README.txt
│
├── model/
│   └── handwritten_character_model.h5
│
├── notebook/
│   └── Handwritten_Character_Recognition.ipynb
│
├── screenshots/
│   ├── dataset_loaded.png
│   ├── sample_digit.png
│   ├── training_output.png
│   ├── accuracy.png
│   ├── accuracy_graph.png
│   └── prediction_result.png
│
├── handwritten_character_recognition.py
├── requirements.txt
└── README.md
```

---

## Installation

Install the required libraries:

```bash
pip install -r requirements.txt
```

---

## How to Run

Run the Python file:

```bash
python handwritten_character_recognition.py
```

The program will:

* Load the dataset
* Display a sample handwritten digit
* Train the neural network
* Evaluate the model
* Predict handwritten digits
* Save the trained model

---

## Model Architecture

* Flatten Layer
* Dense Layer (128 neurons, ReLU)
* Dropout Layer (0.2)
* Output Layer (10 neurons, Softmax)

Optimizer:

* Adam

Loss Function:

* Sparse Categorical Crossentropy

Metric:

* Accuracy

---

## Results

* Test Accuracy: Approximately 97% to 99%
* Successfully recognizes handwritten digits from 0 to 9.

---

## Future Enhancements

* Recognize handwritten alphabets
* Real-time digit recognition using webcam
* Deploy as a web application using Flask or Streamlit
* Improve accuracy using Convolutional Neural Networks (CNN)

---

## Author

**Subramaniyan**

Machine Learning Intern

CodeAlpha Internship

```
```

