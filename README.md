# Real-Time Sign Language Recognition

A real-time sign language recognition application that uses a webcam to detect hand gestures and convert them into corresponding text.

## Project Overview

This project recognizes Indian Sign Language alphabet and digit gestures using hand landmarks extracted from webcam images.

The system processes the hand using MediaPipe, extracts 126 landmark features, and uses a trained neural network to classify the gesture.

## Features

- Real-time webcam-based hand gesture recognition
- Recognition of 35 classes
- Digits 1–9
- Alphabets A–Z
- Hand landmark extraction
- Normalized landmark-based classification
- Real-time text output
- Streamlit web interface
- Camera permission support

## Classes

The model recognizes:

**Digits:**

1, 2, 3, 4, 5, 6, 7, 8, 9

**Alphabets:**

A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z

## System Workflow

```text
Webcam
   ↓
Hand Detection
   ↓
MediaPipe Hand Landmarks
   ↓
126 Landmark Features
   ↓
Landmark Normalization
   ↓
Neural Network Model
   ↓
Gesture Prediction
   ↓
Text Output


Model

The trained model uses 126 hand landmark features.

Model architecture:

Input: 126 features
        ↓
Dense Layer: 256 neurons
        ↓
Dropout
        ↓
Dense Layer: 128 neurons
        ↓
Dropout
        ↓
Dense Layer: 64 neurons
        ↓
Dense Output Layer: 35 classes

The normalized landmark model achieved approximately 99.88% test accuracy on the prepared dataset.

Dataset

The project uses a dataset containing 35 gesture classes:

9 digit classes
26 alphabet classes
1200 images per class
42,000 images in the original dataset

During landmark extraction, some images could not be processed successfully. The resulting landmark dataset contains 41,385 successfully processed images.

Project Structure
sign-language-recognition/
│
├── backend/
│   ├── create_landmark_dataset.py
│   ├── normalized_landmark_webcam.py
│   ├── sign_language_text.py
│   └── train_normalized_landmark_model.py
│
├── frontend/
│   └── app.py
│
├── model/
│   ├── normalized_class_names.npy
│   └── normalized_landmark_sign_language_model.keras
│
├── models/
│   └── hand_landmarker.task
│
├── .gitignore
├── requirements.txt
└── README.md

Installation

Clone the repository:

git clone https://github.com/ssakshithaas123-ctrl/sign-language-recognition.git

Open the project:

cd sign-language-recognition

Create and activate a Python environment.

Install the required packages:

pip install -r requirements.txt
Run the Application

Start the Streamlit application:

streamlit run frontend/app.py

The application will open in your browser.

Allow camera access when requested.

Technologies
Python
TensorFlow
MediaPipe
OpenCV
Streamlit
Streamlit-WebRTC
NumPy
Scikit-learn
Future Enhancement
Text-to-speech output
Word and sentence formation
Improved recognition under different lighting conditions
Support for additional sign language gestures

