#  Emotion Detection from Text

This project is an AI-powered text classification model that detects emotions such as joy, sadness, anger, etc., using machine learning.

## Features
- Cleans raw text using `neattext`
- TF-IDF vectorization
- Logistic Regression classifier
- Supports 6 emotions: sadness, joy, love, anger, fear, surprise

## Usage
1. Install dependencies:
pip install -r requirements.txt

2. Train the model:
python model/train.py

3. Run emotion prediction:
python model/predict.py

## Model Performance
Achieved **86.19% accuracy** on the test dataset.

## Author
Nikhil Sehgal
