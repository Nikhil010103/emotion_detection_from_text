import joblib
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.preprocess import clean_text


def predict_emotion(text):
    model = joblib.load('emotion_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    cleaned_text = clean_text(text)
    vec = vectorizer.transform([cleaned_text])
    prediction = model.predict(vec)
    
    emotions = {
        0: "sadness",
        1: "joy",
        2: "love",
        3: "anger",
        4: "fear",
        5: "surprise"
    }
    return emotions[int(prediction[0])]

if __name__ == "__main__":
    text = input("Enter a sentence: ")
    print("Predicted Emotion:", predict_emotion(text))
