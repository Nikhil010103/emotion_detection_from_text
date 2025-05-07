import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os


def clean_text(text):
    import re
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@w+|\#','', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    return text

# Load Data
df = pd.read_csv("data/emotion_dataset.csv")  

# Preprocessing
df['text'] = df['text'].apply(clean_text)
X = df['text']
y = df['label']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Vectorization
tfidf = TfidfVectorizer(ngram_range=(1,2), max_df=0.95, min_df=3)
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)

# Save Vectorizer
joblib.dump(tfidf, 'model/vectorizer.joblib')

# Classifier with class_weight balancing
clf = LogisticRegression(class_weight='balanced', max_iter=300)
clf.fit(X_train_vec, y_train)

# Save Model
joblib.dump(clf, 'model/emotion_model.joblib')

# Evaluation
y_pred = clf.predict(X_test_vec)

acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix Plot
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=clf.classes_, yticklabels=clf.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig("model/confusion_matrix.png")
plt.close()
