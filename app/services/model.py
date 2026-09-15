import json
from pathlib import Path

import joblib
import nltk

from app.services.preprocessing import preprocess_message

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

model_dir = Path(__file__).resolve().parents[1]
clf_path = model_dir / "model" / "spam_classifier.pkl"
vectorizer_path = model_dir / "model" / "vectorizer.pkl"
metadata_path = model_dir / "model" / "metadata.json"

with open(clf_path, "rb") as f:
    model = joblib.load(f)

with open(vectorizer_path, "rb") as f:
    vectorizer = joblib.load(f)

with open(metadata_path, "r") as f:
    metadata = json.load(f)


def predict_message(message):
    processed_message = preprocess_message(message)
    message_vector = vectorizer.transform([processed_message])
    prediction = model.predict(message_vector)[0]
    confidence = model.predict_proba(message_vector).max()
    return prediction, float(confidence)
