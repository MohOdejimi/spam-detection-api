import json
from pathlib import Path

import joblib
import nltk
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from app.services import preprocess_message

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)  # required by newer NLTK versions
nltk.download("stopwords", quiet=True)

DATA_PATH = Path("spam.csv")
MODEL_DIR = Path("model")
MODEL_DIR.mkdir(exist_ok=True)

STOPWORDS = set(stopwords.words("english"))


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, encoding="latin-1")
    df = df.drop(columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], errors="ignore")
    df = df.rename(columns={"v1": "label", "v2": "msg"})
    df["msg"] = df["msg"].apply(preprocess_message)
    return df


def train():
    df = load_data()

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["msg"], df["label"], test_size=0.25, random_state=26
    )

    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    clf = MultinomialNB()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(classification_report(y_test, y_pred))

    joblib.dump(vectorizer, MODEL_DIR / "vectorizer.pkl")
    joblib.dump(clf, MODEL_DIR / "spam_classifier.pkl")

    metadata = {
        "algorithm": "MultinomialNB",
        "vectorizer": "CountVectorizer",
        "vocabulary_size": len(vectorizer.vocabulary_),
        "accuracy": accuracy,
        "classification_report": report,
        "train_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
    }
    with open(MODEL_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved model artifacts to {MODEL_DIR.resolve()}")


if __name__ == "__main__":
    train()
