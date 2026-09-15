from app.services.model import model, vectorizer
from app.services.preprocessing import preprocess_message


def predict_message(message):
    processed_message = preprocess_message(message)
    message_vector = vectorizer.transform([processed_message])
    prediction = model.predict(message_vector)[0]
    confidence = model.predict_proba(message_vector).max()
    return prediction, float(confidence)
