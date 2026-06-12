import os
import joblib
from preprocessing import clean_email
model_directory="models"
model_path = os.path.join(model_directory, "model.joblib")
vectorizer_path = os.path.join(model_directory, "vectorizer.joblib")
def load_model_and_vectorizer():
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        raise FileNotFoundError("Model or vectorizer file not found. Please train the model first.")
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer
def analysis(text, model, vectorizer):
    text = clean_email(text)
    tfidf_vector = vectorizer.transform([text]) # tf idf matrix 
    model_prediction = model.predict(tfidf_vector)[0]
    probability = model.predict_proba(tfidf_vector)[0]
    confidence = max(probability)
    return model_prediction, confidence