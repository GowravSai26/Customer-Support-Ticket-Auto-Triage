from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import re

# Load model and encoder
model = joblib.load("ticket_classifier_model.joblib")
encoder = joblib.load("label_encoder.joblib")

app = FastAPI(title="Customer Support Ticket Auto-Triage API")

class Ticket(BaseModel):
    subject: str
    description: str

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return text

@app.post("/predict")
def predict(ticket: Ticket):
    text = preprocess_text(ticket.subject + " " + ticket.description)
    pred = model.predict([text])[0]
    category = encoder.inverse_transform([pred])[0]
    return {"predicted_category": category}
