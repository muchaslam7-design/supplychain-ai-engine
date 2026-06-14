from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import os
import requests

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "/tmp/best_delivery_model.pkl"
GOOGLE_DRIVE_FILE_ID = "14sMS0ltOZur0uIPYwBlrh_eAqO1NbweL"

def download_model_from_drive(file_id, destination):
    print("⏳ Downloading model...")
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
            break
    if token:
        response = session.get(URL, params={'id': file_id, 'confirm': token}, stream=True)
    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size=32768):
            if chunk in chunk:
                f.write(chunk)
    print("📌 Download complete!")

model = None

def get_model():
    global model
    if model is None:
        if not os.path.exists(MODEL_PATH):
            download_model_from_drive(GOOGLE_DRIVE_FILE_ID, MODEL_PATH)
        model = joblib.load(MODEL_PATH)
    return model

class ShipmentData(BaseModel):
    payment_method: str
    actual_days: float
    scheduled_days: float
    benefit_per_order: float
    sales_per_customer: float

@app.post("/predict")
@app.post("/predict/")
def predict_risk(data: ShipmentData):
    try:
        clf = get_model()
        
        # Mapping logic (Ensure this matches your training data order)
        payment_encoded = 0.0
        channel_upper = data.payment_method.upper()
        if "DEBIT" in channel_upper:
            payment_encoded = 1.0
        elif "TRANSFER" in channel_upper:
            payment_encoded = 2.0
        elif "CASH" in channel_upper:
            payment_encoded = 3.0

        # IMPORTANT: Check your training code for the exact order of these columns
        # Features array creation
        features = np.array([[
            payment_encoded,
            data.actual_days,
            data.scheduled_days,
            data.benefit_per_order,
            data.sales_per_customer
        ]], dtype=np.float64)

        raw_prediction = clf.predict(features)[0]

        # Mapping output based on standard classification logic
        if raw_prediction == 1:
            status = "LATE DELIVERY"
        elif raw_prediction == 2:
            status = "ADVANCE DISPATCH"
        else:
            status = "ON TIME PERFECT"

        return {
            "prediction": status,
            "raw_output": int(raw_prediction)
        }

    except Exception as e:
        return {"prediction": "ERROR", "error": str(e)}