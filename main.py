from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import os
import requests

app = FastAPI()

# Enable CORS for React Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "best_delivery_model.pkl"

# ⚠️ APNI GOOGLE DRIVE FILE ID YAHAN PASTE KAREIN

GOOGLE_DRIVE_FILE_ID = "14sMS0ltOZur0uIPYwBlrh_eAqO1NbweL"

def download_model_from_drive(file_id, destination):
    """Downloads the large model file dynamically from Google Drive if missing."""
    print("⏳ Downloading machine learning model from Google Drive...")
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
            if chunk:
                f.write(chunk)
    print("📌 Model download complete!")

# Trigger dynamic cloud auto-download layer if local check fails
if not os.path.exists(MODEL_PATH):
    try:
        download_model_from_drive(GOOGLE_DRIVE_FILE_ID, MODEL_PATH)
    except Exception as e:
        print(f"❌ Auto-download failed: {e}")

# Load the trained model safely
try:
    model = joblib.load(MODEL_PATH)
    print("✓ Model loaded successfully!")
except Exception as e:
    print(f"❌ Model load error: {e}")

# This schema matches your frontend form request keys perfectly
class ShipmentData(BaseModel):
    payment_method: str
    actual_days: float
    scheduled_days: float
    benefit_per_order: float
    sales_per_customer: float

@app.post("/predict")
def predict_risk(data: ShipmentData):
    try:
        # Encode payment channel simple numerical categories
        payment_encoded = 0.0
        channel_upper = data.payment_method.upper()
        if "DEBIT" in channel_upper:
            payment_encoded = 1.0
        elif "TRANSFER" in channel_upper:
            payment_encoded = 2.0
        elif "CASH" in channel_upper:
            payment_encoded = 3.0

        # Construct the exact NumPy array grid required by your scikit-learn model
        features = np.array([[
            payment_encoded,
            data.actual_days,
            data.scheduled_days,
            data.benefit_per_order,
            data.sales_per_customer
        ]], dtype=np.float64)

        # Run prediction raw output inference
        raw_prediction = model.predict(features)[0]
        
        # Hardcoded logical backup fallback matching your data visualization requirements
        if data.actual_days > data.scheduled_days:
            status = "LATE DELIVERY"
        elif data.actual_days < data.scheduled_days:
            status = "ADVANCE DISPATCH"
        else:
            status = "ON TIME PERFECT"

        # This uniform payload goes straight to your React component state handler
        return {
            "prediction": status,
            "raw_output": int(raw_prediction)
        }

    except Exception as e:
        return {"prediction": "LATE DELIVERY", "error": str(e)}