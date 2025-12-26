import joblib
import time
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Model path - relative to the deployment root (one level up from api/)
model_path = os.path.join(
    SCRIPT_DIR,
    "..",
    "model",
    "logreg-80k.joblib"
)

# Model load (lazy loading pattern for serverless)
model = None

def get_model():
    global model
    if model is None:
        try:
            model = joblib.load(model_path)
            print(f"[LOG] Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"[LOG] Failed to load model: {e}")
            raise RuntimeError("Model could not be loaded.")
    return model

# Input Schema
class TextInput(BaseModel):
    text: str

# FastAPI Init
app = FastAPI(title="Text Classifier API (Sklearn)", version="1.0")

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
@app.post("/api/predict")
def predict(input_data: TextInput):
    try:
        start_time = time.time()
        
        # Get model (lazy load)
        current_model = get_model()
        
        # Run prediction
        prediction = current_model.predict([input_data.text])[0]
        
        inference_time = round((time.time() - start_time) * 1000, 2)  # ms
        
        sentiment = "POSITIVE" if prediction == 1 else "NEGATIVE"
        print(
            f"Text: '{input_data.text[:50]}...' | "
            f"Sentiment: {sentiment} | "
            f"Time: {inference_time} ms"
        )
        
        return {
            "input": input_data.text,
            "prediction": int(prediction),
            "inference_time_ms": inference_time,
            "Jax model": False
        }

    except Exception as e:
        print(f"[LOG] Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Model inference failed.")

@app.get("/api/predict")
def health_check():
    return {
        "message": "Sklearn Text Classifier API is running!",
        "model": "Logistic Regression",
        "status": "healthy"
    }
