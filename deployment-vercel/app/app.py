import joblib
import time
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from datetime import datetime

# Middleware
from fastapi.middleware.cors import CORSMiddleware

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(
    SCRIPT_DIR,
    "..",
    "model",
    "logreg-80k.joblib"
)

# Logging Setup - save logs in the backend directory
log_file_path = os.path.join(SCRIPT_DIR, "inference.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Model load
try:
    model = joblib.load(model_path)
    logger.info(f"[LOG] Model loaded successfully from {model_path}")
except Exception as e:
    logger.error(f"[LOG] Failed to load model: {e}")
    raise RuntimeError("Model could not be loaded.")

# Input Schema
class TextInput(BaseModel):
    text: str

# FastAPI Init
app = FastAPI(title="Text Classifier API (Sklearn)", version="1.0")

# Routes
@app.post("/predict")
def predict(input_data: TextInput):
    try:
        start_time = time.time()
        
        # Run prediction
        prediction = model.predict([input_data.text])[0]
        
        inference_time = round((time.time() - start_time) * 1000, 2)  # ms
        
        sentiment = "POSITIVE" if prediction == 1 else "NEGATIVE"
        logger.info(
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
        logger.error(f"[LOG] Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Model inference failed.")

# UI inference
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {
        "message": "Sklearn Text Classifier API is running!",
        "model": "Logistic Regression",
        "endpoints": {
            "predict": "/predict (POST)",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    import uvicorn
    logger.info("[INFO] Starting Sklearn FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)