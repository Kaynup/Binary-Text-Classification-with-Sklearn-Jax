"""
Unified Sentiment Analysis API
Supports both Sklearn and JAX models for production deployment
"""
import joblib
import time
import logging
import os
import re
from typing import Optional
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# ============================================================================
# Configuration
# ============================================================================

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Model paths
SKLEARN_MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "sklearn", "logreg-80k.joblib")
JAX_MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "jax", "bag_of_words-with_pooling", "pooling_batch-300_pipeline.joblib")

# Environment configuration
PORT = int(os.environ.get("PORT", 8000))
ENABLE_JAX = os.environ.get("ENABLE_JAX", "true").lower() == "true"

# ============================================================================
# Logging Setup
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ============================================================================
# JAX Model Components (conditionally loaded)
# ============================================================================

jax_model = None
jax_available = False

if ENABLE_JAX:
    try:
        from flax import linen as nn, serialization
        from flax.training import train_state
        import optax
        import jax
        import jax.numpy as jnp

        PAD, UNK = "<PAD>", "<UNK>"

        def custom_tokenizer(text):
            text = text.lower()
            text = re.sub(r"http\S+", "<URL>", text)
            text = re.sub(r"@\w+", "<USER>", text)
            tokens = re.findall(
                r"[#@]?\w+|[:;=xX8][-^']?[)DPOp3]+|[^\w\s]", 
                text, flags=re.UNICODE
            )
            return tokens

        def encode_tokens(tokens, word2idx, max_len=34):
            ids = [word2idx.get(tok, word2idx[UNK]) for tok in tokens]
            if len(ids) >= max_len:
                return ids[:max_len]
            return ids + [word2idx[PAD]] * (max_len - len(ids))

        class TrainState(train_state.TrainState):
            pass

        class Classifier(nn.Module):
            vocab_size: int
            embed_dim: int = 128
            hidden_dim: int = 128

            @nn.compact
            def __call__(self, x):
                emb = nn.Embed(self.vocab_size, self.embed_dim)(x)
                pooled = emb.mean(axis=1)
                h = nn.Dense(self.hidden_dim)(pooled)
                h = nn.relu(h)
                out = nn.Dense(1)(h)
                return out.squeeze(-1)

        class InferencePipeline:
            def __init__(self, model, params, word2idx, max_len=34, threshold=0.5):
                self.model = model
                self.params = params
                self.word2idx = word2idx
                self.max_len = max_len
                self.threshold = threshold

            def preprocess(self, texts):
                tok = [custom_tokenizer(t) for t in texts]
                ids = [encode_tokens(t, self.word2idx, self.max_len) for t in tok]
                return jnp.array(ids, dtype=jnp.int32)

            def predict_proba(self, texts):
                X = self.preprocess(texts)
                logits = self.model.apply({"params": self.params}, X)
                return jax.nn.sigmoid(logits)

            def predict(self, texts):
                probs = self.predict_proba(texts)
                return (probs >= self.threshold).astype(int)

            @classmethod
            def load(cls, path, model):
                data = joblib.load(path)
                return cls(model, data["params"], data["word2idx"],
                           data["max_len"], data["threshold"])

        jax_available = True
        logger.info("[JAX] JAX/Flax libraries loaded successfully")
    except ImportError as e:
        logger.warning(f"[JAX] JAX/Flax not available: {e}. JAX model will be disabled.")
        jax_available = False

# ============================================================================
# Model Loading
# ============================================================================

sklearn_model = None
jax_model = None

# Load Sklearn model
try:
    sklearn_model = joblib.load(SKLEARN_MODEL_PATH)
    logger.info(f"[SKLEARN] Model loaded successfully from {SKLEARN_MODEL_PATH}")
except Exception as e:
    logger.error(f"[SKLEARN] Failed to load model: {e}")

# Load JAX model (if available)
if jax_available and ENABLE_JAX:
    try:
        base_model = Classifier(vocab_size=256995)
        jax_model = InferencePipeline.load(JAX_MODEL_PATH, base_model)
        logger.info(f"[JAX] Model loaded successfully from {JAX_MODEL_PATH}")
    except Exception as e:
        logger.warning(f"[JAX] Failed to load model: {e}")
        jax_model = None

# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="Sentiment Analysis API",
    description="Unified API for sentiment analysis using Sklearn and JAX models",
    version="2.0"
)

# CORS Configuration - Allow all origins for deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Request/Response Models
# ============================================================================

class TextInput(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    input: str
    prediction: int
    sentiment: str
    inference_time_ms: float
    model_used: str
    jax_model: bool = False  # Backwards compatibility

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
def root():
    """Health check and API info"""
    return {
        "message": "Sentiment Analysis API is running!",
        "version": "2.0",
        "models": {
            "sklearn": sklearn_model is not None,
            "jax": jax_model is not None
        },
        "endpoints": {
            "predict": "/predict (POST) - Analyze sentiment",
            "health": "/health (GET) - Health check",
            "docs": "/docs - API documentation"
        }
    }

@app.get("/health")
def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "models": {
            "sklearn": {
                "available": sklearn_model is not None,
                "path": SKLEARN_MODEL_PATH
            },
            "jax": {
                "available": jax_model is not None,
                "enabled": ENABLE_JAX,
                "path": JAX_MODEL_PATH if ENABLE_JAX else None
            }
        }
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(
    input_data: TextInput,
    model: Optional[str] = Query(default="auto", description="Model to use: 'sklearn', 'jax', or 'auto'")
):
    """
    Analyze sentiment of the input text.
    
    - **text**: The text to analyze
    - **model**: Which model to use ('sklearn', 'jax', or 'auto')
    """
    try:
        start_time = time.time()
        
        # Determine which model to use
        use_jax = False
        if model == "jax":
            if jax_model is None:
                raise HTTPException(status_code=400, detail="JAX model is not available")
            use_jax = True
        elif model == "sklearn":
            if sklearn_model is None:
                raise HTTPException(status_code=400, detail="Sklearn model is not available")
            use_jax = False
        else:  # auto - prefer sklearn for faster inference
            use_jax = False if sklearn_model is not None else (jax_model is not None)
        
        # Run prediction
        if use_jax:
            prediction = int(jax_model.predict([input_data.text])[0])
            model_name = "JAX Neural Network"
        else:
            if sklearn_model is None:
                raise HTTPException(status_code=500, detail="No models available")
            prediction = int(sklearn_model.predict([input_data.text])[0])
            model_name = "Sklearn Logistic Regression"
        
        inference_time = round((time.time() - start_time) * 1000, 2)
        sentiment = "POSITIVE" if prediction == 1 else "NEGATIVE"
        
        logger.info(
            f"Text: '{input_data.text[:50]}...' | "
            f"Sentiment: {sentiment} | "
            f"Model: {model_name} | "
            f"Time: {inference_time} ms"
        )
        
        return PredictionResponse(
            input=input_data.text,
            prediction=prediction,
            sentiment=sentiment,
            inference_time_ms=inference_time,
            model_used=model_name,
            jax_model=use_jax  # Backwards compatibility with frontend
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[ERROR] Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Model inference failed.")

# Alias for backwards compatibility
@app.post("/predict/sklearn")
def predict_sklearn(input_data: TextInput):
    """Sklearn-specific endpoint (backwards compatibility)"""
    return predict(input_data, model="sklearn")

@app.post("/predict/jax")
def predict_jax(input_data: TextInput):
    """JAX-specific endpoint (backwards compatibility)"""
    return predict(input_data, model="jax")

# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    logger.info(f"[INFO] Starting Unified FastAPI server on port {PORT}...")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
