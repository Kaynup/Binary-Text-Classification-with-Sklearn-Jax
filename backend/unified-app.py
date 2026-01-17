import joblib
import time
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import re
import jax
import jax.numpy as jnp
from flax import linen as nn, serialization
from flax.training import train_state
import optax

# Middleware
from fastapi.middleware.cors import CORSMiddleware

# =============================================================================
# TOKENIZATION & ENCODING (shared by JAX models)
# =============================================================================

def custom_tokenizer(text):
    text = text.lower()
    text = re.sub(r"http\S+", "<URL>", text)
    text = re.sub(r"@\w+", "<USER>", text)
    tokens = re.findall(
        r"[#@]?\w+|[:;=xX8][-^']?[)DPOp3]+|[^\w\s]", 
        text, flags=re.UNICODE
    )
    return tokens

PAD, UNK = "<PAD>", "<UNK>"

def encode_tokens(tokens, word2idx, max_len=34):
    ids = [word2idx.get(tok, word2idx[UNK]) for tok in tokens]
    if len(ids) >= max_len:
        return ids[:max_len]
    return ids + [word2idx[PAD]] * (max_len - len(ids))

# =============================================================================
# JAX MODEL DEFINITIONS
# =============================================================================

class TrainState(train_state.TrainState):
    pass

class Classifier(nn.Module):
    """Model: jax-pooling-300 (hidden_dim=128)"""
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

class ClassifierHidden256(nn.Module):
    """Model: jax-pooling-200-h256"""
    vocab_size: int
    embed_dim: int = 128
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, x):
        emb = nn.Embed(self.vocab_size, self.embed_dim)(x)
        pooled = emb.mean(axis=1)
        h = nn.Dense(self.hidden_dim)(pooled)
        h = nn.relu(h)
        out = nn.Dense(1)(h)
        return out.squeeze(-1)

class ClassifierTwoHidden(nn.Module):
    """Model: jax-pooling-200-2h (256->128)"""
    vocab_size: int
    embed_dim: int = 128
    hidden_dim_1: int = 256
    hidden_dim_2: int = 128

    @nn.compact
    def __call__(self, x):
        emb = nn.Embed(self.vocab_size, self.embed_dim)(x)
        pooled = emb.mean(axis=1)
        h = nn.Dense(self.hidden_dim_1)(pooled)
        h = nn.relu(h)
        h = nn.Dense(self.hidden_dim_2)(pooled)
        h = nn.relu(h)
        out = nn.Dense(1)(h)
        return out.squeeze(-1)

# =============================================================================
# JAX INFERENCE PIPELINE
# =============================================================================

class JaxInferencePipeline:
    def __init__(self, model, params, word2idx, max_len=34, threshold=0.5):
        self.model = model
        self.params = params
        self.word2idx = word2idx
        self.max_len = max_len
        self.threshold = threshold
        self.is_jax = True

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

# =============================================================================
# SKLEARN WRAPPER (for unified interface)
# =============================================================================

class SklearnWrapper:
    def __init__(self, model):
        self.model = model
        self.is_jax = False
    
    def predict(self, texts):
        return self.model.predict(texts)

# =============================================================================
# PATHS & LOGGING
# =============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "..", "models")

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

# =============================================================================
# LOAD ALL MODELS
# =============================================================================

MODEL_REGISTRY = {}
DEFAULT_MODEL = "sklearn-logreg"

# Sklearn model
sklearn_path = os.path.join(MODELS_DIR, "sklearn", "logreg-80k.joblib")
try:
    sklearn_model = joblib.load(sklearn_path)
    MODEL_REGISTRY["sklearn-logreg"] = SklearnWrapper(sklearn_model)
    logger.info(f"[LOG] Loaded sklearn model: sklearn-logreg")
except Exception as e:
    logger.error(f"[LOG] Failed to load sklearn model: {e}")

# JAX models
JAX_CONFIGS = [
    {
        "name": "jax-pooling-300",
        "file": "bag_of_words-with_pooling/pooling_batch-300_pipeline.joblib",
        "class": Classifier,
    },
    {
        "name": "jax-pooling-200-h256",
        "file": "bag_of_words-with_pooling/pooling_batch-200-hidden=256_pipeline.joblib",
        "class": ClassifierHidden256,
    },
    {
        "name": "jax-pooling-200-2h",
        "file": "bag_of_words-with_pooling/pooling_batch-200_2-hidden=256-128_pipeline.joblib",
        "class": ClassifierTwoHidden,
    },
]

for cfg in JAX_CONFIGS:
    try:
        model_path = os.path.join(MODELS_DIR, "jax", cfg["file"])
        model_instance = cfg["class"](vocab_size=256995)
        MODEL_REGISTRY[cfg["name"]] = JaxInferencePipeline.load(model_path, model_instance)
        logger.info(f"[LOG] Loaded JAX model: {cfg['name']}")
    except Exception as e:
        logger.error(f"[LOG] Failed to load JAX model {cfg['name']}: {e}")

logger.info(f"[LOG] Total models loaded: {len(MODEL_REGISTRY)}")

# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

class TextInput(BaseModel):
    text: str
    model: str = DEFAULT_MODEL

app = FastAPI(title="Unified Sentiment Classifier API", version="2.0")

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
        "message": "Unified Sentiment Classifier API is running!",
        "available_models": list(MODEL_REGISTRY.keys()),
        "default_model": DEFAULT_MODEL,
        "endpoints": {
            "predict": "/predict (POST)",
            "models": "/models (GET)",
            "docs": "/docs"
        }
    }

@app.get("/models")
def list_models():
    """List all available models"""
    return {
        "available_models": list(MODEL_REGISTRY.keys()),
        "default": DEFAULT_MODEL,
        "count": len(MODEL_REGISTRY)
    }

@app.post("/predict")
def predict(input_data: TextInput):
    # Validate model selection
    if input_data.model not in MODEL_REGISTRY:
        raise HTTPException(
            status_code=400, 
            detail=f"Unknown model: '{input_data.model}'. Available: {list(MODEL_REGISTRY.keys())}"
        )
    
    try:
        start_time = time.time()
        
        selected_model = MODEL_REGISTRY[input_data.model]
        prediction = selected_model.predict([input_data.text])[0]
        
        inference_time = round((time.time() - start_time) * 1000, 2)
        
        sentiment = "POSITIVE" if prediction == 1 else "NEGATIVE"
        is_jax = getattr(selected_model, 'is_jax', False)
        
        logger.info(
            f"Model: {input_data.model} | "
            f"Text: '{input_data.text[:50]}...' | "
            f"Sentiment: {sentiment} | "
            f"Time: {inference_time} ms"
        )
        
        return {
            "input": input_data.text,
            "prediction": int(prediction),
            "inference_time_ms": inference_time,
            "model_used": input_data.model,
            "Jax model": is_jax
        }

    except Exception as e:
        logger.error(f"[LOG] Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Model inference failed.")

if __name__ == "__main__":
    import uvicorn
    logger.info("[INFO] Starting Unified FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
