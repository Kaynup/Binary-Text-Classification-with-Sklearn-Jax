import joblib
import time
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from flax import linen as nn, serialization
from flax.training import train_state
import optax
import re
import jax
import jax.numpy as jnp

# Middleware
from fastapi.middleware.cors import CORSMiddleware

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

def restore_state(msgpack_path, model, vocab_size, max_len=34):
    rng = jax.random.PRNGKey(0)
    dummy = jnp.ones((1, max_len), dtype=jnp.int32)
    params = model.init(rng, dummy)["params"]

    tx = optax.adam(1e-3)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    with open(msgpack_path, "rb") as f:
        state = serialization.from_bytes(state, f.read())

    return state

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

    def save(self, path):
        joblib.dump({
            "params": self.params,
            "word2idx": self.word2idx,
            "max_len": self.max_len,
            "threshold": self.threshold,
        }, path)

    @classmethod
    def load(cls, path, model):
        data = joblib.load(path)
        return cls(model, data["params"], data["word2idx"],
                   data["max_len"], data["threshold"])

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(
    SCRIPT_DIR,
    "..",
    "model",
    "pooling_batch-300_pipeline.joblib"   # Note:- only this model is supported for now
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
    model = Classifier(vocab_size=256995)
    model = InferencePipeline.load(model_path, model)
    logger.info(f"[LOG] JAX Model loaded successfully from {model_path}")
except Exception as e:
    logger.error(f"[LOG] Failed to load model: {e}")
    raise RuntimeError("Model could not be loaded.")

# Input Schema
class TextInput(BaseModel):
    text: str

# FastAPI Init
app = FastAPI(title="Text Classifier API (JAX)", version="1.0")

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
            "Jax model": True
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
        "message": "JAX Text Classifier API is running!",
        "model": "JAX Neural Network",
        "endpoints": {
            "predict": "/predict (POST)",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    import uvicorn
    logger.info("[INFO] Starting JAX FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)