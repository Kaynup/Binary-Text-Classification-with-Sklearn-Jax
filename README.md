# Sentiment Classification with Scikit-learn and JAX/Flax/Optax

## Overview
This repository explores binary sentiment classification using two complementary approaches:

- **Scikit-learn baseline**: Logistic Regression with TF-IDF features.  
- **JAX/Flax/Optax models**: Neural architectures starting from an embedding + pooling classifier, with plans to extend to LSTMs.

The aim is to combine fast prototyping with sklearn and flexible deep learning with JAX, while providing deployable inference endpoints via FastAPI.

---

## Dataset
- **Source**: [Juggernaut Sentiment Analysis by Adeoluwa Adeboye](https://www.kaggle.com/datasets/adeoluwa/juggernaut-sentiment-analysis).  
- **Note**: This dataset is not owned by me. All credit goes to the original creator.  

The dataset was split into multiple configurations (different vocab sizes, n-gram ranges, and train/test splits) for experimentation.

### Dataset Limitations
While the dataset provides a good foundation for binary sentiment classification, it has several limitations that affect model performance:

- **Sarcasm and Irony**: The dataset does not contain strong coverage of sarcastic or ironic expressions. Models trained on it may fail to capture sentiment in such cases (e.g., "Yeah, this was *really* helpful" used sarcastically).  
- **Contextual Nuance**: Sentiment labels are often assigned at the sentence level, without accounting for larger context (paragraphs, conversations). Models may misinterpret statements that depend on prior text.  
- **Domain Generalization**: The dataset is focused on general sentiment and may not transfer well to specific domains (e.g., product reviews vs. social media vs. news articles).  
- **Class Balance**: Depending on preprocessing choices, there may be class imbalance which can bias the model toward the majority label.  
- **Vocabulary Coverage**: Fixed vocabulary sizes (e.g., 80k in sklearn baseline) can lead to rare or emerging terms being ignored or mapped to `<UNK>`.

These limitations should be considered when evaluating model performance and deploying in real-world scenarios.

---

## Current Progress
- Baseline sklearn model achieved ~82 F1 score with Logistic Regression + TF-IDF (1–5 grams, 80k vocabulary).  
- JAX prototype (embedding + pooling) has caught up to sklearn baseline performance.  
- All JAX experiments were run on **CPU only**, since local JAX CUDA installation defaults to the latest CUDA version which is not supported by the RTX 4060 GPU used.  
- Ongoing experiments with hidden layer depth/width.  
- Next steps include LSTM models in Flax for improved sequence modeling.

---

## Deployment
Both sklearn and JAX models can be deployed locally with FastAPI.

- `sklearn-app.py`: Serves the Logistic Regression model.  
- `jax-app.py`: Serves the Flax neural network (embedding + pooling).  
- All inference activity is logged to `local-deploy/inference.log`.

### Run locally
```bash
pip install -r requirements.txt
```

Start either server:
```bash
uvicorn local-deploy.sklearn-app:app --reload
uvicorn local-deploy.jax-app:app --reload
```

### Example request
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "I absolutely loved this movie!"}'
```

### Example response (sklearn)
```json
{
  "input": "I absolutely loved this movie!",
  "prediction": 1,
  "inference_time_ms": 3.42
}
```

### Example response (jax)
```json
{
  "input": "I absolutely loved this movie!",
  "prediction": 1,
  "inference_time_ms": 5.12,
  "Jax model": true
}
```

---

## Deployment Logs Summary
Logs from local testing show the following:

- **Model Initialization**  
  - Multiple successful model loads across sessions.  
  - Occasional initialization errors (e.g., undefined variables, misloaded artifacts).  
  - Hardware fallback messages observed: TPU unavailable, CUDA-enabled JAX not installed (defaulted to CPU).

- **Inference Behavior**  
  - Predictions worked consistently on both positive and negative inputs.  
  - Example positive predictions:  
    - "This product is amazing!" → 1  
    - "hi" → 1  
    - "This is fabulous" → 1  
  - Example negative predictions:  
    - "The boy looked so sad." → 0  
    - "I had a very bad day." → 0  
    - "You are the worst" → 0  
    - "I like this place very much" → 0  

- **Performance**  
  - Typical inference times: **1–3 ms** for short texts on CPU.  
  - Some outliers (e.g., ~800 ms) likely due to cold start or CPU fallback.  
  - After warm-up, predictions stabilized in the 1–20 ms range.

- **Errors Encountered**  
  - `'dict' object has no attribute 'predict'` → indicates incorrect object passed into inference pipeline.  
  - `name 'vocab_size' is not defined` → issue with model restoration code.  
  - Backend initialization warnings regarding TPU and GPU support.  

Overall, inference pipeline is functional and performant, with occasional initialization issues that need handling.

---

## License
MIT License.
