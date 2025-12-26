# API Reference

## Base URL

- **Local**: `http://localhost:8000`
- **Production**: Your Vercel deployment URL

---

## Endpoints

### POST `/api/predict`

Analyzes text sentiment and returns prediction.

**Request Body:**
```json
{
  "text": "Your text to analyze"
}
```

**Response:**
```json
{
  "input": "Your text to analyze",
  "prediction": 1,
  "inference_time_ms": 1.23,
  "Jax model": false
}
```

| Field | Type | Description |
|-------|------|-------------|
| `input` | string | The original input text |
| `prediction` | int | `1` = Positive, `0` = Negative |
| `inference_time_ms` | float | Model inference time in milliseconds |
| `Jax model` | bool | Whether JAX model was used (currently always `false`) |

**Status Codes:**
- `200` - Success
- `500` - Model inference failed

---

### GET `/api/predict`

Health check endpoint.

**Response:**
```json
{
  "message": "Sklearn Text Classifier API is running!",
  "model": "Logistic Regression",
  "status": "healthy"
}
```

---

## CORS

All origins are allowed (`*`). This can be restricted for production.

---

## Rate Limits

Vercel Hobby plan limits:
- 100 GB bandwidth/month
- 100,000 serverless function invocations/month
- 10 second max execution time

---

## Example cURL Commands

```bash
# Test positive sentiment
curl -X POST "https://your-app.vercel.app/api/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this!"}'

# Health check
curl "https://your-app.vercel.app/api/predict"
```
