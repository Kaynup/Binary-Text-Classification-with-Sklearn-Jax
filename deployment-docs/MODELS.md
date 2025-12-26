# Model History

## Current Model

| Property | Value |
|----------|-------|
| **File** | `logreg-80k.joblib` |
| **Type** | Logistic Regression (Sklearn) |
| **Size** | ~3.8 MB |
| **Training Samples** | 80,000 |
| **Output** | Binary (0 = Negative, 1 = Positive) |

---

## Version History

### v1.0 - 2025-12-26

**Model**: `logreg-80k.joblib`

- **Algorithm**: Logistic Regression
- **Framework**: Scikit-learn
- **Vectorizer**: TF-IDF (included in pipeline)
- **Training Data**: 80,000 text samples
- **Task**: Binary sentiment classification

**Performance:**
- Cold start: ~1.4 seconds
- Inference: ~1 ms

---

## Updating the Model

To update the model:

1. Train new model and export as `.joblib`
2. Replace `deployment/model/logreg-80k.joblib`
3. Test locally:
   ```bash
   cd deployment
   uvicorn api.predict:app --reload
   ```
4. Update this document with new model details
5. Add entry to `CHANGELOG.md`
6. Commit and deploy

---

## Future Models

Placeholder for future model versions (e.g., JAX neural network):

```
### vX.X - YYYY-MM-DD

**Model**: `model-name.joblib`

- Algorithm: 
- Framework:
- Training Data:
- Accuracy:
- Notes:
```
