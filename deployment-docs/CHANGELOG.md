# Changelog

All notable changes to this deployment will be documented in this file.

## [1.0.1] - 2025-12-26

### Fixed
- Python runtime now correctly specified as 3.10 using `functions` block in `vercel.json`
- Improved model path resolution using `pathlib` for Vercel serverless environment
- Added debug logging for model path troubleshooting

### Technical Details
- Changed `vercel.json` from deprecated `config.runtime` to top-level `functions` block
- Model loading now uses `Path(__file__).parent.resolve()` for reliable path resolution
- Added file existence check before model load with detailed error logging

### Testing Results (Local)
| Test | Input | Result |
|------|-------|--------|
| Positive | "I love this product!" | ✅ prediction: 1 |
| Negative | "This is terrible!" | ✅ prediction: 0 |
| Health check | GET /api/predict | ✅ Returns status |

---

## [1.0.0] - 2025-12-26

### Added
- Initial Vercel deployment configuration
- Created `vercel.json` for routing API and static files
- Created `api/predict.py` serverless function with lazy model loading
- Created `requirements.txt` with Python dependencies
- Frontend API URL updated from localhost to relative path `/api/predict`

### Technical Details
- **Model**: Logistic Regression (`logreg-80k.joblib`, ~3.8MB)
- **Framework**: FastAPI with Pydantic v2
- **Runtime**: Python (Vercel Serverless)
- **Static Files**: HTML/CSS/JS served from `app/` folder

### Testing Results
| Test | Input | Expected | Result |
|------|-------|----------|--------|
| Positive sentiment | "I love this product, it is amazing!" | prediction: 1 | ✅ Pass |
| Negative sentiment | "This is terrible, I hate it." | prediction: 0 | ✅ Pass |

### Notes
- Cold start time: ~1.4 seconds (includes model loading)
- Subsequent requests: ~1 ms inference time

---

## Template for Future Entries

```markdown
## [X.X.X] - YYYY-MM-DD

### Added
- New features

### Changed 
- Changes in existing functionality

### Fixed
- Bug fixes

### Removed
- Removed features

### Technical Details
- Any relevant technical information

### Notes
- Additional notes
```
