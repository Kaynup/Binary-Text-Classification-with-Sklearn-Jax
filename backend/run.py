#!/usr/bin/env python3
"""
Backend execution entrypoint.
Usage:
    python backend/run.py
    python run.py (inside backend/)
"""
import os
import sys

# Ensure repository root is on path so backend package can be imported
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import uvicorn
from backend.app.config import settings

if __name__ == "__main__":
    port = int(os.environ.get("PORT", settings.PORT))
    host = os.environ.get("HOST", settings.HOST)
    print(f"Starting Sentiment Analysis API on http://{host}:{port}")
    uvicorn.run("backend.app.main:app", host=host, port=port, reload=False)
