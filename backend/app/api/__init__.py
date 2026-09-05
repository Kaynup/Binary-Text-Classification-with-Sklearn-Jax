"""
API routes package.
"""
from backend.app.api.routes import router as api_router
from backend.app.api.metrics_route import router as metrics_router

__all__ = ["api_router", "metrics_router"]
