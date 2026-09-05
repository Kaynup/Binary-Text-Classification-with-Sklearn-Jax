"""
FastAPI Main Application Factory.
Sentiment Analyzer API v2.0.0 - Production Ready for Railway Deployment.
"""
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import uuid
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware

from backend.app.config import settings
from backend.app.logging_config import setup_logging, get_logger
from backend.app.services.classifier_service import classifier_service
from backend.app.middleware.timing import TimingMiddleware
from backend.app.middleware.rate_limiter import RateLimiterMiddleware
from backend.app.middleware.security import SecurityHeadersMiddleware
from backend.app.api.routes import router as api_router
from backend.app.api.metrics_route import router as metrics_router

# Initialize logging on load
logger = setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler: manages startup and graceful shutdown."""
    logger.info(f"=== Starting {settings.APP_NAME} v{settings.APP_VERSION} ===")
    logger.info(f"Environment: {settings.ENVIRONMENT} | Debug: {settings.DEBUG}")

    # Eagerly load the model pipeline
    classifier_service.load_model()
    if classifier_service.is_loaded:
        logger.info("Classifier pipeline is ready for inference requests.")
    else:
        logger.error("Classifier pipeline failed to load on startup. Health check will report degraded.")

    yield

    logger.info(f"=== Shutting down {settings.APP_NAME} ===")


def create_app() -> FastAPI:
    """Creates and configures the FastAPI application instance."""
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description="High-performance binary sentiment classification API powered by Scikit-Learn.",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan
    )

    # 1. Timing and Metrics Middleware (outermost for accurate measurement)
    app.add_middleware(TimingMiddleware)

    # 2. Rate Limiting Middleware
    app.add_middleware(RateLimiterMiddleware)

    # 3. Security Headers Middleware
    app.add_middleware(SecurityHeadersMiddleware)

    # 4. CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    # -------------------------------------------------------------------------
    # Exception Handlers
    # -------------------------------------------------------------------------

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": True,
                "status_code": exc.status_code,
                "message": exc.detail,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        details = exc.errors()
        messages = [f"{err.get('loc', [])}: {err.get('msg', 'Validation error')}" for err in details]
        return JSONResponse(
            status_code=422,
            content={
                "error": True,
                "status_code": 422,
                "message": "Input validation failed.",
                "details": messages,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        error_id = str(uuid.uuid4())[:8]
        logger.error(f"[Unhandled Error ID {error_id}] {request.method} {request.url.path}: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": True,
                "status_code": 500,
                "message": f"An unexpected internal error occurred (Reference: {error_id}).",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

    # -------------------------------------------------------------------------
    # Include API Routers
    # -------------------------------------------------------------------------
    app.include_router(api_router)
    app.include_router(metrics_router)

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
