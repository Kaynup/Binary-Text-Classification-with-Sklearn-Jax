"""
Request Timing and Observability Middleware.
Measures wall-clock latency, records metrics into Prometheus collector, and logs requests.
"""
import time
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from backend.app.services.metrics_collector import metrics_collector
from backend.app.logging_config import get_logger

logger = get_logger("sentiment_analyzer.access")


class TimingMiddleware(BaseHTTPMiddleware):
    """Measures latency, updates Prometheus histograms, and adds timing header."""

    async def dispatch(self, request: Request, call_next):
        metrics_collector.inc_in_flight()
        start_time = time.perf_counter()

        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception as exc:
            status_code = 500
            metrics_collector.dec_in_flight()
            raise exc

        duration_sec = time.perf_counter() - start_time
        duration_ms = duration_sec * 1000

        metrics_collector.dec_in_flight()
        metrics_collector.record_request(
            method=request.method,
            endpoint=request.url.path,
            status_code=status_code,
            duration_sec=duration_sec
        )

        response.headers["X-Process-Time-Ms"] = f"{duration_ms:.2f}"

        # Avoid spamming logs for frequent metrics/health scrapes
        if request.url.path not in ("/metrics", "/health"):
            logger.info(
                f"{request.method} {request.url.path} | "
                f"Status: {status_code} | "
                f"Latency: {duration_ms:.2f} ms | "
                f"IP: {request.client.host if request.client else 'unknown'}"
            )

        return response
