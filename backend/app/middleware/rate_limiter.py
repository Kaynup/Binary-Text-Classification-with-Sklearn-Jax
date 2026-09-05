"""
In-Memory Sliding-Window Rate Limiting Middleware.
Protects the API against denial-of-service, abuse, and brute-force bursts without Redis dependency.
"""
import time
import threading
from typing import Dict, List, Tuple
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from backend.app.config import settings
from backend.app.services.metrics_collector import metrics_collector
from backend.app.logging_config import get_logger

logger = get_logger("sentiment_analyzer.ratelimit")


class RateLimiterMiddleware(BaseHTTPMiddleware):
    """
    Sliding-window IP rate limiter.
    """

    def __init__(self, app):
        super().__init__(app)
        self._lock = threading.Lock()
        # Mapping: ip -> list of float timestamps
        self._history: Dict[str, List[float]] = {}
        self._cleanup_counter = 0

    def _get_client_ip(self, request: Request) -> str:
        """Extracts client IP, respecting standard reverse-proxy headers."""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            # First IP in list is the originating client
            return forwarded.split(",")[0].strip()
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip.strip()
        if request.client and request.client.host:
            return request.client.host
        return "127.0.0.1"

    def _check_rate_limit(self, ip: str, path: str) -> Tuple[bool, int]:
        """
        Check if request should be allowed.
        Returns: (allowed: bool, retry_after_seconds: int)
        """
        if not settings.RATE_LIMIT_ENABLED:
            return True, 0

        # Whitelist internal monitoring or root
        if path in ("/health", "/metrics", "/docs", "/openapi.json"):
            limit = settings.RATE_LIMIT_GENERAL_PER_MINUTE
        else:
            limit = settings.RATE_LIMIT_PREDICT_PER_MINUTE

        now = time.time()
        window_start = now - 60.0

        with self._lock:
            # Periodically purge old IPs
            self._cleanup_counter += 1
            if self._cleanup_counter > 500:
                self._purge_stale_ips(window_start)
                self._cleanup_counter = 0

            timestamps = self._history.get(ip, [])
            # Filter timestamps within the 60 second window
            valid_timestamps = [t for t in timestamps if t > window_start]

            if len(valid_timestamps) >= limit:
                # Calculate remaining seconds until oldest timestamp expires
                oldest = valid_timestamps[0]
                retry_after = max(1, int(60.0 - (now - oldest)))
                self._history[ip] = valid_timestamps
                return False, retry_after

            valid_timestamps.append(now)
            self._history[ip] = valid_timestamps
            return True, 0

    def _purge_stale_ips(self, window_start: float) -> None:
        """Removes IP entries that have no requests in current window."""
        stale_keys = [
            ip for ip, times in self._history.items()
            if not times or max(times) < window_start
        ]
        for k in stale_keys:
            del self._history[k]

    async def dispatch(self, request: Request, call_next):
        client_ip = self._get_client_ip(request)
        path = request.url.path

        allowed, retry_after = self._check_rate_limit(client_ip, path)
        if not allowed:
            logger.warning(
                f"Rate limit exceeded for IP '{client_ip}' on path '{path}'. "
                f"Retry-After: {retry_after}s"
            )
            metrics_collector.record_rate_limit()
            return JSONResponse(
                status_code=429,
                content={
                    "error": True,
                    "status_code": 429,
                    "message": f"Too Many Requests. Rate limit exceeded. Try again in {retry_after} seconds.",
                    "retry_after": retry_after,
                    "client_ip": client_ip
                },
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(settings.RATE_LIMIT_PREDICT_PER_MINUTE),
                    "X-RateLimit-Remaining": "0"
                }
            )

        response = await call_next(request)
        return response
