"""
Prometheus Metrics Collector and Exporter.
Tracks request counts, response latency histograms, sentiment distribution,
and rate-limit blocks for Prometheus scraping and Grafana visualization.
"""
import time
import threading
from typing import Dict, List, Tuple


class MetricsCollector:
    """
    Lightweight, thread-safe Prometheus metrics collector.
    Exposes metrics in standard Prometheus text exposition format (version 0.0.4).
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._http_requests: Dict[Tuple[str, str, int], int] = {}
        self._predictions: Dict[str, int] = {"positive": 0, "negative": 0}
        self._rate_limits_exceeded: int = 0
        self._in_flight_requests: int = 0
        self._start_time: float = time.time()

        # Histogram buckets in seconds
        self._latency_buckets = [0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
        self._request_latencies: Dict[str, Dict[float, int]] = {}
        self._latency_sums: Dict[str, float] = {}
        self._latency_counts: Dict[str, int] = {}

    def inc_in_flight(self) -> None:
        with self._lock:
            self._in_flight_requests += 1

    def dec_in_flight(self) -> None:
        with self._lock:
            self._in_flight_requests = max(0, self._in_flight_requests - 1)

    def record_request(self, method: str, endpoint: str, status_code: int, duration_sec: float) -> None:
        with self._lock:
            key = (method, endpoint, status_code)
            self._http_requests[key] = self._http_requests.get(key, 0) + 1

            if endpoint not in self._request_latencies:
                self._request_latencies[endpoint] = {b: 0 for b in self._latency_buckets}
                self._latency_sums[endpoint] = 0.0
                self._latency_counts[endpoint] = 0

            self._latency_sums[endpoint] += duration_sec
            self._latency_counts[endpoint] += 1

            for bucket in self._latency_buckets:
                if duration_sec <= bucket:
                    self._request_latencies[endpoint][bucket] += 1

    def record_prediction(self, sentiment: str) -> None:
        key = sentiment.lower()
        with self._lock:
            if key in self._predictions:
                self._predictions[key] += 1
            else:
                self._predictions[key] = 1

    def record_rate_limit(self) -> None:
        with self._lock:
            self._rate_limits_exceeded += 1

    def generate_prometheus_output(self) -> str:
        """Serializes current state into Prometheus exposition format."""
        with self._lock:
            lines: List[str] = [
                "# HELP app_uptime_seconds Total seconds the service has been running",
                "# TYPE app_uptime_seconds gauge",
                f"app_uptime_seconds {time.time() - self._start_time:.2f}",
                "",
                "# HELP http_requests_in_flight Number of HTTP requests currently being processed",
                "# TYPE http_requests_in_flight gauge",
                f"http_requests_in_flight {self._in_flight_requests}",
                "",
                "# HELP http_requests_total Total number of HTTP requests processed",
                "# TYPE http_requests_total counter",
            ]

            for (method, endpoint, status), count in sorted(self._http_requests.items()):
                lines.append(f'http_requests_total{{method="{method}",endpoint="{endpoint}",status="{status}"}} {count}')

            lines.extend([
                "",
                "# HELP sentiment_predictions_total Total sentiment predictions classified by category",
                "# TYPE sentiment_predictions_total counter",
                f'sentiment_predictions_total{{sentiment="positive"}} {self._predictions.get("positive", 0)}',
                f'sentiment_predictions_total{{sentiment="negative"}} {self._predictions.get("negative", 0)}',
                "",
                "# HELP rate_limit_exceeded_total Total number of requests rejected by rate limiting (HTTP 429)",
                "# TYPE rate_limit_exceeded_total counter",
                f"rate_limit_exceeded_total {self._rate_limits_exceeded}",
                "",
                "# HELP http_request_duration_seconds HTTP request duration in seconds histogram",
                "# TYPE http_request_duration_seconds histogram",
            ])

            for endpoint, buckets in sorted(self._request_latencies.items()):
                for b, count in sorted(buckets.items()):
                    lines.append(f'http_request_duration_seconds_bucket{{endpoint="{endpoint}",le="{b}"}} {count}')
                lines.append(f'http_request_duration_seconds_bucket{{endpoint="{endpoint}",le="+Inf"}} {self._latency_counts[endpoint]}')
                lines.append(f'http_request_duration_seconds_sum{{endpoint="{endpoint}"}} {self._latency_sums[endpoint]:.6f}')
                lines.append(f'http_request_duration_seconds_count{{endpoint="{endpoint}"}} {self._latency_counts[endpoint]}')

            lines.append("")
            return "\n".join(lines)


metrics_collector = MetricsCollector()
