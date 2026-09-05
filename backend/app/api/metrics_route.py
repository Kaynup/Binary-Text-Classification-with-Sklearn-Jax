"""
Prometheus Metrics Endpoint.
"""
from fastapi import APIRouter
from fastapi.responses import PlainTextResponse
from backend.app.services.metrics_collector import metrics_collector

router = APIRouter()


@router.get("/metrics", response_class=PlainTextResponse, summary="Prometheus Metrics Exporter")
def get_metrics():
    """Returns Prometheus exposition formatted runtime metrics."""
    return metrics_collector.generate_prometheus_output()
