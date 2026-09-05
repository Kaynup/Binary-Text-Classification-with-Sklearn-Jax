"""
Tests for Sliding-Window Rate Limiter Middleware.
"""
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.config import settings

client = TestClient(app)


def test_rate_limiter_allows_normal_traffic():
    for _ in range(5):
        resp = client.post(
            "/predict",
            json={"text": "Standard testing message within normal rate limits."},
            headers={"X-Forwarded-For": "192.0.2.1"}
        )
        assert resp.status_code in (200, 429)


def test_rate_limiter_blocks_on_excessive_burst():
    # Temporarily set a lower rate limit for this test client IP
    test_ip = "198.51.100.99"
    limit = settings.RATE_LIMIT_PREDICT_PER_MINUTE

    status_codes = []
    # Burst 20 requests beyond the limit
    for _ in range(limit + 5):
        resp = client.post(
            "/predict",
            json={"text": "Burst load rate limiting test."},
            headers={"X-Forwarded-For": test_ip}
        )
        status_codes.append(resp.status_code)

    assert 429 in status_codes, f"Expected 429 in status codes but got {set(status_codes)}"
    last_resp = client.post(
        "/predict",
        json={"text": "Post burst test."},
        headers={"X-Forwarded-For": test_ip}
    )
    assert last_resp.status_code == 429
    assert "Retry-After" in last_resp.headers
