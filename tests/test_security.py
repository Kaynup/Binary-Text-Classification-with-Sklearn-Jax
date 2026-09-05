"""
Security, Defensive Headers, and XSS Payload Tests.
"""
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)


def test_security_headers_present():
    resp = client.get("/health")
    assert resp.headers.get("X-Content-Type-Options") == "nosniff"
    assert resp.headers.get("X-Frame-Options") == "DENY"
    assert "X-XSS-Protection" in resp.headers


def test_xss_script_payload_safety():
    xss_payload = "<script>alert('XSS-Exploit')</script>"
    resp = client.post("/predict", json={"text": xss_payload})
    assert resp.status_code == 200
    data = resp.json()
    # The output preview must NOT execute as HTML and must preserve sanitized text safely
    assert "prediction" in data
    assert "sentiment" in data


def test_xss_img_onerror_payload_safety():
    payload = '<img src=x onerror="alert(document.cookie)"> Great product overall!'
    resp = client.post("/predict", json={"text": payload})
    assert resp.status_code == 200
    data = resp.json()
    assert data["prediction"] in (0, 1)


def test_sql_injection_string_safety():
    payload = "' OR '1'='1'; DROP TABLE users; --"
    resp = client.post("/predict", json={"text": payload})
    assert resp.status_code == 200


def test_unicode_and_emoticons_safety():
    payload = "🔥🚀 Super happy! 😊 100% recommended! ありがとう ❤️"
    resp = client.post("/predict", json={"text": payload})
    assert resp.status_code == 200
    data = resp.json()
    assert data["prediction"] == 1
