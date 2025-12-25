from __future__ import annotations
from fastapi.testclient import TestClient
from app.main import create_app

def test_health() ->None:
    app = create_app()
    client = TestClient(app)
    r = client.get('/health')
    assert r.status_code ==200
    assert r.json()['status'] == 'ok'