"""
Tests for API key authentication.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from app.main import app


# Mock settings for testing
MOCK_API_KEYS = ["sk-test-key-123", "sk-test-key-456"]
MOCK_ADMIN_KEY = "sk-admin-test-key"


@pytest.fixture
def client():
    """Create test client."""
    with patch("app.config.settings") as mock_settings:
        mock_settings.api_keys = MOCK_API_KEYS
        mock_settings.admin_api_key = MOCK_ADMIN_KEY
        mock_settings.model_name = "test-model"
        yield TestClient(app)


def test_health_endpoint_no_auth(client):
    """Test that health endpoint doesn't require authentication."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_models_endpoint_requires_auth(client):
    """Test that models endpoint requires authentication."""
    response = client.get("/v1/models")
    assert response.status_code == 403  # No auth header


def test_models_endpoint_with_valid_key(client):
    """Test models endpoint with valid API key."""
    response = client.get(
        "/v1/models",
        headers={"Authorization": f"Bearer {MOCK_API_KEYS[0]}"}
    )
    assert response.status_code == 200


def test_models_endpoint_with_invalid_key(client):
    """Test models endpoint with invalid API key."""
    response = client.get(
        "/v1/models",
        headers={"Authorization": "Bearer invalid-key"}
    )
    assert response.status_code == 401


def test_chat_endpoint_requires_auth(client):
    """Test that chat endpoint requires authentication."""
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}]
        }
    )
    assert response.status_code == 403


def test_admin_endpoint_requires_admin_key(client):
    """Test that admin endpoints require admin key."""
    # Try with regular API key
    response = client.get(
        "/admin/info",
        headers={"Authorization": f"Bearer {MOCK_API_KEYS[0]}"}
    )
    assert response.status_code == 403

    # Try with admin key
    response = client.get(
        "/admin/info",
        headers={"Authorization": f"Bearer {MOCK_ADMIN_KEY}"}
    )
    assert response.status_code == 200


def test_admin_reload_requires_admin_key(client):
    """Test that model reload requires admin key."""
    # Try with regular API key
    response = client.post(
        "/admin/reload",
        headers={"Authorization": f"Bearer {MOCK_API_KEYS[0]}"}
    )
    assert response.status_code == 403

    # Admin key test would work with proper mocking
    # Not testing actual reload here as it requires model


def test_multiple_valid_keys(client):
    """Test that multiple API keys work."""
    for key in MOCK_API_KEYS:
        response = client.get(
            "/v1/models",
            headers={"Authorization": f"Bearer {key}"}
        )
        assert response.status_code == 200
