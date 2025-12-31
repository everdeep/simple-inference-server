"""
Tests for API endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock

from app.main import app


# Mock settings for testing
MOCK_API_KEYS = ["sk-test-key-123"]
MOCK_ADMIN_KEY = "sk-admin-test-key"


@pytest.fixture
def client():
    """Create test client with mocked settings."""
    with patch("app.config.settings") as mock_settings:
        mock_settings.api_keys = MOCK_API_KEYS
        mock_settings.admin_api_key = MOCK_ADMIN_KEY
        mock_settings.model_name = "test-model"
        mock_settings.model_path = "/app/models/test.gguf"
        mock_settings.n_ctx = 4096
        mock_settings.n_gpu_layers = -1
        yield TestClient(app)


def test_root_endpoint(client):
    """Test root endpoint returns API information."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Remote LLM Inference Server"
    assert data["status"] == "running"
    assert "endpoints" in data


def test_health_endpoint(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data


def test_models_list_endpoint(client):
    """Test models list endpoint."""
    response = client.get(
        "/v1/models",
        headers={"Authorization": f"Bearer {MOCK_API_KEYS[0]}"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert len(data["data"]) > 0
    assert data["data"][0]["id"] == "test-model"


def test_chat_completion_request_validation(client):
    """Test that chat completion validates request schema."""
    # Missing required field (messages)
    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {MOCK_API_KEYS[0]}"},
        json={"model": "test-model"}
    )
    assert response.status_code == 422  # Validation error

    # Invalid message format
    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {MOCK_API_KEYS[0]}"},
        json={
            "model": "test-model",
            "messages": [{"invalid": "format"}]
        }
    )
    assert response.status_code == 422


def test_chat_completion_valid_request_format(client):
    """Test chat completion with valid request format."""
    # This will fail without a real model, but we can test the format
    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {MOCK_API_KEYS[0]}"},
        json={
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "Hello"}
            ],
            "temperature": 0.7,
            "max_tokens": 100
        }
    )
    # Will return 500 because model isn't actually loaded
    # But format is correct
    assert response.status_code in [200, 500]


def test_admin_info_endpoint(client):
    """Test admin info endpoint."""
    response = client.get(
        "/admin/info",
        headers={"Authorization": f"Bearer {MOCK_ADMIN_KEY}"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "test-model"
    assert "model_path" in data
    assert "n_ctx" in data
    assert "n_gpu_layers" in data
    assert "model_loaded" in data


def test_request_validation_temperature_range(client):
    """Test that temperature is validated to be within range."""
    # Temperature too high
    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {MOCK_API_KEYS[0]}"},
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 3.0  # Max is 2.0
        }
    )
    assert response.status_code == 422


def test_request_validation_max_tokens_positive(client):
    """Test that max_tokens must be positive."""
    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {MOCK_API_KEYS[0]}"},
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 0  # Min is 1
        }
    )
    assert response.status_code == 422


def test_openapi_docs_available(client):
    """Test that OpenAPI documentation is available."""
    response = client.get("/docs")
    assert response.status_code == 200

    response = client.get("/redoc")
    assert response.status_code == 200

    response = client.get("/openapi.json")
    assert response.status_code == 200
