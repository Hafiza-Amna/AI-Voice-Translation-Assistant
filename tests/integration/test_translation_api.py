import pytest
from app.config import settings

@pytest.mark.asyncio
async def test_root_endpoint(async_client):
    """Test root API endpoint returns status 200."""
    response = await async_client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert data["health_check"] == "/api/v1/health"

@pytest.mark.asyncio
async def test_health_endpoint(async_client):
    """Test health check endpoint returns 200 and healthy status."""
    response = await async_client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["app_name"] == settings.APP_NAME

def test_config_loading():
    """Test Pydantic configuration loading."""
    assert settings.APP_NAME is not None
    assert settings.PORT == 8000
