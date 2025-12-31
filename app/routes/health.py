"""
Health check and model info endpoints.
"""

from fastapi import APIRouter, Depends

from app.auth import verify_api_key
from app.config import settings
from app.llm import llm_manager
from app.models import HealthResponse, ModelInfo, ModelsResponse

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    Returns the server status and whether the model is loaded.
    Does not require authentication for basic monitoring.
    """
    return HealthResponse(
        status="ok",
        model_loaded=llm_manager.is_loaded()
    )


@router.get("/v1/models", response_model=ModelsResponse)
async def list_models(api_key: str = Depends(verify_api_key)):
    """
    List available models (OpenAI-compatible endpoint).
    Returns information about the currently loaded model.

    Requires API key authentication.
    """
    model_info = ModelInfo(
        id=settings.model_name,
        object="model",
        owned_by="local"
    )

    return ModelsResponse(
        object="list",
        data=[model_info]
    )
