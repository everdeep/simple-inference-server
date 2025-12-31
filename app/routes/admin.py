"""
Admin management endpoints.
Requires admin API key for access.
"""

import logging
from fastapi import APIRouter, Depends, HTTPException, status

from app.auth import verify_admin_key
from app.config import settings
from app.llm import llm_manager
from app.models import ReloadResponse, ServerInfo

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["Admin"])


@router.post("/reload", response_model=ReloadResponse)
async def reload_model(admin_key: str = Depends(verify_admin_key)):
    """
    Reload the LLM model.

    Useful after changing model configuration or swapping model files.
    Requires admin API key authentication.

    Args:
        admin_key: Validated admin API key

    Returns:
        ReloadResponse with status and message

    Raises:
        HTTPException: 500 if reload fails
    """
    try:
        logger.info("Admin requested model reload")
        await llm_manager.reload_model()

        return ReloadResponse(
            status="success",
            message="Model reloaded successfully"
        )

    except Exception as e:
        logger.error(f"Model reload failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model reload failed: {str(e)}"
        )


@router.get("/info", response_model=ServerInfo)
async def get_server_info(admin_key: str = Depends(verify_admin_key)):
    """
    Get server and model information.

    Returns configuration and status information about the server.
    Requires admin API key authentication.

    Args:
        admin_key: Validated admin API key

    Returns:
        ServerInfo with model and server details
    """
    return ServerInfo(
        model_name=settings.model_name,
        model_path=str(settings.model_path),
        n_ctx=settings.n_ctx,
        n_gpu_layers=settings.n_gpu_layers,
        model_loaded=llm_manager.is_loaded()
    )
