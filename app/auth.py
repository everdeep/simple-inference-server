"""
API key authentication for the LLM inference server.
Implements HTTPBearer security scheme with API key validation.
"""

from fastapi import HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import settings

# HTTPBearer security scheme
security = HTTPBearer(
    scheme_name="Bearer Token",
    description="API key for authentication"
)


async def verify_api_key(
    credentials: HTTPAuthorizationCredentials = Security(security)
) -> str:
    """
    Validate API key from Authorization header.

    Args:
        credentials: HTTP Bearer token credentials from request header

    Returns:
        The validated API key

    Raises:
        HTTPException: 401 if API key is invalid
    """
    token = credentials.credentials

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if token not in settings.api_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return token


async def verify_admin_key(
    credentials: HTTPAuthorizationCredentials = Security(security)
) -> str:
    """
    Validate admin API key from Authorization header.

    Admin key is required for management endpoints like model reloading
    and server information.

    Args:
        credentials: HTTP Bearer token credentials from request header

    Returns:
        The validated admin API key

    Raises:
        HTTPException: 401 if key is missing, 403 if key is invalid
    """
    token = credentials.credentials

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if token != settings.admin_api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )

    return token
