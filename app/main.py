"""
FastAPI application entry point.
LLM inference server with OpenAI-compatible API.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.llm import llm_manager
from app.routes import admin, chat, health

# Configure logging
logging.basicConfig(
    level=settings.log_level.upper(),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    Preloads the model on startup and cleans up on shutdown.
    """
    # Startup
    logger.info("Starting LLM inference server...")
    logger.info(f"Model: {settings.model_name}")
    logger.info(f"Port: {settings.port}")

    try:
        # Preload model on startup
        logger.info("Preloading model...")
        await llm_manager.get_llm()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.warning(f"Failed to preload model: {e}")
        logger.warning("Model will be loaded on first request")

    yield

    # Shutdown
    logger.info("Shutting down LLM inference server...")


# Create FastAPI application
app = FastAPI(
    title="Remote LLM Inference Server",
    description="OpenAI-compatible LLM inference server using llama.cpp with CUDA support",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure based on your needs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Handle uncaught exceptions globally.
    """
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Internal server error",
            "type": "internal_error"
        }
    )


# Register routers
app.include_router(health.router)
app.include_router(chat.router)
app.include_router(admin.router)


@app.get("/")
async def root():
    """
    Root endpoint with API information.
    """
    return {
        "name": "Remote LLM Inference Server",
        "version": "0.1.0",
        "model": settings.model_name,
        "status": "running",
        "docs": "/docs",
        "endpoints": {
            "health": "GET /health",
            "models": "GET /v1/models",
            "chat_completions": "POST /v1/chat/completions",
            "admin_info": "GET /admin/info",
            "admin_reload": "POST /admin/reload"
        }
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
    )
