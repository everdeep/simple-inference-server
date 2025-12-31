"""
Pydantic models for OpenAI-compatible API requests and responses.
"""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


# Request Models

class Message(BaseModel):
    """Chat message with role and content."""
    role: Literal["system", "user", "assistant"] = Field(
        description="Role of the message sender"
    )
    content: str = Field(
        description="Content of the message"
    )


class ChatCompletionRequest(BaseModel):
    """Request model for chat completions endpoint."""
    model: str = Field(
        default="llama-3-8b",
        description="Model to use for completion"
    )
    messages: List[Message] = Field(
        description="List of messages in the conversation"
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (0.0 to 2.0)"
    )
    max_tokens: int = Field(
        default=500,
        ge=1,
        le=32000,
        description="Maximum number of tokens to generate"
    )
    top_p: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling probability"
    )
    stream: bool = Field(
        default=False,
        description="Whether to stream the response"
    )
    stop: Optional[List[str]] = Field(
        default=None,
        description="Stop sequences for generation"
    )


# Response Models

class Usage(BaseModel):
    """Token usage information."""
    prompt_tokens: int = Field(
        description="Number of tokens in the prompt"
    )
    completion_tokens: int = Field(
        description="Number of tokens in the completion"
    )
    total_tokens: int = Field(
        description="Total tokens used (prompt + completion)"
    )


class Choice(BaseModel):
    """A single completion choice."""
    index: int = Field(
        description="Index of this choice"
    )
    message: Message = Field(
        description="The generated message"
    )
    finish_reason: Literal["stop", "length", "error"] = Field(
        description="Reason the generation stopped"
    )


class ChatCompletionResponse(BaseModel):
    """Response model for chat completions endpoint."""
    id: str = Field(
        description="Unique identifier for this completion"
    )
    object: str = Field(
        default="chat.completion",
        description="Object type"
    )
    created: int = Field(
        description="Unix timestamp of creation"
    )
    model: str = Field(
        description="Model used for this completion"
    )
    choices: List[Choice] = Field(
        description="List of completion choices"
    )
    usage: Usage = Field(
        description="Token usage information"
    )


# Health and Info Models

class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(
        default="ok",
        description="Health status"
    )
    model_loaded: bool = Field(
        description="Whether the model is loaded"
    )


class ModelInfo(BaseModel):
    """Model information."""
    id: str = Field(
        description="Model identifier"
    )
    object: str = Field(
        default="model",
        description="Object type"
    )
    owned_by: str = Field(
        default="local",
        description="Model owner"
    )


class ModelsResponse(BaseModel):
    """List of available models."""
    object: str = Field(
        default="list",
        description="Object type"
    )
    data: List[ModelInfo] = Field(
        description="List of available models"
    )


class ServerInfo(BaseModel):
    """Server information for admin endpoint."""
    model_name: str = Field(
        description="Currently loaded model name"
    )
    model_path: str = Field(
        description="Path to model file"
    )
    n_ctx: int = Field(
        description="Context window size"
    )
    n_gpu_layers: int = Field(
        description="Number of GPU layers"
    )
    model_loaded: bool = Field(
        description="Whether model is loaded"
    )


class ReloadResponse(BaseModel):
    """Response for model reload endpoint."""
    status: str = Field(
        description="Reload status"
    )
    message: str = Field(
        description="Status message"
    )
