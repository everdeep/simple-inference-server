"""
Configuration management using Pydantic Settings.
Loads configuration from environment variables and .env file.
"""

from pathlib import Path
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Server Configuration
    port: int = Field(default=1133, description="Server port")
    host: str = Field(default="0.0.0.0", description="Server host")
    log_level: str = Field(default="info", description="Logging level")

    # Authentication
    api_keys: List[str] = Field(
        default_factory=list,
        description="List of valid API keys (comma-separated in env)"
    )
    admin_api_key: str = Field(
        default="",
        description="Admin API key for management endpoints"
    )

    # Model Configuration
    model_path: Path = Field(
        default=Path("/app/models/model.gguf"),
        description="Path to GGUF model file"
    )
    model_name: str = Field(
        default="llama-3-8b",
        description="Model name for API responses"
    )

    # GPU/CUDA Configuration
    n_gpu_layers: int = Field(
        default=-1,
        description="Number of layers to offload to GPU (-1 = all)"
    )
    n_ctx: int = Field(
        default=4096,
        description="Context window size in tokens"
    )
    n_batch: int = Field(
        default=512,
        description="Batch size for prompt processing"
    )
    n_threads: int = Field(
        default=8,
        description="Number of CPU threads for processing"
    )

    # Performance Tuning
    use_mlock: bool = Field(
        default=True,
        description="Lock model in RAM to prevent swapping"
    )
    use_mmap: bool = Field(
        default=True,
        description="Use memory-mapped file I/O"
    )
    rope_freq_base: float = Field(
        default=10000.0,
        description="RoPE frequency base (model-specific)"
    )
    rope_freq_scale: float = Field(
        default=1.0,
        description="RoPE frequency scale (model-specific)"
    )

    # Model configuration
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    @field_validator("api_keys", mode="before")
    @classmethod
    def parse_api_keys(cls, v):
        """Parse comma-separated API keys from environment variable."""
        if isinstance(v, str):
            return [key.strip() for key in v.split(",") if key.strip()]
        return v

    @field_validator("model_path", mode="before")
    @classmethod
    def parse_model_path(cls, v):
        """Convert string to Path object."""
        if isinstance(v, str):
            return Path(v)
        return v

    def validate_required_fields(self) -> None:
        """Validate that required fields are set."""
        if not self.api_keys:
            raise ValueError(
                "At least one API key must be configured. "
                "Set API_KEYS environment variable."
            )
        if not self.admin_api_key:
            raise ValueError(
                "Admin API key must be configured. "
                "Set ADMIN_API_KEY environment variable."
            )


# Global settings instance
settings = Settings()

# Validate required fields on import
try:
    settings.validate_required_fields()
except ValueError as e:
    # Allow import to succeed but log warning
    # This allows scripts and tests to import without full config
    import warnings
    warnings.warn(f"Configuration validation failed: {e}")
