"""
LLM model manager with CUDA support.
Handles model loading, inference, and thread-safe access.
"""

import asyncio
import logging
from typing import Optional

from llama_cpp import Llama

from app.config import settings

logger = logging.getLogger(__name__)


class LLMManager:
    """
    Singleton manager for LLM model instance.
    Ensures thread-safe access and proper lifecycle management.
    """

    def __init__(self):
        """Initialize the LLM manager."""
        self._llm: Optional[Llama] = None
        self._lock = asyncio.Lock()
        self._loading = False

    async def get_llm(self) -> Llama:
        """
        Get or initialize the LLM instance.
        Uses double-check locking pattern for thread-safe lazy initialization.

        Returns:
            Initialized Llama instance

        Raises:
            RuntimeError: If model fails to load
        """
        if self._llm is None:
            async with self._lock:
                if self._llm is None:  # Double-check locking
                    logger.info("Loading LLM model...")
                    self._llm = await self._load_model()
                    logger.info("LLM model loaded successfully")
        return self._llm

    async def _load_model(self) -> Llama:
        """
        Load the LLM model with CUDA configuration.

        Returns:
            Initialized Llama instance

        Raises:
            RuntimeError: If model file not found or loading fails
        """
        try:
            model_path = settings.model_path

            # Check if model file exists
            if not model_path.exists():
                raise RuntimeError(
                    f"Model file not found: {model_path}. "
                    f"Please download a GGUF model to the models directory."
                )

            logger.info(f"Loading model from: {model_path}")
            logger.info(f"GPU layers: {settings.n_gpu_layers}")
            logger.info(f"Context size: {settings.n_ctx}")

            # Load model with configuration
            llm = Llama(
                model_path=str(model_path),
                n_gpu_layers=settings.n_gpu_layers,
                n_ctx=settings.n_ctx,
                n_batch=settings.n_batch,
                n_threads=settings.n_threads,
                use_mlock=settings.use_mlock,
                use_mmap=settings.use_mmap,
                rope_freq_base=settings.rope_freq_base,
                rope_freq_scale=settings.rope_freq_scale,
                verbose=False,
            )

            return llm

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Failed to load model: {e}")

    async def reload_model(self) -> None:
        """
        Reload the model (for admin endpoint).
        Useful after changing model configuration or file.

        Raises:
            RuntimeError: If model reload fails
        """
        async with self._lock:
            logger.info("Reloading model...")

            # Clean up existing model
            if self._llm is not None:
                del self._llm
                self._llm = None

            # Load new model
            self._llm = await self._load_model()
            logger.info("Model reloaded successfully")

    def is_loaded(self) -> bool:
        """
        Check if model is currently loaded.

        Returns:
            True if model is loaded, False otherwise
        """
        return self._llm is not None

    async def generate_completion(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stop: Optional[list] = None,
    ) -> dict:
        """
        Generate a completion for the given prompt.

        Args:
            prompt: The input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            stop: Stop sequences

        Returns:
            Dictionary containing the completion and token usage

        Raises:
            RuntimeError: If generation fails
        """
        llm = await self.get_llm()

        try:
            # Generate completion
            response = llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                echo=False,
            )

            return response

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise RuntimeError(f"Generation failed: {e}")

    async def create_chat_completion(
        self,
        messages: list,
        max_tokens: int = 500,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stop: Optional[list] = None,
    ) -> dict:
        """
        Generate a chat completion for the given messages.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            stop: Stop sequences

        Returns:
            Dictionary containing the completion and token usage

        Raises:
            RuntimeError: If generation fails
        """
        llm = await self.get_llm()

        try:
            # Use llama-cpp-python's chat completion format
            response = llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
            )

            return response

        except Exception as e:
            logger.error(f"Chat completion failed: {e}")
            raise RuntimeError(f"Chat completion failed: {e}")


# Global LLM manager instance
llm_manager = LLMManager()
