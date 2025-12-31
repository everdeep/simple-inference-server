"""
Chat completion endpoints (OpenAI-compatible).
"""

import time
import uuid
import logging
from fastapi import APIRouter, Depends, HTTPException, status

from app.auth import verify_api_key
from app.config import settings
from app.llm import llm_manager
from app.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    Message,
    Usage,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Chat"])


@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(
    request: ChatCompletionRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Create a chat completion (OpenAI-compatible endpoint).

    Generates a response to a conversation with the LLM.
    Requires API key authentication.

    Args:
        request: Chat completion request with messages and parameters
        api_key: Validated API key from authentication

    Returns:
        ChatCompletionResponse with generated completion and usage info

    Raises:
        HTTPException: 500 if generation fails
    """
    try:
        logger.info(f"Chat completion request for model: {request.model}")
        logger.info(f"Messages: {len(request.messages)}")

        # Convert Pydantic models to dictionaries for llama-cpp-python
        messages = [
            {"role": msg.role, "content": msg.content}
            for msg in request.messages
        ]

        # Generate completion using LLM manager
        response = await llm_manager.create_chat_completion(
            messages=messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=request.stop,
        )

        # Extract response data
        # llama-cpp-python returns OpenAI-compatible format
        choice_data = response["choices"][0]
        usage_data = response["usage"]

        # Build OpenAI-compatible response
        completion_response = ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
            object="chat.completion",
            created=int(time.time()),
            model=settings.model_name,
            choices=[
                Choice(
                    index=0,
                    message=Message(
                        role=choice_data["message"]["role"],
                        content=choice_data["message"]["content"]
                    ),
                    finish_reason=choice_data.get("finish_reason", "stop")
                )
            ],
            usage=Usage(
                prompt_tokens=usage_data["prompt_tokens"],
                completion_tokens=usage_data["completion_tokens"],
                total_tokens=usage_data["total_tokens"]
            )
        )

        logger.info(
            f"Completion generated: "
            f"{usage_data['completion_tokens']} tokens"
        )

        return completion_response

    except RuntimeError as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Generation failed: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}"
        )
