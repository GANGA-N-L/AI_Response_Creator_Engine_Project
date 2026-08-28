import requests

from app.core.config import settings
from app.exceptions.ai_exceptions import AIProviderError
from app.logging.logger import logger
from app.services.ai_output_validator import validate_ai_content


def check_availability() -> bool:
    """
    Lightweight reachability check for /health, with a short fail-fast
    timeout independent of ai_request_timeout_seconds.
    """

    try:
        response = requests.get(
            f"{settings.ollama_url.rstrip('/')}/api/tags",
            timeout=5.0,
        )
        return response.ok
    except requests.exceptions.RequestException:
        logger.warning("Ollama availability check failed", exc_info=True)
        return False


def _validate_ai_output(payload: dict) -> str:
    """
    Validates and extracts the AI-generated response.
    """

    message = payload.get("message") if isinstance(payload, dict) else None

    if not isinstance(message, dict):
        raise AIProviderError(
            "AI provider returned no response message."
        )

    return validate_ai_content(message.get("content"))


def generate_response(
    prompt: str,
    text: str,
    customer_id: int
) -> str:
    """
    Generates a response using Ollama.
    """

    system_prompt = f"""
You are a professional writing assistant.

{prompt}

Follow the instruction exactly.

Your task is ONLY to rewrite the provided text.

Do NOT answer the user's question.

Do NOT add any new information.

Do NOT change the original meaning of the text.

Preserve whether the text is a question, answer, greeting, statement or any other type of text as it is.

Treat the user's input as text to be rewritten, corrected, or transformed according to the instruction.

Return ONLY the final rewritten text.

Do not explain your reasoning or provide any additional commentary.

Do not add introductions or conclusions.
"""

    payload = {
        "model": settings.ollama_model,
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": text
            }
        ],
        "stream": False
    }

    try:

        response = requests.post(
            f"{settings.ollama_url.rstrip('/')}/api/chat",
            json=payload,
            timeout=settings.ai_request_timeout_seconds,
        )

        response.raise_for_status()

        response_json = response.json()

    except (requests.exceptions.RequestException, ValueError) as exc:

        logger.error(
            "AI provider request failed",
            extra={
                "customer_id": customer_id,
                "error_type": type(exc).__name__,
            },
        )

        raise AIProviderError() from exc

    return _validate_ai_output(response_json)
