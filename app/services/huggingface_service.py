from openai import OpenAI, APIError

from app.core.config import settings
from app.exceptions.ai_exceptions import AIProviderError
from app.logging.logger import logger
from app.services.ai_output_validator import validate_ai_content


client = OpenAI(
    base_url=settings.hf_base_url,
    api_key=settings.hf_token,
    timeout=settings.ai_request_timeout_seconds,
)


def check_availability() -> bool:
    """
    Lightweight reachability check for /health. Lists available models
    rather than running a real generation, with a short fail-fast
    timeout independent of ai_request_timeout_seconds.
    """

    try:
        client.with_options(timeout=5.0).models.list()
        return True
    except Exception:
        logger.warning(
            "Hugging Face availability check failed", exc_info=True
        )
        return False


def _validate_ai_output(response) -> str:
    """
    Validates and extracts the AI-generated response.
    """

    if not response.choices:
        raise AIProviderError(
            "AI provider returned no response choices."
        )

    return validate_ai_content(response.choices[0].message.content)


def generate_response(
    prompt: str,
    text: str,
    customer_id: int
) -> str:
    """
    Generates a response using Hugging Face GPT-OSS 20B.
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

    try:

        response = client.chat.completions.create(
            model=settings.hf_model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": text
                }
            ],
            temperature=0
        )

    except APIError as exc:

        logger.error(
            "AI provider request failed",
            extra={
                "customer_id": customer_id,
                "error_type": type(exc).__name__,
            },
        )

        raise AIProviderError() from exc

    return _validate_ai_output(response)
