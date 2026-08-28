from app.exceptions.ai_exceptions import AIProviderError


def validate_ai_content(content: str | None) -> str:
    """
    Validates that AI-generated content is present and non-blank.
    """

    if content is None:
        raise AIProviderError(
            "AI provider returned empty response content."
        )

    content = content.strip()

    if not content:
        raise AIProviderError(
            "AI provider returned empty response content."
        )

    return content
