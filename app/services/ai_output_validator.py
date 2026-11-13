import re

from app.exceptions.ai_exceptions import AIProviderError
from app.logging.logger import logger
from app.services.ai_prompt import LEAKAGE_MARKERS
from app.validators.content_moderator import ContentModerator

# A rewrite should track the size of what it's rewriting, not grow
# unbounded - generously wide to allow legitimate expansion (e.g.
# "rewrite as a formal paragraph"), just wide enough to catch a
# degenerate/looping generation.
_MAX_OUTPUT_TO_INPUT_RATIO = 6

# Catches the model breaking character - answering/generating instead
# of rewriting - by its "voice": genuine rewritten prose never talks
# about itself, but a model that slips into chatbot mode reliably does
# ("As an AI, I can explain..."). Deliberately narrow and self-
# referential only, not generic conversational openers like "Sure,
# here's..." or "I'd be happy to help" - those phrases can legitimately
# appear in real customer correspondence being rewritten (e.g. "I'd be
# happy to help you move this weekend"), so including them would reject
# genuine content, the same false-positive trap the input-side
# injection screen has to avoid. \b boundaries prevent a match inside
# an unrelated word (e.g. "served as an aide" contains "as an ai" as
# raw characters, but not as a whole word).
_ASSISTANT_VOICE_PATTERNS = tuple(
    re.compile(rf"\b{re.escape(marker)}\b", re.IGNORECASE)
    for marker in (
        "as an ai",
        "as a language model",
        "as an ai language model",
        "i'm an ai",
        "i am an ai",
        "i'm just an ai",
        "i am just an ai",
        "i don't have personal opinions",
        "i don't have personal experiences",
        "as of my last update",
        "as of my knowledge cutoff",
    )
)


def validate_ai_content(
    content: str | None,
    input_text: str,
    customer_id: int | None = None,
) -> str:
    """
    Validates and extracts the AI-generated response. Beyond "present
    and non-blank", also rejects a response that's implausibly long
    relative to the input (likely a degenerate/looping generation),
    echoes the system prompt's own instructions back instead of doing
    the rewrite (prompt leakage), reads as the model breaking character
    to answer/generate instead of rewrite (assistant voice - a scope
    violation for a rewrite-only service), or fails the same content-
    policy screen applied to input - the model can generate policy-
    violating text even from a clean input.

    All failures raise AIProviderError with an internal-only reason;
    ai_response_service.py re-raises a fresh default-message
    AIProviderError before it reaches an HTTP caller, so which specific
    check failed is never exposed externally - only logged.
    """

    if content is None:
        logger.warning(
            "AI provider returned empty response content",
            extra={"customer_id": customer_id},
        )
        raise AIProviderError("AI provider returned empty response content.")

    content = content.strip()

    if not content:
        logger.warning(
            "AI provider returned empty response content",
            extra={"customer_id": customer_id},
        )
        raise AIProviderError("AI provider returned empty response content.")

    _validate_length_ratio(content, input_text, customer_id)
    _validate_no_prompt_leakage(content, customer_id)
    _validate_no_assistant_voice(content, customer_id)
    _validate_content_policy(content, customer_id)

    return content


def _validate_length_ratio(
    content: str, input_text: str, customer_id: int | None
) -> None:
    input_length = len(input_text.strip())

    if input_length and len(content) > input_length * _MAX_OUTPUT_TO_INPUT_RATIO:
        logger.warning(
            "AI provider returned an unexpectedly long response",
            extra={"customer_id": customer_id},
        )
        raise AIProviderError(
            "AI provider returned an unexpectedly long response."
        )


def _validate_no_prompt_leakage(content: str, customer_id: int | None) -> None:
    if any(marker in content for marker in LEAKAGE_MARKERS):
        logger.warning(
            "AI provider echoed its own instructions instead of "
            "rewriting the text",
            extra={"customer_id": customer_id},
        )
        raise AIProviderError(
            "AI provider echoed its own instructions instead of "
            "rewriting the text."
        )


def _validate_no_assistant_voice(content: str, customer_id: int | None) -> None:
    if any(pattern.search(content) for pattern in _ASSISTANT_VOICE_PATTERNS):
        logger.warning(
            "AI provider responded as an assistant instead of "
            "rewriting the text",
            extra={"customer_id": customer_id},
        )
        raise AIProviderError(
            "AI provider responded as an assistant instead of "
            "rewriting the text."
        )


def _validate_content_policy(content: str, customer_id: int | None) -> None:
    result = ContentModerator.moderate(content)

    if not result.is_valid:
        logger.warning(
            "AI provider returned content that violates content policy",
            extra={"customer_id": customer_id},
        )
        raise AIProviderError(
            "AI provider returned content that violates content policy."
        )
