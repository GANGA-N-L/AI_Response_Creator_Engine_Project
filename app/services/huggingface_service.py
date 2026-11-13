from openai import AsyncOpenAI, APIError

from app.core.config import settings
from app.exceptions.ai_exceptions import AIProviderError
from app.logging.logger import logger
from app.services.ai_output_validator import validate_ai_content
from app.services.ai_prompt import (
    build_hate_speech_judge_system_prompt,
    build_hate_speech_judge_user_message,
    build_judge_system_prompt,
    build_judge_user_message,
    build_self_harm_judge_system_prompt,
    build_self_harm_judge_user_message,
    build_system_prompt,
    generate_boundary_tag,
    wrap_user_text,
)
from app.services.token_usage import TokenUsage

client = AsyncOpenAI(
    base_url=settings.hf_base_url,
    api_key=settings.hf_token,
    timeout=settings.ai_request_timeout_seconds,
)

# The judge only ever answers a single word - capped independently of
# ai_max_output_tokens (the main rewrite's much larger cap), since a
# runaway judge response would be a bug in the prompt, not something
# that needs headroom.
_JUDGE_MAX_TOKENS = 5

# Caps how much of an APIError's message gets logged - it originates
# from Hugging Face's side, not customer input, but still worth
# bounding defensively rather than trusting an upstream error body to
# be a reasonable size.
_MAX_LOGGED_ERROR_MESSAGE = 500


def _truncate_for_log(message: str) -> str:
    if len(message) > _MAX_LOGGED_ERROR_MESSAGE:
        return message[:_MAX_LOGGED_ERROR_MESSAGE] + "...(truncated)"

    return message


def _api_error_log_fields(exc: APIError) -> dict:
    """
    Extracts the fields actually useful for diagnosing an APIError
    without re-running the request - status_code is None for a
    connection-level failure (no HTTP response was ever received), not
    a logging bug. Previously only `type(exc).__name__` was logged,
    which distinguishes "connection failed" from "bad status code" but
    not *which* status code or why - indistinguishable in logs whether
    the cause was an invalid HF_TOKEN, a model unavailable via the
    router, rate limiting, or a transient upstream error.
    """

    return {
        "status_code": getattr(exc, "status_code", None),
        "provider_message": _truncate_for_log(
            getattr(exc, "message", str(exc))
        ),
    }


async def check_availability() -> bool:
    """
    Lightweight reachability check for /health. Lists available models
    rather than running a real generation, with a short fail-fast
    timeout independent of ai_request_timeout_seconds.
    """

    try:
        await client.with_options(timeout=5.0).models.list()
        return True
    except Exception:
        logger.warning(
            "Hugging Face availability check failed", exc_info=True
        )
        return False


def _validate_ai_output(
    response, input_text: str = "", customer_id: int | None = None
) -> str:
    """
    Validates and extracts the AI-generated response.
    """

    if not response.choices:
        logger.warning(
            "AI provider returned no response choices",
            extra={"customer_id": customer_id},
        )
        raise AIProviderError(
            "AI provider returned no response choices."
        )

    return validate_ai_content(
        response.choices[0].message.content, input_text, customer_id
    )


def _extract_usage(response, customer_id: int | None = None) -> TokenUsage:
    usage = getattr(response, "usage", None)

    if usage is None:
        logger.warning(
            "AI provider response missing usage data",
            extra={"customer_id": customer_id},
        )
        return TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)

    return TokenUsage(
        prompt_tokens=usage.prompt_tokens or 0,
        completion_tokens=usage.completion_tokens or 0,
        total_tokens=usage.total_tokens or 0,
    )


async def generate_response(
    prompt: str,
    text: str,
    customer_id: int
) -> tuple[str, TokenUsage]:
    """
    Generates a response using Hugging Face GPT-OSS 20B.
    """

    tag = generate_boundary_tag()
    system_prompt = build_system_prompt(prompt, tag)

    try:

        response = await client.chat.completions.create(
            model=settings.hf_model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": wrap_user_text(text, tag)
                }
            ],
            temperature=0,
            max_tokens=settings.ai_max_output_tokens
        )

    except APIError as exc:

        logger.error(
            "AI provider request failed",
            extra={
                "customer_id": customer_id,
                "error_type": type(exc).__name__,
                **_api_error_log_fields(exc),
            },
        )

        raise AIProviderError() from exc

    validated = _validate_ai_output(response, text, customer_id)
    return validated, _extract_usage(response, customer_id)


async def judge_is_rewrite(
    original_text: str,
    ai_output: str,
    customer_id: int,
) -> tuple[bool | None, TokenUsage]:
    """
    Returns (verdict, usage). verdict is True if ai_output reads as a
    genuine rewrite of original_text, False if it reads as a scope
    violation (the model answered/generated instead), or None if the
    judge call itself couldn't be completed - callers should fail open
    on None (allow the response through) since this is a secondary
    quality check layered on top of the primary generation, not the
    generation itself; a flaky judge call shouldn't make the core
    product unreliable. A confirmed False, in contrast, is a real
    detected violation and should fail closed. usage is zeroed out
    only when the call itself never completed (no response was ever
    received to bill against); every other path spent real tokens.
    """

    original_tag = generate_boundary_tag()
    output_tag = generate_boundary_tag()

    try:
        response = await client.chat.completions.create(
            model=settings.judge_model,
            messages=[
                {
                    "role": "system",
                    "content": build_judge_system_prompt(original_tag, output_tag),
                },
                {
                    "role": "user",
                    "content": build_judge_user_message(
                        original_text, ai_output, original_tag, output_tag
                    ),
                },
            ],
            temperature=0,
            max_tokens=_JUDGE_MAX_TOKENS,
        )
    except APIError as exc:
        logger.warning(
            "Judge model call failed - failing open",
            extra={
                "customer_id": customer_id,
                "error_type": type(exc).__name__,
                **_api_error_log_fields(exc),
            },
            exc_info=True,
        )
        return None, TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)

    usage = _extract_usage(response, customer_id)

    if not response.choices:
        logger.warning(
            "Judge model returned no response choices - failing open",
            extra={"customer_id": customer_id},
        )
        return None, usage

    verdict = (response.choices[0].message.content or "").strip().upper()

    # startswith, not exact equality: verified live that some models
    # (e.g. phi-4) correctly answer but trail extra tokens within the
    # judge's short token budget (e.g. "VIOLATION\n\nThe") - an exact
    # match would wrongly discard a verdict the model actually got right.
    if verdict.startswith("REWRITE"):
        return True, usage

    if verdict.startswith("VIOLATION"):
        return False, usage

    logger.warning(
        "Judge model returned an unparseable verdict - failing open",
        extra={"customer_id": customer_id, "verdict": verdict},
    )
    return None, usage


async def check_self_harm_risk(
    text: str,
    customer_id: int,
) -> tuple[bool | None, TokenUsage]:
    """
    Returns (verdict, usage). verdict is True if `text` expresses
    self-harm/suicide risk, False if it reads as safe, or None if the
    judge call itself couldn't be completed - same fail-open-on-None
    contract as judge_is_rewrite. Runs on the customer's raw input
    text, before any rewrite is attempted - see ai_prompt.py's
    _SELF_HARM_JUDGE_INSTRUCTION_TEMPLATE docstring for what was
    verified live before this was wired in.
    """

    tag = generate_boundary_tag()

    try:
        response = await client.chat.completions.create(
            model=settings.judge_model,
            messages=[
                {
                    "role": "system",
                    "content": build_self_harm_judge_system_prompt(tag),
                },
                {
                    "role": "user",
                    "content": build_self_harm_judge_user_message(text, tag),
                },
            ],
            temperature=0,
            max_tokens=_JUDGE_MAX_TOKENS,
        )
    except APIError as exc:
        logger.warning(
            "Self-harm judge call failed - failing open",
            extra={
                "customer_id": customer_id,
                "error_type": type(exc).__name__,
                **_api_error_log_fields(exc),
            },
            exc_info=True,
        )
        return None, TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)

    usage = _extract_usage(response, customer_id)

    if not response.choices:
        logger.warning(
            "Self-harm judge returned no response choices - failing open",
            extra={"customer_id": customer_id},
        )
        return None, usage

    verdict = (response.choices[0].message.content or "").strip().upper()

    if verdict.startswith("RISK"):
        return True, usage

    if verdict.startswith("SAFE"):
        return False, usage

    logger.warning(
        "Self-harm judge returned an unparseable verdict - failing open",
        extra={"customer_id": customer_id, "verdict": verdict},
    )
    return None, usage


async def check_hate_speech_risk(
    text: str,
    customer_id: int,
) -> tuple[bool | None, TokenUsage]:
    """
    Returns (verdict, usage). verdict is True if `text` expresses
    hateful/dehumanizing hostility toward a person or group, False if
    it reads as safe, or None if the judge call itself couldn't be
    completed - same fail-open-on-None contract as judge_is_rewrite
    and check_self_harm_risk. Runs on the customer's raw input text,
    before any rewrite is attempted - see ai_prompt.py's
    _HATE_SPEECH_JUDGE_INSTRUCTION_TEMPLATE docstring for what was
    verified live before this was wired in.
    """

    tag = generate_boundary_tag()

    try:
        response = await client.chat.completions.create(
            model=settings.judge_model,
            messages=[
                {
                    "role": "system",
                    "content": build_hate_speech_judge_system_prompt(tag),
                },
                {
                    "role": "user",
                    "content": build_hate_speech_judge_user_message(text, tag),
                },
            ],
            temperature=0,
            max_tokens=_JUDGE_MAX_TOKENS,
        )
    except APIError as exc:
        logger.warning(
            "Hate-speech judge call failed - failing open",
            extra={
                "customer_id": customer_id,
                "error_type": type(exc).__name__,
                **_api_error_log_fields(exc),
            },
            exc_info=True,
        )
        return None, TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)

    usage = _extract_usage(response, customer_id)

    if not response.choices:
        logger.warning(
            "Hate-speech judge returned no response choices - failing open",
            extra={"customer_id": customer_id},
        )
        return None, usage

    verdict = (response.choices[0].message.content or "").strip().upper()

    if verdict.startswith("HATEFUL"):
        return True, usage

    if verdict.startswith("SAFE"):
        return False, usage

    logger.warning(
        "Hate-speech judge returned an unparseable verdict - failing open",
        extra={"customer_id": customer_id, "verdict": verdict},
    )
    return None, usage
