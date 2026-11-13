from app.core.config import settings
from app.logging.logger import logger
from app.services import huggingface_service, ollama_service
from app.services.token_usage import TokenUsage


async def generate_response(
    prompt: str,
    text: str,
    customer_id: int
) -> tuple[str, TokenUsage]:
    """
    Generates a response using the AI provider configured via
    settings.ai_provider.
    """

    if settings.ai_provider == "ollama":
        return await ollama_service.generate_response(prompt, text, customer_id)

    return await huggingface_service.generate_response(prompt, text, customer_id)


async def check_availability() -> bool:
    """
    Checks whether the configured AI provider is currently reachable.
    """

    if settings.ai_provider == "ollama":
        return await ollama_service.check_availability()

    return await huggingface_service.check_availability()


async def _dispatch_judge(
    original_text: str, ai_output: str, customer_id: int
) -> tuple[bool | None, TokenUsage]:
    if settings.ai_provider == "ollama":
        return await ollama_service.judge_is_rewrite(
            original_text, ai_output, customer_id
        )

    return await huggingface_service.judge_is_rewrite(
        original_text, ai_output, customer_id
    )


async def judge_is_rewrite(
    original_text: str,
    ai_output: str,
    customer_id: int,
) -> tuple[bool | None, TokenUsage]:
    """
    Dispatches to the configured provider's judge_is_rewrite - see
    huggingface_service.judge_is_rewrite's docstring for the
    (verdict, usage) contract - retrying once on a False verdict before
    trusting it. Returned usage is the sum of both calls' tokens when a
    retry happens - both are real spend against the AI provider,
    regardless of which verdict ends up trusted.

    Verified live: identical input, temperature=0, 5 consecutive calls
    to the judge produced 3 VIOLATION and 2 REWRITE verdicts - hosted
    GPU inference doesn't guarantee determinism even at temperature=0
    (batching effects across concurrent requests on shared
    infrastructure), so a single False isn't reliable enough on its own
    to fail closed on what may well be a legitimate rewrite. The
    retry's verdict is trusted outright -
    including a second False (now a real, repeated signal worth
    failing closed on) or a None (fails open, same as any other judge-
    call failure) - rather than compounding retries further.
    """

    verdict, usage = await _dispatch_judge(original_text, ai_output, customer_id)

    if verdict is not False:
        return verdict, usage

    logger.warning(
        "Judge model flagged a scope violation on first attempt - "
        "retrying once before failing closed",
        extra={"customer_id": customer_id},
    )

    retry_verdict, retry_usage = await _dispatch_judge(
        original_text, ai_output, customer_id
    )
    return retry_verdict, usage + retry_usage


async def _dispatch_self_harm_judge(
    text: str, customer_id: int
) -> tuple[bool | None, TokenUsage]:
    if settings.ai_provider == "ollama":
        return await ollama_service.check_self_harm_risk(text, customer_id)

    return await huggingface_service.check_self_harm_risk(text, customer_id)


async def check_self_harm_risk(
    text: str,
    customer_id: int,
) -> tuple[bool | None, TokenUsage]:
    """
    Dispatches to the configured provider's check_self_harm_risk,
    retrying once on a True (risk) verdict before trusting it - the
    same nondeterminism mitigation judge_is_rewrite already needed on
    this same model/infrastructure (hosted GPU batching effects mean
    temperature=0 doesn't guarantee identical verdicts across calls -
    see judge_is_rewrite's docstring). Applied here by that same
    reasoning, not independently re-verified for this specific prompt:
    a lone RISK verdict shouldn't fail closed and reject a customer's
    request on what might be a one-off nondeterministic flip, but a
    second RISK verdict is a repeated signal worth trusting - a False
    or None retry result is trusted outright, same as judge_is_rewrite.
    """

    verdict, usage = await _dispatch_self_harm_judge(text, customer_id)

    if verdict is not True:
        return verdict, usage

    logger.warning(
        "Self-harm judge flagged risk on first attempt - retrying once "
        "before failing closed",
        extra={"customer_id": customer_id},
    )

    retry_verdict, retry_usage = await _dispatch_self_harm_judge(
        text, customer_id
    )
    return retry_verdict, usage + retry_usage


async def _dispatch_hate_speech_judge(
    text: str, customer_id: int
) -> tuple[bool | None, TokenUsage]:
    if settings.ai_provider == "ollama":
        return await ollama_service.check_hate_speech_risk(text, customer_id)

    return await huggingface_service.check_hate_speech_risk(text, customer_id)


async def check_hate_speech_risk(
    text: str,
    customer_id: int,
) -> tuple[bool | None, TokenUsage]:
    """
    Dispatches to the configured provider's check_hate_speech_risk,
    retrying once on a True (hateful) verdict before trusting it - same
    nondeterminism mitigation as check_self_harm_risk/judge_is_rewrite
    on this same model/infrastructure, applied here by that same
    reasoning rather than independently re-verified for this specific
    prompt.
    """

    verdict, usage = await _dispatch_hate_speech_judge(text, customer_id)

    if verdict is not True:
        return verdict, usage

    logger.warning(
        "Hate-speech judge flagged risk on first attempt - retrying "
        "once before failing closed",
        extra={"customer_id": customer_id},
    )

    retry_verdict, retry_usage = await _dispatch_hate_speech_judge(
        text, customer_id
    )
    return retry_verdict, usage + retry_usage
