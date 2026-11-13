import asyncio

from transformers import pipeline

from app.logging.logger import logger

# Below this contradiction-label probability, the pair is treated as
# consistent. Verified live: faithful rewrites (including a tone
# change, which the model scored "neutral" rather than "entailment" -
# still fine, since only high-confidence contradiction rejects) scored
# well under 0.01; a changed date and a flipped negation ("I agree" ->
# "I do not agree") both scored above 0.98. Wide margin either side of
# 0.5, so this isn't a finely-tuned knife-edge threshold.
_CONTRADICTION_THRESHOLD = 0.5

# Loaded once at import time, same precedent as ner_pii_detector.py's
# spaCy model - this is a small, dedicated NLI (Natural Language
# Inference) model, not the general-purpose judge LLM: classifying
# whether one sentence contradicts another is a narrower, better-
# matched task for a purpose-built entailment model than for a chat
# model reasoning about it in a handful of tokens, and it runs
# locally - no AI provider cost, no added per-request billing, unlike
# judge_is_rewrite.
try:
    _classifier = pipeline(
        "text-classification",
        model="cross-encoder/nli-MiniLM2-L6-H768",
        top_k=None,
        # Skips a network round-trip to check for a newer version of an
        # already-downloaded model - the model is expected to already
        # be cached (see CLAUDE.md's environment setup), same precedent
        # as en_core_web_sm not being fetched at runtime either. Roughly
        # halves this module's contribution to app startup time.
        local_files_only=True,
    )
except Exception:
    _classifier = None
    logger.warning(
        "NLI model 'cross-encoder/nli-MiniLM2-L6-H768' failed to load "
        "- factual-consistency checking is disabled.",
        exc_info=True,
    )


def _contradiction_score(original_text: str, ai_output: str) -> float:
    scores = _classifier({"text": original_text, "text_pair": ai_output})
    return next(s["score"] for s in scores if s["label"] == "contradiction")


async def check_factual_consistency(
    original_text: str,
    ai_output: str,
) -> bool | None:
    """
    Returns False if ai_output contradicts original_text - e.g. a
    changed date, amount, or a flipped meaning ("I agree" -> "I do not
    agree") - a category of distortion neither the length-ratio check
    nor judge_is_rewrite is built to catch (judge_is_rewrite asks "is
    this a rewrite at all", not "did it silently change a fact while
    rewriting"). Returns True if no contradiction is detected, or None
    if the model isn't available (load failure) or the check itself
    errors - callers should fail open on None, same contract as
    judge_is_rewrite, rather than block the core product on a
    secondary check.

    Runs in a thread executor: this is CPU-bound local inference
    (~150ms), and awaiting it directly would block the event loop for
    that whole duration, undercutting the point of this being an
    async codebase.
    """

    if _classifier is None:
        return None

    try:
        loop = asyncio.get_running_loop()
        score = await loop.run_in_executor(
            None, _contradiction_score, original_text, ai_output
        )
    except Exception:
        logger.warning(
            "Factual-consistency check failed - failing open",
            exc_info=True,
        )
        return None

    return score < _CONTRADICTION_THRESHOLD
