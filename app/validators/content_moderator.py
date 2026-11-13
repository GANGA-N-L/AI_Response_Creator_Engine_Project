import re

from better_profanity import profanity

from app.models.validation_result import ValidationResult

profanity.load_censor_words()

# Self-harm/violence threat phrasing isn't reliably covered by a
# profanity wordlist (these are ordinary words used threateningly, not
# obscenities), so it's checked separately with a small pattern set.
_THREAT_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bkill\s+(?:yourself|urself|ur\s*self)\b",
        r"\b(?:i'?m going to|i will|i'?ll)\s+(?:kill|hurt|attack|murder)\s+you\b",
        r"\bgo\s+kill\s+yourself\b",
    )
)


class ContentModerator:
    """
    Heuristic, pattern-based content-policy screen for text going into
    or out of the AI provider (profanity/hate speech via the
    better-profanity wordlist, plus a small set of violence/self-harm
    threat patterns). This is a first line of defense, not a trained
    classifier - false negatives are expected for anything not covered
    by the wordlist or these patterns.

    Deliberately does NOT flag PII (emails, phone numbers, etc.): this
    service rewrites real customer correspondence, which routinely and
    legitimately contains contact details - treating that as a
    rejection reason would break the product's core use case rather
    than protect anyone.
    """

    _VALID = ValidationResult(is_valid=True, error_code="", reason="")

    @staticmethod
    def moderate(text: str) -> ValidationResult:
        if profanity.contains_profanity(text):
            return ValidationResult(
                is_valid=False,
                error_code="INAPPROPRIATE_CONTENT",
                reason="Input contains content that violates content policy.",
            )

        if any(pattern.search(text) for pattern in _THREAT_PATTERNS):
            return ValidationResult(
                is_valid=False,
                error_code="INAPPROPRIATE_CONTENT",
                reason="Input contains content that violates content policy.",
            )

        return ContentModerator._VALID
