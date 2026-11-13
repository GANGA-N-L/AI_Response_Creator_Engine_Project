import re
from dataclasses import dataclass

from app.exceptions.ai_exceptions import AIProviderError
from app.logging.logger import logger
from app.services.pii_detector import (
    CREDIT_CARD_PATTERN,
    EMAIL_PATTERN,
    PHONE_PATTERN,
    SSN_PATTERN,
)

# The regex-shaped categories from pii_detector.py that are actually
# masked - a narrower set than that module's full visibility coverage
# (which also includes NER-based names/addresses and international
# phone parsing, neither masked - see ner_pii_detector.py). Kept
# separate from content_moderator.py: masking is a data-minimization
# measure (reduce what the AI provider sees), not a policy screen - PII
# is never a rejection reason (see content_moderator.py's own
# docstring for why).
#
# Order matters: ssn/credit_card are checked before phone since all
# three are digit-grouping patterns that can overlap on the same
# digits (e.g. a 16-digit card number) - Python's alternation takes
# the first branch that matches at a given position, so the more
# specific patterns must come first or phone would grab a fragment of
# a longer number first.
_PII_PATTERN = re.compile(
    f"(?P<email>{EMAIL_PATTERN})"
    f"|(?P<ssn>{SSN_PATTERN})"
    f"|(?P<credit_card>{CREDIT_CARD_PATTERN})"
    f"|(?P<phone>{PHONE_PATTERN})"
)

# A deliberately unnatural delimiter (section sign - essentially never
# appears in real customer correspondence) so the placeholder doesn't
# read as a formatting artifact the model might "clean up," unlike a
# bracket-style token (e.g. [EMAIL_1]).
_PLACEHOLDER_PATTERN = re.compile(r"§PII\d+§")


def _placeholder(index: int) -> str:
    return f"§PII{index}§"


@dataclass(frozen=True)
class MaskResult:
    masked_text: str
    mapping: dict[str, str]


def mask_pii(text: str) -> MaskResult:
    """
    Replaces detected email, phone-number, SSN-like, and credit-card-
    like spans with unique placeholder tokens, so the AI provider never
    sees the real values - a data-minimization measure on top of (not
    instead of) a data processing agreement with the provider.
    Heuristic/regex-based, same coverage limitations as any
    pattern-based PII detector - false negatives are expected for
    formats it doesn't recognize (see ner_pii_detector.py for the
    categories - names, addresses - that regex structurally can't
    catch).
    """

    mapping: dict[str, str] = {}

    def _replace(match: re.Match) -> str:
        placeholder = _placeholder(len(mapping))
        mapping[placeholder] = match.group(0)
        return placeholder

    masked_text = _PII_PATTERN.sub(_replace, text)

    if mapping:
        logger.info(
            "Masked PII before AI provider call",
            extra={"masked_count": len(mapping)},
        )

    return MaskResult(masked_text=masked_text, mapping=mapping)


def unmask_and_verify(
    content: str, mapping: dict[str, str], customer_id: int | None = None
) -> str:
    """
    Restores the real PII values in place of their placeholders.

    A rewrite task requires the model to reproduce every span of the
    input somewhere in its output, including masked ones - unlike
    extraction/classification tasks, there's no reliable way to force a
    hosted model to preserve an arbitrary literal token through a
    free-form rewrite. Rather than trust that it did, this verifies
    every placeholder survived unmodified before substituting real
    values back in. If any didn't survive - dropped, altered, or a
    stray placeholder-shaped token appears that wasn't one that was
    issued - the output is treated as untrustworthy and rejected via
    the same AIProviderError path used by every other output-
    validation failure, rather than returning corrupted or leaked text.
    """

    if not mapping:
        return content

    for placeholder in mapping:
        if placeholder not in content:
            logger.warning(
                "AI provider dropped or altered a masked PII placeholder",
                extra={"customer_id": customer_id},
            )
            raise AIProviderError(
                "AI provider did not preserve a masked PII placeholder."
            )

    restored = content

    for placeholder, original in mapping.items():
        restored = restored.replace(placeholder, original)

    if _PLACEHOLDER_PATTERN.search(restored):
        logger.warning(
            "AI provider output contains an unexpected PII placeholder",
            extra={"customer_id": customer_id},
        )
        raise AIProviderError(
            "AI provider output contains an unexpected PII placeholder."
        )

    return restored
