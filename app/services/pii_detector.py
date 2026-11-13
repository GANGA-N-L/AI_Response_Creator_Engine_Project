import re

import phonenumbers

# Single source of truth for PII patterns. pii_masker.py imports the
# email/phone patterns from here for masking - the two categories that
# round-trip safely through a rewrite (see pii_masker.py). The other
# categories here are visibility-only: detected and counted for a
# data-handling/DPA conversation, never masked or rejected, since
# there's no evidence yet that masking them is worth the added
# round-trip-failure risk for how rarely they actually occur.
EMAIL_PATTERN = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
PHONE_PATTERN = r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
SSN_PATTERN = r"\b\d{3}-\d{2}-\d{4}\b"
CREDIT_CARD_PATTERN = r"\b\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{1,4}\b"

_CATEGORY_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(EMAIL_PATTERN),
    "phone": re.compile(PHONE_PATTERN),
    "ssn": re.compile(SSN_PATTERN),
    "credit_card": re.compile(CREDIT_CARD_PATTERN),
}


def _has_international_phone(text: str) -> bool:
    """
    Uses phonenumbers (the Python port of Google's libphonenumber) to
    catch phone number formats PHONE_PATTERN structurally can't - it
    was shaped around the North American digit grouping, but there is
    no single format worldwide (UK, Indian, German numbers all group
    digits differently). "US" is only the fallback region for numbers
    with no explicit country code - a number written with its own "+"
    country code (e.g. "+44 20 7946 0958") is still correctly matched
    regardless. Unlike PHONE_PATTERN, this also validates plausibility
    (real area code/exchange ranges), so it won't flag an obviously
    fake number a naive shape-only regex would.
    """

    matcher = phonenumbers.PhoneNumberMatcher(text, "US")
    return next(iter(matcher), None) is not None


def detect_pii_categories(text: str) -> set[str]:
    """
    Returns the set of PII categories detected in `text`. Broader than
    pii_masker.py's masking coverage (email/phone/ssn/credit_card only)
    - used purely for visibility/metrics, never for masking or
    rejection. Same heuristic/regex limitations as any pattern-based
    detector: expect false negatives on formats these patterns don't
    recognize (see ner_pii_detector.py for names/addresses, which have
    no fixed shape a pattern could match at all).

    "international_phone" (via the phonenumbers library) can overlap
    with "phone" (PHONE_PATTERN) on plain domestic numbers too - it's
    an independent, more rigorous signal, not an exclusively-non-US one.
    """

    categories = {
        category
        for category, pattern in _CATEGORY_PATTERNS.items()
        if pattern.search(text)
    }

    if _has_international_phone(text):
        categories.add("international_phone")

    return categories
