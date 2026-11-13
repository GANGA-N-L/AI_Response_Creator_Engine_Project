from app.services.pii_detector import detect_pii_categories


def test_detects_email():
    assert detect_pii_categories("Contact jane@example.com") == {"email"}


def test_detects_phone():
    assert detect_pii_categories("Call 555-123-4567") == {"phone"}


def test_detects_ssn():
    assert detect_pii_categories("SSN: 123-45-6789") == {"ssn"}


def test_detects_credit_card():
    assert detect_pii_categories("Card: 4111 1111 1111 1111") == {"credit_card"}


def test_detects_international_phone_not_caught_by_regex():
    # UK-format number: PHONE_PATTERN is shaped around NANP grouping and
    # doesn't match this, so "phone" should NOT also be present here.
    categories = detect_pii_categories("Call +44 20 7946 0958")

    assert categories == {"international_phone"}


def test_phone_pattern_rejects_implausible_number_phonenumbers_also_rejects():
    # 555 exchanges are reserved/fictional in NANP - phonenumbers
    # validates plausibility and correctly excludes it, unlike the
    # shape-only regex, which still matches it.
    categories = detect_pii_categories("Call 555-123-4567")

    assert categories == {"phone"}


def test_detects_multiple_categories():
    text = "Email jane@example.com or call 555-123-4567, SSN 123-45-6789."

    categories = detect_pii_categories(text)

    assert {"email", "phone", "ssn"}.issubset(categories)


def test_no_pii_returns_empty_set():
    assert detect_pii_categories("Please make this sound more professional.") == set()
