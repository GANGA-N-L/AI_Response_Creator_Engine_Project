import pytest

from app.exceptions.ai_exceptions import AIProviderError
from app.services.pii_masker import mask_pii, unmask_and_verify


def test_mask_pii_detects_email():
    result = mask_pii("Contact me at jane@example.com please.")

    assert "jane@example.com" not in result.masked_text
    assert "§PII0§" in result.masked_text
    assert result.mapping == {"§PII0§": "jane@example.com"}


def test_mask_pii_detects_phone():
    result = mask_pii("Call me at 555-123-4567 today.")

    assert "555-123-4567" not in result.masked_text
    assert len(result.mapping) == 1


def test_mask_pii_detects_ssn():
    result = mask_pii("My SSN is 123-45-6789, please rewrite this politely.")

    assert "123-45-6789" not in result.masked_text
    assert len(result.mapping) == 1


def test_mask_pii_detects_credit_card():
    result = mask_pii("Charge card 4111 1111 1111 1111 for this order.")

    assert "4111 1111 1111 1111" not in result.masked_text
    assert len(result.mapping) == 1


def test_mask_pii_detects_multiple_items_with_distinct_placeholders():
    result = mask_pii("Email jane@example.com or call 555-123-4567.")

    assert len(result.mapping) == 2
    assert "§PII0§" in result.masked_text
    assert "§PII1§" in result.masked_text


def test_mask_pii_no_pii_present():
    result = mask_pii("Please make this sound more professional.")

    assert result.masked_text == "Please make this sound more professional."
    assert result.mapping == {}


def test_unmask_and_verify_restores_original_values():
    mapping = {"§PII0§": "jane@example.com"}

    restored = unmask_and_verify("Reach out at §PII0§ anytime.", mapping)

    assert restored == "Reach out at jane@example.com anytime."


def test_unmask_and_verify_restores_multiple_values():
    mapping = {"§PII0§": "jane@example.com", "§PII1§": "555-123-4567"}

    restored = unmask_and_verify(
        "Email §PII0§ or call §PII1§ to reach her.", mapping
    )

    assert restored == "Email jane@example.com or call 555-123-4567 to reach her."


def test_unmask_and_verify_no_mapping_returns_content_unchanged():
    assert (
        unmask_and_verify("Nothing to restore here.", {})
        == "Nothing to restore here."
    )


def test_unmask_and_verify_raises_when_placeholder_missing():
    mapping = {"§PII0§": "jane@example.com"}

    with pytest.raises(AIProviderError):
        unmask_and_verify("Reach out anytime.", mapping)


def test_unmask_and_verify_raises_on_stray_placeholder():
    mapping = {"§PII0§": "jane@example.com"}

    content = "Reach out at §PII0§ or §PII9§ anytime."

    with pytest.raises(AIProviderError):
        unmask_and_verify(content, mapping)
