import pytest

from app.validators.content_moderator import ContentModerator


@pytest.mark.parametrize(
    "text",
    [
        "You are a stupid bastard and I hate you.",
        "Kill yourself, nobody wants you here.",
        "I will kill you if you do that again.",
    ],
)
def test_moderate_rejects_policy_violating_content(text):
    result = ContentModerator.moderate(text)

    assert result.is_valid is False
    assert result.error_code == "INAPPROPRIATE_CONTENT"


@pytest.mark.parametrize(
    "text",
    [
        "Please rewrite this email to sound more professional.",
        "Contact me at jane.doe@example.com or 555-123-4567.",
        "The quarterly report is due Friday, please review it.",
        "I'm really frustrated with this delay, please help.",
    ],
)
def test_moderate_accepts_ordinary_and_pii_bearing_content(text):
    result = ContentModerator.moderate(text)

    assert result.is_valid is True
    assert result.error_code == ""
