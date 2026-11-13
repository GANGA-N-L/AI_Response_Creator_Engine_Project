import pytest
from unittest.mock import MagicMock

from app.services.huggingface_service import _validate_ai_output
from app.exceptions.ai_exceptions import AIProviderError


def test_no_choices():
    response = MagicMock()
    response.choices = []

    with pytest.raises(AIProviderError):
        _validate_ai_output(response)


def test_none_content():
    response = MagicMock()
    response.choices = [
        MagicMock(
            message=MagicMock(content=None)
        )
    ]

    with pytest.raises(AIProviderError):
        _validate_ai_output(response)


def test_empty_content():
    response = MagicMock()
    response.choices = [
        MagicMock(
            message=MagicMock(content="")
        )
    ]

    with pytest.raises(AIProviderError):
        _validate_ai_output(response)


def test_whitespace_content():
    response = MagicMock()
    response.choices = [
        MagicMock(
            message=MagicMock(content="   ")
        )
    ]

    with pytest.raises(AIProviderError):
        _validate_ai_output(response)


def test_valid_content():
    response = MagicMock()
    response.choices = [
        MagicMock(
            message=MagicMock(
                content="  Thanks for your help.  "
            )
        )
    ]

    result = _validate_ai_output(response)

    assert result == "Thanks for your help."


def test_length_ratio_violation_rejected():
    response = MagicMock()
    response.choices = [
        MagicMock(
            message=MagicMock(content="word " * 500)
        )
    ]

    with pytest.raises(AIProviderError):
        _validate_ai_output(response, "short input")


def test_length_ratio_within_bound_accepted():
    response = MagicMock()
    response.choices = [
        MagicMock(
            message=MagicMock(content="A somewhat longer rewritten sentence.")
        )
    ]

    result = _validate_ai_output(response, "short input")

    assert result == "A somewhat longer rewritten sentence."


def test_prompt_leakage_rejected():
    response = MagicMock()
    response.choices = [
        MagicMock(
            message=MagicMock(
                content="Do NOT answer the user's question."
            )
        )
    ]

    with pytest.raises(AIProviderError):
        _validate_ai_output(response, "please help me")


def test_content_policy_violation_rejected():
    response = MagicMock()
    response.choices = [
        MagicMock(
            message=MagicMock(content="You are a stupid bastard.")
        )
    ]

    with pytest.raises(AIProviderError):
        _validate_ai_output(response, "please rewrite this")


def test_assistant_voice_rejected():
    response = MagicMock()
    response.choices = [
        MagicMock(
            message=MagicMock(
                content="As an AI, I can explain that quantum "
                "entanglement is a phenomenon where two particles "
                "become linked."
            )
        )
    ]

    with pytest.raises(AIProviderError):
        _validate_ai_output(response, "explain quantum entanglement to me")


def test_assistant_voice_does_not_false_positive_on_similar_words():
    response = MagicMock()
    response.choices = [
        MagicMock(
            message=MagicMock(
                content="He served as an aide to the senator for "
                "many years."
            )
        )
    ]

    result = _validate_ai_output(
        response, "he served as an aid to the senator for years"
    )

    assert result == (
        "He served as an aide to the senator for many years."
    )