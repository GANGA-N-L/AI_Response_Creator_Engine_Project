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