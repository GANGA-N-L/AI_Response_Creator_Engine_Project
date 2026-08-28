from unittest.mock import MagicMock, patch

import pytest
import requests

from app.exceptions.ai_exceptions import AIProviderError
from app.services.ollama_service import (
    _validate_ai_output,
    check_availability,
    generate_response,
)


def test_validate_ai_output_no_message():
    with pytest.raises(AIProviderError):
        _validate_ai_output({})


def test_validate_ai_output_none_content():
    with pytest.raises(AIProviderError):
        _validate_ai_output({"message": {"content": None}})


def test_validate_ai_output_empty_content():
    with pytest.raises(AIProviderError):
        _validate_ai_output({"message": {"content": ""}})


def test_validate_ai_output_whitespace_content():
    with pytest.raises(AIProviderError):
        _validate_ai_output({"message": {"content": "   "}})


def test_validate_ai_output_valid_content():
    result = _validate_ai_output(
        {"message": {"content": "  Thanks for contacting us.  "}}
    )

    assert result == "Thanks for contacting us."


def test_generate_response_success():
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "message": {"content": "Thank you for contacting us."}
    }

    with patch(
        "app.services.ollama_service.requests.post",
        return_value=mock_response,
    ) as mock_post:

        response = generate_response(
            "Correct the grammar of the following text.",
            "Thanks for contacting us.",
            1,
        )

    assert response == "Thank you for contacting us."

    mock_post.assert_called_once()

    called_url = mock_post.call_args.args[0]
    called_kwargs = mock_post.call_args.kwargs

    assert called_url == "http://localhost:11434/api/chat"
    assert called_kwargs["json"]["model"] == "mistral"
    assert called_kwargs["timeout"] == 30.0


def test_generate_response_connection_error():
    with patch(
        "app.services.ollama_service.requests.post",
        side_effect=requests.exceptions.ConnectionError("boom"),
    ):
        with pytest.raises(AIProviderError):
            generate_response("prompt", "text", 1)


def test_generate_response_http_error():
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
        "500 Server Error"
    )

    with patch(
        "app.services.ollama_service.requests.post",
        return_value=mock_response,
    ):
        with pytest.raises(AIProviderError):
            generate_response("prompt", "text", 1)


def test_check_availability_success():
    mock_response = MagicMock(ok=True)

    with patch(
        "app.services.ollama_service.requests.get",
        return_value=mock_response,
    ) as mock_get:

        result = check_availability()

    assert result is True

    called_url = mock_get.call_args.args[0]
    called_kwargs = mock_get.call_args.kwargs

    assert called_url == "http://localhost:11434/api/tags"
    assert called_kwargs["timeout"] == 5.0


def test_check_availability_bad_status():
    with patch(
        "app.services.ollama_service.requests.get",
        return_value=MagicMock(ok=False),
    ):
        result = check_availability()

    assert result is False


def test_check_availability_connection_error():
    with patch(
        "app.services.ollama_service.requests.get",
        side_effect=requests.exceptions.ConnectionError("boom"),
    ):
        result = check_availability()

    assert result is False


def test_generate_response_invalid_json():
    mock_response = MagicMock()
    mock_response.json.side_effect = ValueError("not json")

    with patch(
        "app.services.ollama_service.requests.post",
        return_value=mock_response,
    ):
        with pytest.raises(AIProviderError):
            generate_response("prompt", "text", 1)
