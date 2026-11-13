from unittest.mock import MagicMock, patch

import httpx
import pytest

from app.exceptions.ai_exceptions import AIProviderError
from app.services.ollama_service import (
    _validate_ai_output,
    check_availability,
    check_hate_speech_risk,
    check_self_harm_risk,
    generate_response,
    judge_is_rewrite,
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


def test_validate_ai_output_length_ratio_violation():
    with pytest.raises(AIProviderError):
        _validate_ai_output(
            {"message": {"content": "word " * 500}}, "short input"
        )


def test_validate_ai_output_content_policy_violation():
    with pytest.raises(AIProviderError):
        _validate_ai_output(
            {"message": {"content": "You are a stupid bastard."}},
            "please rewrite this",
        )


def test_validate_ai_output_assistant_voice_rejected():
    with pytest.raises(AIProviderError):
        _validate_ai_output(
            {
                "message": {
                    "content": "As an AI, I can explain that "
                    "quantum entanglement is a phenomenon."
                }
            },
            "explain quantum entanglement to me",
        )


async def test_generate_response_success():
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "message": {"content": "Thank you for contacting us."},
        "prompt_eval_count": 12,
        "eval_count": 7,
    }

    with patch(
        "app.services.ollama_service.generate_boundary_tag",
        return_value="test-boundary",
    ), patch(
        "app.services.ollama_service.client.post",
        return_value=mock_response,
    ) as mock_post:

        response, usage = await generate_response(
            "Correct the grammar of the following text.",
            "Thanks for contacting us.",
            1,
        )

    assert response == "Thank you for contacting us."
    assert usage.prompt_tokens == 12
    assert usage.completion_tokens == 7
    assert usage.total_tokens == 19

    mock_post.assert_called_once()

    called_url = mock_post.call_args.args[0]
    called_kwargs = mock_post.call_args.kwargs

    assert called_url == "http://localhost:11434/api/chat"
    assert called_kwargs["json"]["model"] == "mistral"
    assert called_kwargs["timeout"] == 30.0

    system_message = called_kwargs["json"]["messages"][0]["content"]
    user_message = called_kwargs["json"]["messages"][1]["content"]

    assert user_message == (
        "<test-boundary>\nThanks for contacting us.\n</test-boundary>"
    )
    assert "<test-boundary>" in system_message
    assert called_kwargs["json"]["options"]["num_predict"] == 1024


async def test_generate_response_missing_usage_data():
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "message": {"content": "Thank you for contacting us."}
    }

    with patch(
        "app.services.ollama_service.client.post",
        return_value=mock_response,
    ):
        _, usage = await generate_response(
            "Correct the grammar of the following text.",
            "Thanks for contacting us.",
            1,
        )

    assert usage.prompt_tokens == 0
    assert usage.completion_tokens == 0
    assert usage.total_tokens == 0


async def test_generate_response_connection_error():
    with patch(
        "app.services.ollama_service.client.post",
        side_effect=httpx.ConnectError("boom"),
    ):
        with pytest.raises(AIProviderError):
            await generate_response("prompt", "text", 1)


async def test_generate_response_http_error():
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "500 Server Error", request=MagicMock(), response=MagicMock()
    )

    with patch(
        "app.services.ollama_service.client.post",
        return_value=mock_response,
    ):
        with pytest.raises(AIProviderError):
            await generate_response("prompt", "text", 1)


async def test_check_availability_success():
    mock_response = MagicMock(is_success=True)

    with patch(
        "app.services.ollama_service.client.get",
        return_value=mock_response,
    ) as mock_get:

        result = await check_availability()

    assert result is True

    called_url = mock_get.call_args.args[0]
    called_kwargs = mock_get.call_args.kwargs

    assert called_url == "http://localhost:11434/api/tags"
    assert called_kwargs["timeout"] == 5.0


async def test_check_availability_bad_status():
    with patch(
        "app.services.ollama_service.client.get",
        return_value=MagicMock(is_success=False),
    ):
        result = await check_availability()

    assert result is False


async def test_check_availability_connection_error():
    with patch(
        "app.services.ollama_service.client.get",
        side_effect=httpx.ConnectError("boom"),
    ):
        result = await check_availability()

    assert result is False


async def test_generate_response_invalid_json():
    mock_response = MagicMock()
    mock_response.json.side_effect = ValueError("not json")

    with patch(
        "app.services.ollama_service.client.post",
        return_value=mock_response,
    ):
        with pytest.raises(AIProviderError):
            await generate_response("prompt", "text", 1)


def _judge_response(verdict: str) -> MagicMock:
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "message": {"content": verdict},
        "prompt_eval_count": 2,
        "eval_count": 1,
    }
    return mock_response


async def test_judge_is_rewrite_true():
    with patch(
        "app.services.ollama_service.client.post",
        return_value=_judge_response("REWRITE"),
    ):
        result, usage = await judge_is_rewrite("original", "rewritten", 1)

    assert result is True
    assert usage.total_tokens == 3


async def test_judge_is_rewrite_false():
    with patch(
        "app.services.ollama_service.client.post",
        return_value=_judge_response("VIOLATION"),
    ):
        result, usage = await judge_is_rewrite(
            "explain quantum entanglement", "Quantum entanglement is...", 1
        )

    assert result is False
    assert usage.total_tokens == 3


async def test_judge_is_rewrite_fails_open_on_connection_error():
    with patch(
        "app.services.ollama_service.client.post",
        side_effect=httpx.ConnectError("boom"),
    ):
        result, usage = await judge_is_rewrite("original", "rewritten", 1)

    assert result is None
    assert usage.total_tokens == 0


async def test_judge_is_rewrite_fails_open_on_no_message():
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "prompt_eval_count": 2,
        "eval_count": 0,
    }

    with patch(
        "app.services.ollama_service.client.post",
        return_value=mock_response,
    ):
        result, usage = await judge_is_rewrite("original", "rewritten", 1)

    assert result is None
    assert usage.total_tokens == 2


async def test_judge_is_rewrite_fails_open_on_unparseable_verdict():
    with patch(
        "app.services.ollama_service.client.post",
        return_value=_judge_response("MAYBE"),
    ):
        result, usage = await judge_is_rewrite("original", "rewritten", 1)

    assert result is None
    assert usage.total_tokens == 3


async def test_check_self_harm_risk_true():
    with patch(
        "app.services.ollama_service.client.post",
        return_value=_judge_response("RISK"),
    ):
        result, usage = await check_self_harm_risk(
            "I don't want to be here anymore", 1
        )

    assert result is True
    assert usage.total_tokens == 3


async def test_check_self_harm_risk_false():
    with patch(
        "app.services.ollama_service.client.post",
        return_value=_judge_response("SAFE"),
    ):
        result, usage = await check_self_harm_risk(
            "This bug is killing me, please fix it ASAP", 1
        )

    assert result is False
    assert usage.total_tokens == 3


async def test_check_self_harm_risk_fails_open_on_connection_error():
    with patch(
        "app.services.ollama_service.client.post",
        side_effect=httpx.ConnectError("boom"),
    ):
        result, usage = await check_self_harm_risk("text", 1)

    assert result is None
    assert usage.total_tokens == 0


async def test_check_self_harm_risk_fails_open_on_no_message():
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "prompt_eval_count": 2,
        "eval_count": 0,
    }

    with patch(
        "app.services.ollama_service.client.post",
        return_value=mock_response,
    ):
        result, usage = await check_self_harm_risk("text", 1)

    assert result is None
    assert usage.total_tokens == 2


async def test_check_self_harm_risk_fails_open_on_unparseable_verdict():
    with patch(
        "app.services.ollama_service.client.post",
        return_value=_judge_response("MAYBE"),
    ):
        result, usage = await check_self_harm_risk("text", 1)

    assert result is None
    assert usage.total_tokens == 3


async def test_check_hate_speech_risk_true():
    with patch(
        "app.services.ollama_service.client.post",
        return_value=_judge_response("HATEFUL"),
    ):
        result, usage = await check_hate_speech_risk(
            "you people are all worthless idiots and should just die", 1
        )

    assert result is True
    assert usage.total_tokens == 3


async def test_check_hate_speech_risk_false():
    with patch(
        "app.services.ollama_service.client.post",
        return_value=_judge_response("SAFE"),
    ):
        result, usage = await check_hate_speech_risk(
            "this company is garbage and your support team is useless", 1
        )

    assert result is False
    assert usage.total_tokens == 3


async def test_check_hate_speech_risk_fails_open_on_connection_error():
    with patch(
        "app.services.ollama_service.client.post",
        side_effect=httpx.ConnectError("boom"),
    ):
        result, usage = await check_hate_speech_risk("text", 1)

    assert result is None
    assert usage.total_tokens == 0


async def test_check_hate_speech_risk_fails_open_on_no_message():
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "prompt_eval_count": 2,
        "eval_count": 0,
    }

    with patch(
        "app.services.ollama_service.client.post",
        return_value=mock_response,
    ):
        result, usage = await check_hate_speech_risk("text", 1)

    assert result is None
    assert usage.total_tokens == 2


async def test_check_hate_speech_risk_fails_open_on_unparseable_verdict():
    with patch(
        "app.services.ollama_service.client.post",
        return_value=_judge_response("MAYBE"),
    ):
        result, usage = await check_hate_speech_risk("text", 1)

    assert result is None
    assert usage.total_tokens == 3
