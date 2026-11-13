from unittest.mock import AsyncMock, MagicMock, patch

from openai import APIError

from app.core.config import settings
from app.services.huggingface_service import (
    check_availability,
    check_hate_speech_risk,
    check_self_harm_risk,
    generate_response,
    judge_is_rewrite,
)


async def test_generate_response():
    mock_response = MagicMock()

    mock_response.choices = [
        MagicMock(
            message=MagicMock(
                content="Thank you for contacting us."
            )
        )
    ]
    mock_response.usage = MagicMock(
        prompt_tokens=12, completion_tokens=7, total_tokens=19
    )

    with patch(
        "app.services.huggingface_service.generate_boundary_tag",
        return_value="test-boundary",
    ), patch(
        "app.services.huggingface_service.client.chat.completions.create",
        new_callable=AsyncMock,
        return_value=mock_response,
    ) as mock_create:

        response, usage = await generate_response(
            "Correct the grammar of the following text.",
            "Thanks for contacting us.",
            1,
        )

    assert response == "Thank you for contacting us."
    assert usage.prompt_tokens == 12
    assert usage.completion_tokens == 7
    assert usage.total_tokens == 19

    mock_create.assert_called_once()

    called_kwargs = mock_create.call_args.kwargs
    system_message = called_kwargs["messages"][0]["content"]
    user_message = called_kwargs["messages"][1]["content"]

    # The per-request boundary tag (see ai_prompt.generate_boundary_tag)
    # must match between the system prompt and the wrapped user text -
    # both are derived from the same generated tag.
    assert user_message == (
        "<test-boundary>\nThanks for contacting us.\n</test-boundary>"
    )
    assert "<test-boundary>" in system_message
    assert called_kwargs["max_tokens"] == 1024


async def test_generate_response_missing_usage_data():
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content="Thank you for contacting us."))
    ]
    mock_response.usage = None

    with patch(
        "app.services.huggingface_service.client.chat.completions.create",
        new_callable=AsyncMock,
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


async def test_check_availability_success():
    mock_scoped_client = MagicMock()
    mock_scoped_client.models.list = AsyncMock(return_value=MagicMock())

    with patch(
        "app.services.huggingface_service.client.with_options",
        return_value=mock_scoped_client,
    ) as mock_with_options:

        result = await check_availability()

    assert result is True
    mock_with_options.assert_called_once_with(timeout=5.0)
    mock_scoped_client.models.list.assert_called_once()


async def test_check_availability_failure():
    mock_scoped_client = MagicMock()
    mock_scoped_client.models.list = AsyncMock(
        side_effect=ConnectionError("boom")
    )

    with patch(
        "app.services.huggingface_service.client.with_options",
        return_value=mock_scoped_client,
    ):
        result = await check_availability()

    assert result is False


def _judge_response(verdict: str) -> MagicMock:
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content=verdict))
    ]
    mock_response.usage = MagicMock(
        prompt_tokens=2, completion_tokens=1, total_tokens=3
    )
    return mock_response


async def test_judge_is_rewrite_true():
    with patch(
        "app.services.huggingface_service.client.chat.completions.create",
        new_callable=AsyncMock,
        return_value=_judge_response("REWRITE"),
    ) as mock_create:
        result, usage = await judge_is_rewrite("original", "rewritten", 1)

    assert result is True
    assert usage.total_tokens == 3

    # Judge must use its own model, not hf_model (the model being
    # judged) - the whole point of a separate judge_model setting.
    assert mock_create.call_args.kwargs["model"] == settings.judge_model
    assert mock_create.call_args.kwargs["model"] != settings.hf_model


async def test_judge_is_rewrite_parses_verdict_with_trailing_tokens():
    # Verified live: some models (e.g. phi-4) correctly answer but
    # trail extra tokens within the judge's short token budget (e.g.
    # "VIOLATION\n\nThe") - startswith parsing must still catch this.
    with patch(
        "app.services.huggingface_service.client.chat.completions.create",
        new_callable=AsyncMock,
        return_value=_judge_response("VIOLATION\n\nThe"),
    ):
        result, _ = await judge_is_rewrite("original", "output", 1)

    assert result is False


async def test_judge_is_rewrite_false():
    with patch(
        "app.services.huggingface_service.client.chat.completions.create",
        new_callable=AsyncMock,
        return_value=_judge_response("VIOLATION"),
    ):
        result, usage = await judge_is_rewrite(
            "explain quantum entanglement", "Quantum entanglement is...", 1
        )

    assert result is False
    assert usage.total_tokens == 3


async def test_judge_is_rewrite_fails_open_on_api_error():
    with patch(
        "app.services.huggingface_service.client.chat.completions.create",
        new_callable=AsyncMock,
        side_effect=APIError("boom", request=MagicMock(), body=None),
    ):
        result, usage = await judge_is_rewrite("original", "rewritten", 1)

    assert result is None
    assert usage.total_tokens == 0


async def test_judge_is_rewrite_fails_open_on_no_choices():
    mock_response = MagicMock()
    mock_response.choices = []
    mock_response.usage = MagicMock(
        prompt_tokens=2, completion_tokens=0, total_tokens=2
    )

    with patch(
        "app.services.huggingface_service.client.chat.completions.create",
        new_callable=AsyncMock,
        return_value=mock_response,
    ):
        result, usage = await judge_is_rewrite("original", "rewritten", 1)

    assert result is None
    assert usage.total_tokens == 2


async def test_judge_is_rewrite_fails_open_on_unparseable_verdict():
    with patch(
        "app.services.huggingface_service.client.chat.completions.create",
        new_callable=AsyncMock,
        return_value=_judge_response("MAYBE"),
    ):
        result, usage = await judge_is_rewrite("original", "rewritten", 1)

    assert result is None
    assert usage.total_tokens == 3


async def test_check_self_harm_risk_true():
    with patch(
        "app.services.huggingface_service.client.chat.completions.create",
        new_callable=AsyncMock,
        return_value=_judge_response("RISK"),
    ) as mock_create:
        result, usage = await check_self_harm_risk(
            "I don't want to be here anymore", 1
        )

    assert result is True
    assert usage.total_tokens == 3

    # Same judge_model as judge_is_rewrite, not hf_model - see
    # ai_prompt.py's _SELF_HARM_JUDGE_INSTRUCTION_TEMPLATE docstring for
    # why this needed a judge call rather than a pattern/classifier.
    assert mock_create.call_args.kwargs["model"] == settings.judge_model
    assert mock_create.call_args.kwargs["model"] != settings.hf_model


async def test_check_self_harm_risk_false():
    with patch(
        "app.services.huggingface_service.client.chat.completions.create",
        new_callable=AsyncMock,
        return_value=_judge_response("SAFE"),
    ):
        result, usage = await check_self_harm_risk(
            "This bug is killing me, please fix it ASAP", 1
        )

    assert result is False
    assert usage.total_tokens == 3


async def test_check_self_harm_risk_parses_verdict_with_trailing_tokens():
    with patch(
        "app.services.huggingface_service.client.chat.completions.create",
        new_callable=AsyncMock,
        return_value=_judge_response("RISK\n\nThe"),
    ):
        result, _ = await check_self_harm_risk("text", 1)

    assert result is True


async def test_check_self_harm_risk_fails_open_on_api_error():
    with patch(
        "app.services.huggingface_service.client.chat.completions.create",
        new_callable=AsyncMock,
        side_effect=APIError("boom", request=MagicMock(), body=None),
    ):
        result, usage = await check_self_harm_risk("text", 1)

    assert result is None
    assert usage.total_tokens == 0


async def test_check_self_harm_risk_fails_open_on_no_choices():
    mock_response = MagicMock()
    mock_response.choices = []
    mock_response.usage = MagicMock(
        prompt_tokens=2, completion_tokens=0, total_tokens=2
    )

    with patch(
        "app.services.huggingface_service.client.chat.completions.create",
        new_callable=AsyncMock,
        return_value=mock_response,
    ):
        result, usage = await check_self_harm_risk("text", 1)

    assert result is None
    assert usage.total_tokens == 2


async def test_check_self_harm_risk_fails_open_on_unparseable_verdict():
    with patch(
        "app.services.huggingface_service.client.chat.completions.create",
        new_callable=AsyncMock,
        return_value=_judge_response("MAYBE"),
    ):
        result, usage = await check_self_harm_risk("text", 1)

    assert result is None
    assert usage.total_tokens == 3


async def test_check_hate_speech_risk_true():
    with patch(
        "app.services.huggingface_service.client.chat.completions.create",
        new_callable=AsyncMock,
        return_value=_judge_response("HATEFUL"),
    ) as mock_create:
        result, usage = await check_hate_speech_risk(
            "you people are all worthless idiots and should just die", 1
        )

    assert result is True
    assert usage.total_tokens == 3

    # Same judge_model as judge_is_rewrite/check_self_harm_risk, not
    # hf_model - see ai_prompt.py's
    # _HATE_SPEECH_JUDGE_INSTRUCTION_TEMPLATE docstring for why this
    # needed a judge call rather than a pattern/wordlist.
    assert mock_create.call_args.kwargs["model"] == settings.judge_model
    assert mock_create.call_args.kwargs["model"] != settings.hf_model


async def test_check_hate_speech_risk_false():
    with patch(
        "app.services.huggingface_service.client.chat.completions.create",
        new_callable=AsyncMock,
        return_value=_judge_response("SAFE"),
    ):
        result, usage = await check_hate_speech_risk(
            "this company is garbage and your support team is useless", 1
        )

    assert result is False
    assert usage.total_tokens == 3


async def test_check_hate_speech_risk_fails_open_on_api_error():
    with patch(
        "app.services.huggingface_service.client.chat.completions.create",
        new_callable=AsyncMock,
        side_effect=APIError("boom", request=MagicMock(), body=None),
    ):
        result, usage = await check_hate_speech_risk("text", 1)

    assert result is None
    assert usage.total_tokens == 0


async def test_check_hate_speech_risk_fails_open_on_no_choices():
    mock_response = MagicMock()
    mock_response.choices = []
    mock_response.usage = MagicMock(
        prompt_tokens=2, completion_tokens=0, total_tokens=2
    )

    with patch(
        "app.services.huggingface_service.client.chat.completions.create",
        new_callable=AsyncMock,
        return_value=mock_response,
    ):
        result, usage = await check_hate_speech_risk("text", 1)

    assert result is None
    assert usage.total_tokens == 2


async def test_check_hate_speech_risk_fails_open_on_unparseable_verdict():
    with patch(
        "app.services.huggingface_service.client.chat.completions.create",
        new_callable=AsyncMock,
        return_value=_judge_response("MAYBE"),
    ):
        result, usage = await check_hate_speech_risk("text", 1)

    assert result is None
    assert usage.total_tokens == 3
