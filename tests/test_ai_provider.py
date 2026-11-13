from unittest.mock import MagicMock, patch

from app.services.ai_provider import (
    check_availability,
    check_hate_speech_risk,
    check_self_harm_risk,
    generate_response,
    judge_is_rewrite,
)
from app.services.token_usage import TokenUsage

_USAGE = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)


async def test_generate_response_uses_huggingface_by_default():
    with patch(
        "app.services.ai_provider.settings",
        MagicMock(ai_provider="huggingface"),
    ), patch(
        "app.services.huggingface_service.generate_response",
        return_value=("hf answer", _USAGE),
    ) as mock_hf, patch(
        "app.services.ollama_service.generate_response",
    ) as mock_ollama:

        result = await generate_response("prompt", "text", 1)

    assert result == ("hf answer", _USAGE)
    mock_hf.assert_called_once_with("prompt", "text", 1)
    mock_ollama.assert_not_called()


async def test_generate_response_uses_ollama_when_configured():
    with patch(
        "app.services.ai_provider.settings",
        MagicMock(ai_provider="ollama"),
    ), patch(
        "app.services.ollama_service.generate_response",
        return_value=("ollama answer", _USAGE),
    ) as mock_ollama, patch(
        "app.services.huggingface_service.generate_response",
    ) as mock_hf:

        result = await generate_response("prompt", "text", 1)

    assert result == ("ollama answer", _USAGE)
    mock_ollama.assert_called_once_with("prompt", "text", 1)
    mock_hf.assert_not_called()


async def test_check_availability_uses_huggingface_by_default():
    with patch(
        "app.services.ai_provider.settings",
        MagicMock(ai_provider="huggingface"),
    ), patch(
        "app.services.huggingface_service.check_availability",
        return_value=True,
    ) as mock_hf, patch(
        "app.services.ollama_service.check_availability",
    ) as mock_ollama:

        result = await check_availability()

    assert result is True
    mock_hf.assert_called_once()
    mock_ollama.assert_not_called()


async def test_check_availability_uses_ollama_when_configured():
    with patch(
        "app.services.ai_provider.settings",
        MagicMock(ai_provider="ollama"),
    ), patch(
        "app.services.ollama_service.check_availability",
        return_value=False,
    ) as mock_ollama, patch(
        "app.services.huggingface_service.check_availability",
    ) as mock_hf:

        result = await check_availability()

    assert result is False
    mock_ollama.assert_called_once()
    mock_hf.assert_not_called()


_JUDGE_CALL_USAGE = TokenUsage(prompt_tokens=2, completion_tokens=1, total_tokens=3)


async def test_judge_is_rewrite_uses_huggingface_by_default():
    with patch(
        "app.services.ai_provider.settings",
        MagicMock(ai_provider="huggingface"),
    ), patch(
        "app.services.huggingface_service.judge_is_rewrite",
        return_value=(True, _JUDGE_CALL_USAGE),
    ) as mock_hf, patch(
        "app.services.ollama_service.judge_is_rewrite",
    ) as mock_ollama:

        result, usage = await judge_is_rewrite("original", "output", 1)

    assert result is True
    assert usage == _JUDGE_CALL_USAGE
    mock_hf.assert_called_once_with("original", "output", 1)
    mock_ollama.assert_not_called()


async def test_judge_is_rewrite_uses_ollama_when_configured():
    # Both calls return False (no flip on retry) - the dispatch route
    # (ollama vs huggingface) is what this test verifies, not the
    # retry mechanics themselves - see the dedicated retry tests below.
    with patch(
        "app.services.ai_provider.settings",
        MagicMock(ai_provider="ollama"),
    ), patch(
        "app.services.ollama_service.judge_is_rewrite",
        return_value=(False, _JUDGE_CALL_USAGE),
    ) as mock_ollama, patch(
        "app.services.huggingface_service.judge_is_rewrite",
    ) as mock_hf:

        result, usage = await judge_is_rewrite("original", "output", 1)

    assert result is False
    assert usage == _JUDGE_CALL_USAGE + _JUDGE_CALL_USAGE
    assert mock_ollama.call_count == 2
    mock_ollama.assert_called_with("original", "output", 1)
    mock_hf.assert_not_called()


async def test_judge_is_rewrite_retries_once_on_violation_verdict():
    # A lone False isn't trusted outright - verified live that identical
    # judge calls at temperature=0 can flip between REWRITE/VIOLATION on
    # the same input (hosted-inference nondeterminism), so a first False
    # triggers exactly one retry rather than failing closed immediately.
    with patch(
        "app.services.ai_provider.settings",
        MagicMock(ai_provider="huggingface"),
    ), patch(
        "app.services.huggingface_service.judge_is_rewrite",
        side_effect=[(False, _JUDGE_CALL_USAGE), (True, _JUDGE_CALL_USAGE)],
    ) as mock_hf:

        result, usage = await judge_is_rewrite("original", "output", 1)

    assert result is True
    # Both the original and retry call spent real tokens - both count.
    assert usage == _JUDGE_CALL_USAGE + _JUDGE_CALL_USAGE
    assert mock_hf.call_count == 2


async def test_judge_is_rewrite_fails_closed_on_repeated_violation():
    with patch(
        "app.services.ai_provider.settings",
        MagicMock(ai_provider="huggingface"),
    ), patch(
        "app.services.huggingface_service.judge_is_rewrite",
        side_effect=[(False, _JUDGE_CALL_USAGE), (False, _JUDGE_CALL_USAGE)],
    ) as mock_hf:

        result, usage = await judge_is_rewrite("original", "output", 1)

    assert result is False
    assert usage == _JUDGE_CALL_USAGE + _JUDGE_CALL_USAGE
    assert mock_hf.call_count == 2


async def test_judge_is_rewrite_retry_fails_open_on_none():
    with patch(
        "app.services.ai_provider.settings",
        MagicMock(ai_provider="huggingface"),
    ), patch(
        "app.services.huggingface_service.judge_is_rewrite",
        side_effect=[
            (False, _JUDGE_CALL_USAGE),
            (None, TokenUsage(0, 0, 0)),
        ],
    ) as mock_hf:

        result, usage = await judge_is_rewrite("original", "output", 1)

    assert result is None
    assert usage == _JUDGE_CALL_USAGE
    assert mock_hf.call_count == 2


async def test_judge_is_rewrite_does_not_retry_on_true_verdict():
    with patch(
        "app.services.ai_provider.settings",
        MagicMock(ai_provider="huggingface"),
    ), patch(
        "app.services.huggingface_service.judge_is_rewrite",
        return_value=(True, _JUDGE_CALL_USAGE),
    ) as mock_hf:

        result, usage = await judge_is_rewrite("original", "output", 1)

    assert result is True
    assert usage == _JUDGE_CALL_USAGE
    mock_hf.assert_called_once_with("original", "output", 1)


async def test_judge_is_rewrite_does_not_retry_on_none_verdict():
    with patch(
        "app.services.ai_provider.settings",
        MagicMock(ai_provider="huggingface"),
    ), patch(
        "app.services.huggingface_service.judge_is_rewrite",
        return_value=(None, TokenUsage(0, 0, 0)),
    ) as mock_hf:

        result, usage = await judge_is_rewrite("original", "output", 1)

    assert result is None
    assert usage == TokenUsage(0, 0, 0)
    mock_hf.assert_called_once_with("original", "output", 1)


async def test_check_self_harm_risk_uses_huggingface_by_default():
    with patch(
        "app.services.ai_provider.settings",
        MagicMock(ai_provider="huggingface"),
    ), patch(
        "app.services.huggingface_service.check_self_harm_risk",
        return_value=(False, _JUDGE_CALL_USAGE),
    ) as mock_hf, patch(
        "app.services.ollama_service.check_self_harm_risk",
    ) as mock_ollama:

        result, usage = await check_self_harm_risk("text", 1)

    assert result is False
    assert usage == _JUDGE_CALL_USAGE
    mock_hf.assert_called_once_with("text", 1)
    mock_ollama.assert_not_called()


async def test_check_self_harm_risk_uses_ollama_when_configured():
    # Both calls return True (no flip on retry) - the dispatch route
    # (ollama vs huggingface) is what this test verifies, not the retry
    # mechanics themselves - see the dedicated retry tests below.
    with patch(
        "app.services.ai_provider.settings",
        MagicMock(ai_provider="ollama"),
    ), patch(
        "app.services.ollama_service.check_self_harm_risk",
        return_value=(True, _JUDGE_CALL_USAGE),
    ) as mock_ollama, patch(
        "app.services.huggingface_service.check_self_harm_risk",
    ) as mock_hf:

        result, usage = await check_self_harm_risk("text", 1)

    assert result is True
    assert usage == _JUDGE_CALL_USAGE + _JUDGE_CALL_USAGE
    assert mock_ollama.call_count == 2
    mock_ollama.assert_called_with("text", 1)
    mock_hf.assert_not_called()


async def test_check_self_harm_risk_retries_once_on_risk_verdict():
    # Mirrors judge_is_rewrite's nondeterminism mitigation (see that
    # test's comment) - applied here on the same model/infra, not
    # independently re-verified for this specific prompt. A lone RISK
    # verdict triggers exactly one retry rather than failing closed
    # immediately.
    with patch(
        "app.services.ai_provider.settings",
        MagicMock(ai_provider="huggingface"),
    ), patch(
        "app.services.huggingface_service.check_self_harm_risk",
        side_effect=[(True, _JUDGE_CALL_USAGE), (False, _JUDGE_CALL_USAGE)],
    ) as mock_hf:

        result, usage = await check_self_harm_risk("text", 1)

    assert result is False
    assert usage == _JUDGE_CALL_USAGE + _JUDGE_CALL_USAGE
    assert mock_hf.call_count == 2


async def test_check_self_harm_risk_fails_closed_on_repeated_risk():
    with patch(
        "app.services.ai_provider.settings",
        MagicMock(ai_provider="huggingface"),
    ), patch(
        "app.services.huggingface_service.check_self_harm_risk",
        side_effect=[(True, _JUDGE_CALL_USAGE), (True, _JUDGE_CALL_USAGE)],
    ) as mock_hf:

        result, usage = await check_self_harm_risk("text", 1)

    assert result is True
    assert usage == _JUDGE_CALL_USAGE + _JUDGE_CALL_USAGE
    assert mock_hf.call_count == 2


async def test_check_self_harm_risk_retry_fails_open_on_none():
    with patch(
        "app.services.ai_provider.settings",
        MagicMock(ai_provider="huggingface"),
    ), patch(
        "app.services.huggingface_service.check_self_harm_risk",
        side_effect=[
            (True, _JUDGE_CALL_USAGE),
            (None, TokenUsage(0, 0, 0)),
        ],
    ) as mock_hf:

        result, usage = await check_self_harm_risk("text", 1)

    assert result is None
    assert usage == _JUDGE_CALL_USAGE
    assert mock_hf.call_count == 2


async def test_check_self_harm_risk_does_not_retry_on_false_verdict():
    with patch(
        "app.services.ai_provider.settings",
        MagicMock(ai_provider="huggingface"),
    ), patch(
        "app.services.huggingface_service.check_self_harm_risk",
        return_value=(False, _JUDGE_CALL_USAGE),
    ) as mock_hf:

        result, usage = await check_self_harm_risk("text", 1)

    assert result is False
    assert usage == _JUDGE_CALL_USAGE
    mock_hf.assert_called_once_with("text", 1)


async def test_check_self_harm_risk_does_not_retry_on_none_verdict():
    with patch(
        "app.services.ai_provider.settings",
        MagicMock(ai_provider="huggingface"),
    ), patch(
        "app.services.huggingface_service.check_self_harm_risk",
        return_value=(None, TokenUsage(0, 0, 0)),
    ) as mock_hf:

        result, usage = await check_self_harm_risk("text", 1)

    assert result is None
    assert usage == TokenUsage(0, 0, 0)
    mock_hf.assert_called_once_with("text", 1)


async def test_check_hate_speech_risk_uses_huggingface_by_default():
    with patch(
        "app.services.ai_provider.settings",
        MagicMock(ai_provider="huggingface"),
    ), patch(
        "app.services.huggingface_service.check_hate_speech_risk",
        return_value=(False, _JUDGE_CALL_USAGE),
    ) as mock_hf, patch(
        "app.services.ollama_service.check_hate_speech_risk",
    ) as mock_ollama:

        result, usage = await check_hate_speech_risk("text", 1)

    assert result is False
    assert usage == _JUDGE_CALL_USAGE
    mock_hf.assert_called_once_with("text", 1)
    mock_ollama.assert_not_called()


async def test_check_hate_speech_risk_uses_ollama_when_configured():
    # Both calls return True (no flip on retry) - the dispatch route
    # (ollama vs huggingface) is what this test verifies, not the retry
    # mechanics themselves - see the dedicated retry tests below.
    with patch(
        "app.services.ai_provider.settings",
        MagicMock(ai_provider="ollama"),
    ), patch(
        "app.services.ollama_service.check_hate_speech_risk",
        return_value=(True, _JUDGE_CALL_USAGE),
    ) as mock_ollama, patch(
        "app.services.huggingface_service.check_hate_speech_risk",
    ) as mock_hf:

        result, usage = await check_hate_speech_risk("text", 1)

    assert result is True
    assert usage == _JUDGE_CALL_USAGE + _JUDGE_CALL_USAGE
    assert mock_ollama.call_count == 2
    mock_ollama.assert_called_with("text", 1)
    mock_hf.assert_not_called()


async def test_check_hate_speech_risk_retries_once_on_hateful_verdict():
    # Mirrors check_self_harm_risk's nondeterminism mitigation (see that
    # test's comment) - applied here on the same model/infra, not
    # independently re-verified for this specific prompt.
    with patch(
        "app.services.ai_provider.settings",
        MagicMock(ai_provider="huggingface"),
    ), patch(
        "app.services.huggingface_service.check_hate_speech_risk",
        side_effect=[(True, _JUDGE_CALL_USAGE), (False, _JUDGE_CALL_USAGE)],
    ) as mock_hf:

        result, usage = await check_hate_speech_risk("text", 1)

    assert result is False
    assert usage == _JUDGE_CALL_USAGE + _JUDGE_CALL_USAGE
    assert mock_hf.call_count == 2


async def test_check_hate_speech_risk_fails_closed_on_repeated_hateful_verdict():
    with patch(
        "app.services.ai_provider.settings",
        MagicMock(ai_provider="huggingface"),
    ), patch(
        "app.services.huggingface_service.check_hate_speech_risk",
        side_effect=[(True, _JUDGE_CALL_USAGE), (True, _JUDGE_CALL_USAGE)],
    ) as mock_hf:

        result, usage = await check_hate_speech_risk("text", 1)

    assert result is True
    assert usage == _JUDGE_CALL_USAGE + _JUDGE_CALL_USAGE
    assert mock_hf.call_count == 2


async def test_check_hate_speech_risk_retry_fails_open_on_none():
    with patch(
        "app.services.ai_provider.settings",
        MagicMock(ai_provider="huggingface"),
    ), patch(
        "app.services.huggingface_service.check_hate_speech_risk",
        side_effect=[
            (True, _JUDGE_CALL_USAGE),
            (None, TokenUsage(0, 0, 0)),
        ],
    ) as mock_hf:

        result, usage = await check_hate_speech_risk("text", 1)

    assert result is None
    assert usage == _JUDGE_CALL_USAGE
    assert mock_hf.call_count == 2


async def test_check_hate_speech_risk_does_not_retry_on_false_verdict():
    with patch(
        "app.services.ai_provider.settings",
        MagicMock(ai_provider="huggingface"),
    ), patch(
        "app.services.huggingface_service.check_hate_speech_risk",
        return_value=(False, _JUDGE_CALL_USAGE),
    ) as mock_hf:

        result, usage = await check_hate_speech_risk("text", 1)

    assert result is False
    assert usage == _JUDGE_CALL_USAGE
    mock_hf.assert_called_once_with("text", 1)


async def test_check_hate_speech_risk_does_not_retry_on_none_verdict():
    with patch(
        "app.services.ai_provider.settings",
        MagicMock(ai_provider="huggingface"),
    ), patch(
        "app.services.huggingface_service.check_hate_speech_risk",
        return_value=(None, TokenUsage(0, 0, 0)),
    ) as mock_hf:

        result, usage = await check_hate_speech_risk("text", 1)

    assert result is None
    assert usage == TokenUsage(0, 0, 0)
    mock_hf.assert_called_once_with("text", 1)
