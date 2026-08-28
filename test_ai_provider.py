from unittest.mock import MagicMock, patch

from app.services.ai_provider import check_availability, generate_response


def test_generate_response_uses_huggingface_by_default():
    with patch(
        "app.services.ai_provider.settings",
        MagicMock(ai_provider="huggingface"),
    ), patch(
        "app.services.huggingface_service.generate_response",
        return_value="hf answer",
    ) as mock_hf, patch(
        "app.services.ollama_service.generate_response",
    ) as mock_ollama:

        result = generate_response("prompt", "text", 1)

    assert result == "hf answer"
    mock_hf.assert_called_once_with("prompt", "text", 1)
    mock_ollama.assert_not_called()


def test_generate_response_uses_ollama_when_configured():
    with patch(
        "app.services.ai_provider.settings",
        MagicMock(ai_provider="ollama"),
    ), patch(
        "app.services.ollama_service.generate_response",
        return_value="ollama answer",
    ) as mock_ollama, patch(
        "app.services.huggingface_service.generate_response",
    ) as mock_hf:

        result = generate_response("prompt", "text", 1)

    assert result == "ollama answer"
    mock_ollama.assert_called_once_with("prompt", "text", 1)
    mock_hf.assert_not_called()


def test_check_availability_uses_huggingface_by_default():
    with patch(
        "app.services.ai_provider.settings",
        MagicMock(ai_provider="huggingface"),
    ), patch(
        "app.services.huggingface_service.check_availability",
        return_value=True,
    ) as mock_hf, patch(
        "app.services.ollama_service.check_availability",
    ) as mock_ollama:

        result = check_availability()

    assert result is True
    mock_hf.assert_called_once()
    mock_ollama.assert_not_called()


def test_check_availability_uses_ollama_when_configured():
    with patch(
        "app.services.ai_provider.settings",
        MagicMock(ai_provider="ollama"),
    ), patch(
        "app.services.ollama_service.check_availability",
        return_value=False,
    ) as mock_ollama, patch(
        "app.services.huggingface_service.check_availability",
    ) as mock_hf:

        result = check_availability()

    assert result is False
    mock_ollama.assert_called_once()
    mock_hf.assert_not_called()
