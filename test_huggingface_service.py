from unittest.mock import MagicMock, patch

from app.services.huggingface_service import check_availability, generate_response


def test_generate_response():
    mock_response = MagicMock()

    mock_response.choices = [
        MagicMock(
            message=MagicMock(
                content="Thank you for contacting us."
            )
        )
    ]

    with patch(
        "app.services.huggingface_service.client.chat.completions.create",
        return_value=mock_response,
    ) as mock_create:

        response = generate_response(
            "Correct the grammar of the following text.",
            "Thanks for contacting us.",
            1,
        )

    assert response == "Thank you for contacting us."

    mock_create.assert_called_once()


def test_check_availability_success():
    mock_scoped_client = MagicMock()

    with patch(
        "app.services.huggingface_service.client.with_options",
        return_value=mock_scoped_client,
    ) as mock_with_options:

        result = check_availability()

    assert result is True
    mock_with_options.assert_called_once_with(timeout=5.0)
    mock_scoped_client.models.list.assert_called_once()


def test_check_availability_failure():
    mock_scoped_client = MagicMock()
    mock_scoped_client.models.list.side_effect = ConnectionError("boom")

    with patch(
        "app.services.huggingface_service.client.with_options",
        return_value=mock_scoped_client,
    ):
        result = check_availability()

    assert result is False