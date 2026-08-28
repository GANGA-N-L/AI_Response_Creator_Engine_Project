import pytest

from app.validators.input_validator import InputValidator


@pytest.mark.parametrize(
    "text, expected_error_code",
    [
        ("", "EMPTY_INPUT"),
        (" ", "EMPTY_INPUT"),
        ("????", "NO_MEANINGFUL_CONTENT"),
        ("!!!!!", "NO_MEANINGFUL_CONTENT"),
        ("😊😊😊", "NO_MEANINGFUL_CONTENT"),
        ("🔥🔥🔥", "NO_MEANINGFUL_CONTENT"),
        ("！！！", "NO_MEANINGFUL_CONTENT"),
        ("12345", "ONLY_NUMBERS"),
        ("-42", "ONLY_NUMBERS"),
        ("12.5", "ONLY_NUMBERS"),
        ("a", "INPUT_TOO_SHORT"),
        ("x", "INPUT_TOO_SHORT"),
        ("k", "INPUT_TOO_SHORT"),
    ],
)
def test_invalid_inputs_are_rejected(text, expected_error_code):
    result = InputValidator.validate(text)

    assert result.is_valid is False
    assert result.error_code == expected_error_code


@pytest.mark.parametrize(
    "text",
    [
        "Hi",
        "OK",
        "No",
        "Hello",
        "Hi!!!",
        "Thanks 😊",
        "Order #1234",
        "₹500",
        "നമസ്കാരം",
        "नमस्ते",
        "Order 12345",
        "Invoice 9876",
        "Room 205",
        "Thnks",
        "A" * 1001,
    ],
)
def test_valid_inputs_are_accepted(text):
    result = InputValidator.validate(text)

    assert result.is_valid is True
    assert result.error_code == ""
