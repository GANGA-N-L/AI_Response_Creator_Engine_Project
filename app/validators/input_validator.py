import re
import unicodedata

from app.core.config import settings
from app.models.validation_result import ValidationResult


class InputValidator:
    """
    Validates whether the given text is meaningful enough
    to be processed by the AI Response Creator.
    """

    _VALID = ValidationResult(is_valid=True, error_code="", reason="")

    _NUMERIC_PATTERN = re.compile(r"[+-]?\d+(?:,\d+)*(?:\.\d+)?")

    @staticmethod
    def validate(text: str) -> ValidationResult:

        result = InputValidator._validate_empty(text)
        if not result.is_valid:
            return result

        result = InputValidator._validate_min_length(text)
        if not result.is_valid:
            return result

        result = InputValidator._validate_meaningful_content(text)
        if not result.is_valid:
            return result

        result = InputValidator._validate_only_numbers(text)
        if not result.is_valid:
            return result

        return InputValidator._VALID

    @staticmethod
    def _validate_empty(text: str) -> ValidationResult:

        if text.strip() == "":
            return ValidationResult(
                is_valid=False,
                error_code="EMPTY_INPUT",
                reason="Input cannot be empty or whitespace only."
            )

        return InputValidator._VALID

    @staticmethod
    def _validate_min_length(text: str) -> ValidationResult:

        stripped = text.strip()

        if len(stripped) < settings.min_input_length:
            return ValidationResult(
                is_valid=False,
                error_code="INPUT_TOO_SHORT",
                reason=(
                    f"Input must be at least "
                    f"{settings.min_input_length} characters long."
                )
            )

        return InputValidator._VALID

    @staticmethod
    def _validate_meaningful_content(text: str) -> ValidationResult:

        stripped = text.strip()

        has_meaningful_content = any(
            unicodedata.category(char).startswith(("L", "N"))
            for char in stripped
        )

        if not has_meaningful_content:
            return ValidationResult(
                is_valid=False,
                error_code="NO_MEANINGFUL_CONTENT",
                reason="Input does not contain meaningful textual or numeric content."
            )

        return InputValidator._VALID

    @staticmethod
    def _validate_only_numbers(text: str) -> ValidationResult:

        stripped = text.strip()

        if InputValidator._NUMERIC_PATTERN.fullmatch(stripped):
            return ValidationResult(
                is_valid=False,
                error_code="ONLY_NUMBERS",
                reason="Input contains only numbers."
            )

        return InputValidator._VALID
