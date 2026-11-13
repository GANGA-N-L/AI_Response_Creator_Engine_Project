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

    # Common phrasing used to try to override the system prompt from
    # within the user-supplied text. Heuristic/pattern-based, not
    # exhaustive - and deliberately NOT used to reject requests (see
    # detect_injection_attempt below). Phrasing alone is an unreliable
    # signal of intent: legitimate text can contain it verbatim (e.g.
    # rewriting a quoted email, or dialogue), and it's trivial to evade
    # by rephrasing (e.g. "ignore all the instructions given to you"
    # matches none of these patterns, found live during Swagger
    # testing). The real defense against actual hijacking is delimiter
    # isolation (ai_prompt.py's wrap_user_text/system prompt) plus
    # output-side verification (judge_is_rewrite,
    # check_factual_consistency, ai_output_validator's leakage/
    # assistant-voice checks) - this pattern set only feeds a
    # non-blocking audit-log signal in ai_response_service.py.
    # Up to 3 filler words (e.g. "all your", "any of the") allowed
    # between the verb and "previous/prior/above instructions" - the
    # original fixed optional groups (just "all "/"any "/"the ") missed
    # phrasing like "ignore all *your* previous instructions" (caught
    # live during Swagger testing) since "your" wasn't an accepted
    # filler word. A small bounded window instead of a fixed word list
    # covers that class of variation without going as loose as an
    # unbounded ".*", which would start matching unrelated text that
    # happens to contain both words far apart.
    _INJECTION_PATTERNS = tuple(
        re.compile(pattern, re.IGNORECASE)
        for pattern in (
            r"\bignore\b(?:\s+\w+){0,3}?\s+(?:previous|prior|above|preceding)\s+instructions\b",
            r"\bdisregard\b(?:\s+\w+){0,3}?\s+(?:previous|prior|above|preceding)\s+instructions\b",
            r"\bforget\b(?:\s+\w+){0,3}?\s+(?:previous|prior|above)\s+(?:instructions|context)\b",
            r"\byou are now (?:a|an)\b",
            r"\b(?:reveal|show|print|output) (?:your |the )?system prompt\b",
            r"^\s*new instructions\s*:",
        )
    )

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

        result = InputValidator._validate_english_only(text)
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

    @staticmethod
    def _validate_english_only(text: str) -> ValidationResult:

        stripped = text.strip()

        for char in stripped:
            if not unicodedata.category(char).startswith("L"):
                continue

            try:
                char_name = unicodedata.name(char)
            except ValueError:
                continue

            if "LATIN" not in char_name:
                return ValidationResult(
                    is_valid=False,
                    error_code="UNSUPPORTED_LANGUAGE",
                    reason="Input must be in English."
                )

        return InputValidator._VALID

    @staticmethod
    def detect_injection_attempt(text: str) -> bool:
        """
        Non-blocking heuristic signal, used only for audit logging (see
        generate_ai_response in ai_response_service.py) - never rejects
        a request. See the _INJECTION_PATTERNS comment above for why
        phrase-matching alone isn't used as a rejection reason.
        """

        stripped = text.strip()

        return any(
            pattern.search(stripped)
            for pattern in InputValidator._INJECTION_PATTERNS
        )
