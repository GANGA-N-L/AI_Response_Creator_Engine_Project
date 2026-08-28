from dataclasses import dataclass


@dataclass(frozen=True)
class ValidationResult:
    is_valid: bool
    error_code: str
    reason: str
