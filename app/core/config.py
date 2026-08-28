from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_FILE = Path(__file__).resolve().parent.parent.parent / ".env"


class Settings(BaseSettings):
    app_name: str = Field(
        default="AI Response Creator Engine",
        min_length=1,
        max_length=100,
    )
    app_version: str = Field(
        default="1.0.0",
        min_length=1,
        max_length=30,
    )

    log_level: Literal[
        "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
    ] = "INFO"

    database_url: str = Field(min_length=1, repr=False)
    test_database_url: str = Field(min_length=1, repr=False)

    ai_provider: Literal["huggingface", "ollama"] = "huggingface"

    hf_token: str | None = Field(default=None, repr=False)

    hf_base_url: str = Field(
        default="https://router.huggingface.co/v1",
        min_length=1,
    )
    hf_model: str = Field(
        default="openai/gpt-oss-20b",
        min_length=1,
        max_length=200,
    )

    ollama_url: str = Field(
        default="http://localhost:11434",
        min_length=1,
    )
    ollama_model: str = Field(
        default="mistral",
        min_length=1,
        max_length=200,
    )

    default_credits: int = Field(
        default=100,
        ge=0,
        le=1_000_000,
    )

    max_input_length: int = Field(
        default=1000,
        ge=1,
        le=100_000,
    )
    min_input_length: int = Field(
        default=2,
        ge=1,
        le=100_000,
    )

    characters_per_credit: int = Field(
        default=200,
        ge=1,
        le=100_000,
        description=(
            "Number of input characters that cost one credit. "
            "The cost of a request is ceil(len(text) / characters_per_credit), "
            "with a minimum of 1 credit."
        ),
    )

    ai_request_timeout_seconds: float = Field(
        default=30.0,
        gt=0,
        le=600.0,
        description="Max time to wait for an AI provider response, in seconds.",
    )

    jwt_secret_key: str = Field(min_length=32, repr=False)
    jwt_algorithm: str = Field(default="HS256", min_length=1)
    jwt_expiry_minutes: int = Field(default=60, ge=1, le=1440)

    auth_client_id: str = Field(min_length=1, repr=False)
    auth_client_secret: str = Field(min_length=1, repr=False)

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="forbid",
        validate_default=True,
        frozen=True,
    )

    @field_validator("log_level", mode="before")
    @classmethod
    def normalize_log_level(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().upper()

        return value

    @field_validator("database_url", "test_database_url")
    @classmethod
    def validate_database_url(cls, value: str) -> str:
        value = value.strip()
        parsed = urlparse(value)

        allowed_schemes = {
            "sqlite",
            "sqlite+pysqlite",
            "postgresql",
            "postgresql+psycopg",
            "postgresql+psycopg2",
        }

        if parsed.scheme not in allowed_schemes:
            raise ValueError(
                "Database URL must use a supported database scheme."
            )

        if parsed.scheme.startswith("postgresql") and not parsed.hostname:
            raise ValueError(
                "PostgreSQL database URL must include a hostname."
            )

        return value

    @field_validator("hf_base_url", "ollama_url")
    @classmethod
    def validate_http_url(cls, value: str) -> str:
        value = value.strip()
        parsed = urlparse(value)

        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(
                "AI service URL must be a valid HTTP or HTTPS URL."
            )

        return value

    @field_validator("app_name", "app_version", "hf_model", "ollama_model")
    @classmethod
    def strip_text_settings(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("Configuration value cannot be blank.")

        return value

    @model_validator(mode="after")
    def validate_input_lengths(self) -> "Settings":
        if self.min_input_length > self.max_input_length:
            raise ValueError(
                "min_input_length cannot exceed max_input_length."
            )

        return self

    @model_validator(mode="after")
    def validate_hf_token_required(self) -> "Settings":
        if self.ai_provider == "huggingface" and not self.hf_token:
            raise ValueError(
                "hf_token is required when ai_provider is 'huggingface'."
            )

        return self


settings = Settings()
