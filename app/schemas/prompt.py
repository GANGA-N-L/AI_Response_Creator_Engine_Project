import unicodedata

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StrictStr,
    field_validator,
)

from app.constants import PROMPT_NAME_MAX_LENGTH, PROMPT_TEXT_MAX_LENGTH


class PromptFields(BaseModel):
    prompt_name: StrictStr = Field(
        min_length=2,
        max_length=PROMPT_NAME_MAX_LENGTH
    )
    prompt_text: StrictStr = Field(
        min_length=1,
        max_length=PROMPT_TEXT_MAX_LENGTH
    )

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True
    )

    @field_validator("prompt_name")
    @classmethod
    def validate_prompt_name(cls, value: str) -> str:
        value = unicodedata.normalize("NFC", value).strip()

        if not value:
            raise ValueError("Prompt name cannot be blank.")

        if any(
            unicodedata.category(char).startswith("C")
            for char in value
        ):
            raise ValueError(
                "Prompt name contains unsupported control characters."
            )

        return value

    @field_validator("prompt_text")
    @classmethod
    def validate_prompt_text(cls, value: str) -> str:
        value = unicodedata.normalize("NFC", value).strip()

        if not value:
            raise ValueError("Prompt text cannot be blank.")

        if any(
            unicodedata.category(char) == "Cc"
            and char not in "\n\r\t"
            for char in value
        ):
            raise ValueError(
                "Prompt text contains unsupported control characters."
            )

        return value


class PromptCreate(PromptFields):
    pass


class PromptUpdate(PromptFields):
    pass


class PromptResponse(BaseModel):
    id: StrictInt
    prompt_name: StrictStr
    prompt_text: StrictStr

    model_config = ConfigDict(from_attributes=True)


class PromptDeleteResponse(BaseModel):
    message: StrictStr

    model_config = ConfigDict(extra="forbid")