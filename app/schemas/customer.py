import unicodedata

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StrictStr,
    field_validator,
)

from app.constants import CUSTOMER_NAME_MAX_LENGTH


class CustomerCreate(BaseModel):
    customer_name: StrictStr = Field(
        min_length=2,
        max_length=CUSTOMER_NAME_MAX_LENGTH,
    )

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    @field_validator("customer_name")
    @classmethod
    def validate_customer_name(cls, value: str) -> str:
        value = unicodedata.normalize("NFC", value)
        value = " ".join(value.split())

        if not value:
            raise ValueError(
                "Customer name cannot be empty or whitespace only."
            )

        has_letter = any(
            unicodedata.category(char).startswith("L")
            for char in value
        )

        if not has_letter:
            raise ValueError(
                "Customer name must contain at least one letter."
            )

        for char in value:
            category = unicodedata.category(char)

            if category.startswith(("L", "M")):
                continue

            if char not in " '-.":
                raise ValueError(
                    "Customer name contains invalid characters."
                )

        return value


class CustomerResponse(BaseModel):
    id: StrictInt
    customer_name: StrictStr

    model_config = ConfigDict(
        from_attributes=True,
        extra="forbid",
    )