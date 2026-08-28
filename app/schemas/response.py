from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr

from app.core.config import settings


class ResponseRequest(BaseModel):
    customer_id: StrictInt = Field(
        gt=0,
        description="Positive customer identifier.",
    )
    prompt_id: StrictInt = Field(
        gt=0,
        description="Positive prompt identifier.",
    )
    text: StrictStr = Field(
        max_length=settings.max_input_length,
        description="Text that will be processed by the AI provider.",
    )

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )


class SuccessResponse(BaseModel):
    status_code: Literal[200] = 200
    answer: StrictStr = Field(
        min_length=1,
        description="Generated AI response.",
    )
    credits: StrictInt = Field(
        ge=0,
        description="Remaining customer credits.",
    )
    credits_used: StrictInt = Field(
        ge=1,
        description="Credits consumed by this request.",
    )

    model_config = ConfigDict(
        extra="forbid",
    )


class ErrorField(BaseModel):
    field: StrictStr
    message: StrictStr

    model_config = ConfigDict(
        extra="forbid",
    )


class ErrorResponse(BaseModel):
    status_code: StrictInt = Field(
        ge=400,
        le=599,
    )
    error_code: StrictStr = Field(
        min_length=1,
        max_length=100,
    )
    message: StrictStr = Field(
        min_length=1,
        max_length=500,
    )
    fields: list[ErrorField] | None = None

    model_config = ConfigDict(
        extra="forbid",
    )