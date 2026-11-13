from pydantic import BaseModel, ConfigDict, Field, StrictInt


class PiiMetricsResponse(BaseModel):
    total_requests: StrictInt = Field(ge=0)
    requests_with_pii: StrictInt = Field(ge=0)
    pii_percentage: float = Field(ge=0, le=100)
    by_category: dict[str, int]

    model_config = ConfigDict(
        extra="forbid",
    )


class CacheMetricsResponse(BaseModel):
    hits: StrictInt = Field(ge=0)
    misses: StrictInt = Field(ge=0)
    hit_rate_percentage: float = Field(ge=0, le=100)

    model_config = ConfigDict(
        extra="forbid",
    )
