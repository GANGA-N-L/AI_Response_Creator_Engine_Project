from pydantic import BaseModel, ConfigDict, StrictInt, StrictStr


class TokenRequest(BaseModel):
    client_id: StrictStr
    client_secret: StrictStr

    model_config = ConfigDict(extra="forbid")


class TokenResponse(BaseModel):
    access_token: StrictStr
    token_type: StrictStr = "bearer"
    expires_in: StrictInt

    model_config = ConfigDict(extra="forbid")
