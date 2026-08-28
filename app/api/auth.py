import hmac

from fastapi import APIRouter, HTTPException, Request, status

from app.core.config import settings
from app.core.rate_limiter import limiter
from app.core.security import create_access_token
from app.schemas.auth import TokenRequest, TokenResponse

router = APIRouter(
    prefix="/auth",
    tags=["Authentication"]
)


@router.post("/token", response_model=TokenResponse)
@limiter.limit("5/minute")
def issue_token(request: Request, credentials: TokenRequest):
    valid_id = hmac.compare_digest(
        credentials.client_id, settings.auth_client_id
    )
    valid_secret = hmac.compare_digest(
        credentials.client_secret, settings.auth_client_secret
    )

    if not (valid_id and valid_secret):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid client credentials"
        )

    access_token, expires_in = create_access_token(subject=credentials.client_id)

    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in=expires_in,
    )
