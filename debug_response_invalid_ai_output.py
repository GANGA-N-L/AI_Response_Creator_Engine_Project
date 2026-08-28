from unittest.mock import patch, MagicMock

from app.database.database import SessionLocal
from app.models.customer import Customer
from app.api.response import generate_ai_response
from app.schemas.response import ResponseRequest


db = SessionLocal()

try:
    customer = (
        db.query(Customer)
        .filter(Customer.id == 1)
        .first()
    )

    credits_before = customer.credits

    request = ResponseRequest(
        customer_id=1,
        prompt_id=1,
        text="Thnks for your help"
    )

    mock_response = MagicMock()
    mock_response.choices = []

    with patch(
        "app.api.response.generate_response",
        side_effect=__import__(
            "app.exceptions.ai_exceptions",
            fromlist=["AIProviderError"]
        ).AIProviderError(
            "AI provider returned no response choices."
        )
    ):
        response = generate_ai_response(
            request=request,
            db=db
        )

    print("Response status:", response.status_code)
    print("Response body:", response.body.decode())

    db.expire_all()

    customer = (
        db.query(Customer)
        .filter(Customer.id == 1)
        .first()
    )

    print("Credits before:", credits_before)
    print("Credits after:", customer.credits)

finally:
    db.close()