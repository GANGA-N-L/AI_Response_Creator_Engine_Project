from datetime import datetime

from pydantic import BaseModel, ConfigDict, StrictInt, StrictStr


class CreditTransactionResponse(BaseModel):
    id: StrictInt
    customer_id: StrictInt
    amount: StrictInt
    balance_after: StrictInt
    transaction_type: StrictStr
    reason: StrictStr
    created_at: datetime

    model_config = ConfigDict(
        from_attributes=True,
        extra="forbid",
    )
