from datetime import datetime

from pydantic import BaseModel, ConfigDict, StrictInt, StrictStr


class ResponseHistoryEntry(BaseModel):
    id: StrictInt
    prompt_id: StrictInt
    input_text: StrictStr
    output_text: StrictStr
    created_at: datetime

    model_config = ConfigDict(
        from_attributes=True,
        extra="forbid",
    )
