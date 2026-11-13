from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, func
from sqlalchemy.orm import Mapped, mapped_column

from app.database.database import Base


class AiUsageLog(Base):
    __tablename__ = "ai_usage_logs"

    id: Mapped[int] = mapped_column(primary_key=True)
    customer_id: Mapped[int] = mapped_column(
        ForeignKey("customers.id"), nullable=False, index=True
    )
    prompt_id: Mapped[int] = mapped_column(
        ForeignKey("prompts.id"), nullable=False, index=True
    )
    input_text_length: Mapped[int] = mapped_column(nullable=False)
    output_text_length: Mapped[int] = mapped_column(nullable=False)
    prompt_tokens: Mapped[int] = mapped_column(nullable=False)
    completion_tokens: Mapped[int] = mapped_column(nullable=False)
    total_tokens: Mapped[int] = mapped_column(nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
