from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.database.database import Base


class ResponseHistory(Base):
    """
    A customer-scoped record of what was actually submitted and
    generated for a request - kept for support/dispute resolution
    ("what did I get for this request?"), not for AI-quality analysis
    (ai_usage_logs already covers that without storing raw text).
    Deliberately narrower than a general audit log: retention-limited
    (see response_history_service.purge_expired_response_history) and
    only ever read scoped to one explicit customer_id (app/api/
    response_history.py) - there is no endpoint that lists across
    customers.
    """

    __tablename__ = "response_history"

    id: Mapped[int] = mapped_column(primary_key=True)
    customer_id: Mapped[int] = mapped_column(
        ForeignKey("customers.id"), nullable=False, index=True
    )
    prompt_id: Mapped[int] = mapped_column(
        ForeignKey("prompts.id"), nullable=False
    )
    input_text: Mapped[str] = mapped_column(Text, nullable=False)
    output_text: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
