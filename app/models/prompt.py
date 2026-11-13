from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

from app.constants import PROMPT_NAME_MAX_LENGTH, PROMPT_TEXT_MAX_LENGTH
from app.database.database import Base


class Prompt(Base):
    __tablename__ = "prompts"

    id: Mapped[int] = mapped_column(primary_key=True)
    prompt_name: Mapped[str] = mapped_column(
        String(PROMPT_NAME_MAX_LENGTH), nullable=False
    )
    prompt_text: Mapped[str] = mapped_column(
        String(PROMPT_TEXT_MAX_LENGTH), nullable=False
    )
