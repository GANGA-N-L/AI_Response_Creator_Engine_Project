from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

from app.constants import CUSTOMER_NAME_MAX_LENGTH
from app.database.database import Base


class Customer(Base):
    __tablename__ = "customers"

    id: Mapped[int] = mapped_column(primary_key=True)
    customer_name: Mapped[str] = mapped_column(
        String(CUSTOMER_NAME_MAX_LENGTH), nullable=False
    )
