from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.exceptions.response_exceptions import CustomerNotFoundError
from app.models.customer import Customer


async def create_customer(db: AsyncSession, customer_name: str) -> Customer:
    new_customer = Customer(customer_name=customer_name)

    db.add(new_customer)
    await db.commit()
    await db.refresh(new_customer)

    return new_customer


async def list_customers(
    db: AsyncSession, limit: int, offset: int
) -> list[Customer]:
    result = await db.execute(
        select(Customer).offset(offset).limit(limit)
    )

    return list(result.scalars().all())


async def get_customer(db: AsyncSession, customer_id: int) -> Customer:
    result = await db.execute(
        select(Customer).where(Customer.id == customer_id)
    )

    customer = result.scalar_one_or_none()

    if customer is None:
        raise CustomerNotFoundError("Customer not found.")

    return customer
