from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.database import get_db
from app.schemas.customer import CustomerCreate, CustomerResponse
from app.services import customer_service

router = APIRouter(
    prefix="/customers",
    tags=["Customers"]
)


@router.post("/", response_model=CustomerResponse)
async def create_customer(
    customer: CustomerCreate,
    db: AsyncSession = Depends(get_db)
):
    return await customer_service.create_customer(db, customer.customer_name)


@router.get("/", response_model=list[CustomerResponse])
async def get_all_customers(
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db)
):
    return await customer_service.list_customers(db, limit, offset)


@router.get("/{customer_id}", response_model=CustomerResponse)
async def get_customer(
    customer_id: int,
    db: AsyncSession = Depends(get_db)
):
    return await customer_service.get_customer(db, customer_id)
