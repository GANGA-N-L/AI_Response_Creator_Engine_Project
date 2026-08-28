from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.database.database import get_db
from app.schemas.credit_transaction import CreditTransactionResponse
from app.schemas.customer import CustomerCreate, CustomerResponse, CreditUpdate
from app.services import customer_service

router = APIRouter(
    prefix="/customers",
    tags=["Customers"]
)


@router.post("/", response_model=CustomerResponse)
def create_customer(
    customer: CustomerCreate,
    db: Session = Depends(get_db)
):
    return customer_service.create_customer(db, customer.customer_name)


@router.get("/", response_model=list[CustomerResponse])
def get_all_customers(
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db)
):
    return customer_service.list_customers(db, limit, offset)


@router.get("/{customer_id}", response_model=CustomerResponse)
def get_customer(
    customer_id: int,
    db: Session = Depends(get_db)
):
    return customer_service.get_customer(db, customer_id)


@router.put("/{customer_id}/credits", response_model=CustomerResponse)
def update_customer_credits(
    customer_id: int,
    credit_update: CreditUpdate,
    db: Session = Depends(get_db)
):
    return customer_service.update_customer_credits(
        db, customer_id, credit_update.credits
    )


@router.get(
    "/{customer_id}/transactions",
    response_model=list[CreditTransactionResponse],
)
def get_customer_transactions(
    customer_id: int,
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db)
):
    return customer_service.list_customer_transactions(
        db, customer_id, limit, offset
    )
