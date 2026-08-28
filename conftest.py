import pytest

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from fastapi.testclient import TestClient

from app.core.config import settings
from app.core.rate_limiter import limiter, reset_customer_rate_limits
from app.core.security import create_access_token
from app.main import app
from app.database.database import get_db
from app.models.customer import Customer
from app.models.prompt import Prompt


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    limiter.reset()
    reset_customer_rate_limits()


@pytest.fixture(scope="session")
def test_engine():
    engine = create_engine(
        settings.test_database_url
    )

    yield engine

    engine.dispose()


@pytest.fixture
def db(test_engine):
    connection = test_engine.connect()
    transaction = connection.begin()

    TestingSessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=connection,
    )

    session = TestingSessionLocal(
        join_transaction_mode="create_savepoint"
    )

    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()


@pytest.fixture
def unauthenticated_client(db):
    def override_get_db():
        yield db

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


@pytest.fixture
def client(unauthenticated_client):
    token, _ = create_access_token(subject=settings.auth_client_id)
    unauthenticated_client.headers["Authorization"] = f"Bearer {token}"

    return unauthenticated_client


@pytest.fixture
def customer(db):
    customer = Customer(
        customer_name="Test Customer",
        credits=20,
    )

    db.add(customer)
    db.commit()
    db.refresh(customer)

    return customer


@pytest.fixture
def prompt(db):
    prompt = Prompt(
        prompt_name="Grammar Correction",
        prompt_text="Correct the grammar of the following text.",
    )

    db.add(prompt)
    db.commit()
    db.refresh(prompt)

    return prompt