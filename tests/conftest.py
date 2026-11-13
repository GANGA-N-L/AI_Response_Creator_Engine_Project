from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core.config import settings
from app.core.rate_limiter import limiter, reset_customer_rate_limits
from app.core.security import create_access_token
from app.main import app
from app.database.database import get_db
from app.models.customer import Customer
from app.models.prompt import Prompt
from app.services.pii_metrics_service import clear_pii_metrics
from app.services.response_cache_service import clear_cache_metrics, clear_response_cache
from app.services.token_usage import TokenUsage

_ZERO_USAGE = TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)


def assert_error_body(response, expected: dict) -> None:
    """
    Asserts an error response body matches `expected`, except for
    `request_id` - a fresh UUID per request (RequestIdMiddleware), so it
    can't be pinned to a literal value in a test. Asserts it's present
    and non-empty instead of comparing it exactly, and that it matches
    the X-Request-ID response header (the same ID is meant to be echoed
    in both places).
    """

    body = response.json()
    request_id = body.pop("request_id", None)

    assert request_id
    assert response.headers.get("X-Request-ID") == request_id
    assert body == expected


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    limiter.reset()
    reset_customer_rate_limits()


@pytest.fixture(autouse=True)
def _reset_response_cache():
    clear_response_cache()
    clear_cache_metrics()


@pytest.fixture(autouse=True)
def _reset_pii_metrics():
    clear_pii_metrics()


@pytest.fixture(autouse=True)
def _default_judge_verdict():
    """
    Defaults judge_is_rewrite to "not a violation" for every test, so
    existing tests that mock generate_response's return value (but
    aren't specifically testing judge behavior) don't each need their
    own patch - without this, they'd fall through to the real
    judge_is_rewrite, which makes a real AI provider call. A test that
    specifically wants to exercise judge behavior can still override
    this with its own `with patch(...)` block, which correctly wins for
    its duration.
    """

    with patch(
        "app.services.ai_response_service.judge_is_rewrite",
        return_value=(True, _ZERO_USAGE),
    ):
        yield


@pytest.fixture(autouse=True)
def _default_self_harm_verdict():
    """
    Defaults check_self_harm_risk to "safe" for every test, same
    reasoning as _default_judge_verdict - without this, every test that
    reaches generate_ai_response past input validation would fall
    through to the real check_self_harm_risk, which makes a real AI
    provider call. A test exercising this check specifically can still
    override it with its own `with patch(...)` block.
    """

    with patch(
        "app.services.ai_response_service.check_self_harm_risk",
        return_value=(False, _ZERO_USAGE),
    ):
        yield


@pytest.fixture(autouse=True)
def _default_hate_speech_verdict():
    """
    Defaults check_hate_speech_risk to "safe" for every test, same
    reasoning as _default_self_harm_verdict.
    """

    with patch(
        "app.services.ai_response_service.check_hate_speech_risk",
        return_value=(False, _ZERO_USAGE),
    ):
        yield


@pytest.fixture(autouse=True)
def _default_consistency_verdict():
    """
    Defaults check_factual_consistency to "consistent" for every test,
    same reasoning as _default_judge_verdict - without this, every test
    that lets generate_response succeed would run the real local NLI
    model (~150ms each) instead of the near-instant mock it's meant to
    be testing against. A test exercising this check specifically can
    still override it with its own `with patch(...)` block.
    """

    with patch(
        "app.services.ai_response_service.check_factual_consistency",
        return_value=True,
    ):
        yield


@pytest.fixture(scope="module")
async def test_engine():
    engine = create_async_engine(
        settings.test_database_url
    )

    yield engine

    await engine.dispose()


@pytest.fixture
async def db(test_engine):
    async with test_engine.connect() as connection:
        await connection.begin()

        TestingSessionLocal = async_sessionmaker(
            autocommit=False,
            autoflush=False,
            expire_on_commit=False,
            bind=connection,
        )

        session = TestingSessionLocal(
            join_transaction_mode="create_savepoint"
        )

        try:
            yield session
        finally:
            await session.close()
            await connection.rollback()


@pytest.fixture
async def unauthenticated_client(db):
    async def override_get_db():
        yield db

    app.dependency_overrides[get_db] = override_get_db

    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as test_client:
        yield test_client

    app.dependency_overrides.clear()


@pytest.fixture
async def client(unauthenticated_client):
    token, _ = create_access_token(subject=settings.auth_client_id)
    unauthenticated_client.headers["Authorization"] = f"Bearer {token}"

    return unauthenticated_client


@pytest.fixture
async def customer(db):
    customer = Customer(
        customer_name="Test Customer",
    )

    db.add(customer)
    await db.commit()
    await db.refresh(customer)

    return customer


@pytest.fixture
async def prompt(db):
    prompt = Prompt(
        prompt_name="Grammar Correction",
        prompt_text="Correct the grammar of the following text.",
    )

    db.add(prompt)
    await db.commit()
    await db.refresh(prompt)

    return prompt
