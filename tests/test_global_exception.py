from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.exceptions.handlers import register_exception_handlers


def test_unexpected_exception_handler():
    test_app = FastAPI()

    register_exception_handlers(test_app)

    @test_app.get("/test-unexpected-error")
    def test_unexpected_error():
        raise RuntimeError("Simulated internal failure.")

    client = TestClient(
        test_app,
        raise_server_exceptions=False
    )

    response = client.get("/test-unexpected-error")

    assert response.status_code == 500

    assert response.json() == {
        "status_code": 500,
        "error_code": "INTERNAL_SERVER_ERROR",
        "message": "An unexpected error occurred. Please try again later."
    }