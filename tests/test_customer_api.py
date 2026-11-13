from app.models.customer import Customer

from tests.conftest import assert_error_body


# Testing the successful Customer Creation API request
async def test_create_customer(client):
    response = await client.post(
        "/customers/",
        json={
            "customer_name": "Test Customer",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["customer_name"] == "Test Customer"
    assert isinstance(data["id"], int)


# Testing the Customer Validation API request
async def test_create_customer_invalid_name(client):
    response = await client.post(
        "/customers/",
        json={
            "customer_name": "John@123",
        },
    )

    assert response.status_code == 422

    data = response.json()

    assert data["status_code"] == 422
    assert data["error_code"] == "VALIDATION_ERROR"
    assert data["message"] == "Invalid request data."
    assert data["fields"] == [
        {
            "field": "customer_name",
            "message": "Value error, Customer name contains invalid characters.",
        }
    ]


# Testing the Get Customer by ID API request
async def test_get_customer(client, db):
    customer = Customer(
        customer_name="Get Customer Test",
    )

    db.add(customer)
    await db.commit()
    await db.refresh(customer)

    response = await client.get(
        f"/customers/{customer.id}"
    )

    assert response.status_code == 200

    assert response.json() == {
        "id": customer.id,
        "customer_name": "Get Customer Test",
    }


# Testing the Get Customer by ID Not Found API request
async def test_get_customer_not_found(client):
    response = await client.get("/customers/999999")

    assert response.status_code == 404

    assert_error_body(response, {
        "status_code": 404,
        "error_code": "CUSTOMER_NOT_FOUND",
        "message": "Customer not found.",
    })


# Testing the Get All Customers API request
async def test_get_all_customers(client, db):
    customer1 = Customer(
        customer_name="Customer One",
    )

    customer2 = Customer(
        customer_name="Customer Two",
    )

    db.add_all([customer1, customer2])
    await db.commit()

    response = await client.get("/customers/")

    assert response.status_code == 200

    data = response.json()

    assert len(data) == 2

    assert data[0]["customer_name"] == "Customer One"
    assert data[1]["customer_name"] == "Customer Two"
