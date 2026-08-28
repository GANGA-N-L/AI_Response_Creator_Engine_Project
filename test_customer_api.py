from app.models.customer import Customer


# Testing the successful Customer Creation API request
def test_create_customer(client):
    response = client.post(
        "/customers/",
        json={
            "customer_name": "Test Customer",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["customer_name"] == "Test Customer"
    assert data["credits"] == 100
    assert isinstance(data["id"], int)


# Testing the Customer Validation API request
def test_create_customer_invalid_name(client):
    response = client.post(
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


# Testing the Get Customer by ID API request
def test_get_customer(client, db):
    customer = Customer(
        customer_name="Get Customer Test",
        credits=75,
    )

    db.add(customer)
    db.commit()
    db.refresh(customer)

    response = client.get(
        f"/customers/{customer.id}"
    )

    assert response.status_code == 200

    assert response.json() == {
        "id": customer.id,
        "customer_name": "Get Customer Test",
        "credits": 75,
    }


# Testing the Get Customer by ID Not Found API request
def test_get_customer_not_found(client):
    response = client.get("/customers/999999")

    assert response.status_code == 404

    assert response.json() == {
        "status_code": 404,
        "error_code": "CUSTOMER_NOT_FOUND",
        "message": "Customer not found",
    }


# Testing the Get All Customers API request
def test_get_all_customers(client, db):
    customer1 = Customer(
        customer_name="Customer One",
        credits=100,
    )

    customer2 = Customer(
        customer_name="Customer Two",
        credits=75,
    )

    db.add_all([customer1, customer2])
    db.commit()

    response = client.get("/customers/")

    assert response.status_code == 200

    data = response.json()

    assert len(data) == 2

    assert data[0]["customer_name"] == "Customer One"
    assert data[0]["credits"] == 100

    assert data[1]["customer_name"] == "Customer Two"
    assert data[1]["credits"] == 75


# Testing the successful Update Customer Credits API request
def test_update_customer_credits(client, db):
    customer = Customer(
        customer_name="Credit Update Test",
        credits=100,
    )

    db.add(customer)
    db.commit()
    db.refresh(customer)

    response = client.put(
        f"/customers/{customer.id}/credits",
        json={
            "credits": 50
        },
    )

    assert response.status_code == 200

    assert response.json() == {
        "id": customer.id,
        "customer_name": "Credit Update Test",
        "credits": 50,
    }

    db.refresh(customer)

    assert customer.credits == 50


# Testing the Update Customer Credits API request when customer is not found
def test_update_customer_credits_not_found(client):
    response = client.put(
        "/customers/999999/credits",
        json={
            "credits": 50
        },
    )

    assert response.status_code == 404

    assert response.json() == {
        "status_code": 404,
        "error_code": "CUSTOMER_NOT_FOUND",
        "message": "Customer not found",
    }


# Testing the Get Customer Transactions API request
def test_get_customer_transactions(client, db):
    customer = Customer(
        customer_name="Transactions Test",
        credits=100,
    )

    db.add(customer)
    db.commit()
    db.refresh(customer)

    client.put(
        f"/customers/{customer.id}/credits",
        json={
            "credits": 50
        },
    )

    client.put(
        f"/customers/{customer.id}/credits",
        json={
            "credits": 80
        },
    )

    response = client.get(
        f"/customers/{customer.id}/transactions"
    )

    assert response.status_code == 200

    data = response.json()

    assert len(data) == 2

    assert data[0]["amount"] == 30
    assert data[0]["balance_after"] == 80
    assert data[0]["transaction_type"] == "ADMIN_ADJUSTMENT"

    assert data[1]["amount"] == -50
    assert data[1]["balance_after"] == 50
    assert data[1]["transaction_type"] == "ADMIN_ADJUSTMENT"


# Testing the Get Customer Transactions API request with no transactions
def test_get_customer_transactions_empty(client, db):
    customer = Customer(
        customer_name="No Transactions Test",
        credits=100,
    )

    db.add(customer)
    db.commit()
    db.refresh(customer)

    response = client.get(
        f"/customers/{customer.id}/transactions"
    )

    assert response.status_code == 200
    assert response.json() == []


# Testing the Get Customer Transactions API request when customer is not found
def test_get_customer_transactions_not_found(client):
    response = client.get("/customers/999999/transactions")

    assert response.status_code == 404

    assert response.json() == {
        "status_code": 404,
        "error_code": "CUSTOMER_NOT_FOUND",
        "message": "Customer not found",
    }