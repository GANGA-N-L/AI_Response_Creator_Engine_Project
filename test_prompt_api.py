from app.models.prompt import Prompt


# Testing the successful Prompt Creation API request
def test_create_prompt(client):
    response = client.post(
        "/prompts/",
        json={
            "prompt_name": "Grammar Correction",
            "prompt_text": "Correct the grammar of the following text.",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert isinstance(data["id"], int)
    assert data["prompt_name"] == "Grammar Correction"
    assert (
        data["prompt_text"]
        == "Correct the grammar of the following text."
    )


# Testing the Get All Prompts API request
def test_get_all_prompts(client, db):
    prompt1 = Prompt(
        prompt_name="Grammar Correction",
        prompt_text="Correct the grammar of the following text.",
    )

    prompt2 = Prompt(
        prompt_name="Formal Response",
        prompt_text="Rewrite the following text in a formal and professional manner.",
    )

    db.add_all([prompt1, prompt2])
    db.commit()

    response = client.get("/prompts/")

    assert response.status_code == 200

    data = response.json()

    assert len(data) == 2

    assert data[0]["prompt_name"] == "Grammar Correction"
    assert (
        data[0]["prompt_text"]
        == "Correct the grammar of the following text."
    )

    assert data[1]["prompt_name"] == "Formal Response"
    assert (
        data[1]["prompt_text"]
        == "Rewrite the following text in a formal and professional manner."
    )


# Testing the Get Prompt by ID API request for an existing prompt
def test_get_prompt(client, db):
    prompt = Prompt(
        prompt_name="Get Prompt Test",
        prompt_text="Correct the grammar of the following text.",
    )

    db.add(prompt)
    db.commit()
    db.refresh(prompt)

    response = client.get(
        f"/prompts/{prompt.id}"
    )

    assert response.status_code == 200

    assert response.json() == {
        "id": prompt.id,
        "prompt_name": "Get Prompt Test",
        "prompt_text": "Correct the grammar of the following text.",
    }


# Testing the Get Prompt by ID API request for a non-existing prompt
def test_get_prompt_not_found(client):
    response = client.get("/prompts/999999")

    assert response.status_code == 404

    assert response.json() == {
        "status_code": 404,
        "error_code": "PROMPT_NOT_FOUND",
        "message": "Prompt not found.",
    }


# Testing the Update Prompt API request
def test_update_prompt(client, db):
    prompt = Prompt(
        prompt_name="Original Prompt",
        prompt_text="Original prompt text.",
    )

    db.add(prompt)
    db.commit()
    db.refresh(prompt)

    response = client.put(
        f"/prompts/{prompt.id}",
        json={
            "prompt_name": "Updated Prompt",
            "prompt_text": "Updated prompt text.",
        },
    )

    assert response.status_code == 200

    assert response.json() == {
        "id": prompt.id,
        "prompt_name": "Updated Prompt",
        "prompt_text": "Updated prompt text.",
    }

    db.refresh(prompt)

    assert prompt.prompt_name == "Updated Prompt"
    assert prompt.prompt_text == "Updated prompt text."


# Testing the Update Prompt API request when prompt is not found
def test_update_prompt_not_found(client):
    response = client.put(
        "/prompts/999999",
        json={
            "prompt_name": "Updated Prompt",
            "prompt_text": "Updated prompt text.",
        },
    )

    assert response.status_code == 404

    assert response.json() == {
        "status_code": 404,
        "error_code": "PROMPT_NOT_FOUND",
        "message": "Prompt not found.",
    }


# Testing the Delete Prompt API request for an existing prompt
def test_delete_prompt(client, db):
    prompt = Prompt(
        prompt_name="Delete Prompt Test",
        prompt_text="This prompt should be deleted.",
    )

    db.add(prompt)
    db.commit()
    db.refresh(prompt)

    prompt_id = prompt.id

    response = client.delete(
        f"/prompts/{prompt_id}"
    )

    assert response.status_code == 200

    assert response.json() == {
        "message": "Prompt deleted successfully."
    }

    deleted_prompt = (
        db.query(Prompt)
        .filter(Prompt.id == prompt_id)
        .first()
    )

    assert deleted_prompt is None


# Testing the Delete Prompt API request for a non-existing prompt
def test_delete_prompt_not_found(client):
    response = client.delete("/prompts/999999")

    assert response.status_code == 404

    assert response.json() == {
        "status_code": 404,
        "error_code": "PROMPT_NOT_FOUND",
        "message": "Prompt not found.",
    }