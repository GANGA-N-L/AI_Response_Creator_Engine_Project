from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.database.database import get_db
from app.schemas.prompt import (
    PromptCreate,
    PromptUpdate,
    PromptResponse,
    PromptDeleteResponse,
)
from app.services import prompt_service

router = APIRouter(
    prefix="/prompts",
    tags=["Prompt Management"]
)


@router.post(
    "/",
    response_model=PromptResponse
)
def create_prompt(
    prompt: PromptCreate,
    db: Session = Depends(get_db)
):
    return prompt_service.create_prompt(
        db, prompt.prompt_name, prompt.prompt_text
    )


@router.get(
    "/",
    response_model=list[PromptResponse]
)
def get_all_prompts(
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db)
):
    return prompt_service.list_prompts(db, limit, offset)


@router.get(
    "/{prompt_id}",
    response_model=PromptResponse
)
def get_prompt(
    prompt_id: int,
    db: Session = Depends(get_db)
):
    return prompt_service.get_prompt(db, prompt_id)


@router.put(
    "/{prompt_id}",
    response_model=PromptResponse
)
def update_prompt(
    prompt_id: int,
    request: PromptUpdate,
    db: Session = Depends(get_db)
):
    return prompt_service.update_prompt(
        db, prompt_id, request.prompt_name, request.prompt_text
    )


@router.delete(
    "/{prompt_id}",
    response_model=PromptDeleteResponse
)
def delete_prompt(
    prompt_id: int,
    db: Session = Depends(get_db)
):
    prompt_service.delete_prompt(db, prompt_id)

    return {
        "message": "Prompt deleted successfully."
    }
