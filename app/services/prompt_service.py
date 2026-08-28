from sqlalchemy.orm import Session

from app.exceptions.response_exceptions import PromptNotFoundError
from app.models.prompt import Prompt


def create_prompt(db: Session, prompt_name: str, prompt_text: str) -> Prompt:
    new_prompt = Prompt(
        prompt_name=prompt_name,
        prompt_text=prompt_text,
    )

    db.add(new_prompt)
    db.commit()
    db.refresh(new_prompt)

    return new_prompt


def list_prompts(db: Session, limit: int, offset: int) -> list[Prompt]:
    return (
        db.query(Prompt)
        .offset(offset)
        .limit(limit)
        .all()
    )


def get_prompt(db: Session, prompt_id: int) -> Prompt:
    prompt = db.query(Prompt).filter(Prompt.id == prompt_id).first()

    if prompt is None:
        raise PromptNotFoundError("Prompt not found.")

    return prompt


def update_prompt(
    db: Session,
    prompt_id: int,
    prompt_name: str,
    prompt_text: str,
) -> Prompt:
    prompt = db.query(Prompt).filter(Prompt.id == prompt_id).first()

    if prompt is None:
        raise PromptNotFoundError("Prompt not found.")

    prompt.prompt_name = prompt_name
    prompt.prompt_text = prompt_text

    db.commit()
    db.refresh(prompt)

    return prompt


def delete_prompt(db: Session, prompt_id: int) -> None:
    prompt = (
        db.query(Prompt)
        .filter(Prompt.id == prompt_id)
        .first()
    )

    if prompt is None:
        raise PromptNotFoundError("Prompt not found.")

    db.delete(prompt)
    db.commit()
