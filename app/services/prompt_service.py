from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.exceptions.response_exceptions import PromptNotFoundError
from app.models.prompt import Prompt


async def create_prompt(
    db: AsyncSession, prompt_name: str, prompt_text: str
) -> Prompt:
    new_prompt = Prompt(
        prompt_name=prompt_name,
        prompt_text=prompt_text,
    )

    db.add(new_prompt)
    await db.commit()
    await db.refresh(new_prompt)

    return new_prompt


async def list_prompts(
    db: AsyncSession, limit: int, offset: int
) -> list[Prompt]:
    result = await db.execute(
        select(Prompt).offset(offset).limit(limit)
    )

    return list(result.scalars().all())


async def get_prompt(db: AsyncSession, prompt_id: int) -> Prompt:
    result = await db.execute(
        select(Prompt).where(Prompt.id == prompt_id)
    )

    prompt = result.scalar_one_or_none()

    if prompt is None:
        raise PromptNotFoundError("Prompt not found.")

    return prompt


async def update_prompt(
    db: AsyncSession,
    prompt_id: int,
    prompt_name: str,
    prompt_text: str,
) -> Prompt:
    result = await db.execute(
        select(Prompt).where(Prompt.id == prompt_id)
    )

    prompt = result.scalar_one_or_none()

    if prompt is None:
        raise PromptNotFoundError("Prompt not found.")

    prompt.prompt_name = prompt_name
    prompt.prompt_text = prompt_text

    await db.commit()
    await db.refresh(prompt)

    return prompt


async def delete_prompt(db: AsyncSession, prompt_id: int) -> None:
    result = await db.execute(
        select(Prompt).where(Prompt.id == prompt_id)
    )

    prompt = result.scalar_one_or_none()

    if prompt is None:
        raise PromptNotFoundError("Prompt not found.")

    await db.delete(prompt)
    await db.commit()
