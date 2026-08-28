from app.core.config import settings
from app.services import huggingface_service, ollama_service


def generate_response(
    prompt: str,
    text: str,
    customer_id: int
) -> str:
    """
    Generates a response using the AI provider configured via
    settings.ai_provider.
    """

    if settings.ai_provider == "ollama":
        return ollama_service.generate_response(prompt, text, customer_id)

    return huggingface_service.generate_response(prompt, text, customer_id)


def check_availability() -> bool:
    """
    Checks whether the configured AI provider is currently reachable.
    """

    if settings.ai_provider == "ollama":
        return ollama_service.check_availability()

    return huggingface_service.check_availability()
