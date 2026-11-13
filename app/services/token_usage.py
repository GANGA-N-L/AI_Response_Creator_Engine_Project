from dataclasses import dataclass


@dataclass(frozen=True)
class TokenUsage:
    """
    Token counts for a single AI provider call, in the OpenAI-compatible
    shape both huggingface_service and ollama_service normalize to
    (Ollama's own /api/chat response reports the same counts under
    different field names - prompt_eval_count/eval_count).
    """

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )
