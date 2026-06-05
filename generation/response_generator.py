import logging
from pydantic import BaseModel
from openai import OpenAI
from config import settings

logger = logging.getLogger(__name__)


class GeneratedResponse(BaseModel):
    response_text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ResponseGenerator:

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        max_tokens: int = 512,
        temperature: float = 0.2,
    ):
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._client = OpenAI(api_key=settings.openai_api_key)

    def generate(self, messages: list[dict]) -> GeneratedResponse:
        api_resp = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )
        return GeneratedResponse(
            response_text=api_resp.choices[0].message.content,
            model=api_resp.model,
            prompt_tokens=api_resp.usage.prompt_tokens,
            completion_tokens=api_resp.usage.completion_tokens,
            total_tokens=api_resp.usage.total_tokens,
        )

    def generate_with_fallback(
        self,
        messages: list[dict],
        fallback_messages: list[dict],
    ) -> GeneratedResponse:
        primary = self.generate(messages)
        is_too_short = len(primary.response_text.strip()) < 20
        is_refusal = "i don't know" in primary.response_text.lower()

        if is_too_short or is_refusal:
            logger.info("Primary response insufficient. Attempting fallback.")
            fallback = self.generate(fallback_messages)
            if len(fallback.response_text) > len(primary.response_text):
                return fallback

        return primary
