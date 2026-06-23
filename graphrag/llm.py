from __future__ import annotations

from openai import OpenAI

from graphrag.config import LlmSettings


class LlmClient:
    def __init__(self, settings: LlmSettings) -> None:
        self._model = settings.model
        self._client = OpenAI(
            api_key=settings.api_key.get_secret_value(),
            base_url=settings.base_url,
        )

    def close(self) -> None:
        self._client.close()

    def complete(
        self,
        prompt: str,
        *,
        max_tokens: int = 1200,
        temperature: float = 0.0,
    ) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        content = response.choices[0].message.content
        if content is None or not content.strip():
            raise EmptyLlmResponseError
        return content.strip()

    def __enter__(self) -> LlmClient:
        return self

    def __exit__(self, *_args: type[BaseException] | BaseException | None) -> None:
        self.close()


class EmptyLlmResponseError(RuntimeError):
    pass
