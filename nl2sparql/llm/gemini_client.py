"""Google Gemini LLM client implementation."""

import os
from typing import Optional

from .base import LLMClient


class GeminiClient(LLMClient):
    """Google Gemini API client."""

    def __init__(self, model: str, api_key: Optional[str] = None, **kwargs):
        super().__init__(model, api_key, **kwargs)

        if self.api_key is None:
            self.api_key = os.environ.get("GEMINI_API_KEY")

        if not self.api_key:
            raise ValueError(
                "Gemini API key not found. Set GEMINI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError(
                "Google Generative AI package not installed. "
                "Install with: pip install nl2sparql[gemini]"
            )

        genai.configure(api_key=self.api_key)
        self._model = genai.GenerativeModel(self.model)

    def complete(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 2048,
        **kwargs,
    ) -> str:
        # Convert messages to Gemini format
        # Gemini uses 'user' and 'model' roles, and handles system differently
        system_instruction = None
        history = []
        last_user_content = ""

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                system_instruction = content
            elif role == "user":
                last_user_content = content
            elif role == "assistant":
                history.append({"role": "model", "parts": [content]})

        # Start chat with history
        chat = self._model.start_chat(history=history)

        # Prepend system instruction to user message if present
        if system_instruction:
            last_user_content = f"{system_instruction}\n\n{last_user_content}"

        # Generate response
        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }

        response = chat.send_message(
            last_user_content,
            generation_config=generation_config,
        )

        return response.text
