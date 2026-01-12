"""OpenAI LLM client implementation."""

import json
import os
from typing import Any, Optional

from .base import LLMClient

# Models that require max_completion_tokens instead of max_tokens
NEWER_MODELS = {"gpt-5.2","gpt-5.1", "gpt-5", "o1", "o1-mini", "o1-preview", "o3", "o3-mini"}


class OpenAIClient(LLMClient):
    """OpenAI API client."""

    def __init__(self, model: str, api_key: Optional[str] = None, **kwargs):
        super().__init__(model, api_key, **kwargs)

        if self.api_key is None:
            self.api_key = os.environ.get("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "OpenAI package not installed. Install with: pip install nl2sparql[openai]"
            )

        self._client = OpenAI(api_key=self.api_key)

    def _uses_max_completion_tokens(self) -> bool:
        """Check if model uses max_completion_tokens instead of max_tokens."""
        # Check exact match or prefix match for newer models
        for newer_model in NEWER_MODELS:
            if self.model == newer_model or self.model.startswith(f"{newer_model}-"):
                return True
        return False

    def complete(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 2048,
        **kwargs,
    ) -> str:
        # Build request parameters
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            **kwargs,
        }

        # Use appropriate token limit parameter based on model
        if self._uses_max_completion_tokens():
            params["max_completion_tokens"] = max_tokens
        else:
            params["max_tokens"] = max_tokens

        response = self._client.chat.completions.create(**params)
        return response.choices[0].message.content

    def complete_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 2048,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Generate a completion with tool/function calling.

        Args:
            messages: Conversation messages
            tools: Tool definitions in OpenAI format
            temperature: Sampling temperature
            max_tokens: Max tokens

        Returns:
            Dict with 'content' and 'tool_calls'
        """
        # Convert messages to OpenAI format (handle tool results)
        openai_messages = []
        for msg in messages:
            if msg["role"] == "tool":
                # OpenAI uses 'tool' role with tool_call_id
                openai_messages.append({
                    "role": "tool",
                    "content": msg.get("content", ""),
                    "tool_call_id": msg.get("tool_call_id", ""),
                })
            elif msg["role"] == "assistant" and "tool_calls" in msg:
                # Assistant message with tool calls
                tool_calls = []
                for tc in msg["tool_calls"]:
                    tool_calls.append({
                        "id": tc.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["arguments"]) if isinstance(tc["arguments"], dict) else tc["arguments"],
                        }
                    })
                openai_messages.append({
                    "role": "assistant",
                    "content": msg.get("content") or None,
                    "tool_calls": tool_calls,
                })
            else:
                openai_messages.append(msg)

        # Build request parameters
        params = {
            "model": self.model,
            "messages": openai_messages,
            "tools": tools,
            "temperature": temperature,
            **kwargs,
        }

        # Use appropriate token limit parameter based on model
        if self._uses_max_completion_tokens():
            params["max_completion_tokens"] = max_tokens
        else:
            params["max_tokens"] = max_tokens

        response = self._client.chat.completions.create(**params)

        message = response.choices[0].message
        result = {
            "content": message.content or "",
            "tool_calls": [],
        }

        # Extract tool calls if present
        if message.tool_calls:
            for tc in message.tool_calls:
                # Parse arguments from JSON string
                try:
                    args = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    args = {}

                result["tool_calls"].append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": args,
                })

        return result
