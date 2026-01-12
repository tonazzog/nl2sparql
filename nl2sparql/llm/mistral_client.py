"""Mistral AI LLM client implementation."""

import json
import os
from typing import Any, Optional

from .base import LLMClient


class MistralClient(LLMClient):
    """Mistral AI API client with tool calling support."""

    def __init__(self, model: str, api_key: Optional[str] = None, **kwargs):
        super().__init__(model, api_key, **kwargs)

        if self.api_key is None:
            self.api_key = os.environ.get("MISTRAL_API_KEY")

        if not self.api_key:
            raise ValueError(
                "Mistral API key not found. Set MISTRAL_API_KEY environment variable "
                "or pass api_key parameter."
            )

        try:
            from mistralai import Mistral
        except ImportError:
            raise ImportError(
                "Mistral package not installed. Install with: pip install nl2sparql[mistral]"
            )

        self._client = Mistral(api_key=self.api_key)

    def complete(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 2048,
        **kwargs,
    ) -> str:
        response = self._client.chat.complete(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
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
        # Convert messages to Mistral format (handle tool results)
        mistral_messages = []
        for msg in messages:
            if msg["role"] == "tool":
                # Mistral uses 'tool' role with tool_call_id
                mistral_messages.append({
                    "role": "tool",
                    "name": msg.get("name", ""),
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
                mistral_messages.append({
                    "role": "assistant",
                    "content": msg.get("content", ""),
                    "tool_calls": tool_calls,
                })
            else:
                mistral_messages.append(msg)

        response = self._client.chat.complete(
            model=self.model,
            messages=mistral_messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

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
