"""Anthropic (Claude) LLM client implementation."""

import json
import os
from typing import Any, Optional

from .base import LLMClient


class AnthropicClient(LLMClient):
    """Anthropic Claude API client."""

    def __init__(self, model: str, api_key: Optional[str] = None, **kwargs):
        super().__init__(model, api_key, **kwargs)

        if self.api_key is None:
            self.api_key = os.environ.get("ANTHROPIC_API_KEY")

        if not self.api_key:
            raise ValueError(
                "Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key parameter."
            )

        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "Anthropic package not installed. Install with: pip install nl2sparql[anthropic]"
            )

        self._client = anthropic.Anthropic(api_key=self.api_key)

    def complete(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 2048,
        **kwargs,
    ) -> str:
        # Anthropic handles system messages separately
        system_content = ""
        filtered_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            else:
                filtered_messages.append(msg)

        response = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system_content if system_content else None,
            messages=filtered_messages,
            temperature=temperature,
            **kwargs,
        )

        # Extract text from content blocks
        return response.content[0].text

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
        # Convert OpenAI tool format to Anthropic format
        anthropic_tools = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool["function"]
                anthropic_tools.append({
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
                })

        # Anthropic handles system messages separately
        system_content = ""
        anthropic_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            elif msg["role"] == "tool":
                # Anthropic uses tool_result content blocks within user messages
                anthropic_messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": msg.get("tool_call_id", ""),
                        "content": msg.get("content", ""),
                    }],
                })
            elif msg["role"] == "assistant" and "tool_calls" in msg:
                # Assistant message with tool calls - convert to Anthropic format
                content = []
                if msg.get("content"):
                    content.append({"type": "text", "text": msg["content"]})
                for tc in msg["tool_calls"]:
                    content.append({
                        "type": "tool_use",
                        "id": tc.get("id", ""),
                        "name": tc["name"],
                        "input": tc["arguments"] if isinstance(tc["arguments"], dict) else json.loads(tc["arguments"]),
                    })
                anthropic_messages.append({
                    "role": "assistant",
                    "content": content,
                })
            else:
                anthropic_messages.append(msg)

        response = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system_content if system_content else None,
            messages=anthropic_messages,
            tools=anthropic_tools if anthropic_tools else None,
            temperature=temperature,
            **kwargs,
        )

        # Extract content and tool calls from response
        result = {
            "content": "",
            "tool_calls": [],
        }

        for block in response.content:
            if block.type == "text":
                result["content"] = block.text
            elif block.type == "tool_use":
                result["tool_calls"].append({
                    "id": block.id,
                    "name": block.name,
                    "arguments": block.input,
                })

        return result
