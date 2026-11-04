"""Mistral AI LLM client implementation."""

import json
import os
import time
from typing import Any, Optional

from .base import LLMClient


def mistral_text(content) -> str:
    """Answer text of a Mistral message: a plain string, or the text chunks of a chunk list
    (reasoning models return [ThinkChunk, TextChunk, ...]; the thinking is discarded).
    Also accepts LangChain-style dict chunks ({"type": "text", "text": ...})."""
    if isinstance(content, str):
        return content.strip()
    parts = []
    for ch in content or []:
        if isinstance(ch, dict):
            if ch.get("type") == "text":
                parts.append(ch.get("text", ""))
        elif getattr(ch, "type", None) == "text":
            parts.append(getattr(ch, "text", ""))
    return "".join(parts).strip()


def log_mistral_call(model: str, response, seconds: float) -> None:
    """Append one JSON line per call to $MISTRAL_LOG (if set), with token usage incl. reasoning."""
    path = os.environ.get("MISTRAL_LOG")
    if not path:
        return
    usage = response.usage
    details = getattr(usage, "completion_tokens_details", None) or {}
    reasoning = details.get("reasoning_tokens") if isinstance(details, dict) else getattr(details, "reasoning_tokens", None)
    record = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "requested_model": model,
        "model_used": response.model,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "reasoning_tokens": reasoning,
        "finish_reason": response.choices[0].finish_reason,
        "duration_s": round(seconds, 2),
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


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

        from mistralai.utils import RetryConfig, BackoffStrategy
        # Retry rate limits (429) and 5xx with exponential backoff (ms), up to 10 min per call
        retry = RetryConfig("backoff", BackoffStrategy(2000, 60000, 2.0, 600000), True)
        self._client = Mistral(api_key=self.api_key, retry_config=retry)

    MIN_MAX_TOKENS = 16000  # floor for reasoning models (thinking counts towards max_tokens)

    def complete(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 2048,
        **kwargs,
    ) -> str:
        started = time.time()
        response = self._client.chat.complete(
            model=self.model,
            messages=messages,
            temperature=temperature,
            top_p=1.0,  # required by recent Mistral models with greedy sampling (temperature 0)
            # Reasoning models (e.g. mistral-large-4) spend part of max_tokens on thinking
            max_tokens=max(max_tokens, self.MIN_MAX_TOKENS),
            **kwargs,
        )
        log_mistral_call(self.model, response, time.time() - started)
        choice = response.choices[0]
        text = mistral_text(choice.message.content)
        if not text and choice.finish_reason == "length":
            raise RuntimeError("Mistral response truncated before any answer text (max_tokens reached while reasoning)")
        return text

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
