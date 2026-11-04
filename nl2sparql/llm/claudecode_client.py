"""Claude via the Claude Code CLI in headless mode (`claude -p`).

Uses the local Claude Code login, so no API key is needed. Mirrors
ClaudeCodeLLM in mosaic-liita/shared/llm.py: tools disabled, one turn, no
session persistence, the system prompt replaced by ours and the call run in an
empty temporary directory, so the model only sees our prompt.

The CLI does not expose temperature or max_tokens, so those arguments are
ignored. Set CLAUDECODE_LOG to a file path to append one JSON line per call
with the exact model that answered (useful because aliases such as "sonnet"
move to newer models over time).
"""

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Optional

from .base import LLMClient


MAX_ATTEMPTS = 30  # usage-limit waits can span several 5-hour windows


def run_claude_code(
    executable: str,
    model: str,
    prompt: str,
    system: Optional[str] = None,
    timeout: int = 300,
) -> str:
    """Run one headless Claude Code call, waiting out usage limits and CLI updates."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return _run_claude_code_once(executable, model, prompt, system, timeout)
        except FileNotFoundError:
            # The CLI can briefly disappear while it updates itself: wait and look again
            time.sleep(60)
            executable = shutil.which("claude") or executable
        except RuntimeError as e:
            wait = usage_limit_wait_seconds(str(e))
            if wait is None or attempt == MAX_ATTEMPTS:
                raise
            print(f"[claudecode] usage limit reached, waiting {wait / 60:.0f} min: {e}", flush=True)
            time.sleep(wait)
    raise RuntimeError("Claude Code CLI: too many retries")


def usage_limit_wait_seconds(message: str) -> Optional[float]:
    """Seconds to wait if message is a Claude usage-limit error, else None.

    Parses e.g. "You've hit your 5-hour limit - resets 1:30pm (Europe/Rome)" (local time);
    falls back to 15 minutes when the reset time cannot be read.
    """
    if "limit" not in message.lower() or "hit your" not in message.lower():
        return None
    m = re.search(r"resets\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)", message, re.IGNORECASE)
    if not m:
        return 15 * 60
    hour = int(m.group(1)) % 12 + (12 if m.group(3).lower() == "pm" else 0)
    now = datetime.now()
    reset = now.replace(hour=hour, minute=int(m.group(2) or 0), second=0, microsecond=0)
    if reset <= now:
        if now - reset < timedelta(hours=1):
            return 120  # the reset has just happened: retry shortly
        reset += timedelta(days=1)
    return (reset - now).total_seconds() + 120  # small margin after the reset


def _run_claude_code_once(
    executable: str,
    model: str,
    prompt: str,
    system: Optional[str] = None,
    timeout: int = 300,
) -> str:
    """Run one headless Claude Code call and return the text of the answer."""
    # Run in an empty temp dir so no project CLAUDE.md is picked up
    with tempfile.TemporaryDirectory() as workdir:
        cmd = [
            executable, "-p",
            "--output-format", "json",
            "--model", model,
            "--tools", "",
            "--max-turns", "1",
            "--strict-mcp-config",
            "--no-session-persistence",
        ]

        # System prompt goes through a file and the prompt through stdin,
        # to stay under the Windows command-line length limit
        if system:
            system_file = os.path.join(workdir, "system_prompt.txt")
            with open(system_file, "w", encoding="utf-8") as f:
                f.write(system)
            cmd += ["--system-prompt-file", system_file]

        started = time.time()
        proc = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            encoding="utf-8",
            cwd=workdir,
            timeout=timeout,
        )

    try:
        output = json.loads(proc.stdout)
    except json.JSONDecodeError:
        raise RuntimeError(
            f"Claude Code CLI failed (exit {proc.returncode}): {proc.stderr.strip() or proc.stdout.strip()}"
        )

    _log_call(model, output, time.time() - started, len(prompt) + len(system or ""))

    if output.get("is_error"):
        raise RuntimeError(f"Claude Code CLI error: {output.get('result') or output.get('subtype')}")

    return output["result"].strip()


def _log_call(model: str, output: dict, seconds: float, prompt_chars: int) -> None:
    path = os.environ.get("CLAUDECODE_LOG")
    if not path:
        return
    record = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "requested_model": model,
        "models_used": list(output.get("modelUsage", {})),
        "is_error": output.get("is_error"),
        "duration_s": round(seconds, 2),
        "prompt_chars": prompt_chars,
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


class ClaudeCodeClient(LLMClient):
    """LLMClient backed by the Claude Code CLI."""

    DEFAULT_TIMEOUT = 300

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,  # Not needed for Claude Code
        timeout: int = DEFAULT_TIMEOUT,
        **kwargs,
    ):
        super().__init__(model, api_key, **kwargs)
        self.timeout = timeout
        self.executable = shutil.which("claude")
        if not self.executable:
            raise ImportError("Claude Code CLI not found on PATH: https://docs.claude.com/en/docs/claude-code")

    def complete(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 2048,
        **kwargs,
    ) -> str:
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        turns = [m for m in messages if m["role"] != "system"]
        if len(turns) == 1 and turns[0]["role"] == "user":
            prompt = turns[0]["content"]
        else:
            # The CLI takes a single prompt: flatten any earlier turns into it
            prompt = "\n\n".join(f"[{m['role']}]\n{m['content']}" for m in turns)
        return run_claude_code(self.executable, self.model, prompt, system or None, self.timeout)


class ClaudeCodeChat:
    """Minimal LangChain-style chat model (invoke(messages) -> .content) for the agent."""

    _ROLES = {"system": "system", "human": "user", "ai": "assistant"}

    def __init__(self, model: str, timeout: int = ClaudeCodeClient.DEFAULT_TIMEOUT):
        self.model = model
        self._client = ClaudeCodeClient(model=model, timeout=timeout)

    def invoke(self, messages, **kwargs):
        converted = [
            {"role": self._ROLES.get(getattr(m, "type", "human"), "user"), "content": m.content}
            for m in messages
        ]
        return SimpleNamespace(content=self._client.complete(converted))
