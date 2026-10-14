"""LLM providers behind one small interface: send a system + user prompt, get JSON back that matches a
JSON schema (`json`) or plain prose (`text`, for the briefing: a 4B model writing a long string inside
schema-constrained JSON sometimes degenerates into a list of numbers), plus timing and token counts.

- none:      no LLM; every call raises LLMUnavailable and callers keep their rule-based result
- ollama:    a local model through Ollama's HTTP API (`format` = the JSON schema). The model stays loaded
             during a stage and `unload()` (keep_alive 0) frees its memory right after: on an 8 GB Mac
             it must never sit next to the NLP models.
- anthropic: the Claude API through the official SDK, structured outputs (output_config.format), low effort.
             Token counts come from response.usage; cost from ANTHROPIC_PRICE_INPUT/OUTPUT.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import requests

from config import Settings, settings


class LLMError(Exception):
    """The provider failed or returned something unusable; the caller falls back."""


class LLMUnavailable(LLMError):
    """No LLM configured (LLM_PROVIDER=none) or not reachable."""


@dataclass
class LLMResult:
    data: dict[str, Any]
    seconds: float
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


class Provider(Protocol):
    name: str
    model: str

    def json(self, system: str, prompt: str, schema: dict[str, Any], max_tokens: int = 1024) -> LLMResult: ...

    def text(self, system: str, prompt: str, max_tokens: int = 1024) -> LLMResult:
        """Plain prose in LLMResult.data["text"]."""
        ...

    def unload(self) -> None: ...


def strict(schema: dict[str, Any]) -> dict[str, Any]:
    """Objects get additionalProperties: false and every property required (the Claude API's
    structured outputs need both; Ollama is fine with them too)."""
    out = dict(schema)
    if out.get("type") == "object":
        props = {k: strict(v) for k, v in out.get("properties", {}).items()}
        out.update(properties=props, required=list(props), additionalProperties=False)
    if out.get("type") == "array" and isinstance(out.get("items"), dict):
        out["items"] = strict(out["items"])
    return out


def parse_json(text: str) -> dict[str, Any]:
    try:
        data = json.loads(text)
    except (TypeError, ValueError) as exc:
        raise LLMError(f"not JSON: {str(text)[:120]!r}") from exc
    if not isinstance(data, dict):
        raise LLMError("JSON is not an object")
    return data


class NoneProvider:
    name, model = "none", "rule-based"

    def json(self, system: str, prompt: str, schema: dict[str, Any], max_tokens: int = 1024) -> LLMResult:
        raise LLMUnavailable("LLM_PROVIDER=none")

    def text(self, system: str, prompt: str, max_tokens: int = 1024) -> LLMResult:
        raise LLMUnavailable("LLM_PROVIDER=none")

    def unload(self) -> None:
        pass


class OllamaProvider:
    name = "ollama"

    def __init__(self, cfg: Settings = settings, post: Callable[..., Any] = requests.post,
                 get: Callable[..., Any] = requests.get) -> None:
        self.url, self.model, self.timeout = cfg.ollama_url, cfg.ollama_model, cfg.llm_timeout
        self.post, self.get = post, get

    def available(self) -> tuple[bool, str]:
        """(ok, detail): is the server up and the model pulled?"""
        try:
            tags = self.get(f"{self.url}/api/tags", timeout=5).json()
        except Exception as exc:  # noqa: BLE001
            return False, f"Ollama not reachable at {self.url} ({type(exc).__name__})"
        names = {m.get("name") for m in tags.get("models", [])}
        if self.model in names or f"{self.model}:latest" in names:
            return True, f"{self.model} at {self.url}"
        return False, f"model {self.model} not pulled (ollama pull {self.model})"

    def _chat(self, system: str, prompt: str, max_tokens: int, schema: dict[str, Any] | None) -> tuple[str, dict[str, Any]]:
        body: dict[str, Any] = {
            "model": self.model, "stream": False, "keep_alive": "5m",
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            "options": {"temperature": 0.2, "num_predict": max_tokens, "num_ctx": 4096},
        }
        if schema is not None:
            body["format"] = strict(schema)
        try:
            response = self.post(f"{self.url}/api/chat", json=body, timeout=self.timeout)
        except requests.ConnectionError as exc:
            raise LLMUnavailable(f"Ollama not reachable at {self.url}") from exc
        except requests.Timeout as exc:
            raise LLMError(f"Ollama timed out after {self.timeout:.0f}s") from exc
        if response.status_code != 200:
            raise LLMError(f"Ollama answered {response.status_code}: {response.text[:160]}")
        payload = response.json()
        return (payload.get("message") or {}).get("content", ""), payload

    def json(self, system: str, prompt: str, schema: dict[str, Any], max_tokens: int = 1024) -> LLMResult:
        started = time.perf_counter()
        tokens_in = tokens_out = 0
        for attempt in (1, 2):  # small models now and then emit broken JSON; a second sample is usually fine
            try:
                content, payload = self._chat(system, prompt, max_tokens, schema)
            except LLMUnavailable:
                raise
            except LLMError as exc:  # Ollama stops a model stuck repeating itself with a 500: sample again
                if attempt == 1 and "repeat limit" in str(exc):
                    continue
                raise
            tokens_in += int(payload.get("prompt_eval_count") or 0)
            tokens_out += int(payload.get("eval_count") or 0)
            try:
                data = parse_json(content)
                break
            except LLMError as exc:
                if attempt == 2:
                    raise LLMError(f"{exc} (done: {payload.get('done_reason')}, {len(content)} chars, "
                                   f"ends {content[-60:]!r})") from exc
        return LLMResult(data, time.perf_counter() - started, tokens_in, tokens_out)

    def text(self, system: str, prompt: str, max_tokens: int = 1024) -> LLMResult:
        started = time.perf_counter()
        content, payload = self._chat(system, prompt, max_tokens, None)
        if payload.get("done_reason") == "length":
            raise LLMError(f"Ollama ran out of tokens ({max_tokens}) mid-answer")
        return LLMResult({"text": content.strip()}, time.perf_counter() - started,
                         int(payload.get("prompt_eval_count") or 0), int(payload.get("eval_count") or 0))

    def unload(self) -> None:
        """keep_alive 0 on an empty request unloads the model now (frees ~2 GB)."""
        try:
            self.post(f"{self.url}/api/generate", json={"model": self.model, "keep_alive": 0}, timeout=30)
        except Exception:  # noqa: BLE001 — nothing to free if the server is gone
            pass


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, cfg: Settings = settings, client: Any = None) -> None:
        self.cfg, self.model, self._client = cfg, cfg.anthropic_model, client

    @property
    def client(self) -> Any:
        if self._client is None:
            import anthropic

            # the key comes from .env (ANTHROPIC_API_KEY); timeouts are retried by the SDK (max_retries 2)
            self._client = anthropic.Anthropic(api_key=self.cfg.anthropic_api_key or None, timeout=self.cfg.llm_timeout)
        return self._client

    def _create(self, system: str, prompt: str, max_tokens: int, schema: dict[str, Any] | None) -> tuple[str, LLMResult]:
        import anthropic

        output_config: dict[str, Any] = {"effort": "low"}
        if schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": strict(schema)}
        started = time.perf_counter()
        try:
            response = self.client.messages.create(
                model=self.model, max_tokens=max_tokens + 2048,  # room for (adaptive) thinking before the answer
                system=system, messages=[{"role": "user", "content": prompt}], output_config=output_config,
            )
        except anthropic.AuthenticationError as exc:
            raise LLMUnavailable("ANTHROPIC_API_KEY missing or invalid") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMUnavailable("Claude API not reachable") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"Claude API error {exc.status_code}: {exc.message[:160]}") from exc
        if response.stop_reason in ("refusal", "max_tokens"):
            raise LLMError(f"Claude stopped: {response.stop_reason}")
        text = next((b.text for b in response.content if b.type == "text"), "")
        usage = response.usage
        cost = (usage.input_tokens * self.cfg.anthropic_price_input
                + usage.output_tokens * self.cfg.anthropic_price_output) / 1_000_000
        return text, LLMResult({}, time.perf_counter() - started, usage.input_tokens, usage.output_tokens, cost)

    def json(self, system: str, prompt: str, schema: dict[str, Any], max_tokens: int = 1024) -> LLMResult:
        text, result = self._create(system, prompt, max_tokens, schema)
        result.data = parse_json(text)
        return result

    def text(self, system: str, prompt: str, max_tokens: int = 1024) -> LLMResult:
        text, result = self._create(system, prompt, max_tokens, None)
        result.data = {"text": text.strip()}
        return result

    def unload(self) -> None:
        pass


def get_provider(cfg: Settings = settings) -> Provider:
    if cfg.llm_provider == "ollama":
        return OllamaProvider(cfg)
    if cfg.llm_provider == "anthropic":
        return AnthropicProvider(cfg)
    if cfg.llm_provider not in ("none", ""):
        raise ValueError(f"LLM_PROVIDER must be none, ollama or anthropic, not {cfg.llm_provider!r}")
    return NoneProvider()
