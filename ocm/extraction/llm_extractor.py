"""Opt-in OpenAI-compatible LLM extractor (W1).

``LLMExtractor`` is the optional, configuration-gated counterpart to the default
offline ``Mock_Extractor``. It is selected when ``settings.extractor == "llm"``
(Req 3.6) and calls an OpenAI-compatible chat-completions endpoint in JSON mode
using the extraction prompt from the design ("W1 — Extractor").

The returned JSON is validated into an :class:`ExtractionResult` Pydantic model
(Req 3.2). Any failure — request timeout, a non-JSON / malformed response body,
or Pydantic validation failure — is surfaced as an :class:`ExtractionError` so
the Write_Pipeline can reject the input and record a validation failure
(Req 3.3, 3.6).

The HTTP layer is intentionally injectable: a ``client`` callable (taking the
request payload dict and returning the parsed response dict) can be supplied at
construction time, or :meth:`_post` can be overridden. This keeps the class
fully offline-testable — no network call happens until :meth:`extract` is
invoked, and tests can inject a fake client.

Requirements: 3.2, 3.3, 3.6.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Optional

from pydantic import ValidationError

from ocm.core.config import Settings
from ocm.memory.contracts import ExtractionResult

# Prefer importing the shared Extractor protocol + ExtractionError from the
# extraction base module (task 10.1). If it is not present yet, fall back to a
# locally-defined ExtractionError so this module is independently importable.
try:  # pragma: no cover - exercised by whichever ordering tasks run in
    from ocm.extraction.base import ExtractionError
except ImportError:  # pragma: no cover - fallback when base.py not yet created

    class ExtractionError(Exception):
        """Raised when extraction fails (timeout, non-JSON, or invalid output)."""


# The JSON-mode extraction prompt from design.md ("W1 — Extractor").
SYSTEM_PROMPT = (
    "You are an information extraction engine for an ontology-constrained "
    "memory.\n"
    "Extract entities, events, claims, documents, decisions, and relations "
    "from the input.\n"
    "Return ONLY valid JSON matching this schema (no prose):\n"
    "{\n"
    '  "entities":  [{"type": "Person|Organization|Project|Task|...", '
    '"name": "...", "fields": {...}}],\n'
    '  "events":    [{"type":"...","timestamp_start":"ISO8601",'
    '"timestamp_end":"ISO8601|null","description":"..."}],\n'
    '  "claims":    [{"text":"...","confidence":0.0-1.0}],\n'
    '  "documents": [{"title":"...","path_or_url":"...","tags":[]}],\n'
    '  "decisions": [{"summary":"...","timestamp":"ISO8601",'
    '"made_by":"...|null","status":"draft|final|..."}],\n'
    '  "relations": [{"subject":"<name|id>","predicate":"OWNS|ASSIGNED_TO|...",'
    '"object":"<name|id>","confidence":0.0-1.0,'
    '"write_intent":"new_fact|correction|..."}]\n'
    "}\n"
    "Rules: use only registered predicates; do not invent IDs; confidence in "
    "[0,1];\n"
    'if a fact corrects a prior fact, set write_intent="correction".'
)


class LLMExtractor:
    """Optional OpenAI-compatible extractor enabled via configuration (Req 3.6).

    Parameters
    ----------
    settings:
        The OCM :class:`Settings`; supplies ``llm_base_url``, ``llm_api_key``,
        and ``llm_model``. The container/factory is responsible for only
        selecting this class when ``settings.extractor == "llm"``.
    client:
        Optional injectable transport. A callable taking the request payload
        ``dict`` and returning the parsed response ``dict`` (the decoded JSON
        body of the chat-completions call). When provided it is used instead of
        the built-in HTTP client, which keeps the class offline-testable.
    timeout:
        Request timeout in seconds for the built-in HTTP path.
    """

    version: str = "llm-extractor-v1"

    def __init__(
        self,
        settings: Settings,
        client: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
        timeout: float = 30.0,
    ) -> None:
        self.settings = settings
        self._client = client
        self.timeout = timeout

    # -- Public API ---------------------------------------------------------
    def extract(self, text: str, source_ref: str) -> ExtractionResult:
        """Extract candidate memory items from ``text`` via the LLM endpoint.

        Raises :class:`ExtractionError` on timeout, a non-JSON response, or
        Pydantic validation failure (Req 3.3, 3.6).
        """
        payload = self._build_payload(text, source_ref)

        try:
            response = self._post(payload)
        except ExtractionError:
            raise
        except Exception as exc:  # network/timeout/transport errors
            raise ExtractionError(
                f"LLM extraction request failed: {exc!r}"
            ) from exc

        content = self._content_from_response(response)

        try:
            data = json.loads(content)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ExtractionError(
                "LLM extractor returned non-JSON content"
            ) from exc

        if not isinstance(data, dict):
            raise ExtractionError(
                "LLM extractor returned JSON that is not an object"
            )

        # The model is prompted with a schema that omits extractor_version;
        # stamp it so the result is traceable to this extractor.
        data.setdefault("extractor_version", self.version)

        try:
            return ExtractionResult.model_validate(data)
        except ValidationError as exc:
            raise ExtractionError(
                f"LLM extractor output failed Pydantic validation: {exc}"
            ) from exc

    # -- Internals ----------------------------------------------------------
    def _build_messages(self, text: str, source_ref: str) -> list[dict[str, str]]:
        """Build the system + user chat messages for the extraction prompt."""
        user_prompt = f"source_ref={source_ref}\n<<<{text}>>>"
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

    def _build_payload(self, text: str, source_ref: str) -> dict[str, Any]:
        """Build the OpenAI-compatible chat-completions request payload."""
        payload: dict[str, Any] = {
            "model": self.settings.llm_model,
            "messages": self._build_messages(text, source_ref),
            "temperature": 0,
        }
        # JSON mode is opt-out for local servers that reject ``response_format``
        # (the system prompt still demands JSON-only output regardless).
        if getattr(self.settings, "llm_use_json_mode", True):
            payload["response_format"] = {"type": "json_object"}
        return payload

    def _content_from_response(self, response: dict[str, Any]) -> str:
        """Pull the assistant message content out of a chat-completion body."""
        try:
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ExtractionError(
                "LLM extractor response missing choices[0].message.content"
            ) from exc

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST the payload to the chat-completions endpoint.

        Uses the injected ``client`` if one was provided (preferred for tests),
        otherwise performs a real HTTP request. Override this method to supply a
        custom transport.
        """
        if self._client is not None:
            return self._client(payload)
        return self._http_post(payload)

    def _http_post(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Perform the real network call (httpx if available, else urllib)."""
        base_url = (self.settings.llm_base_url or "").rstrip("/")
        if not base_url:
            raise ExtractionError(
                "LLM extractor requires settings.llm_base_url to be configured"
            )
        url = f"{base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.settings.llm_api_key or ''}",
        }

        try:
            import httpx  # type: ignore

            try:
                resp = httpx.post(
                    url, headers=headers, json=payload, timeout=self.timeout
                )
                resp.raise_for_status()
                return resp.json()
            except httpx.TimeoutException as exc:
                raise ExtractionError("LLM extraction request timed out") from exc
            except httpx.HTTPError as exc:
                raise ExtractionError(
                    f"LLM extraction request failed: {exc!r}"
                ) from exc
        except ImportError:
            return self._urllib_post(url, headers, payload)

    def _urllib_post(
        self, url: str, headers: dict[str, str], payload: dict[str, Any]
    ) -> dict[str, Any]:
        """Standard-library fallback transport using urllib."""
        import socket
        import urllib.error
        import urllib.request

        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url, data=data, headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8")
        except socket.timeout as exc:
            raise ExtractionError("LLM extraction request timed out") from exc
        except urllib.error.URLError as exc:
            reason = getattr(exc, "reason", exc)
            if isinstance(reason, socket.timeout):
                raise ExtractionError(
                    "LLM extraction request timed out"
                ) from exc
            raise ExtractionError(
                f"LLM extraction request failed: {exc!r}"
            ) from exc

        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise ExtractionError(
                "LLM extractor returned a non-JSON HTTP body"
            ) from exc
