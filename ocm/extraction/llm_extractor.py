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
    "memory. Read the input text and return ONLY a single JSON object (no "
    "prose, no markdown fences).\n"
    "\n"
    "The JSON object MUST have EXACTLY these six top-level keys, each an array "
    "(use [] when there is nothing for that key). Do NOT invent other "
    "top-level keys (no \"tasks\", no \"people\", etc.):\n"
    "{\n"
    '  "entities":  [{"type": "Person|Organization|Project|Task|Event|Decision", '
    '"name": "...", "fields": {...}}],\n'
    '  "events":    [{"type":"...","timestamp_start":"ISO8601|null",'
    '"timestamp_end":"ISO8601|null","description":"..."}],\n'
    '  "claims":    [{"text":"...","confidence":0.0}],\n'
    '  "documents": [{"title":"...","path_or_url":"...","tags":[]}],\n'
    '  "decisions": [{"summary":"...","timestamp":"ISO8601|null",'
    '"made_by":"...|null","status":"draft|final"}],\n'
    '  "relations": [{"subject":"<entity name>","predicate":"OWNS|ASSIGNED_TO|...",'
    '"object":"<entity name>","confidence":0.0,'
    '"write_intent":"new_fact|correction|update"}]\n'
    "}\n"
    "\n"
    "Entity rules: EVERY person, organization, project, task, event, and "
    "decision is an item in \"entities\" with the matching \"type\". Tasks are "
    "entities with type \"Task\" (NOT a separate \"tasks\" key). Put a task's "
    "status in fields, e.g. {\"type\":\"Task\",\"name\":\"T1\","
    '"fields":{"status":"done"}}.\n'
    "\n"
    "STATUS / COMPLETION is NOT a relation. There is no COMPLETED/STARTED/"
    "FINISHED predicate. To say a task is completed/started/blocked/cancelled, "
    "emit the Task entity with fields.status set to one of: todo, in_progress, "
    "blocked, done, cancelled. Examples: \"Bob completed Task T1\" -> "
    '{"type":"Task","name":"T1","fields":{"status":"done"}} (plus, if the doer '
    "matters, an ASSIGNED_TO relation T1->Bob). \"Task T1 is not started\" -> "
    '{"type":"Task","name":"T1","fields":{"status":"todo"}}. Project/Person '
    "status goes in fields.status the same way.\n"
    "\n"
    "Relation direction is STRICT — subject and object types are fixed:\n"
    "- OWNS: subject=Person/Organization, object=Project. (\"Alice owns "
    "Project Orion\" -> subject=Alice, object=Project Orion)\n"
    "- ASSIGNED_TO: subject=Task, object=Person. (\"Bob is assigned to Task "
    "T1\" -> subject=T1, object=Bob — the TASK is the subject)\n"
    "- CONTAINS: subject=Project, object=Task.\n"
    "- MEMBER_OF: subject=Person, object=Organization.\n"
    "- PARTICIPATES_IN: subject=Person, object=Event.\n"
    "- PRECEDES: subject=Event, object=Event.\n"
    "Use only these registered predicates: OWNS, ASSIGNED_TO, CONTAINS, "
    "MEMBER_OF, PARTICIPATES_IN, PRECEDES, SUPPORTS, CONTRADICTS, "
    "EVIDENCE_FOR, RESULTS_IN, ABOUT.\n"
    "\n"
    "Other rules: do not invent IDs (use names); confidence in [0,1]; if a "
    "fact corrects/changes a prior fact set write_intent=\"correction\"; use "
    "null (not the string \"null\") for unknown timestamps.\n"
    "\n"
    "Worked example.\n"
    "Input: \"Alice owns Project Orion. Bob is assigned to Task T1.\"\n"
    "Output: {\"entities\":[{\"type\":\"Person\",\"name\":\"Alice\",\"fields\":{}},"
    "{\"type\":\"Project\",\"name\":\"Project Orion\",\"fields\":{}},"
    "{\"type\":\"Person\",\"name\":\"Bob\",\"fields\":{}},"
    "{\"type\":\"Task\",\"name\":\"T1\",\"fields\":{}}],"
    "\"events\":[],\"claims\":[],\"documents\":[],\"decisions\":[],"
    "\"relations\":[{\"subject\":\"Alice\",\"predicate\":\"OWNS\","
    "\"object\":\"Project Orion\",\"confidence\":0.95,\"write_intent\":\"new_fact\"},"
    "{\"subject\":\"T1\",\"predicate\":\"ASSIGNED_TO\",\"object\":\"Bob\","
    "\"confidence\":0.95,\"write_intent\":\"new_fact\"}]}\n"
    "\n"
    "Worked example (status/completion).\n"
    "Input: \"Bob completed Task T1.\"\n"
    "Output: {\"entities\":[{\"type\":\"Task\",\"name\":\"T1\","
    "\"fields\":{\"status\":\"done\"}},{\"type\":\"Person\",\"name\":\"Bob\","
    "\"fields\":{}}],\"events\":[],\"claims\":[],\"documents\":[],"
    "\"decisions\":[],\"relations\":[{\"subject\":\"T1\","
    "\"predicate\":\"ASSIGNED_TO\",\"object\":\"Bob\",\"confidence\":0.9,"
    "\"write_intent\":\"new_fact\"}]}"
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
