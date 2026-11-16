# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Captured evidence from one selected branch of a Pi v3 session."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict, JsonValue, model_validator

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources.connectors import ConnectorRunCompletion, ConnectorRunSession, ConnectorRunStatus
from powercontext.sources.definitions import AdapterSourceDefinition
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence

PI_EVENT_SOURCE_NAME = "pi-event"
PI_SESSION_CONNECTOR_NAME = "pi-session-history"
_CHECKPOINT_SCHEMA = "powercontext.pi-session-checkpoint.v1"


class PiSessionInput(BaseModel):
    """One complete offline Pi v3 JSONL archive and an optional selected leaf."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    content: bytes
    leaf_id: str | None = None

    @model_validator(mode="after")
    def validate_leaf_id(self) -> PiSessionInput:
        if self.leaf_id is not None and not self.leaf_id:
            raise ValueError("leaf_id must be non-empty when provided")  # noqa: TRY003
        return self


class PiEventInput(BaseModel):
    """One visible event on a selected Pi session branch."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    session_id: str
    event_id: str
    parent_id: str | None
    event_type: str
    role: str
    content: str
    record: dict[str, JsonValue]


class PiEventSource(Source):
    """Captured immutable evidence for one Pi session event."""

    source_id: str
    session_id: str
    event_id: str
    parent_id: str | None
    event_type: str
    role: str
    content: str
    record: dict[str, JsonValue]


class _ParsedSession:
    def __init__(
        self,
        *,
        session_id: str,
        records: tuple[dict[str, JsonValue], ...],
        by_id: dict[str, dict[str, JsonValue]],
        selected: tuple[PiEventInput, ...],
    ) -> None:
        self.session_id = session_id
        self.records = records
        self.by_id = by_id
        self.selected = selected


def pi_event_source_id(session_id: str, event_id: str) -> str:
    """Return the stable, delimiter-safe identity for one Pi event."""

    _require_non_empty_string("session_id", session_id)
    _require_non_empty_string("event_id", event_id)
    identity = f"pi:{quote(session_id, safe='')}:{quote(event_id, safe='')}"
    if len(identity) > MAX_SOURCE_ID_LENGTH:
        raise ValueError(f"Pi event source identity must not exceed {MAX_SOURCE_ID_LENGTH} characters")  # noqa: TRY003
    return identity


def parse_pi_session(value: PiSessionInput, /) -> tuple[PiEventInput, ...]:
    """Validate a Pi v3 archive and return visible events on its selected branch."""

    return _parse_session(value).selected


def _parse_session(value: PiSessionInput) -> _ParsedSession:
    records = _decode_records(value.content)
    session_id = _session_id(records)
    event_records = records[1:]
    if not event_records:
        if value.leaf_id is not None:
            raise ValueError("selected Pi session leaf does not exist")  # noqa: TRY003
        return _ParsedSession(session_id=session_id, records=(), by_id={}, selected=())

    by_id, parent_by_id = _index_events(event_records)
    leaf_id = value.leaf_id if value.leaf_id is not None else event_records[-1]["id"]
    if leaf_id not in by_id:
        raise ValueError("selected Pi session leaf does not exist")  # noqa: TRY003
    branch_ids = _branch_ids(leaf_id, parent_by_id)
    selected = tuple(
        visible for event_id in branch_ids if (visible := _visible_event(session_id, by_id[event_id])) is not None
    )
    return _ParsedSession(
        session_id=session_id,
        records=tuple(event_records),
        by_id=by_id,
        selected=selected,
    )


def _decode_records(content: bytes) -> list[dict[str, JsonValue]]:
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("Pi session content must be valid UTF-8") from error  # noqa: TRY003

    records: list[dict[str, JsonValue]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            decoded = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError("Pi session content must be valid JSONL") from error  # noqa: TRY003
        if not isinstance(decoded, dict):
            raise ValueError("each Pi session record must be a JSON object")  # noqa: TRY003, TRY004
        records.append(decoded)
    if not records:
        raise ValueError("Pi session archive must contain a header")  # noqa: TRY003
    return records


def _session_id(records: list[dict[str, JsonValue]]) -> str:
    header = records[0]
    session_id = header.get("id")
    if header.get("type") != "session" or header.get("version") != 3:
        raise ValueError("first Pi session record must be a version 3 session header")  # noqa: TRY003
    _require_non_empty_string("session id", session_id)
    return session_id


def _index_events(
    event_records: list[dict[str, JsonValue]],
) -> tuple[dict[str, dict[str, JsonValue]], dict[str, str | None]]:
    by_id: dict[str, dict[str, JsonValue]] = {}
    parent_by_id: dict[str, str | None] = {}
    for record in event_records:
        event_id = record.get("id")
        _require_non_empty_string("event id", event_id)
        if event_id in by_id:
            raise ValueError("Pi session event ids must be unique")  # noqa: TRY003
        if "parentId" not in record:
            raise ValueError("Pi session events must contain parentId")  # noqa: TRY003
        parent_id = record["parentId"]
        if parent_id is not None:
            _require_non_empty_string("parentId", parent_id)
            if parent_id not in by_id:
                raise ValueError("Pi session parent must refer to an earlier event")  # noqa: TRY003
        by_id[event_id] = record
        parent_by_id[event_id] = parent_id
    return by_id, parent_by_id


def _branch_ids(leaf_id: str, parent_by_id: dict[str, str | None]) -> tuple[str, ...]:
    branch_ids: list[str] = []
    visited: set[str] = set()
    current: str | None = leaf_id
    while current is not None:
        if current in visited:
            raise ValueError("Pi session graph contains a cycle")  # noqa: TRY003
        visited.add(current)
        branch_ids.append(current)
        current = parent_by_id[current]
    branch_ids.reverse()
    return tuple(branch_ids)


def _visible_event(session_id: str, record: dict[str, JsonValue]) -> PiEventInput | None:
    event_id = record["id"]
    parent_id = record["parentId"]
    event_type = record.get("type")
    if not isinstance(event_type, str) or not event_type:
        raise ValueError("Pi session event type must be a non-empty string")  # noqa: TRY003

    role: str
    body: str | None
    if event_type == "message":
        message = record.get("message")
        if not isinstance(message, dict):
            raise ValueError("Pi message events must contain a message object")  # noqa: TRY003
        role_value = message.get("role")
        _require_non_empty_string("message role", role_value)
        role = role_value
        if role == "toolResult":
            body = _tool_result_body(message)
        elif role == "bashExecution":
            body = _bash_execution_body(message)
        else:
            body = _visible_content(message.get("content"))
    elif event_type in {"compaction", "branch_summary"}:
        role = event_type
        summary = record.get("summary")
        body = summary if isinstance(summary, str) and summary else None
    elif event_type == "custom_message":
        role = "custom"
        body = _visible_content(record.get("content"))
    else:
        return None

    if body is None:
        return None
    return PiEventInput(
        session_id=session_id,
        event_id=event_id,
        parent_id=parent_id,
        event_type=event_type,
        role=role,
        content=f"{role}: {body}",
        record=record,
    )


def _visible_content(value: JsonValue | None) -> str | None:
    if isinstance(value, str):
        return value if value else None
    if not isinstance(value, list):
        return None
    parts: list[str] = []
    for block in value:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text")
            if isinstance(text, str) and text:
                parts.append(text)
        elif block_type == "toolCall":
            name = block.get("name")
            call_id = block.get("id")
            arguments = block.get("arguments")
            if isinstance(name, str) and name and isinstance(call_id, str) and call_id:
                encoded = json.dumps(arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                parts.append(f"[tool_call {name} {call_id}] {encoded}")
    return "\n".join(parts) if parts else None


def _tool_result_body(message: dict[str, JsonValue]) -> str | None:
    tool_name = message.get("toolName")
    tool_call_id = message.get("toolCallId")
    _require_non_empty_string("toolName", tool_name)
    _require_non_empty_string("toolCallId", tool_call_id)
    text = _visible_content(message.get("content"))
    if text is None:
        return None
    return f"[tool_result {tool_name} {tool_call_id}] {text}"


def _bash_execution_body(message: dict[str, JsonValue]) -> str | None:
    command = message.get("command")
    output = message.get("output")
    if not isinstance(command, str) or not isinstance(output, str):
        return None
    return f"$ {command}\n{output}"


def _require_non_empty_string(field: str, value: object) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")  # noqa: TRY003


class PiEventSourceAdapter:
    """Resolve and read immutable Pi event Sources."""

    input_class = PiEventInput
    name = PI_EVENT_SOURCE_NAME
    source_class = PiEventSource

    async def resolve(self, value: PiEventInput, /) -> PiEventSource:
        identity = pi_event_source_id(value.session_id, value.event_id)
        return PiEventSource(
            name=identity,
            source_id=identity,
            materialization=SourceMaterialization.CAPTURED,
            session_id=value.session_id,
            event_id=value.event_id,
            parent_id=value.parent_id,
            event_type=value.event_type,
            role=value.role,
            content=value.content,
            record=value.record,
        )

    async def read(self, source: PiEventSource, /) -> PiEventInput:
        return PiEventInput(
            session_id=source.session_id,
            event_id=source.event_id,
            parent_id=source.parent_id,
            event_type=source.event_type,
            role=source.role,
            content=source.content,
            record=source.record,
        )


class PiEventTextEvidenceProjection:
    """Project a complete visible Pi event as canonical text evidence."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PiEventSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: PiEventSource, /) -> TextEvidence:
        return TextEvidence(
            source_type=PI_EVENT_SOURCE_NAME,
            source_id=source.source_id,
            content=source.content,
            metadata={
                "session_id": source.session_id,
                "event_id": source.event_id,
                "parent_id": source.parent_id,
                "event_type": source.event_type,
                "role": source.role,
            },
        )


PI_EVENT_SOURCE_ADAPTER = PiEventSourceAdapter()
PI_EVENT_SOURCE_DEFINITION = AdapterSourceDefinition(
    PI_EVENT_SOURCE_ADAPTER,
    projections=(PiEventTextEvidenceProjection(),),
)


class PiSessionConnector:
    """Incrementally submit visible events from one selected Pi session branch."""

    name = PI_SESSION_CONNECTOR_NAME
    version = "1"
    source_definitions = frozenset({PI_EVENT_SOURCE_NAME})

    def __init__(self, path: str | Path, leaf_id: str | None = None) -> None:
        self.path = Path(path)
        self.leaf_id = leaf_id

    async def run(self, session: ConnectorRunSession, /) -> ConnectorRunCompletion:
        parsed = _parse_session(PiSessionInput(content=self.path.read_bytes(), leaf_id=self.leaf_id))
        imported = _decode_checkpoint(session.checkpoint, parsed)
        updated = dict(imported)
        for event in parsed.selected:
            digest = _record_digest(event.record)
            previous = imported.get(event.event_id)
            if previous is not None:
                if previous != digest:
                    raise ValueError(f"previously imported Pi event {event.event_id!r} was modified")  # noqa: TRY003
                continue
            await session.submit(pi_event_source_id(event.session_id, event.event_id), PI_EVENT_SOURCE_NAME, event)
            updated[event.event_id] = digest
        return ConnectorRunCompletion(
            status=ConnectorRunStatus.COMPLETE,
            checkpoint={"schema": _CHECKPOINT_SCHEMA, "session_id": parsed.session_id, "events": updated},
        )


def _decode_checkpoint(checkpoint: JsonValue | None, parsed: _ParsedSession) -> dict[str, str]:
    if checkpoint is None:
        return {}
    if not isinstance(checkpoint, dict) or set(checkpoint) != {"schema", "session_id", "events"}:
        raise ValueError("invalid Pi session checkpoint")  # noqa: TRY003
    if checkpoint["schema"] != _CHECKPOINT_SCHEMA or checkpoint["session_id"] != parsed.session_id:
        raise ValueError("Pi session checkpoint does not match the archive")  # noqa: TRY003
    events = checkpoint["events"]
    if not isinstance(events, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in events.items()
    ):
        raise ValueError("invalid Pi session checkpoint events")  # noqa: TRY003
    imported = dict(events)
    for event_id, digest in imported.items():
        record = parsed.by_id.get(event_id)
        if record is None or _record_digest(record) != digest:
            raise ValueError(f"previously imported Pi event {event_id!r} was modified")  # noqa: TRY003
    return imported


def _record_digest(record: dict[str, JsonValue]) -> str:
    encoded = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


__all__ = [
    "PI_EVENT_SOURCE_ADAPTER",
    "PI_EVENT_SOURCE_DEFINITION",
    "PiEventInput",
    "PiEventSource",
    "PiSessionConnector",
    "PiSessionInput",
    "parse_pi_session",
    "pi_event_source_id",
]
