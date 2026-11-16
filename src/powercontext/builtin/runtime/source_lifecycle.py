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

"""Persistent Source availability propagation across Memory and Topic Memory."""

# ruff: noqa: TRY003 - Lifecycle contract errors intentionally carry boundary-specific detail.

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, field_validator
from sqlalchemy import delete, insert, select, update

from powercontext.artifacts import ArtifactRef, MemoryCitation
from powercontext.builtin.persistence.database import AsyncDatabase
from powercontext.builtin.persistence.tables import (
    SOURCE_LIFECYCLE_EDGES_TABLE,
    SOURCE_LIFECYCLE_EVENTS_TABLE,
    SOURCE_LIFECYCLE_NODES_TABLE,
)
from powercontext.builtin.runtime.models import GetMemoryEntryRequest
from powercontext.builtin.sources import validate_scope_id
from powercontext.builtin.statistics import ModelUsagePurpose
from powercontext.sources import SourceRef

LifecyclePolicy = Literal["all", "any"]
LifecyclePurpose = Literal["generation", "audit"]


class SourceLifecycleError(ValueError):
    """The lifecycle graph, event, or requested use is invalid."""


class SourceLifecycleEvent(BaseModel):
    """One versioned availability transition for a registered Source node."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    node_id: str
    revision: StrictInt = Field(gt=0)
    sequence: StrictInt = Field(gt=0)
    available: StrictBool
    reason: str

    @field_validator("node_id", "reason")
    @classmethod
    def require_nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("lifecycle text fields must not be blank")
        return value


@dataclass(frozen=True, slots=True)
class _Node:
    node_id: str
    kind: Literal["source", "memory", "topic"]
    reference: SourceRef | MemoryCitation | ArtifactRef
    revision: int | None
    parents: tuple[str, ...]
    policy: LifecyclePolicy | None
    available: bool


class SourceLifecycleApplication:
    """Select a persistent lifecycle service for one scope."""

    def __init__(self, runtime: Any, database: AsyncDatabase) -> None:
        self._runtime = runtime
        self._database = database

    def for_scope(self, scope_id: str, /) -> ScopedSourceLifecycleApplication:
        return ScopedSourceLifecycleApplication(self._runtime, self._database, validate_scope_id(scope_id))


class ScopedSourceLifecycleApplication:
    """Register and propagate one scope-local evidence dependency DAG."""

    def __init__(self, runtime: Any, database: AsyncDatabase, scope_id: str) -> None:
        self._runtime = runtime
        self._database = database
        self.scope_id = scope_id

    async def register_source(self, node_id: str, reference: SourceRef, /, *, revision: int) -> None:
        _require_node_id(node_id)
        if type(revision) is not int or revision < 1:
            raise SourceLifecycleError("Source lifecycle revision must be a positive integer")
        try:
            await self._runtime.records.for_scope(self.scope_id).get_source(reference.source_type, reference.source_id)
        except Exception as error:
            raise SourceLifecycleError("registered Source reference is not readable") from error
        await self._register(node_id, "source", reference, revision=revision, parents=(), policy=None, available=True)

    async def register_memory(
        self,
        node_id: str,
        reference: MemoryCitation,
        /,
        *,
        parents: tuple[str, ...],
        policy: LifecyclePolicy,
    ) -> None:
        _validate_derived(node_id, parents, policy)
        try:
            await self._runtime.memory.for_scope(self.scope_id).get(GetMemoryEntryRequest(citation=reference))
        except Exception as error:
            raise SourceLifecycleError("registered Memory citation is not readable") from error
        await self._register_derived(node_id, "memory", reference, parents, policy)

    async def register_topic(
        self,
        node_id: str,
        reference: ArtifactRef,
        /,
        *,
        parents: tuple[str, ...],
        policy: LifecyclePolicy,
    ) -> None:
        _validate_derived(node_id, parents, policy)
        if reference.family != "topic-memory":
            raise SourceLifecycleError("Topic lifecycle reference must use the topic-memory family")
        try:
            await self._runtime.records.for_scope(self.scope_id).get_artifact_revision(
                reference.family, reference.artifact_id, reference.revision
            )
        except Exception as error:
            raise SourceLifecycleError("registered Topic Memory reference is not readable") from error
        await self._register_derived(node_id, "topic", reference, parents, policy)

    async def apply(self, event: SourceLifecycleEvent, /) -> list[str]:
        async with self._runtime._scoped_operation(self.scope_id), self._runtime._locked(self.scope_id):
            nodes = await self._load_nodes()
            node = nodes.get(event.node_id)
            if node is None or node.kind != "source" or node.revision != event.revision:
                raise SourceLifecycleError("event does not match a registered Source lifecycle revision")
            replay = await self._load_event(event.node_id)
            if replay is not None:
                previous_sequence, previous_available, previous_reason = replay
                if event.sequence < previous_sequence:
                    raise SourceLifecycleError("event sequence must increase monotonically")
                if event.sequence == previous_sequence:
                    if (event.available, event.reason) != (previous_available, previous_reason):
                        raise SourceLifecycleError("same-sequence lifecycle event content differs")
                    return sorted(item.node_id for item in nodes.values() if item.available)

            target = _target_availability(nodes, event.node_id, event.available)
            await self._apply_memory_states(nodes, target, event.reason)
            async with self._database.transaction() as connection:
                for node_id, available in target.items():
                    await connection.execute(
                        update(SOURCE_LIFECYCLE_NODES_TABLE)
                        .where(
                            SOURCE_LIFECYCLE_NODES_TABLE.c.scope_id == self.scope_id,
                            SOURCE_LIFECYCLE_NODES_TABLE.c.node_id == node_id,
                        )
                        .values(available=available)
                    )
                await connection.execute(
                    delete(SOURCE_LIFECYCLE_EVENTS_TABLE).where(
                        SOURCE_LIFECYCLE_EVENTS_TABLE.c.scope_id == self.scope_id,
                        SOURCE_LIFECYCLE_EVENTS_TABLE.c.node_id == event.node_id,
                    )
                )
                await connection.execute(
                    insert(SOURCE_LIFECYCLE_EVENTS_TABLE).values(
                        scope_id=self.scope_id,
                        node_id=event.node_id,
                        sequence=event.sequence,
                        available=event.available,
                        reason=event.reason,
                    )
                )
            return sorted(node_id for node_id, available in target.items() if available)

    async def resolve(self, node_id: str, /, *, purpose: LifecyclePurpose) -> SourceRef | MemoryCitation | ArtifactRef:
        if purpose not in {"generation", "audit"}:
            raise SourceLifecycleError("unknown lifecycle resolution purpose")
        nodes = await self._load_nodes()
        node = nodes.get(node_id)
        if node is None:
            raise SourceLifecycleError("unknown lifecycle node")
        if purpose == "generation" and not node.available:
            raise SourceLifecycleError("lifecycle node is unavailable for generation")
        return node.reference

    async def available_topic_refs(self, refs: tuple[ArtifactRef, ...], /) -> tuple[ArtifactRef, ...]:
        nodes = await self._load_nodes()
        unavailable = {
            _artifact_identity(node.reference)
            for node in nodes.values()
            if node.kind == "topic" and not node.available and isinstance(node.reference, ArtifactRef)
        }
        return tuple(ref for ref in refs if _artifact_identity(ref) not in unavailable)

    async def _register_derived(
        self,
        node_id: str,
        kind: Literal["memory", "topic"],
        reference: MemoryCitation | ArtifactRef,
        parents: tuple[str, ...],
        policy: LifecyclePolicy,
    ) -> None:
        async with self._runtime._scoped_operation(self.scope_id), self._runtime._locked(self.scope_id):
            nodes = await self._load_nodes()
            if any(parent not in nodes for parent in parents):
                raise SourceLifecycleError("lifecycle parents must be registered first")
            available = (
                all(nodes[parent].available for parent in parents)
                if policy == "all"
                else any(nodes[parent].available for parent in parents)
            )
            existing = nodes.get(node_id)
            expected = _Node(node_id, kind, reference, None, parents, policy, available)
            if existing is not None:
                if _same_binding(existing, expected):
                    return
                raise SourceLifecycleError("lifecycle node identity is already bound differently")
            if kind == "memory":
                await self._set_memory_state(reference, available, "initial lifecycle registration")
            await self._insert_node(expected)

    async def _register(
        self,
        node_id: str,
        kind: Literal["source", "memory", "topic"],
        reference: SourceRef | MemoryCitation | ArtifactRef,
        *,
        revision: int | None,
        parents: tuple[str, ...],
        policy: LifecyclePolicy | None,
        available: bool,
    ) -> None:
        async with self._runtime._scoped_operation(self.scope_id), self._runtime._locked(self.scope_id):
            nodes = await self._load_nodes()
            expected = _Node(node_id, kind, reference, revision, parents, policy, available)
            existing = nodes.get(node_id)
            if existing is not None:
                if _same_binding(existing, expected):
                    return
                raise SourceLifecycleError("lifecycle node identity is already bound differently")
            await self._insert_node(expected)

    async def _insert_node(self, node: _Node) -> None:
        async with self._database.transaction() as connection:
            await connection.execute(
                insert(SOURCE_LIFECYCLE_NODES_TABLE).values(
                    scope_id=self.scope_id,
                    node_id=node.node_id,
                    kind=node.kind,
                    reference=_dump_reference(node.reference),
                    lifecycle_revision=node.revision,
                    policy=node.policy,
                    available=node.available,
                )
            )
            if node.parents:
                await connection.execute(
                    insert(SOURCE_LIFECYCLE_EDGES_TABLE),
                    [
                        {"scope_id": self.scope_id, "node_id": node.node_id, "parent_id": parent}
                        for parent in node.parents
                    ],
                )

    async def _load_nodes(self) -> dict[str, _Node]:
        async with self._database.transaction() as connection:
            rows = (
                (
                    await connection.execute(
                        select(SOURCE_LIFECYCLE_NODES_TABLE).where(
                            SOURCE_LIFECYCLE_NODES_TABLE.c.scope_id == self.scope_id
                        )
                    )
                )
                .mappings()
                .all()
            )
            edge_rows = (
                (
                    await connection.execute(
                        select(SOURCE_LIFECYCLE_EDGES_TABLE).where(
                            SOURCE_LIFECYCLE_EDGES_TABLE.c.scope_id == self.scope_id
                        )
                    )
                )
                .mappings()
                .all()
            )
        parents: dict[str, list[str]] = {}
        for row in edge_rows:
            parents.setdefault(str(row["node_id"]), []).append(str(row["parent_id"]))
        return {
            str(row["node_id"]): _Node(
                node_id=str(row["node_id"]),
                kind=row["kind"],
                reference=_load_reference(str(row["kind"]), str(row["reference"])),
                revision=None if row["lifecycle_revision"] is None else int(row["lifecycle_revision"]),
                parents=tuple(parents.get(str(row["node_id"]), ())),
                policy=row["policy"],
                available=bool(row["available"]),
            )
            for row in rows
        }

    async def _load_event(self, node_id: str) -> tuple[int, bool, str] | None:
        async with self._database.transaction() as connection:
            row = (
                (
                    await connection.execute(
                        select(SOURCE_LIFECYCLE_EVENTS_TABLE).where(
                            SOURCE_LIFECYCLE_EVENTS_TABLE.c.scope_id == self.scope_id,
                            SOURCE_LIFECYCLE_EVENTS_TABLE.c.node_id == node_id,
                        )
                    )
                )
                .mappings()
                .one_or_none()
            )
        return None if row is None else (int(row["sequence"]), bool(row["available"]), str(row["reason"]))

    async def _apply_memory_states(self, nodes: dict[str, _Node], target: dict[str, bool], reason: str) -> None:
        for node in nodes.values():
            if node.kind == "memory" and node.available != target[node.node_id]:
                await self._set_memory_state(node.reference, target[node.node_id], reason)

    async def _set_memory_state(
        self, reference: MemoryCitation | ArtifactRef | SourceRef, available: bool, reason: str
    ) -> None:
        if not isinstance(reference, MemoryCitation):
            raise SourceLifecycleError("Memory lifecycle node has an invalid reference")
        async with self._runtime._context(
            self.scope_id,
            embedding_purpose=ModelUsagePurpose.MEMORY_INDEXING,
        ) as context:
            service = context.artifacts.memory
            current = await service.head(reference.memory_ref.artifact_id)
            entries = await service.entries(current)
            entry = next((item for item in entries if item.entry_id == reference.entry_id), None)
            if entry is None or entry.entry_version_id != reference.entry_version_id:
                raise SourceLifecycleError("registered Memory citation is no longer the current body version")
            state = next(item.state for item in current.content.manifest.entries if item.entry_id == entry.entry_id)
            desired = "active" if available else "inactive"
            if state == desired:
                return
            if available:
                await service.reactivate(current, entries=(entry,), reason=reason)
            else:
                await service.forget(current, entries=(entry,), reason=reason)


def _target_availability(nodes: dict[str, _Node], changed: str, available: bool) -> dict[str, bool]:
    target = {node_id: node.available for node_id, node in nodes.items()}
    target[changed] = available
    unresolved = {node_id for node_id, node in nodes.items() if node.kind != "source"}
    while unresolved:
        progressed = False
        for node_id in tuple(unresolved):
            node = nodes[node_id]
            if any(parent in unresolved for parent in node.parents):
                continue
            support = tuple(target[parent] for parent in node.parents)
            target[node_id] = all(support) if node.policy == "all" else any(support)
            unresolved.remove(node_id)
            progressed = True
        if not progressed:
            raise SourceLifecycleError("lifecycle dependency graph contains a cycle")
    return target


def _validate_derived(node_id: str, parents: tuple[str, ...], policy: object) -> None:
    _require_node_id(node_id)
    if policy not in {"all", "any"}:
        raise SourceLifecycleError("lifecycle policy must be all or any")
    if not parents:
        raise SourceLifecycleError("derived lifecycle nodes require parents")
    if node_id in parents or len(set(parents)) != len(parents):
        raise SourceLifecycleError("lifecycle parents must be unique and cannot contain the node")
    for parent in parents:
        _require_node_id(parent)


def _require_node_id(value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise SourceLifecycleError("lifecycle node id must be a non-blank string")


def _dump_reference(reference: SourceRef | MemoryCitation | ArtifactRef) -> str:
    return json.dumps(reference.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))


def _load_reference(kind: str, value: str) -> SourceRef | MemoryCitation | ArtifactRef:
    payload = json.loads(value)
    if kind == "source":
        return SourceRef.model_validate(payload)
    if kind == "memory":
        return MemoryCitation.model_validate(payload)
    if kind == "topic":
        return ArtifactRef.model_validate(payload)
    raise SourceLifecycleError("stored lifecycle node kind is invalid")


def _artifact_identity(reference: ArtifactRef) -> tuple[str, str, int]:
    return reference.family, reference.artifact_id, reference.revision


def _same_binding(left: _Node, right: _Node) -> bool:
    return (
        left.node_id,
        left.kind,
        left.reference,
        left.revision,
        left.parents,
        left.policy,
    ) == (
        right.node_id,
        right.kind,
        right.reference,
        right.revision,
        right.parents,
        right.policy,
    )


__all__ = ["SourceLifecycleApplication", "SourceLifecycleError", "SourceLifecycleEvent"]
