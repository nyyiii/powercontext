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

"""Portable full and incremental wire representations of prepared context."""

# ruff: noqa: TRY003 - Protocol validation errors intentionally carry boundary-specific detail.

from __future__ import annotations

import base64
import binascii
import hashlib
import json
from dataclasses import dataclass

_PACK_SCHEMA = "powercontext.context-pack.v1"
_DELTA_SCHEMA = "powercontext.context-pack-delta.v1"


class ContextPackError(ValueError):
    """A Context Pack or delta violates its wire or reconstruction contract."""


@dataclass(frozen=True, slots=True)
class ContextPack:
    """A complete prepared context split into independently valid UTF-8 chunks."""

    items: tuple[bytes, ...]
    digest: str
    byte_count: int

    @classmethod
    def build(cls, items: tuple[bytes, ...], /) -> ContextPack:
        validated = _validate_items(items, field="items")
        encoded = _encoded_items(validated)
        digest = _pack_digest(encoded)
        byte_count = 0
        while True:
            payload = {"schema": _PACK_SCHEMA, "items": encoded, "digest": digest, "byte_count": byte_count}
            actual = len(_encode_json(payload))
            if actual == byte_count:
                return cls(items=validated, digest=digest, byte_count=actual)
            byte_count = actual

    @classmethod
    def from_content(cls, content: str, /, *, chunk_bytes: int = 512) -> ContextPack:
        if not isinstance(content, str):
            raise ContextPackError("content must be a string")
        if type(chunk_bytes) is not int or chunk_bytes <= 0:
            raise ContextPackError("chunk_bytes must be a positive integer")
        if not content:
            return cls.build(())

        chunks: list[bytes] = []
        for line in content.splitlines(keepends=True):
            chunks.extend(_bounded_utf8_chunks(line, chunk_bytes))
        return cls.build(tuple(chunks))

    def to_bytes(self) -> bytes:
        _validate_items(self.items, field="items")
        payload = {
            "schema": _PACK_SCHEMA,
            "items": _encoded_items(self.items),
            "digest": self.digest,
            "byte_count": self.byte_count,
        }
        encoded = _encode_json(payload)
        if len(encoded) != self.byte_count or _pack_digest(payload["items"]) != self.digest:
            raise ContextPackError("Context Pack fields are inconsistent")
        return encoded

    @classmethod
    def from_bytes(cls, value: bytes, /) -> ContextPack:
        payload = _decode_object(value, fields={"schema", "items", "digest", "byte_count"})
        if payload["schema"] != _PACK_SCHEMA:
            raise ContextPackError("unsupported Context Pack schema")
        if type(payload["byte_count"]) is not int or payload["byte_count"] < 0:
            raise ContextPackError("invalid Context Pack byte_count")
        if not isinstance(payload["digest"], str):
            raise ContextPackError("invalid Context Pack digest")
        items = _decode_items(payload["items"], field="items")
        pack = cls.build(items)
        if (
            payload["digest"] != pack.digest
            or payload["byte_count"] != pack.byte_count
            or len(value) != pack.byte_count
        ):
            raise ContextPackError("Context Pack integrity check failed")
        return pack


def _bounded_utf8_chunks(value: str, limit: int) -> tuple[bytes, ...]:
    chunks: list[bytes] = []
    current = bytearray()
    for character in value:
        encoded = character.encode("utf-8")
        if len(encoded) > limit:
            raise ContextPackError("chunk_bytes cannot contain one UTF-8 character")
        if current and len(current) + len(encoded) > limit:
            chunks.append(bytes(current))
            current.clear()
        current.extend(encoded)
    if current:
        chunks.append(bytes(current))
    return tuple(chunks)


@dataclass(frozen=True, slots=True)
class ContextPackDelta:
    """A target Context Pack represented by a retained base prefix and suffix."""

    base_digest: str
    target_digest: str
    prefix_items: int
    appended: tuple[bytes, ...]

    @property
    def byte_count(self) -> int:
        return len(self.to_bytes())

    @classmethod
    def between(cls, base: ContextPack, target: ContextPack, /) -> ContextPackDelta:
        prefix = 0
        maximum = min(len(base.items), len(target.items))
        while prefix < maximum and base.items[prefix] == target.items[prefix]:
            prefix += 1
        return cls(
            base_digest=base.digest,
            target_digest=target.digest,
            prefix_items=prefix,
            appended=target.items[prefix:],
        )

    def to_bytes(self) -> bytes:
        _validate_delta_fields(self.base_digest, self.target_digest, self.prefix_items, self.appended)
        return _encode_json({
            "schema": _DELTA_SCHEMA,
            "base_digest": self.base_digest,
            "target_digest": self.target_digest,
            "prefix_items": self.prefix_items,
            "appended": _encoded_items(self.appended),
        })

    @classmethod
    def from_bytes(cls, value: bytes, /) -> ContextPackDelta:
        payload = _decode_object(
            value,
            fields={"schema", "base_digest", "target_digest", "prefix_items", "appended"},
        )
        if payload["schema"] != _DELTA_SCHEMA:
            raise ContextPackError("unsupported Context Pack delta schema")
        appended = _decode_items(payload["appended"], field="appended")
        base_digest = payload["base_digest"]
        target_digest = payload["target_digest"]
        prefix_items = payload["prefix_items"]
        _validate_delta_fields(base_digest, target_digest, prefix_items, appended)
        return cls(base_digest, target_digest, prefix_items, appended)


def apply_context_pack_delta(base: ContextPack, delta: ContextPackDelta, /) -> ContextPack:
    """Validate and apply one independently decoded delta to its exact base."""

    _validate_delta_fields(delta.base_digest, delta.target_digest, delta.prefix_items, delta.appended)
    if base.digest != delta.base_digest:
        raise ContextPackError("Context Pack delta base digest does not match")
    if delta.prefix_items > len(base.items):
        raise ContextPackError("Context Pack delta prefix exceeds the base")
    target = ContextPack.build((*base.items[: delta.prefix_items], *delta.appended))
    if target.digest != delta.target_digest:
        raise ContextPackError("Context Pack delta target digest does not match")
    return target


def _validate_items(items: object, *, field: str) -> tuple[bytes, ...]:
    if not isinstance(items, tuple) or not all(isinstance(item, bytes) for item in items):
        raise ContextPackError(f"{field} must be a tuple of bytes")
    for item in items:
        try:
            item.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ContextPackError(f"{field} must contain valid UTF-8 chunks") from error
    return items


def _encoded_items(items: tuple[bytes, ...]) -> list[str]:
    return [base64.b64encode(item).decode("ascii") for item in items]


def _decode_items(value: object, *, field: str) -> tuple[bytes, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ContextPackError(f"{field} must be a Base64 string array")
    try:
        decoded = tuple(base64.b64decode(item, validate=True) for item in value)
    except (binascii.Error, ValueError) as error:
        raise ContextPackError(f"{field} contains invalid Base64") from error
    return _validate_items(decoded, field=field)


def _pack_digest(encoded_items: list[str]) -> str:
    unsigned = _encode_json({"schema": _PACK_SCHEMA, "items": encoded_items})
    return f"sha256:{hashlib.sha256(unsigned).hexdigest()}"


def _encode_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _decode_object(value: bytes, *, fields: set[str]) -> dict[str, object]:
    if not isinstance(value, bytes):
        raise ContextPackError("wire value must be bytes")
    try:
        payload = json.loads(value.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContextPackError("wire value must be valid UTF-8 JSON") from error
    if not isinstance(payload, dict) or set(payload) != fields:
        raise ContextPackError("wire value has missing or extra fields")
    return payload


def _validate_delta_fields(
    base_digest: object,
    target_digest: object,
    prefix_items: object,
    appended: object,
) -> None:
    if not _valid_digest(base_digest) or not _valid_digest(target_digest):
        raise ContextPackError("Context Pack delta contains an invalid digest")
    if type(prefix_items) is not int or prefix_items < 0:
        raise ContextPackError("prefix_items must be a non-negative integer")
    _validate_items(appended, field="appended")


def _valid_digest(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 71 or not value.startswith("sha256:"):
        return False
    hexadecimal = value.removeprefix("sha256:")
    return hexadecimal == hexadecimal.lower() and all(character in "0123456789abcdef" for character in hexadecimal)


__all__ = [
    "ContextPack",
    "ContextPackDelta",
    "ContextPackError",
    "apply_context_pack_delta",
]
