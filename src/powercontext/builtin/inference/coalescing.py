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

"""Concurrent embedding request coalescing with a bounded successful-result cache."""

from __future__ import annotations

import asyncio
import math
from collections import OrderedDict, deque
from dataclasses import dataclass

from powercontext.builtin.inference.models import EmbeddingResult, EmbeddingVector, InferenceUsage
from powercontext.builtin.inference.protocols import EmbeddingModel


@dataclass(frozen=True, slots=True)
class _TextResult:
    vector: EmbeddingVector
    batch_id: int
    usage: InferenceUsage


class CoalescingEmbeddingModel:
    """Share in-flight text embeddings and cache successful vectors by exact text."""

    def __init__(
        self,
        model: EmbeddingModel,
        *,
        max_batch_size: int = 64,
        coalesce_ms: float = 5.0,
        cache_size: int = 1024,
    ) -> None:
        if type(max_batch_size) is not int or max_batch_size < 1:
            raise ValueError("max_batch_size must be a positive integer")  # noqa: TRY003
        if isinstance(coalesce_ms, bool) or not isinstance(coalesce_ms, int | float):
            raise ValueError("coalesce_ms must be a non-negative finite number")  # noqa: TRY003, TRY004
        if not math.isfinite(coalesce_ms) or coalesce_ms < 0:
            raise ValueError("coalesce_ms must be a non-negative finite number")  # noqa: TRY003
        if type(cache_size) is not int or cache_size < 0:
            raise ValueError("cache_size must be a non-negative integer")  # noqa: TRY003
        self.profile = model.profile
        self._model = model
        self._max_batch_size = max_batch_size
        self._coalesce_seconds = float(coalesce_ms) / 1000.0
        self._cache_size = cache_size
        self._cache: OrderedDict[str, EmbeddingVector] = OrderedDict()
        self._inflight: dict[str, asyncio.Future[_TextResult]] = {}
        self._pending: deque[str] = deque()
        self._lock = asyncio.Lock()
        self._dispatcher: asyncio.Task[None] | None = None
        self._next_batch_id = 0

    async def embed(self, texts: tuple[str, ...], /) -> EmbeddingResult:
        """Return ordered vectors while charging each distinct awaited provider batch once."""

        if not texts:
            return EmbeddingResult(vectors=(), usage=InferenceUsage(requests=0))
        unique = tuple(dict.fromkeys(texts))
        vectors: dict[str, EmbeddingVector] = {}
        waiting: dict[str, asyncio.Future[_TextResult]] = {}
        async with self._lock:
            for text in unique:
                cached = self._cache.get(text)
                if cached is not None:
                    self._cache.move_to_end(text)
                    vectors[text] = cached
                    continue
                future = self._inflight.get(text)
                if future is None:
                    future = asyncio.get_running_loop().create_future()
                    future.add_done_callback(_consume_unobserved_exception)
                    self._inflight[text] = future
                    self._pending.append(text)
                waiting[text] = future
            if self._pending and (self._dispatcher is None or self._dispatcher.done()):
                self._dispatcher = asyncio.create_task(self._dispatch(), name="coalescing-embedding-dispatch")

        results = await asyncio.gather(*(asyncio.shield(future) for future in waiting.values()))
        usages: dict[int, InferenceUsage] = {}
        for text, result in zip(waiting, results, strict=True):
            vectors[text] = result.vector
            usages.setdefault(result.batch_id, result.usage)
        return EmbeddingResult(
            vectors=tuple(vectors[text] for text in texts),
            usage=_sum_usage(tuple(usages.values())),
        )

    async def _dispatch(self) -> None:
        await asyncio.sleep(self._coalesce_seconds)
        while True:
            async with self._lock:
                batch = tuple(self._pending.popleft() for _ in range(min(len(self._pending), self._max_batch_size)))
                if not batch:
                    self._dispatcher = None
                    return
                self._next_batch_id += 1
                batch_id = self._next_batch_id
            try:
                result = await self._model.embed(batch)
                _validate_vector_count(result, len(batch))
            except BaseException as error:
                await self._fail_batch(batch, error)
                if isinstance(error, asyncio.CancelledError):
                    raise
            else:
                await self._complete_batch(batch, result, batch_id)

    async def _complete_batch(self, batch: tuple[str, ...], result: EmbeddingResult, batch_id: int) -> None:
        async with self._lock:
            for text, vector in zip(batch, result.vectors, strict=True):
                future = self._inflight.pop(text)
                if self._cache_size:
                    self._cache[text] = vector
                    self._cache.move_to_end(text)
                    while len(self._cache) > self._cache_size:
                        self._cache.popitem(last=False)
                if not future.done():
                    future.set_result(_TextResult(vector=vector, batch_id=batch_id, usage=result.usage))

    async def _fail_batch(self, batch: tuple[str, ...], error: BaseException) -> None:
        async with self._lock:
            for text in batch:
                future = self._inflight.pop(text)
                if not future.done():
                    future.set_exception(error)


def _sum_usage(usages: tuple[InferenceUsage, ...]) -> InferenceUsage:
    return InferenceUsage(
        requests=sum(usage.requests for usage in usages),
        input_tokens=_sum_optional(tuple(usage.input_tokens for usage in usages)),
        output_tokens=_sum_optional(tuple(usage.output_tokens for usage in usages)),
    )


def _validate_vector_count(result: EmbeddingResult, expected: int) -> None:
    if len(result.vectors) != expected:
        raise ValueError("embedding provider returned an unexpected vector count")  # noqa: TRY003


def _sum_optional(values: tuple[int | None, ...]) -> int | None:
    present = tuple(value for value in values if value is not None)
    return None if not present else sum(present)


def _consume_unobserved_exception(future: asyncio.Future[_TextResult]) -> None:
    if not future.cancelled():
        future.exception()


__all__ = ["CoalescingEmbeddingModel"]
