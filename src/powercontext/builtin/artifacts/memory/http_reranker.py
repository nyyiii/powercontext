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

"""HTTP score-based reranking for Memory search candidates."""

from __future__ import annotations

import math

import httpx

from powercontext.builtin.artifacts.memory.models import MemoryHit
from powercontext.builtin.artifacts.memory.reranking import MemoryRerankDecision
from powercontext.builtin.inference import InferenceUsage


class HTTPMemoryReranker:
    """Score bounded candidate batches through a caller-owned HTTP client."""

    def __init__(
        self,
        endpoint: str,
        *,
        client: httpx.AsyncClient,
        batch_size: int = 32,
        policy_id: str = "http-cross-encoder-v1",
    ) -> None:
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")  # noqa: TRY003
        if not isinstance(endpoint, str) or not endpoint:
            raise ValueError("endpoint must be a non-empty string")  # noqa: TRY003
        if not isinstance(policy_id, str) or not policy_id:
            raise ValueError("policy_id must be a non-empty string")  # noqa: TRY003
        self.policy_id = policy_id
        self._endpoint = endpoint
        self._client = client
        self._batch_size = batch_size

    async def rerank(
        self,
        query: str,
        candidates: tuple[MemoryHit, ...],
        limit: int,
        /,
    ) -> MemoryRerankDecision:
        """Score every candidate once and select the globally best original ranks."""

        _validate_bounds(len(candidates), limit)
        scores: dict[int, float] = {}
        requests = 0
        ranked = tuple(enumerate(candidates, start=1))
        for start in range(0, len(ranked), self._batch_size):
            batch = ranked[start : start + self._batch_size]
            response = await self._client.post(
                self._endpoint,
                json={
                    "query": query,
                    "candidates": [{"rank": rank, "text": candidate.text} for rank, candidate in batch],
                },
            )
            requests += 1
            response.raise_for_status()
            scores.update(_response_scores(response, expected_ranks={rank for rank, _candidate in batch}))

        selected = tuple(rank for rank, _score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit])
        return MemoryRerankDecision(
            selected_ranks=selected,
            usage=InferenceUsage(requests=requests),
        )


def _response_scores(response: httpx.Response, *, expected_ranks: set[int]) -> dict[int, float]:
    try:
        payload = response.json()
    except (ValueError, UnicodeDecodeError) as error:
        raise ValueError("reranker response must be valid JSON") from error  # noqa: TRY003
    if not isinstance(payload, dict) or set(payload) != {"scores"}:
        raise ValueError("reranker response must contain exactly scores")  # noqa: TRY003
    raw_scores = payload["scores"]
    if not isinstance(raw_scores, list):
        raise ValueError("reranker scores must be a list")  # noqa: TRY003, TRY004

    scores: dict[int, float] = {}
    for item in raw_scores:
        rank, score = _score_item(item)
        if rank not in expected_ranks or rank in scores:
            raise ValueError("reranker scores must cover each batch rank exactly once")  # noqa: TRY003
        scores[rank] = score
    if set(scores) != expected_ranks:
        raise ValueError("reranker scores must cover each batch rank exactly once")  # noqa: TRY003
    return scores


def _score_item(item: object) -> tuple[int, float]:
    if not isinstance(item, dict) or set(item) != {"rank", "score"}:
        raise ValueError("each reranker score must contain exactly rank and score")  # noqa: TRY003
    rank = item["rank"]
    score = item["score"]
    if type(rank) is not int:
        raise ValueError("reranker score rank must be an integer")  # noqa: TRY003
    if isinstance(score, bool) or not isinstance(score, int | float) or not math.isfinite(score):
        raise ValueError("reranker score must be a finite number")  # noqa: TRY003
    return rank, float(score)


def _validate_bounds(candidate_count: int, limit: int) -> None:
    if candidate_count < 1:
        raise ValueError("rerank requires at least one candidate")  # noqa: TRY003
    if type(limit) is not int or limit < 1 or limit > candidate_count:
        raise ValueError("rerank limit must be between one and the candidate count")  # noqa: TRY003


__all__ = ["HTTPMemoryReranker"]
