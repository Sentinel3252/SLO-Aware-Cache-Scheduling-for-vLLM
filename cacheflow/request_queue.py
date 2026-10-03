"""Bounded-window request queue compatible with vLLM's RequestQueue contract."""

from __future__ import annotations

import heapq
import time
from collections import OrderedDict, deque
from collections.abc import Callable, Iterable, Iterator
from typing import Any

from cacheflow.config import CARASConfig
from cacheflow.cost_model import rank_key
from cacheflow.models import ReorderResult, RequestFeature

FeatureProvider = Callable[[Any, float | None], RequestFeature]
PriorityEntry = tuple[int, float, str, int, Any]


class CARASRequestQueue:
    """Keep baseline order immutable and prepare only a K-sized ranked view.

    FCFS uses an ordered mapping for O(1) append/removal. Priority uses a heap
    plus lazy deletion, preserving O(log N) insertion. ``prepare`` reads only K
    baseline candidates and never sorts or probes the full waiting queue.

    For FCFS, preempted ``prepend_request`` calls live in a protected deque.
    For priority scheduling, prepend follows vLLM's heap semantics and returns
    the request to its priority class. Priority is always an outer ranking key.
    """

    def __init__(
        self,
        policy: str,
        config: CARASConfig,
        feature_provider: FeatureProvider,
    ) -> None:
        if policy not in {"fcfs", "priority"}:
            raise ValueError(f"unsupported vLLM policy: {policy!r}")
        self.policy = policy
        self.config = config
        self.feature_provider = feature_provider
        self._protected: deque[Any] = deque()
        self._fcfs: OrderedDict[str, Any] = OrderedDict()
        self._priority_heap: list[PriorityEntry] = []
        self._priority_active: dict[str, tuple[int, Any]] = {}
        self._priority_generation = 0
        # Ranked view for the current scheduler cycle. Requests stay in their
        # baseline container until pop/remove, so an unfinished cycle can be
        # safely re-ranked with fresh cache state next time.
        self._prepared: deque[Any] = deque()

    @staticmethod
    def _request_id(request: Any) -> str:
        return str(request.request_id)

    @staticmethod
    def _priority_entry(request: Any, generation: int) -> PriorityEntry:
        return (
            int(getattr(request, "priority", 0)),
            float(request.arrival_time),
            str(request.request_id),
            generation,
            request,
        )

    def add_request(self, request: Any) -> None:
        request_id = self._request_id(request)
        if self.policy == "fcfs":
            self._fcfs[request_id] = request
        else:
            self._priority_generation += 1
            generation = self._priority_generation
            self._priority_active[request_id] = (generation, request)
            heapq.heappush(
                self._priority_heap, self._priority_entry(request, generation)
            )

    def pop_request(self) -> Any:
        if self._protected:
            return self._protected.popleft()
        request = self._prepared.popleft() if self._prepared else self._peek_baseline()
        self._remove_baseline(request)
        return request

    def peek_request(self) -> Any:
        if self._protected:
            return self._protected[0]
        if self._prepared:
            return self._prepared[0]
        return self._peek_baseline()

    def prepend_request(self, request: Any) -> None:
        if (
            self.policy == "fcfs"
            and self.config.protect_preempted
            and int(getattr(request, "num_preemptions", 0)) > 0
        ):
            self._protected.appendleft(request)
            return
        if self.policy == "fcfs":
            request_id = self._request_id(request)
            self._fcfs.pop(request_id, None)
            replacement = OrderedDict([(request_id, request)])
            replacement.update(self._fcfs)
            self._fcfs = replacement
        else:
            self.add_request(request)

    def prepend_requests(self, requests: Any) -> None:
        # Repeated appendleft matches FCFSRequestQueue.extendleft: iteration
        # order is reversed at the front.
        for request in requests:
            self.prepend_request(request)

    def remove_request(self, request: Any) -> None:
        try:
            self._protected.remove(request)
            return
        except ValueError:
            pass
        self._remove_baseline(request)

    def remove_requests(self, requests: Iterable[Any]) -> None:
        remove_ids = {self._request_id(request) for request in requests}
        self._protected = deque(
            request
            for request in self._protected
            if self._request_id(request) not in remove_ids
        )
        self._prepared = deque(
            request
            for request in self._prepared
            if self._request_id(request) not in remove_ids
        )
        if self.policy == "fcfs":
            for request_id in remove_ids:
                self._fcfs.pop(request_id, None)
        else:
            for request_id in remove_ids:
                self._priority_active.pop(request_id, None)

    def prepare(self, now: float | None = None) -> ReorderResult:
        """Re-rank exactly the bounded candidate view for one scheduler cycle."""

        started = time.perf_counter_ns()
        current = time.time() if now is None else now
        self._prepared.clear()
        window_size = min(self.config.candidate_window, self._baseline_len())
        before = self._baseline_candidates(window_size)
        featured = [
            (request, self.feature_provider(request, current)) for request in before
        ]

        def key(item: tuple[Any, RequestFeature]) -> tuple[Any, ...]:
            _request, feature = item
            priority_class = feature.priority if self.policy == "priority" else 0
            return (priority_class, *rank_key(feature, self.config))

        featured.sort(key=key)
        after = [request for request, _feature in featured]
        original_position = {id(request): index for index, request in enumerate(before)}
        overtakes = sum(
            max(0, original_position[id(request)] - index)
            for index, request in enumerate(after)
        )
        self._prepared.extend(after)
        elapsed = time.perf_counter_ns() - started
        return ReorderResult(
            candidates_scored=window_size,
            overtakes=overtakes,
            elapsed_ns=elapsed,
            queue_length=len(self),
        )

    def _peek_baseline(self) -> Any:
        candidates = self._baseline_candidates(1)
        if not candidates:
            raise IndexError("peek from an empty queue")
        return candidates[0]

    def _baseline_len(self) -> int:
        return len(self._fcfs) if self.policy == "fcfs" else len(self._priority_active)

    def _baseline_candidates(self, count: int) -> list[Any]:
        if count <= 0:
            return []
        if self.policy == "fcfs":
            candidates: list[Any] = []
            for request in self._fcfs.values():
                candidates.append(request)
                if len(candidates) == count:
                    break
            return candidates

        entries: list[PriorityEntry] = []
        candidates = []
        while self._priority_heap and len(candidates) < count:
            entry = heapq.heappop(self._priority_heap)
            request_id, generation, request = entry[2], entry[3], entry[4]
            if self._priority_active.get(request_id) != (generation, request):
                continue
            entries.append(entry)
            candidates.append(request)
        for entry in entries:
            heapq.heappush(self._priority_heap, entry)
        return candidates

    def _remove_baseline(self, request: Any) -> None:
        request_id = self._request_id(request)
        self._prepared = deque(
            candidate
            for candidate in self._prepared
            if self._request_id(candidate) != request_id
        )
        if self.policy == "fcfs":
            if self._fcfs.pop(request_id, None) is None:
                raise ValueError("request is not in queue")
        else:
            if self._priority_active.pop(request_id, None) is None:
                raise ValueError("request is not in queue")
            # Its heap entry is removed lazily during a future candidate read.

    def __bool__(self) -> bool:
        return bool(self._protected) or self._baseline_len() > 0

    def __len__(self) -> int:
        return len(self._protected) + self._baseline_len()

    def __iter__(self) -> Iterator[Any]:
        yield from self._protected
        prepared_ids = {self._request_id(request) for request in self._prepared}
        yield from self._prepared
        for request in self._baseline_candidates(self._baseline_len()):
            if self._request_id(request) not in prepared_ids:
                yield request
