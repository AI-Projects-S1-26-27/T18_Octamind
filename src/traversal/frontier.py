"""Priority-queue frontier for best-first search (Module 2: frontier + reached)."""
from __future__ import annotations

import heapq
import itertools


class Frontier:
    """Min-heap ordered by f-score. A counter makes ties deterministic and nodes incomparable."""

    def __init__(self):
        self._heap = []
        self._counter = itertools.count()

    def push(self, node, f: float) -> None:
        heapq.heappush(self._heap, (float(f), next(self._counter), node))

    def pop(self):
        f, _, node = heapq.heappop(self._heap)
        return node, f

    def peek_f(self) -> float:
        return self._heap[0][0]

    def __len__(self) -> int:
        return len(self._heap)

    def keep_best(self, k=None, margin=None) -> int:
        """Apply beam width ``k`` and/or score margin; return how many nodes were removed."""
        if not self._heap or (k is None and margin is None):
            return 0
        before = len(self._heap)
        items = sorted(self._heap)
        if margin is not None:
            best = items[0][0]
            items = [item for item in items if item[0] <= best + margin]
        if k is not None:
            items = items[:k]
        self._heap = items
        heapq.heapify(self._heap)
        return before - len(self._heap)
