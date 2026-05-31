from __future__ import annotations

import time
from typing import Callable


class TraceTimer:
    """集中生成链路阶段耗时，避免主编排散落 perf_counter 计算。"""

    def __init__(self, clock: Callable[[], float] | None = None):
        self.clock = clock or time.perf_counter
        self.total_start = self.clock()

    def start_stage(self) -> float:
        return self.clock()

    def elapsed_since(self, started_at: float) -> float:
        return self.clock() - started_at

    def total_elapsed(self) -> float:
        return self.clock() - self.total_start

    def elapsed_map(self, rewrite: float, retrieve: float, generate: float) -> dict[str, float]:
        return {
            "rewrite": rewrite,
            "retrieve": retrieve,
            "generate": generate,
            "total": self.total_elapsed(),
        }
