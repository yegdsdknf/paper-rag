from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class TokenStreamBuffer:
    max_chunks: int = 8
    max_interval_sec: float = 0.08
    _chunks: list[str] = field(default_factory=list)
    _last_flush: float | None = None

    def append(self, chunk: str, now: float | None = None) -> str:
        if not chunk:
            return ""

        current = time.monotonic() if now is None else now
        if self._last_flush is None:
            self._last_flush = current

        self._chunks.append(chunk)
        reached_chunk_limit = len(self._chunks) >= self.max_chunks
        reached_time_limit = (current - self._last_flush) >= self.max_interval_sec
        if reached_chunk_limit or reached_time_limit:
            return self.flush(now=current)
        return ""

    def flush(self, now: float | None = None) -> str:
        if not self._chunks:
            return ""

        text = "".join(self._chunks)
        self._chunks = []
        self._last_flush = time.monotonic() if now is None else now
        return text
