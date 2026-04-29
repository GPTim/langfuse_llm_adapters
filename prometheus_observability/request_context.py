import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Optional

_current_metrics: ContextVar[Optional["RequestMetrics"]] = ContextVar(
    "gptim_request_metrics", default=None
)


@dataclass
class RequestMetrics:
    request_start: float = field(default_factory=time.monotonic)
    llm_total_time: float = 0.0
    embedder_total_time: float = 0.0
    llm_call_count: int = 0
    embedder_call_count: int = 0

    def add_llm_call(self, duration: float) -> None:
        self.llm_total_time += duration
        self.llm_call_count += 1

    def add_embedder_call(self, duration: float) -> None:
        self.embedder_total_time += duration
        self.embedder_call_count += 1

    def elapsed(self) -> float:
        return time.monotonic() - self.request_start


def start_request() -> RequestMetrics:
    metrics = RequestMetrics()
    _current_metrics.set(metrics)
    return metrics


def get_current() -> Optional[RequestMetrics]:
    return _current_metrics.get()


def end_request() -> Optional[RequestMetrics]:
    metrics = _current_metrics.get()
    _current_metrics.set(None)
    return metrics


def is_tracking_active() -> bool:
    return _current_metrics.get() is not None