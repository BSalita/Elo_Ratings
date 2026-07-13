"""Process and cgroup memory metrics for tuning container -m limits."""

from __future__ import annotations

import os
import pathlib
import threading
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MemoryMetrics:
    cgroup_used_bytes: int | None
    cgroup_limit_bytes: int | None
    process_rss_bytes: int | None
    swap_used_bytes: int | None
    swap_limit_bytes: int | None
    cpu_count: int
    thread_count: int


def _gb(value: int) -> float:
    return value / (1024**3)


def _read_int(path: str | os.PathLike[str]) -> int | None:
    try:
        raw = pathlib.Path(path).read_text(encoding="utf-8").strip()
        if raw in {"max", "9223372036854771712"}:
            return None
        return int(raw)
    except (OSError, ValueError):
        return None


def _read_cgroup_memory() -> tuple[int | None, int | None]:
    v2_used = "/sys/fs/cgroup/memory.current"
    v2_max = "/sys/fs/cgroup/memory.max"
    if os.path.exists(v2_used):
        return _read_int(v2_used), _read_int(v2_max)

    v1_used = "/sys/fs/cgroup/memory/memory.usage_in_bytes"
    v1_max = "/sys/fs/cgroup/memory/memory.limit_in_bytes"
    if os.path.exists(v1_used):
        used = _read_int(v1_used)
        limit = _read_int(v1_max)
        if limit is not None and limit > (1 << 60):
            limit = None
        return used, limit
    return None, None


def _read_cgroup_swap() -> tuple[int | None, int | None]:
    v2_used = "/sys/fs/cgroup/memory.swap.current"
    v2_max = "/sys/fs/cgroup/memory.swap.max"
    if os.path.exists(v2_used):
        return _read_int(v2_used), _read_int(v2_max)

    v1_used = "/sys/fs/cgroup/memory/memory.swap.usage_in_bytes"
    if os.path.exists(v1_used):
        return _read_int(v1_used), None
    return None, None


def get_memory_metrics() -> MemoryMetrics:
    """Return current cgroup/process memory metrics."""
    cgroup_used, cgroup_limit = _read_cgroup_memory()
    swap_used, swap_limit = _read_cgroup_swap()
    process_rss = None
    ram_used = cgroup_used
    ram_total = cgroup_limit

    try:
        import psutil

        process_rss = psutil.Process().memory_info().rss
        if ram_used is None or ram_total is None or ram_total <= 0:
            vm = psutil.virtual_memory()
            ram_used, ram_total = vm.used, vm.total
        if swap_used is None or swap_limit is None:
            sm = psutil.swap_memory()
            swap_used, swap_limit = sm.used, sm.total
    except Exception:
        if ram_used is None or ram_total is None or ram_total <= 0:
            ram_used = ram_total = None
        if swap_used is None:
            swap_used = swap_limit = None

    return MemoryMetrics(
        cgroup_used_bytes=cgroup_used,
        cgroup_limit_bytes=cgroup_limit,
        process_rss_bytes=process_rss,
        swap_used_bytes=swap_used,
        swap_limit_bytes=swap_limit,
        cpu_count=os.cpu_count() or 0,
        thread_count=threading.active_count(),
    )


def update_session_peak_memory(session_state: Any, key: str = "_peak_cgroup_bytes") -> MemoryMetrics:
    """Track peak cgroup usage across Streamlit reruns in this session."""
    metrics = get_memory_metrics()
    used = metrics.cgroup_used_bytes
    if used is not None:
        peak = int(session_state.get(key, 0))
        if used > peak:
            session_state[key] = used
    return metrics


def format_memory_metrics(
    metrics: MemoryMetrics,
    *,
    peak_cgroup_bytes: int | None = None,
) -> str:
    """Compact one-line summary for sidebar/footer display."""
    parts: list[str] = []

    cgroup_used = metrics.cgroup_used_bytes
    cgroup_limit = metrics.cgroup_limit_bytes
    if cgroup_used is not None and cgroup_limit is not None and cgroup_limit > 0:
        pct = cgroup_used / cgroup_limit * 100.0
        parts.append(
            f"cgroup {_gb(cgroup_used):.2f}/{_gb(cgroup_limit):.2f} GB ({pct:.0f}%)"
        )
    elif cgroup_used is not None:
        parts.append(f"cgroup {_gb(cgroup_used):.2f} GB")

    if peak_cgroup_bytes is not None and peak_cgroup_bytes > 0:
        parts.append(f"peak {_gb(peak_cgroup_bytes):.2f} GB")

    if metrics.process_rss_bytes is not None:
        parts.append(f"proc RSS {_gb(metrics.process_rss_bytes):.2f} GB")

    if metrics.swap_limit_bytes and metrics.swap_limit_bytes > 0:
        swap_pct = metrics.swap_used_bytes / metrics.swap_limit_bytes * 100.0
        parts.append(
            f"swap {_gb(metrics.swap_used_bytes or 0):.2f}/"
            f"{_gb(metrics.swap_limit_bytes):.2f} GB ({swap_pct:.0f}%)"
        )
    elif metrics.swap_used_bytes is not None:
        parts.append(f"swap {_gb(metrics.swap_used_bytes):.2f} GB")

    parts.append(f"CPU/threads {metrics.cpu_count}/{metrics.thread_count}")

    if not parts:
        return "Memory: unavailable"
    return "Memory: " + " • ".join(parts)


def get_memory_usage_line() -> str:
    """Return a formatted memory usage line (no session peak)."""
    return format_memory_metrics(get_memory_metrics())


def get_memory_usage_dict(session_state: Any | None = None) -> dict[str, Any]:
    """Structured memory metrics for API health endpoints."""
    metrics = get_memory_metrics()
    peak = None
    if session_state is not None:
        update_session_peak_memory(session_state)
        peak = session_state.get("_peak_cgroup_bytes")

    def _maybe_gb(value: int | None) -> float | None:
        return None if value is None else round(_gb(value), 3)

    cgroup_limit = metrics.cgroup_limit_bytes
    cgroup_used = metrics.cgroup_used_bytes
    cgroup_pct = None
    if cgroup_used is not None and cgroup_limit is not None and cgroup_limit > 0:
        cgroup_pct = round(cgroup_used / cgroup_limit * 100.0, 1)

    return {
        "line": format_memory_metrics(metrics, peak_cgroup_bytes=peak),
        "cgroup_used_gb": _maybe_gb(cgroup_used),
        "cgroup_limit_gb": _maybe_gb(cgroup_limit),
        "cgroup_used_pct": cgroup_pct,
        "peak_cgroup_gb": _maybe_gb(peak),
        "process_rss_gb": _maybe_gb(metrics.process_rss_bytes),
        "swap_used_gb": _maybe_gb(metrics.swap_used_bytes),
        "cpu_count": metrics.cpu_count,
        "thread_count": metrics.thread_count,
    }
