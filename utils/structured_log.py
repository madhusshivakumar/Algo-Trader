"""Sprint 7A: structured JSON logging layer.

Wraps the existing ``utils.logger.Logger`` with structlog for machine-
readable JSON output, while keeping the Rich console for local dev.

Usage:
    from utils.structured_log import slog, new_trace

    trace = new_trace()
    slog.info("cycle_start", trace_id=trace, symbol="TSLA", cycle=42)

    # Still works: ``from utils.logger import log`` — unchanged.

When ``STRUCTURED_LOG=true`` (env), every log call additionally writes a
JSON line to ``data/logs/structured.jsonl``. The trace_id ties all log
entries from one engine cycle together.

Design:
    * Zero coupling with the existing Rich logger — this is additive.
    * ``slog`` is a module-level structlog logger.
    * ``new_trace()`` returns a uuid4 hex for per-cycle grouping.
    * ``configure_structlog()`` is called once at import time.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime

import structlog

_ENABLED = os.getenv("STRUCTURED_LOG", "false").lower() == "true"
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LOG_DIR = os.path.join(_REPO_ROOT, "data", "logs")
_LOG_FILE = os.path.join(_LOG_DIR, "structured.jsonl")


def new_trace() -> str:
    """Generate a unique trace ID for grouping log entries by engine cycle."""
    return uuid.uuid4().hex[:12]


class _JSONLFileWriter:
    """Append-only JSONL writer. Best-effort — never raises."""

    def __init__(self, path: str):
        self._path = path
        self._initialized = False

    def _ensure_dir(self):
        if not self._initialized:
            os.makedirs(os.path.dirname(self._path), exist_ok=True)
            self._initialized = True

    def write(self, event: dict) -> None:
        if not _ENABLED:
            return
        try:
            self._ensure_dir()
            with open(self._path, "a") as f:
                f.write(json.dumps(event, default=str) + "\n")
        except OSError:
            pass  # never break the engine


_file_writer = _JSONLFileWriter(_LOG_FILE)


def _add_timestamp(_, __, event_dict: dict) -> dict:
    event_dict["timestamp"] = datetime.now().isoformat(timespec="milliseconds")
    return event_dict


def _file_sink(_, __, event_dict: dict) -> dict:
    """Side-effect processor: writes to JSONL file, then returns unchanged."""
    _file_writer.write(event_dict)
    return event_dict


def configure_structlog():
    """Configure structlog once. Idempotent."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            _add_timestamp,
            structlog.stdlib.add_log_level,
            _file_sink,
            structlog.dev.ConsoleRenderer() if not _ENABLED
            else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(0),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


configure_structlog()

# Module-level logger
slog = structlog.get_logger("algo-trader")
