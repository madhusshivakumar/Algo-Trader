"""Sprint 7C: Sentry error tracking — guarded by SENTRY_DSN env var.

Call ``init_sentry()`` once during startup. If ``SENTRY_DSN`` is empty or
unset, this is a complete no-op (zero overhead, no network calls).

Usage:
    from monitoring.sentry import init_sentry
    init_sentry()  # call once in engine.py / main entry

Captured automatically:
    * Unhandled exceptions in the engine main loop.
    * Agent script failures.
    * Dashboard API handler errors.
"""

from __future__ import annotations

import os

_DSN = os.getenv("SENTRY_DSN", "")
_initialized = False


def init_sentry(dsn: str | None = None) -> bool:
    """Initialize Sentry SDK. Returns True if actually initialized.

    No-op if:
        * ``dsn`` and ``SENTRY_DSN`` are both empty/unset.
        * ``sentry-sdk`` is not installed.
        * Already initialized.
    """
    global _initialized
    if _initialized:
        return True

    dsn = dsn or _DSN
    if not dsn:
        return False

    try:
        import sentry_sdk
        sentry_sdk.init(
            dsn=dsn,
            traces_sample_rate=0.2,
            environment=os.getenv("TRADING_MODE", "paper"),
            release=os.getenv("APP_VERSION", "dev"),
        )
        _initialized = True
        return True
    except Exception:
        return False


def is_initialized() -> bool:
    return _initialized
