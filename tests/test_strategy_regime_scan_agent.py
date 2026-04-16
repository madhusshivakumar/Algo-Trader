"""Tests for agents/strategy_regime_scan.py — Sprint 6F."""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from agents import strategy_regime_scan


def _fake_bars(n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    times = pd.date_range("2024-01-01", periods=n, freq="1h")
    close = 100 + rng.normal(0, 0.2, n).cumsum()
    return pd.DataFrame({
        "time": times, "open": close, "high": close * 1.001,
        "low": close * 0.999, "close": close,
        "volume": np.full(n, 1000),
    })


class TestLoadUniverseBars:
    def test_returns_dict_of_bars(self, monkeypatch):
        captured: list[str] = []

        def _fake_fetch(symbol, days, timeframe, use_cache=True):
            captured.append(symbol)
            return _fake_bars(50)

        fake_mod = MagicMock()
        fake_mod.load_or_fetch_bars = _fake_fetch
        monkeypatch.setattr(
            "agents.strategy_regime_scan.sys.path",
            strategy_regime_scan.sys.path + []
        )
        with patch.dict("sys.modules", {"full_oos_backtest": fake_mod}):
            bars = strategy_regime_scan._load_universe_bars(
                ["TSLA", "AAPL"], days=30, timeframe="hour",
            )
        assert set(bars.keys()) == {"TSLA", "AAPL"}
        assert captured == ["TSLA", "AAPL"]

    def test_empty_bars_skipped(self, monkeypatch):
        def _fake_fetch(symbol, days, timeframe, use_cache=True):
            if symbol == "TSLA":
                return _fake_bars(50)
            return pd.DataFrame()

        fake_mod = MagicMock()
        fake_mod.load_or_fetch_bars = _fake_fetch
        with patch.dict("sys.modules", {"full_oos_backtest": fake_mod}):
            bars = strategy_regime_scan._load_universe_bars(
                ["TSLA", "AAPL"], days=30, timeframe="hour",
            )
        assert list(bars.keys()) == ["TSLA"]


class TestMain:
    def test_dry_run_does_not_write(self, tmp_path, monkeypatch):
        monkeypatch.setattr(strategy_regime_scan,
                            "_load_universe_bars",
                            lambda *a, **kw: {"TSLA": _fake_bars(150)})

        # Redirect the matrix/hard_skip paths into the tmp so we can assert
        # they're NOT created.
        from analytics import strategy_regime_matrix as srm
        matrix_path = tmp_path / "matrix.csv"
        hs_path = tmp_path / "hs.json"
        monkeypatch.setattr(srm, "_MATRIX_CSV", str(matrix_path))
        monkeypatch.setattr(srm, "_HARD_SKIP_JSON", str(hs_path))

        rc = strategy_regime_scan.main(["--dry-run",
                                        "--symbols", "TSLA",
                                        "--days", "30"])
        assert rc == 0
        assert not matrix_path.exists()
        assert not hs_path.exists()

    def test_live_run_writes_both_files(self, tmp_path, monkeypatch):
        monkeypatch.setattr(strategy_regime_scan,
                            "_load_universe_bars",
                            lambda *a, **kw: {"TSLA": _fake_bars(150)})

        from analytics import strategy_regime_matrix as srm
        matrix_path = tmp_path / "matrix.csv"
        hs_path = tmp_path / "hs.json"
        monkeypatch.setattr(srm, "_MATRIX_CSV", str(matrix_path))
        monkeypatch.setattr(srm, "_HARD_SKIP_JSON", str(hs_path))

        rc = strategy_regime_scan.main(["--symbols", "TSLA", "--days", "30"])
        assert rc == 0
        assert matrix_path.exists()
        assert hs_path.exists()

        with open(hs_path) as f:
            payload = json.load(f)
        assert "generated_at" in payload
        assert payload["metadata"]["symbols"] == ["TSLA"]

    def test_no_bars_returns_error_code(self, monkeypatch):
        monkeypatch.setattr(strategy_regime_scan,
                            "_load_universe_bars",
                            lambda *a, **kw: {})
        rc = strategy_regime_scan.main(["--symbols", "TSLA"])
        assert rc == 1

    def test_thresholds_propagate(self, tmp_path, monkeypatch):
        monkeypatch.setattr(strategy_regime_scan,
                            "_load_universe_bars",
                            lambda *a, **kw: {"TSLA": _fake_bars(150)})

        from analytics import strategy_regime_matrix as srm
        matrix_path = tmp_path / "matrix.csv"
        hs_path = tmp_path / "hs.json"
        monkeypatch.setattr(srm, "_MATRIX_CSV", str(matrix_path))
        monkeypatch.setattr(srm, "_HARD_SKIP_JSON", str(hs_path))

        rc = strategy_regime_scan.main(
            ["--symbols", "TSLA", "--days", "30",
             "--min-trades", "7", "--max-avg-pnl", "0.25"])
        assert rc == 0
        with open(hs_path) as f:
            payload = json.load(f)
        assert payload["metadata"]["min_trades"] == 7
        assert payload["metadata"]["max_avg_pnl"] == pytest.approx(0.25)
