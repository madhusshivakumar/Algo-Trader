"""Tests for analytics/strategy_regime_matrix.py — Sprint 6F."""

from __future__ import annotations

import json
import os
from datetime import datetime
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from analytics import strategy_regime_matrix as srm


# ── Helpers ────────────────────────────────────────────────────────────────

def _make_bars(n: int, start: datetime | None = None,
               base: float = 100.0, seed: int = 11,
               freq: str = "1h",
               vol: float = 0.01) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = start or datetime(2024, 1, 1, 9, 0)
    times = pd.date_range(start, periods=n, freq=freq)
    rets = rng.normal(0, vol, n).cumsum()
    close = base * np.exp(rets)
    high = close * (1 + np.abs(rng.normal(0, vol * 0.5, n)))
    low = close * (1 - np.abs(rng.normal(0, vol * 0.5, n)))
    opens = np.r_[close[0], close[:-1]]
    volume = rng.integers(1000, 5000, n)
    return pd.DataFrame({
        "time": times, "open": opens, "high": high, "low": low,
        "close": close, "volume": volume,
    })


def _always_buy_then_sell(step: dict):
    """Factory: returns a compute_signals that flips on each call."""
    def _fn(df):
        step["n"] = step.get("n", 0) + 1
        if step["n"] % 2 == 1:
            return {"action": "buy", "strength": 0.5}
        return {"action": "sell", "strength": 0.5}
    return _fn


def _always_hold(df):
    return {"action": "hold", "strength": 0.0}


def _always_buy(df):
    return {"action": "buy", "strength": 0.5}


# ── _regime_for_bar ────────────────────────────────────────────────────────

class TestRegimeForBar:
    def test_short_series_returns_normal(self):
        s = pd.Series([100.0, 100.1, 99.8])
        assert srm._regime_for_bar(s) == "normal"

    def test_flat_series_returns_normal(self):
        s = pd.Series([100.0] * 50)
        assert srm._regime_for_bar(s) == "normal"

    def test_volatile_series_returns_a_known_label(self):
        rng = np.random.default_rng(0)
        s = pd.Series(100 + rng.normal(0, 3, 60).cumsum())
        assert srm._regime_for_bar(s) in ("low_vol", "normal",
                                          "high_vol", "crisis")


# ── _simulate_strategy_on_symbol ───────────────────────────────────────────

class TestSimulateStrategyOnSymbol:
    def test_empty_df_returns_no_trades(self):
        trades = srm._simulate_strategy_on_symbol(
            "mr", _always_buy, "TSLA", pd.DataFrame(),
        )
        assert trades == []

    def test_insufficient_bars_returns_no_trades(self):
        df = _make_bars(50)  # < warmup
        trades = srm._simulate_strategy_on_symbol(
            "mr", _always_buy, "TSLA", df,
        )
        assert trades == []

    def test_hold_only_returns_no_trades(self):
        df = _make_bars(150)
        trades = srm._simulate_strategy_on_symbol(
            "mr", _always_hold, "TSLA", df,
        )
        assert trades == []

    def test_buy_then_sell_records_trades(self):
        df = _make_bars(150)
        step = {}
        fn = _always_buy_then_sell(step)
        trades = srm._simulate_strategy_on_symbol(
            "mr", fn, "TSLA", df,
        )
        # Each pair buy+sell → one closed trade; we expect at least 1
        assert len(trades) >= 1
        for regime, pnl in trades:
            assert regime in ("low_vol", "normal", "high_vol", "crisis")
            assert isinstance(pnl, float)

    def test_open_position_closed_at_end(self):
        """Buy-only signal should still close at final bar."""
        df = _make_bars(150)
        trades = srm._simulate_strategy_on_symbol(
            "mr", _always_buy, "TSLA", df,
        )
        # Buy-only → single open trade closed at end → 1 trade
        assert len(trades) == 1

    def test_exception_in_signal_is_swallowed(self):
        def _boom(df):
            raise ValueError("bad bar")
        df = _make_bars(150)
        trades = srm._simulate_strategy_on_symbol(
            "mr", _boom, "TSLA", df,
        )
        assert trades == []

    def test_cost_model_deducts_on_roundtrip(self):
        df = _make_bars(150, seed=5)
        step = {}

        cost_model = MagicMock()
        cost_model.estimate.return_value = MagicMock(total_cost=1.0)

        trades = srm._simulate_strategy_on_symbol(
            "mr", _always_buy_then_sell(step),
            "TSLA", df, cost_model=cost_model,
        )
        # Estimate called at least twice (buy + sell)
        assert cost_model.estimate.call_count >= 2
        assert len(trades) >= 1

    def test_non_dict_signal_treated_as_hold(self):
        df = _make_bars(150)

        def _bad(df):
            return "oops"
        trades = srm._simulate_strategy_on_symbol(
            "mr", _bad, "TSLA", df,
        )
        assert trades == []

    def test_gross_too_small_skips_buy(self):
        """Strength=0 → gross=0 → no buy → no trade recorded."""
        df = _make_bars(150)

        def _zero_strength(df):
            return {"action": "buy", "strength": 0.0}
        trades = srm._simulate_strategy_on_symbol(
            "mr", _zero_strength, "TSLA", df,
            starting_cash=0.0,
        )
        assert trades == []


# ── _cell_from_pnls ────────────────────────────────────────────────────────

class TestCellFromPnls:
    def test_empty_pnls_returns_zero_cell(self):
        cell = srm._cell_from_pnls("mr", "normal", [])
        assert cell.n_trades == 0
        assert cell.avg_pnl == 0.0
        assert cell.win_rate == 0.0
        assert cell.sharpe_proxy == 0.0

    def test_aggregates_values_correctly(self):
        cell = srm._cell_from_pnls("mr", "normal", [1.0, 2.0, -1.0, 3.0])
        assert cell.n_trades == 4
        assert cell.total_pnl == pytest.approx(5.0)
        assert cell.avg_pnl == pytest.approx(1.25)
        assert cell.win_rate == pytest.approx(0.75)
        assert cell.sharpe_proxy > 0  # positive mean, non-zero std

    def test_single_trade_has_zero_sharpe(self):
        """Std requires ≥ 2 points; single trade → sharpe=0 (not NaN)."""
        cell = srm._cell_from_pnls("mr", "normal", [1.5])
        assert cell.n_trades == 1
        assert cell.sharpe_proxy == 0.0

    def test_to_row_has_all_fields(self):
        cell = srm._cell_from_pnls("mr", "normal", [1.0, 2.0])
        row = cell.to_row()
        assert set(row.keys()) == {
            "strategy", "regime", "n_trades", "total_pnl", "avg_pnl",
            "win_rate", "sharpe_proxy",
        }


# ── build_matrix ───────────────────────────────────────────────────────────

class TestBuildMatrix:
    def test_no_bars_produces_empty_cells(self):
        strategies = {"mr": _always_hold}
        matrix = srm.build_matrix({"TSLA": pd.DataFrame()},
                                  strategies=strategies)
        # One row per (strategy × regime) with zero trades
        assert len(matrix) == 4
        assert (matrix["n_trades"] == 0).all()

    def test_produces_4_rows_per_strategy(self):
        df = _make_bars(150)
        strategies = {"alpha": _always_hold, "beta": _always_hold}
        matrix = srm.build_matrix({"TSLA": df}, strategies=strategies)
        assert len(matrix) == 8  # 2 strategies × 4 regimes
        assert set(matrix["strategy"]) == {"alpha", "beta"}
        assert set(matrix["regime"]) == {"low_vol", "normal",
                                         "high_vol", "crisis"}

    def test_records_trades_from_active_strategy(self):
        df = _make_bars(150, seed=1)
        step = {}
        strategies = {"churn": _always_buy_then_sell(step)}
        matrix = srm.build_matrix({"TSLA": df}, strategies=strategies)
        assert matrix[matrix["strategy"] == "churn"]["n_trades"].sum() >= 1

    def test_none_df_skipped_gracefully(self):
        strategies = {"mr": _always_hold}
        matrix = srm.build_matrix({"TSLA": None}, strategies=strategies)  # type: ignore[dict-item]
        assert (matrix["n_trades"] == 0).all()

    def test_default_strategies_loaded_from_registry(self, monkeypatch):
        """When strategies=None, the function pulls from router.STRATEGY_REGISTRY."""
        monkeypatch.setattr(srm, "_WARMUP_BARS", 10)

        # Inject a stub registry with one harmless strategy
        fake_registry = {"stub": _always_hold}
        import strategies.router as router_mod
        monkeypatch.setattr(router_mod, "STRATEGY_REGISTRY", fake_registry,
                            raising=True)

        df = _make_bars(30)
        matrix = srm.build_matrix({"TSLA": df})  # strategies=None
        assert set(matrix["strategy"]) == {"stub"}


# ── derive_hard_skip ──────────────────────────────────────────────────────

class TestDeriveHardSkip:
    def test_empty_matrix_returns_empty_list(self):
        assert srm.derive_hard_skip(pd.DataFrame()) == []

    def test_pairs_with_enough_trades_and_negative_avg_included(self):
        df = pd.DataFrame([
            {"strategy": "mr", "regime": "crisis",
             "n_trades": 25, "total_pnl": -10.0, "avg_pnl": -0.4,
             "win_rate": 0.3, "sharpe_proxy": -0.2},
            {"strategy": "mr", "regime": "normal",
             "n_trades": 25, "total_pnl": 10.0, "avg_pnl": 0.4,
             "win_rate": 0.7, "sharpe_proxy": 0.2},
        ])
        pairs = srm.derive_hard_skip(df, min_trades=20, max_avg_pnl=0.0)
        assert pairs == [("mr", "crisis")]

    def test_under_sampled_cells_ignored(self):
        df = pd.DataFrame([
            {"strategy": "mr", "regime": "crisis",
             "n_trades": 3, "total_pnl": -10.0, "avg_pnl": -3.0,
             "win_rate": 0.0, "sharpe_proxy": -1.0},
        ])
        assert srm.derive_hard_skip(df, min_trades=20) == []

    def test_zero_avg_treated_as_fail(self):
        """max_avg_pnl=0.0 means avg_pnl==0 is INCLUDED (defensive)."""
        df = pd.DataFrame([
            {"strategy": "mr", "regime": "normal",
             "n_trades": 30, "total_pnl": 0.0, "avg_pnl": 0.0,
             "win_rate": 0.5, "sharpe_proxy": 0.0},
        ])
        pairs = srm.derive_hard_skip(df, min_trades=20, max_avg_pnl=0.0)
        assert ("mr", "normal") in pairs

    def test_custom_thresholds(self):
        df = pd.DataFrame([
            {"strategy": "mr", "regime": "normal",
             "n_trades": 100, "total_pnl": 5.0, "avg_pnl": 0.05,
             "win_rate": 0.55, "sharpe_proxy": 0.1},
        ])
        # tighter threshold: avg_pnl must be <= 0.1 → should catch this one
        pairs = srm.derive_hard_skip(df, min_trades=20, max_avg_pnl=0.1)
        assert ("mr", "normal") in pairs


# ── write_matrix / write_hard_skip / load_hard_skip ───────────────────────

class TestPersistence:
    def test_write_matrix_creates_file(self, tmp_path):
        df = pd.DataFrame([{"strategy": "mr", "regime": "normal",
                            "n_trades": 0, "total_pnl": 0.0,
                            "avg_pnl": 0.0, "win_rate": 0.0,
                            "sharpe_proxy": 0.0}])
        out = tmp_path / "matrix.csv"
        srm.write_matrix(df, path=str(out))
        assert out.exists()
        read = pd.read_csv(out)
        assert len(read) == 1

    def test_write_matrix_swallows_os_error(self, tmp_path):
        df = pd.DataFrame([{"strategy": "x", "regime": "normal",
                            "n_trades": 0, "total_pnl": 0.0,
                            "avg_pnl": 0.0, "win_rate": 0.0,
                            "sharpe_proxy": 0.0}])
        # No such directory — should not raise
        srm.write_matrix(df, path="/dev/null/cannot/create/here.csv")

    def test_write_hard_skip_writes_metadata(self, tmp_path):
        out = tmp_path / "hs.json"
        path = srm.write_hard_skip(
            [("mr", "crisis"), ("momentum", "low_vol")],
            path=str(out), metadata={"note": "unit"},
        )
        assert path == str(out)
        with open(out) as f:
            payload = json.load(f)
        assert payload["overrides"] == [["mr", "crisis"],
                                        ["momentum", "low_vol"]]
        assert payload["metadata"]["note"] == "unit"
        assert "generated_at" in payload

    def test_write_hard_skip_swallows_os_error(self):
        srm.write_hard_skip([("mr", "crisis")],
                            path="/dev/null/cannot/here.json")

    def test_load_hard_skip_missing_file_returns_empty(self, tmp_path):
        out = srm.load_hard_skip(path=str(tmp_path / "nope.json"))
        assert out == set()

    def test_load_hard_skip_reads_overrides(self, tmp_path):
        out = tmp_path / "hs.json"
        srm.write_hard_skip([("mr", "crisis")], path=str(out))
        loaded = srm.load_hard_skip(path=str(out))
        assert loaded == {("mr", "crisis")}

    def test_load_hard_skip_handles_malformed_json(self, tmp_path):
        out = tmp_path / "bad.json"
        out.write_text("{not valid json")
        assert srm.load_hard_skip(path=str(out)) == set()

    def test_load_hard_skip_ignores_non_pair_entries(self, tmp_path):
        out = tmp_path / "hs.json"
        out.write_text(json.dumps({
            "overrides": [["mr", "crisis"], "bad", ["only one"], None],
        }))
        assert srm.load_hard_skip(path=str(out)) == {("mr", "crisis")}


# ── run() orchestration ───────────────────────────────────────────────────

class TestRun:
    def test_run_writes_both_files(self, tmp_path):
        df = _make_bars(150)
        strategies = {"stub": _always_hold}
        matrix_path = tmp_path / "matrix.csv"
        hs_path = tmp_path / "hs.json"
        summary = srm.run(
            bars_by_symbol={"TSLA": df},
            strategies=strategies,
            matrix_path=str(matrix_path),
            hard_skip_path=str(hs_path),
        )
        assert matrix_path.exists()
        assert hs_path.exists()
        assert summary["matrix_rows"] == 4  # 1 strategy × 4 regimes
        assert "symbols" in summary["metadata"]

    def test_run_surfaces_overrides(self, tmp_path):
        df = _make_bars(200, seed=3)
        step = {}
        strategies = {"churn": _always_buy_then_sell(step)}
        matrix_path = tmp_path / "matrix.csv"
        hs_path = tmp_path / "hs.json"

        # With min_trades=1 we guarantee the derived list is non-empty iff
        # a trade's avg is ≤ 0. That's data-dependent; assert shape only.
        summary = srm.run(
            bars_by_symbol={"TSLA": df},
            strategies=strategies,
            matrix_path=str(matrix_path),
            hard_skip_path=str(hs_path),
            min_trades=1, max_avg_pnl=1e9,  # include everything
        )
        assert isinstance(summary["overrides"], list)
        assert summary["metadata"]["min_trades"] == 1
