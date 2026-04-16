"""Tests for scripts/full_oos_backtest.py — Sprint 6A.

Covers the reusable functions only; the full `main()` CLI smoke-test
mocks the broker so the suite stays fast.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Import module path used by the CLI.
import importlib
_MOD_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                         "scripts")
if _MOD_PATH not in sys.path:
    sys.path.insert(0, _MOD_PATH)
full_oos_backtest = importlib.import_module("full_oos_backtest")


# ── Helpers ──────────────────────────────────────────────────────────────

def _make_bars(n: int, start: datetime | None = None,
               base: float = 100.0, seed: int = 7,
               freq: str = "1h") -> pd.DataFrame:
    """Deterministic OHLCV bars with a `time` column."""
    rng = np.random.default_rng(seed)
    start = start or datetime(2024, 1, 1, 9, 30)
    times = pd.date_range(start, periods=n, freq=freq)
    rets = rng.normal(0, 0.005, n).cumsum()
    close = base * np.exp(rets)
    high = close * (1 + np.abs(rng.normal(0, 0.002, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.002, n)))
    opens = np.r_[close[0], close[:-1]]
    vol = rng.integers(1000, 5000, n)
    return pd.DataFrame({
        "time": times, "open": opens, "high": high, "low": low,
        "close": close, "volume": vol,
    })


def _stub_signal(symbol: str, window: pd.DataFrame,
                 regime: str | None = None) -> dict:
    """Simple alternating buy/sell signal — deterministic, testable."""
    # Use last close parity to flip every other bar
    last = int(window["close"].iloc[-1] * 100)
    if last % 3 == 0:
        return {"action": "buy", "strength": 0.5}
    if last % 3 == 1:
        return {"action": "sell", "strength": 0.5}
    return {"action": "hold", "strength": 0.0}


# ── load_or_fetch_bars ───────────────────────────────────────────────────

class TestLoadOrFetch:
    def test_reads_cache_when_present(self, tmp_path, monkeypatch):
        monkeypatch.setattr(full_oos_backtest, "_CACHE_DIR", str(tmp_path))
        now = datetime(2026, 4, 14)
        df = _make_bars(50)
        cache = full_oos_backtest._cache_path("TSLA", now - timedelta(days=30),
                                              now, "hour")
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        df.to_parquet(cache, index=False)
        broker = MagicMock()
        out = full_oos_backtest.load_or_fetch_bars(
            "TSLA", days=30, timeframe="hour",
            broker=broker, now=now,
        )
        assert len(out) == 50
        broker.get_historical_bars.assert_not_called()

    def test_fetches_and_caches_on_miss(self, tmp_path, monkeypatch):
        monkeypatch.setattr(full_oos_backtest, "_CACHE_DIR", str(tmp_path))
        now = datetime(2026, 4, 14)
        broker = MagicMock()
        broker.get_historical_bars.return_value = _make_bars(50)
        out = full_oos_backtest.load_or_fetch_bars(
            "TSLA", days=30, timeframe="hour",
            broker=broker, now=now,
        )
        assert len(out) > 0
        broker.get_historical_bars.assert_called_once()
        # Cache written
        cache = full_oos_backtest._cache_path("TSLA", now - timedelta(days=30),
                                              now, "hour")
        assert os.path.exists(cache)

    def test_broker_failure_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(full_oos_backtest, "_CACHE_DIR", str(tmp_path))
        broker = MagicMock()
        broker.get_historical_bars.side_effect = RuntimeError("boom")
        out = full_oos_backtest.load_or_fetch_bars(
            "TSLA", days=30, timeframe="hour",
            broker=broker, now=datetime(2026, 4, 14),
        )
        assert out.empty

    def test_no_cache_bypasses_parquet(self, tmp_path, monkeypatch):
        monkeypatch.setattr(full_oos_backtest, "_CACHE_DIR", str(tmp_path))
        now = datetime(2026, 4, 14)
        df = _make_bars(5)
        cache = full_oos_backtest._cache_path("TSLA", now - timedelta(days=30),
                                              now, "hour")
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        df.to_parquet(cache, index=False)
        broker = MagicMock()
        broker.get_historical_bars.return_value = _make_bars(50)
        full_oos_backtest.load_or_fetch_bars(
            "TSLA", days=30, timeframe="hour",
            broker=broker, use_cache=False, now=now,
        )
        broker.get_historical_bars.assert_called_once()


# ── _resample_ohlcv ──────────────────────────────────────────────────────

class TestResample:
    def test_hourly_aggregates(self):
        # Hour-aligned start so 120 min → exactly 2 hour buckets
        df = _make_bars(120, start=datetime(2024, 1, 1, 9, 0), freq="1min")
        out = full_oos_backtest._resample_ohlcv(df, "hour")
        assert len(out) == 2
        assert set(out.columns) >= {"time", "open", "high", "low", "close",
                                    "volume"}

    def test_daily_aggregates(self):
        df = _make_bars(60 * 24 * 3, freq="1min")
        out = full_oos_backtest._resample_ohlcv(df, "day")
        assert len(out) >= 2

    def test_unknown_tf_returns_input(self):
        df = _make_bars(10, freq="1h")
        out = full_oos_backtest._resample_ohlcv(df, "foo")
        assert len(out) == 10


# ── _regime_for_bar ──────────────────────────────────────────────────────

class TestRegimeForBar:
    def test_short_series_returns_normal(self):
        s = pd.Series([100.0] * 5)
        assert full_oos_backtest._regime_for_bar(s) == "normal"

    def test_flat_series_returns_normal(self):
        s = pd.Series([100.0] * 50)
        # Zero vol → classify_vol returns "normal"
        assert full_oos_backtest._regime_for_bar(s) == "normal"

    def test_volatile_series_produces_regime_label(self):
        rng = np.random.default_rng(0)
        s = pd.Series(100 + rng.normal(0, 2, 50).cumsum())
        r = full_oos_backtest._regime_for_bar(s)
        assert r in ("low_vol", "normal", "high_vol", "crisis")


# ── simulate_symbol ──────────────────────────────────────────────────────

class TestSimulateSymbol:
    def test_empty_df_returns_zero_result(self):
        r = full_oos_backtest.simulate_symbol(
            "TSLA", pd.DataFrame(), starting_cash=10_000,
            cost_model=None, annualize=252,
        )
        assert r.trades == 0
        assert r.final_equity == 10_000

    def test_insufficient_data_returns_zero(self):
        r = full_oos_backtest.simulate_symbol(
            "TSLA", _make_bars(50), starting_cash=10_000,
            cost_model=None, annualize=252,
        )
        assert r.trades == 0

    def test_runs_end_to_end_with_stub_signals(self):
        df = _make_bars(200)
        r = full_oos_backtest.simulate_symbol(
            "TSLA", df, starting_cash=10_000,
            cost_model=None, annualize=252,
            compute_signals=_stub_signal,
        )
        assert r.bars == 200
        # Stub should produce at least one trade over 200 bars
        assert r.trades >= 1
        assert isinstance(r.sharpe, float)

    def test_regime_breakdown_populated(self):
        df = _make_bars(250)
        r = full_oos_backtest.simulate_symbol(
            "TSLA", df, starting_cash=10_000,
            cost_model=None, annualize=252,
            compute_signals=_stub_signal,
        )
        if r.trades > 0:
            # Should have at least one regime bucket
            assert len(r.regime_breakdown) >= 1
            for stats in r.regime_breakdown.values():
                assert "n_trades" in stats
                assert "win_rate" in stats

    def test_costs_are_deducted(self):
        """With costs enabled, final equity should be lower for same trades."""
        df = _make_bars(200)

        # No-cost run
        r0 = full_oos_backtest.simulate_symbol(
            "TSLA", df, starting_cash=10_000,
            cost_model=None, annualize=252,
            compute_signals=_stub_signal,
        )

        # High-cost run
        from core.transaction_costs import TransactionCostModel
        model = TransactionCostModel(commission_pct=0.005)
        r1 = full_oos_backtest.simulate_symbol(
            "TSLA", df, starting_cash=10_000,
            cost_model=model, annualize=252,
            compute_signals=_stub_signal,
        )
        if r1.trades > 0:
            assert r1.total_costs > 0.0
            assert r1.final_equity < r0.final_equity

    def test_signal_exception_does_not_crash(self):
        def bad_signal(sym, w, regime=None):
            raise ValueError("bad")
        r = full_oos_backtest.simulate_symbol(
            "TSLA", _make_bars(200), starting_cash=10_000,
            cost_model=None, annualize=252,
            compute_signals=bad_signal,
        )
        assert r.trades == 0


# ── aggregate + monte carlo + acceptance ─────────────────────────────────

class TestAggregateAndMC:
    def _fake_result(self, sym, final=10_500, trades=5, sharpe=1.2):
        return full_oos_backtest.SymbolResult(
            symbol=sym, bars=200, trades=trades, wins=3, losses=2,
            total_return=0.05, sharpe=sharpe, sortino=1.4, calmar=0.3,
            max_drawdown=0.08, win_rate=0.6, total_costs=12.5,
            final_equity=final,
            regime_breakdown={"normal": {"n_trades": trades, "total_pnl": 100,
                                          "avg_pnl": 20.0, "win_rate": 0.6}},
        )

    def test_aggregate_sums_equity(self):
        rs = [self._fake_result("A"), self._fake_result("B", final=9_500)]
        summary = full_oos_backtest.aggregate_portfolio(rs, 10_000, 252)
        assert summary["total_starting"] == 20_000
        assert summary["total_final"] == 20_000  # 10500 + 9500
        assert summary["total_return"] == 0.0
        assert summary["trade_count"] == 10

    def test_monte_carlo_ranges(self):
        rng = np.random.default_rng(42)
        pnls = rng.normal(0, 10, 50).tolist()
        mc = full_oos_backtest.monte_carlo_shuffle(pnls, 10_000, runs=500)
        assert mc["runs"] == 500
        assert mc["p5"] <= mc["p50"] <= mc["p95"]

    def test_monte_carlo_empty_pnls(self):
        mc = full_oos_backtest.monte_carlo_shuffle([], 10_000, runs=100)
        assert mc["runs"] == 0
        assert mc["p50"] == 10_000

    def test_monte_carlo_zero_runs(self):
        mc = full_oos_backtest.monte_carlo_shuffle([1.0, -2.0], 10_000, runs=0)
        assert mc["runs"] == 0

    def test_monte_carlo_deterministic_with_seed(self):
        pnls = [10.0, -5.0, 3.0, 7.0, -2.0]
        a = full_oos_backtest.monte_carlo_shuffle(pnls, 10_000, runs=100,
                                                  seed=42)
        b = full_oos_backtest.monte_carlo_shuffle(pnls, 10_000, runs=100,
                                                  seed=42)
        assert a == b

    def test_evaluate_acceptance_pass(self):
        summary = {"portfolio_sharpe_mean": 1.5}
        res = full_oos_backtest.evaluate_acceptance(summary, min_sharpe=1.0)
        assert res.passed is True
        assert res.actual_sharpe == 1.5

    def test_evaluate_acceptance_fail(self):
        summary = {"portfolio_sharpe_mean": 0.4}
        res = full_oos_backtest.evaluate_acceptance(summary, min_sharpe=1.0)
        assert res.passed is False
        assert "0.40" in res.reason

    def test_evaluate_acceptance_missing_field(self):
        res = full_oos_backtest.evaluate_acceptance({}, min_sharpe=1.0)
        assert res.passed is False
        assert res.actual_sharpe == 0.0


# ── write_artifacts ──────────────────────────────────────────────────────

class TestWriteArtifacts:
    def _result(self, sym):
        return full_oos_backtest.SymbolResult(
            symbol=sym, bars=200, trades=3, wins=2, losses=1,
            total_return=0.02, sharpe=1.1, sortino=1.3, calmar=0.2,
            max_drawdown=0.05, win_rate=0.66, total_costs=4.5,
            final_equity=10_200,
            regime_breakdown={
                "low_vol": {"n_trades": 2, "total_pnl": 80.0,
                            "avg_pnl": 40.0, "win_rate": 1.0},
                "normal": {"n_trades": 1, "total_pnl": -20.0,
                           "avg_pnl": -20.0, "win_rate": 0.0},
            },
        )

    def test_writes_all_files(self, tmp_path):
        results = [self._result("TSLA"), self._result("AAPL")]
        summary = {"portfolio_sharpe_mean": 1.2, "total_return": 0.02}
        mc = {"runs": 1000, "p5": 9_000, "p50": 10_000, "p95": 11_500,
              "mean": 10_100, "std": 500}
        acc = full_oos_backtest.AcceptanceResult(
            passed=True, actual_sharpe=1.2, min_sharpe=1.0, reason="ok"
        )
        full_oos_backtest.write_artifacts(str(tmp_path), results, summary,
                                          mc, acc)
        assert os.path.exists(tmp_path / "per_symbol_metrics.csv")
        assert os.path.exists(tmp_path / "per_regime.csv")
        assert os.path.exists(tmp_path / "summary.json")
        assert os.path.exists(tmp_path / "acceptance.json")

        with open(tmp_path / "summary.json") as f:
            payload = json.load(f)
        assert payload["portfolio"]["portfolio_sharpe_mean"] == 1.2

        with open(tmp_path / "acceptance.json") as f:
            acc_payload = json.load(f)
        assert acc_payload["passed"] is True


# ── End-to-end (mocked broker) ───────────────────────────────────────────

class TestRunBacktest:
    def test_runs_end_to_end(self, tmp_path, monkeypatch):
        monkeypatch.setattr(full_oos_backtest, "_CACHE_DIR", str(tmp_path /
                                                                 "cache"))
        monkeypatch.setattr(full_oos_backtest, "_OUT_ROOT", str(tmp_path /
                                                                "out"))
        broker = MagicMock()
        broker.get_historical_bars.return_value = _make_bars(300)

        out = full_oos_backtest.run_backtest(
            symbols=["TSLA", "AAPL"],
            days=60, timeframe="hour",
            starting_cash=10_000,
            monte_carlo_runs=50,
            # The alternating stub churns on every bar → negative Sharpe.
            # Set gate very low so the "passed=True" branch is exercised.
            min_sharpe=-10.0,
            broker=broker,
            compute_signals=_stub_signal,
            out_root=str(tmp_path / "out"),
            now=datetime(2026, 4, 14, 12, 0, 0),
        )
        assert "out_dir" in out
        assert os.path.exists(os.path.join(out["out_dir"], "summary.json"))
        assert out["acceptance"]["passed"] is True

    def test_acceptance_gate_fails_with_high_threshold(self, tmp_path,
                                                       monkeypatch):
        monkeypatch.setattr(full_oos_backtest, "_CACHE_DIR", str(tmp_path /
                                                                 "cache"))
        broker = MagicMock()
        broker.get_historical_bars.return_value = _make_bars(300)

        out = full_oos_backtest.run_backtest(
            symbols=["TSLA"],
            days=60, timeframe="hour",
            starting_cash=10_000,
            monte_carlo_runs=10,
            min_sharpe=999.0,  # impossible gate
            broker=broker,
            compute_signals=_stub_signal,
            out_root=str(tmp_path / "out"),
            now=datetime(2026, 4, 14, 12, 0, 0),
        )
        assert out["acceptance"]["passed"] is False

    def test_empty_universe_still_writes_artifacts(self, tmp_path,
                                                   monkeypatch):
        """No symbols → summary still produced, acceptance fails gracefully."""
        monkeypatch.setattr(full_oos_backtest, "_CACHE_DIR", str(tmp_path /
                                                                 "cache"))
        broker = MagicMock()
        broker.get_historical_bars.return_value = pd.DataFrame()

        out = full_oos_backtest.run_backtest(
            symbols=["GHOST"], days=60, timeframe="hour",
            starting_cash=10_000, monte_carlo_runs=10, min_sharpe=1.0,
            broker=broker, compute_signals=_stub_signal,
            out_root=str(tmp_path / "out"),
            now=datetime(2026, 4, 14, 12, 0, 0),
        )
        assert out["summary"]["trade_count"] == 0
        assert out["acceptance"]["passed"] is False


# ── CLI smoke test ───────────────────────────────────────────────────────

class TestCLI:
    def test_main_returns_zero_when_acceptance_passes(self, tmp_path,
                                                      monkeypatch):
        monkeypatch.setattr(full_oos_backtest, "_CACHE_DIR", str(tmp_path /
                                                                 "cache"))
        monkeypatch.setattr(full_oos_backtest, "_OUT_ROOT", str(tmp_path /
                                                                "out"))
        broker = MagicMock()
        broker.get_historical_bars.return_value = _make_bars(300)

        with patch.object(full_oos_backtest, "run_backtest",
                          return_value={"acceptance": {"passed": True}}):
            rc = full_oos_backtest.main(["--symbols", "TSLA",
                                         "--days", "30",
                                         "--timeframe", "hour",
                                         "--min-sharpe", "0.0",
                                         "--monte-carlo-runs", "10"])
        assert rc == 0

    def test_main_returns_one_when_acceptance_fails(self):
        with patch.object(full_oos_backtest, "run_backtest",
                          return_value={"acceptance": {"passed": False}}):
            rc = full_oos_backtest.main(["--symbols", "TSLA",
                                         "--min-sharpe", "999.0"])
        assert rc == 1
