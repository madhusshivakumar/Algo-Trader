"""Tests for core/walk_forward.py — walk-forward backtesting."""

import pytest
import numpy as np
import pandas as pd

from core.walk_forward import WalkForwardBacktester, WalkForwardResult


def _make_df(n=500, seed=42, trend=0.001):
    """Create a realistic OHLC DataFrame with a slight trend."""
    rng = np.random.RandomState(seed)
    close = 100 + np.cumsum(rng.randn(n) * 0.5 + trend)
    return pd.DataFrame({
        "open": close + rng.randn(n) * 0.1,
        "high": close + abs(rng.randn(n)) * 0.5,
        "low": close - abs(rng.randn(n)) * 0.5,
        "close": close,
        "volume": rng.randint(100000, 1000000, n),
    })


def _simple_strategy(df):
    """Simple mean-reversion strategy for testing."""
    if len(df) < 20:
        return {"action": "hold", "strength": 0.0, "reason": "insufficient data"}
    close = df["close"].iloc[-1]
    sma = df["close"].tail(20).mean()
    if close < sma * 0.98:
        return {"action": "buy", "strength": 0.7, "reason": "below SMA"}
    elif close > sma * 1.02:
        return {"action": "sell", "strength": 0.7, "reason": "above SMA"}
    return {"action": "hold", "strength": 0.0, "reason": "near SMA"}


@pytest.fixture
def backtester():
    return WalkForwardBacktester(_simple_strategy, strategy_name="test_mr", train_days=60, test_days=20)


class TestWalkForwardRun:
    def test_produces_results(self, backtester):
        df = _make_df(n=500)
        results = backtester.run(df, "AAPL")
        assert len(results) > 0

    def test_result_fields(self, backtester):
        df = _make_df(n=500)
        results = backtester.run(df, "AAPL")
        r = results[0]
        assert isinstance(r, WalkForwardResult)
        assert r.fold == 0
        assert r.strategy == "test_mr"
        assert isinstance(r.in_sample_return, float)
        assert isinstance(r.out_of_sample_return, float)
        assert isinstance(r.in_sample_sharpe, float)
        assert isinstance(r.out_of_sample_sharpe, float)

    def test_multiple_folds(self, backtester):
        df = _make_df(n=1000)
        results = backtester.run(df, "AAPL")
        assert len(results) >= 2
        # Folds should be sequential
        for i, r in enumerate(results):
            assert r.fold == i

    def test_insufficient_data_returns_empty(self, backtester):
        df = _make_df(n=50)
        results = backtester.run(df, "AAPL")
        assert results == []

    def test_none_df_returns_empty(self, backtester):
        results = backtester.run(None, "AAPL")
        assert results == []


class TestSimulate:
    def test_returns_metrics_dict(self, backtester):
        df = _make_df(n=200)
        metrics = backtester._simulate(df, 100000)
        assert "total_return" in metrics
        assert "sharpe" in metrics
        assert "trade_count" in metrics
        assert "win_rate" in metrics

    def test_short_df_returns_zeros(self, backtester):
        df = _make_df(n=10)
        metrics = backtester._simulate(df, 100000)
        assert metrics["total_return"] == 0.0
        assert metrics["trade_count"] == 0

    def test_strategy_exception_handled(self):
        def bad_strategy(df):
            raise RuntimeError("broken")

        bt = WalkForwardBacktester(bad_strategy)
        df = _make_df(n=200)
        metrics = bt._simulate(df, 100000)
        # Should complete without error, just no trades
        assert metrics["trade_count"] == 0


class TestSummary:
    def test_summary_with_results(self, backtester):
        df = _make_df(n=500)
        results = backtester.run(df, "AAPL")
        summary = backtester.summary(results)
        assert summary["folds"] == len(results)
        assert "avg_oos_return" in summary
        assert "avg_oos_sharpe" in summary
        assert "overfitting_probability" in summary
        assert 0 <= summary["overfitting_probability"] <= 1

    def test_summary_empty_results(self, backtester):
        summary = backtester.summary([])
        assert summary["folds"] == 0


class TestOverfittingProbability:
    def test_all_overfit(self, backtester):
        results = [
            WalkForwardResult(fold=0, train_start="", train_end="", test_start="", test_end="",
                              strategy="test", in_sample_return=0.1, in_sample_sharpe=1.0,
                              out_of_sample_return=-0.05, out_of_sample_sharpe=-0.5,
                              trade_count=10, win_rate=0.4),
            WalkForwardResult(fold=1, train_start="", train_end="", test_start="", test_end="",
                              strategy="test", in_sample_return=0.15, in_sample_sharpe=1.2,
                              out_of_sample_return=0.01, out_of_sample_sharpe=0.1,
                              trade_count=8, win_rate=0.5),
        ]
        prob = backtester.overfitting_probability(results)
        assert prob == 1.0  # Both folds: OOS < IS

    def test_no_overfit(self, backtester):
        results = [
            WalkForwardResult(fold=0, train_start="", train_end="", test_start="", test_end="",
                              strategy="test", in_sample_return=0.05, in_sample_sharpe=0.5,
                              out_of_sample_return=0.08, out_of_sample_sharpe=0.8,
                              trade_count=10, win_rate=0.6),
        ]
        prob = backtester.overfitting_probability(results)
        assert prob == 0.0

    def test_empty_results(self, backtester):
        assert backtester.overfitting_probability([]) == 0.0

    def test_mixed(self, backtester):
        results = [
            WalkForwardResult(fold=0, train_start="", train_end="", test_start="", test_end="",
                              strategy="t", in_sample_return=0.1, in_sample_sharpe=1.0,
                              out_of_sample_return=0.12, out_of_sample_sharpe=1.2,
                              trade_count=10, win_rate=0.6),
            WalkForwardResult(fold=1, train_start="", train_end="", test_start="", test_end="",
                              strategy="t", in_sample_return=0.1, in_sample_sharpe=1.0,
                              out_of_sample_return=0.05, out_of_sample_sharpe=0.5,
                              trade_count=8, win_rate=0.5),
        ]
        prob = backtester.overfitting_probability(results)
        assert prob == 0.5  # 1 of 2 folds overfit
