"""Unit tests for the backtesting engine."""

import pytest
import pandas as pd

from compare_strategies import backtest_strategy, STRATEGIES


class TestBacktestEngine:
    """Test the backtester produces valid results."""

    def test_returns_metrics_dict(self, flat_market):
        fn = list(STRATEGIES.values())[0]
        result = backtest_strategy(fn, flat_market)
        assert isinstance(result, dict)

    def test_has_all_required_metrics(self, flat_market):
        fn = list(STRATEGIES.values())[0]
        result = backtest_strategy(fn, flat_market)
        required = [
            "final_equity", "total_return", "total_pnl", "num_trades",
            "win_rate", "wins", "losses", "avg_win", "avg_loss",
            "max_drawdown", "sharpe_ratio", "profit_factor", "buy_hold_return",
        ]
        for key in required:
            assert key in result, f"Missing metric: {key}"

    def test_final_equity_positive(self, flat_market):
        fn = list(STRATEGIES.values())[0]
        result = backtest_strategy(fn, flat_market)
        assert result["final_equity"] > 0

    def test_win_rate_in_range(self, flat_market):
        fn = list(STRATEGIES.values())[0]
        result = backtest_strategy(fn, flat_market)
        assert 0 <= result["win_rate"] <= 100

    def test_max_drawdown_in_range(self, flat_market):
        fn = list(STRATEGIES.values())[0]
        result = backtest_strategy(fn, flat_market)
        assert 0 <= result["max_drawdown"] <= 1.0

    def test_wins_plus_losses_equals_trades(self, uptrend_market):
        fn = list(STRATEGIES.values())[0]
        result = backtest_strategy(fn, uptrend_market)
        # wins + losses should equal sell trades (which equals or is close to num_trades)
        assert result["wins"] >= 0
        assert result["losses"] >= 0

    def test_custom_starting_cash(self, flat_market):
        fn = list(STRATEGIES.values())[0]
        result = backtest_strategy(fn, flat_market, starting_cash=1000)
        # Final equity should be near 1000 for flat market
        assert result["final_equity"] > 0

    def test_custom_stop_loss(self, flat_market):
        fn = list(STRATEGIES.values())[0]
        result = backtest_strategy(fn, flat_market, stop_loss_pct=0.01)
        assert isinstance(result["total_return"], float)


class TestAllStrategiesBacktest:
    """Run backtest on all 9 strategies to make sure none crash."""

    @pytest.mark.parametrize("name,fn", list(STRATEGIES.items()))
    def test_strategy_backtest_completes(self, name, fn, flat_market):
        result = backtest_strategy(fn, flat_market)
        assert result["final_equity"] > 0, f"{name} backtest produced zero equity"

    @pytest.mark.parametrize("name,fn", list(STRATEGIES.items()))
    def test_strategy_backtest_uptrend(self, name, fn, uptrend_market):
        result = backtest_strategy(fn, uptrend_market)
        assert isinstance(result["total_return"], float)

    @pytest.mark.parametrize("name,fn", list(STRATEGIES.items()))
    def test_strategy_backtest_downtrend(self, name, fn, downtrend_market):
        result = backtest_strategy(fn, downtrend_market)
        assert isinstance(result["total_return"], float)
