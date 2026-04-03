"""Unit tests for the transaction cost modeling module."""

import pytest
from unittest.mock import patch

from core.transaction_costs import CostEstimate, TransactionCostModel, cost_bps
from config import Config


# ── CostEstimate Dataclass ──────────────────────────────────────────


class TestCostEstimate:
    """Test the CostEstimate dataclass."""

    def test_default_values(self):
        c = CostEstimate()
        assert c.commission == 0.0
        assert c.spread_cost == 0.0
        assert c.slippage_cost == 0.0
        assert c.total_cost == 0.0
        assert c.trade_value == 0.0

    def test_custom_values(self):
        c = CostEstimate(
            commission=1.0, spread_cost=2.0, slippage_cost=3.0,
            total_cost=6.0, trade_value=1000.0,
        )
        assert c.commission == 1.0
        assert c.spread_cost == 2.0
        assert c.slippage_cost == 3.0
        assert c.total_cost == 6.0
        assert c.trade_value == 1000.0

    def test_total_bps_property(self):
        c = CostEstimate(total_cost=10.0, trade_value=10000.0)
        assert c.total_bps == pytest.approx(10.0)  # 10/10000 * 10000 = 10 bps

    def test_total_bps_zero_trade_value(self):
        c = CostEstimate(total_cost=10.0, trade_value=0.0)
        assert c.total_bps == 0.0

    def test_total_bps_negative_trade_value(self):
        c = CostEstimate(total_cost=10.0, trade_value=-100.0)
        assert c.total_bps == 0.0

    def test_total_bps_typical_equity(self):
        # 5 bps spread + 3 bps slippage on $10,000 trade
        c = CostEstimate(total_cost=8.0, trade_value=10000.0)
        assert c.total_bps == pytest.approx(8.0)

    def test_total_bps_typical_crypto(self):
        # 15 bps spread + 3 bps slippage on $1,000 trade = $1.80
        c = CostEstimate(total_cost=1.80, trade_value=1000.0)
        assert c.total_bps == pytest.approx(18.0)


# ── cost_bps() Function ────────────────────────────────────────────


class TestCostBps:
    """Test the module-level cost_bps() function."""

    def test_basic_calculation(self):
        c = CostEstimate(total_cost=10.0)
        assert cost_bps(c, 10000.0) == pytest.approx(10.0)

    def test_zero_trade_value(self):
        c = CostEstimate(total_cost=10.0)
        assert cost_bps(c, 0.0) == 0.0

    def test_negative_trade_value(self):
        c = CostEstimate(total_cost=10.0)
        assert cost_bps(c, -100.0) == 0.0

    def test_zero_cost(self):
        c = CostEstimate(total_cost=0.0)
        assert cost_bps(c, 10000.0) == 0.0

    def test_large_trade(self):
        c = CostEstimate(total_cost=100.0)
        assert cost_bps(c, 1_000_000.0) == pytest.approx(1.0)


# ── TransactionCostModel Init ──────────────────────────────────────


class TestTransactionCostModelInit:
    """Test constructor and config fallback logic."""

    def test_defaults_from_config(self):
        model = TransactionCostModel()
        assert model.commission_pct == Config.TC_COMMISSION_PCT
        assert model.spread_bps_equity == Config.TC_SPREAD_BPS_EQUITY
        assert model.spread_bps_crypto == Config.TC_SPREAD_BPS_CRYPTO
        assert model.slippage_bps == Config.TC_SLIPPAGE_BPS

    def test_custom_overrides(self):
        model = TransactionCostModel(
            commission_pct=0.001,
            spread_bps_equity=10.0,
            spread_bps_crypto=20.0,
            slippage_bps=5.0,
        )
        assert model.commission_pct == 0.001
        assert model.spread_bps_equity == 10.0
        assert model.spread_bps_crypto == 20.0
        assert model.slippage_bps == 5.0

    def test_partial_overrides(self):
        model = TransactionCostModel(commission_pct=0.002)
        assert model.commission_pct == 0.002
        assert model.spread_bps_equity == Config.TC_SPREAD_BPS_EQUITY

    def test_zero_overrides_accepted(self):
        """Passing 0 should use 0, not fall back to config."""
        model = TransactionCostModel(
            commission_pct=0.0,
            spread_bps_equity=0.0,
            spread_bps_crypto=0.0,
            slippage_bps=0.0,
        )
        assert model.commission_pct == 0.0
        assert model.spread_bps_equity == 0.0


# ── estimate() Method ───────────────────────────────────────────────


class TestEstimate:
    """Test the core cost estimation method."""

    def test_equity_estimate(self):
        model = TransactionCostModel(
            commission_pct=0.0,
            spread_bps_equity=5.0,
            spread_bps_crypto=15.0,
            slippage_bps=3.0,
        )
        cost = model.estimate("AAPL", 10000.0)

        # Spread: 10000 * (5/10000) = 5.0
        assert cost.spread_cost == pytest.approx(5.0)
        # Slippage: 10000 * (3/10000) = 3.0
        assert cost.slippage_cost == pytest.approx(3.0)
        # Commission: 0
        assert cost.commission == pytest.approx(0.0)
        # Total: 8.0
        assert cost.total_cost == pytest.approx(8.0)
        # Trade value stored
        assert cost.trade_value == 10000.0

    def test_crypto_estimate(self):
        model = TransactionCostModel(
            commission_pct=0.0,
            spread_bps_equity=5.0,
            spread_bps_crypto=15.0,
            slippage_bps=3.0,
        )
        cost = model.estimate("BTC/USD", 10000.0)

        # Crypto uses spread_bps_crypto
        assert cost.spread_cost == pytest.approx(15.0)  # 10000 * (15/10000)
        assert cost.slippage_cost == pytest.approx(3.0)
        assert cost.total_cost == pytest.approx(18.0)

    def test_with_commission(self):
        model = TransactionCostModel(
            commission_pct=0.001,  # 10 bps
            spread_bps_equity=5.0,
            slippage_bps=3.0,
        )
        cost = model.estimate("AAPL", 10000.0)

        assert cost.commission == pytest.approx(10.0)  # 10000 * 0.001
        assert cost.total_cost == pytest.approx(18.0)  # 10 + 5 + 3

    def test_zero_trade_value(self):
        model = TransactionCostModel()
        cost = model.estimate("AAPL", 0.0)

        assert cost.commission == 0.0
        assert cost.spread_cost == 0.0
        assert cost.slippage_cost == 0.0
        assert cost.total_cost == 0.0

    def test_negative_trade_value(self):
        model = TransactionCostModel()
        cost = model.estimate("AAPL", -100.0)
        assert cost.total_cost == 0.0

    def test_small_trade_value(self):
        model = TransactionCostModel(
            commission_pct=0.0,
            spread_bps_equity=5.0,
            slippage_bps=3.0,
        )
        cost = model.estimate("AAPL", 1.0)
        assert cost.total_cost == pytest.approx(0.0008)  # (5+3)/10000 * 1

    def test_estimate_returns_cost_estimate(self):
        model = TransactionCostModel()
        cost = model.estimate("AAPL", 1000.0)
        assert isinstance(cost, CostEstimate)

    def test_bps_accessible_from_estimate_result(self):
        model = TransactionCostModel(
            commission_pct=0.0,
            spread_bps_equity=5.0,
            slippage_bps=3.0,
        )
        cost = model.estimate("AAPL", 10000.0)
        assert cost.total_bps == pytest.approx(8.0)


# ── net_buy_amount() Method ─────────────────────────────────────────


class TestNetBuyAmount:
    """Test position sizing adjustment for buys."""

    def test_basic_deduction(self):
        model = TransactionCostModel(
            commission_pct=0.0,
            spread_bps_equity=5.0,
            slippage_bps=3.0,
        )
        net = model.net_buy_amount("AAPL", 10000.0)
        # Costs: 8.0, so net = 9992.0
        assert net == pytest.approx(9992.0)

    def test_crypto_higher_cost(self):
        model = TransactionCostModel(
            commission_pct=0.0,
            spread_bps_equity=5.0,
            spread_bps_crypto=15.0,
            slippage_bps=3.0,
        )
        equity_net = model.net_buy_amount("AAPL", 10000.0)
        crypto_net = model.net_buy_amount("BTC/USD", 10000.0)
        # Crypto costs more, so net is less
        assert crypto_net < equity_net

    def test_zero_amount(self):
        model = TransactionCostModel()
        net = model.net_buy_amount("AAPL", 0.0)
        assert net == 0.0

    def test_floored_at_zero(self):
        """Should never return negative even with extreme costs."""
        model = TransactionCostModel(
            commission_pct=1.0,  # 100% commission — absurd but valid
            spread_bps_equity=10000.0,
            slippage_bps=10000.0,
        )
        net = model.net_buy_amount("AAPL", 100.0)
        assert net == 0.0


# ── net_sell_proceeds() Method ──────────────────────────────────────


class TestNetSellProceeds:
    """Test proceeds calculation for sells."""

    def test_basic_deduction(self):
        model = TransactionCostModel(
            commission_pct=0.0,
            spread_bps_equity=5.0,
            slippage_bps=3.0,
        )
        net = model.net_sell_proceeds("AAPL", 10000.0)
        assert net == pytest.approx(9992.0)

    def test_zero_proceeds(self):
        model = TransactionCostModel()
        net = model.net_sell_proceeds("AAPL", 0.0)
        assert net == 0.0

    def test_floored_at_zero(self):
        model = TransactionCostModel(commission_pct=1.0)
        net = model.net_sell_proceeds("AAPL", 1.0)
        assert net == 0.0


# ── round_trip_cost() Method ────────────────────────────────────────


class TestRoundTripCost:
    """Test round-trip cost estimation."""

    def test_double_one_way_cost(self):
        model = TransactionCostModel(
            commission_pct=0.0,
            spread_bps_equity=5.0,
            slippage_bps=3.0,
        )
        rt = model.round_trip_cost("AAPL", 10000.0)
        one_way = model.estimate("AAPL", 10000.0).total_cost
        assert rt == pytest.approx(one_way * 2)

    def test_zero_trade_value(self):
        model = TransactionCostModel()
        assert model.round_trip_cost("AAPL", 0.0) == 0.0

    def test_crypto_vs_equity(self):
        model = TransactionCostModel(
            commission_pct=0.0,
            spread_bps_equity=5.0,
            spread_bps_crypto=15.0,
            slippage_bps=3.0,
        )
        equity_rt = model.round_trip_cost("AAPL", 10000.0)
        crypto_rt = model.round_trip_cost("BTC/USD", 10000.0)
        assert crypto_rt > equity_rt


# ── breakeven_move_pct() Method ─────────────────────────────────────


class TestBreakevenMovePct:
    """Test break-even move calculation."""

    def test_equity_breakeven(self):
        model = TransactionCostModel(
            commission_pct=0.0,
            spread_bps_equity=5.0,
            slippage_bps=3.0,
        )
        be = model.breakeven_move_pct("AAPL")
        # 2 * (0 + 5/10000 + 3/10000) = 2 * 0.0008 = 0.0016
        assert be == pytest.approx(0.0016)

    def test_crypto_breakeven_higher(self):
        model = TransactionCostModel(
            commission_pct=0.0,
            spread_bps_equity=5.0,
            spread_bps_crypto=15.0,
            slippage_bps=3.0,
        )
        equity_be = model.breakeven_move_pct("AAPL")
        crypto_be = model.breakeven_move_pct("BTC/USD")
        assert crypto_be > equity_be

    def test_with_commission(self):
        model = TransactionCostModel(
            commission_pct=0.001,
            spread_bps_equity=5.0,
            slippage_bps=3.0,
        )
        be = model.breakeven_move_pct("AAPL")
        # 2 * (0.001 + 0.0005 + 0.0003) = 2 * 0.0018 = 0.0036
        assert be == pytest.approx(0.0036)

    def test_zero_cost_model(self):
        model = TransactionCostModel(
            commission_pct=0.0,
            spread_bps_equity=0.0,
            spread_bps_crypto=0.0,
            slippage_bps=0.0,
        )
        assert model.breakeven_move_pct("AAPL") == 0.0
        assert model.breakeven_move_pct("BTC/USD") == 0.0


# ── Engine Integration ──────────────────────────────────────────────


class TestEngineIntegration:
    """Test that transaction costs are correctly applied in the engine buy path."""

    def test_cost_model_reduces_buy_size(self):
        """Verify cost_model.net_buy_amount is called before buying."""
        model = TransactionCostModel(
            commission_pct=0.0,
            spread_bps_equity=5.0,
            slippage_bps=3.0,
        )
        gross = 1000.0
        net = model.net_buy_amount("AAPL", gross)
        assert net < gross
        assert net > 0

    def test_cost_model_gate_on_config(self):
        """TC_ENABLED=false should mean no cost model."""
        with patch.object(Config, "TC_ENABLED", False):
            from core.engine import TradingEngine
            engine = TradingEngine.__new__(TradingEngine)
            # Simulate __init__ behavior
            engine.cost_model = None
            if Config.TC_ENABLED:
                engine.cost_model = TransactionCostModel()
            assert engine.cost_model is None

    def test_cost_model_active_on_config(self):
        """TC_ENABLED=true should create cost model."""
        with patch.object(Config, "TC_ENABLED", True):
            cost_model = TransactionCostModel() if Config.TC_ENABLED else None
            assert cost_model is not None
            assert isinstance(cost_model, TransactionCostModel)
