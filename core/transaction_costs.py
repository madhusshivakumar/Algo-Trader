"""Transaction cost modeling — commission, spread, and slippage estimation.

Models trading costs for position sizing, backtesting, and strategy comparison.
All costs are expressed as fractions of the trade value (e.g., 0.001 = 10 bps).

Alpaca paper/live trading has zero commissions for stocks and crypto, but this
module enables realistic cost modeling for backtesting and portability to other
brokers.
"""

from dataclasses import dataclass

from config import Config


@dataclass
class CostEstimate:
    """Breakdown of estimated transaction costs for a single trade."""
    commission: float = 0.0        # Dollar amount
    spread_cost: float = 0.0       # Dollar amount (half-spread applied)
    slippage_cost: float = 0.0     # Dollar amount
    total_cost: float = 0.0        # Sum of all costs
    trade_value: float = 0.0       # Original trade value (for bps calc)

    @property
    def total_bps(self) -> float:
        """Total cost in basis points relative to trade value."""
        if self.trade_value <= 0:
            return 0.0
        return (self.total_cost / self.trade_value) * 10000


def cost_bps(cost_estimate: CostEstimate, trade_value: float) -> float:
    """Total cost as basis points of trade value."""
    if trade_value <= 0:
        return 0.0
    return (cost_estimate.total_cost / trade_value) * 10000


class TransactionCostModel:
    """Models trading costs for a given asset class.

    Default costs reflect Alpaca's commission-free environment with
    estimated spread and slippage:
      - Equities: 0 commission, ~5 bps spread, ~3 bps slippage
      - Crypto: 0 commission, ~15 bps spread, ~10 bps slippage
    """

    def __init__(self,
                 commission_pct: float | None = None,
                 spread_bps_equity: float | None = None,
                 spread_bps_crypto: float | None = None,
                 slippage_bps: float | None = None):
        """
        Args:
            commission_pct: Commission as fraction (0.001 = 0.1%). Default from config.
            spread_bps_equity: Half-spread for equities in bps. Default from config.
            spread_bps_crypto: Half-spread for crypto in bps. Default from config.
            slippage_bps: Estimated slippage in bps. Default from config.
        """
        self.commission_pct = commission_pct if commission_pct is not None else Config.TC_COMMISSION_PCT
        self.spread_bps_equity = spread_bps_equity if spread_bps_equity is not None else Config.TC_SPREAD_BPS_EQUITY
        self.spread_bps_crypto = spread_bps_crypto if spread_bps_crypto is not None else Config.TC_SPREAD_BPS_CRYPTO
        self.slippage_bps = slippage_bps if slippage_bps is not None else Config.TC_SLIPPAGE_BPS

    def estimate(self, symbol: str, trade_value: float) -> CostEstimate:
        """Estimate transaction costs for a trade.

        Args:
            symbol: Trading symbol (used to determine asset class).
            trade_value: Absolute dollar value of the trade.

        Returns:
            CostEstimate with itemized costs.
        """
        if trade_value <= 0:
            return CostEstimate()

        is_crypto = Config.is_crypto(symbol)

        # Commission
        commission = trade_value * self.commission_pct

        # Spread (half-spread — we pay half the bid-ask on each side)
        spread_bps = self.spread_bps_crypto if is_crypto else self.spread_bps_equity
        spread_cost = trade_value * (spread_bps / 10000)

        # Slippage
        slippage_cost = trade_value * (self.slippage_bps / 10000)

        total = commission + spread_cost + slippage_cost

        return CostEstimate(
            commission=commission,
            spread_cost=spread_cost,
            slippage_cost=slippage_cost,
            total_cost=total,
            trade_value=trade_value,
        )

    def net_buy_amount(self, symbol: str, gross_amount: float) -> float:
        """Amount actually invested after paying costs.

        Use this to adjust position sizing: if you want to invest $1000,
        you actually get $1000 - costs worth of the asset.

        Args:
            symbol: Trading symbol.
            gross_amount: Total cash allocated (including costs).

        Returns:
            Net investable amount after costs.
        """
        costs = self.estimate(symbol, gross_amount)
        return max(0.0, gross_amount - costs.total_cost)

    def net_sell_proceeds(self, symbol: str, gross_proceeds: float) -> float:
        """Cash received after selling, minus costs.

        Args:
            symbol: Trading symbol.
            gross_proceeds: Market value of the position being sold.

        Returns:
            Net cash received after costs.
        """
        costs = self.estimate(symbol, gross_proceeds)
        return max(0.0, gross_proceeds - costs.total_cost)

    def round_trip_cost(self, symbol: str, trade_value: float) -> float:
        """Total cost of a buy + sell round trip.

        Args:
            symbol: Trading symbol.
            trade_value: Dollar value of the trade.

        Returns:
            Total round-trip cost in dollars.
        """
        buy_costs = self.estimate(symbol, trade_value)
        sell_costs = self.estimate(symbol, trade_value)
        return buy_costs.total_cost + sell_costs.total_cost

    def breakeven_move_pct(self, symbol: str) -> float:
        """Minimum price move needed to break even after round-trip costs.

        Returns:
            Required move as fraction (e.g., 0.003 = 0.3%).
        """
        # Cost to enter + cost to exit, as fraction of trade value
        is_crypto = Config.is_crypto(symbol)
        spread_bps = self.spread_bps_crypto if is_crypto else self.spread_bps_equity
        one_way_pct = self.commission_pct + (spread_bps / 10000) + (self.slippage_bps / 10000)
        return 2 * one_way_pct
