"""Monte Carlo simulation for portfolio risk analysis.

Generates randomized equity paths by resampling historical returns to estimate:
- Value at Risk (VaR) — worst expected loss at a confidence level
- Conditional VaR (CVaR / Expected Shortfall) — average loss beyond VaR
- Drawdown distribution — probability of drawdowns exceeding thresholds
- Terminal wealth distribution — range of possible outcomes

Uses bootstrap resampling (sampling returns with replacement) to preserve the
empirical return distribution without assuming normality.
"""

from dataclasses import dataclass, field

import numpy as np

from config import Config


@dataclass
class MonteCarloResult:
    """Results from a Monte Carlo simulation run."""
    num_simulations: int = 0
    horizon_days: int = 0
    starting_equity: float = 0.0

    # Terminal wealth distribution
    terminal_mean: float = 0.0
    terminal_median: float = 0.0
    terminal_std: float = 0.0
    terminal_percentiles: dict[str, float] = field(default_factory=dict)
    # e.g. {"5%": 8500, "25%": 9200, "50%": 10000, "75%": 11000, "95%": 12500}

    # VaR / CVaR (as positive loss fractions, e.g. 0.05 = 5% loss)
    var: dict[str, float] = field(default_factory=dict)
    # e.g. {"95%": 0.08, "99%": 0.15}
    cvar: dict[str, float] = field(default_factory=dict)
    # e.g. {"95%": 0.12, "99%": 0.20}

    # Drawdown distribution
    max_drawdown_mean: float = 0.0
    max_drawdown_median: float = 0.0
    max_drawdown_95th: float = 0.0
    max_drawdown_99th: float = 0.0

    # Probability estimates
    prob_loss: float = 0.0        # P(terminal < starting)
    prob_loss_10pct: float = 0.0  # P(terminal < 90% of starting)
    prob_gain_20pct: float = 0.0  # P(terminal > 120% of starting)

    # Raw simulation paths (optional, for plotting)
    # Not stored by default to save memory — set store_paths=True
    paths: list[list[float]] | None = None


def simulate_paths(returns: np.ndarray, starting_equity: float,
                   num_simulations: int | None = None,
                   horizon_days: int | None = None,
                   seed: int | None = None) -> np.ndarray:
    """Generate simulated equity paths via bootstrap resampling.

    Samples daily returns with replacement from the historical return
    distribution to build forward-looking equity paths.

    Args:
        returns: Historical daily returns (as fractions, e.g. 0.01 = 1%).
        starting_equity: Starting portfolio value.
        num_simulations: Number of simulation paths (default from config).
        horizon_days: Number of days to simulate (default from config).
        seed: Random seed for reproducibility (None = random).

    Returns:
        2D array of shape (num_simulations, horizon_days + 1) with equity values.
        Column 0 is the starting equity for all paths.
    """
    if num_simulations is None:
        num_simulations = Config.MONTE_CARLO_NUM_SIMULATIONS
    if horizon_days is None:
        horizon_days = Config.MONTE_CARLO_HORIZON_DAYS

    if len(returns) == 0 or num_simulations <= 0 or horizon_days <= 0:
        return np.full((max(1, num_simulations), 1), starting_equity)

    rng = np.random.default_rng(seed)

    # Sample returns with replacement: shape (num_simulations, horizon_days)
    sampled_indices = rng.integers(0, len(returns), size=(num_simulations, horizon_days))
    sampled_returns = returns[sampled_indices]

    # Build equity paths: equity[t+1] = equity[t] * (1 + return[t])
    growth_factors = 1.0 + sampled_returns
    cumulative = np.cumprod(growth_factors, axis=1)

    # Prepend starting equity column
    paths = np.empty((num_simulations, horizon_days + 1))
    paths[:, 0] = starting_equity
    paths[:, 1:] = starting_equity * cumulative

    return paths


def compute_max_drawdowns(paths: np.ndarray) -> np.ndarray:
    """Compute max drawdown for each simulated path.

    Args:
        paths: 2D array of shape (num_simulations, horizon_days + 1).

    Returns:
        1D array of max drawdown fractions per path.
    """
    if paths.shape[1] < 2:
        return np.zeros(paths.shape[0])

    running_peak = np.maximum.accumulate(paths, axis=1)
    drawdowns = (running_peak - paths) / np.where(running_peak > 0, running_peak, 1.0)
    return np.max(drawdowns, axis=1)


def compute_var(terminal_returns: np.ndarray,
                confidence_levels: list[float] | None = None) -> dict[str, float]:
    """Compute Value at Risk at given confidence levels.

    VaR is the worst expected loss at a given confidence level over the
    simulation horizon. Expressed as a positive fraction (e.g. 0.05 = 5% loss).

    Args:
        terminal_returns: Array of terminal returns (fractions, can be negative).
        confidence_levels: List of confidence levels (e.g. [0.95, 0.99]).

    Returns:
        Dict mapping "95%" -> VaR value, etc.
    """
    if confidence_levels is None:
        confidence_levels = Config.MONTE_CARLO_CONFIDENCE_LEVELS

    if len(terminal_returns) == 0:
        return {f"{int(cl * 100)}%": 0.0 for cl in confidence_levels}

    result = {}
    for cl in confidence_levels:
        # VaR is the loss at the (1 - cl) percentile
        # e.g. 95% VaR = 5th percentile of returns, negated
        percentile = (1.0 - cl) * 100
        var_value = -float(np.percentile(terminal_returns, percentile))
        result[f"{int(cl * 100)}%"] = max(0.0, var_value)  # VaR is a positive loss
    return result


def compute_cvar(terminal_returns: np.ndarray,
                 confidence_levels: list[float] | None = None) -> dict[str, float]:
    """Compute Conditional VaR (Expected Shortfall) at given confidence levels.

    CVaR is the average loss in the worst (1-confidence) fraction of scenarios.
    Always >= VaR at the same confidence level.

    Args:
        terminal_returns: Array of terminal returns (fractions).
        confidence_levels: List of confidence levels.

    Returns:
        Dict mapping "95%" -> CVaR value, etc.
    """
    if confidence_levels is None:
        confidence_levels = Config.MONTE_CARLO_CONFIDENCE_LEVELS

    if len(terminal_returns) == 0:
        return {f"{int(cl * 100)}%": 0.0 for cl in confidence_levels}

    result = {}
    for cl in confidence_levels:
        percentile = (1.0 - cl) * 100
        threshold = np.percentile(terminal_returns, percentile)
        tail = terminal_returns[terminal_returns <= threshold]
        if len(tail) == 0:
            result[f"{int(cl * 100)}%"] = 0.0
        else:
            result[f"{int(cl * 100)}%"] = max(0.0, -float(np.mean(tail)))
    return result


def run_simulation(returns: np.ndarray, starting_equity: float,
                   num_simulations: int | None = None,
                   horizon_days: int | None = None,
                   confidence_levels: list[float] | None = None,
                   seed: int | None = None,
                   store_paths: bool = False) -> MonteCarloResult:
    """Run a full Monte Carlo simulation and compute all risk metrics.

    Args:
        returns: Historical daily returns (fractions).
        starting_equity: Starting portfolio value.
        num_simulations: Number of paths to simulate.
        horizon_days: Days to project forward.
        confidence_levels: VaR/CVaR confidence levels.
        seed: Random seed for reproducibility.
        store_paths: If True, include raw paths in result (memory intensive).

    Returns:
        MonteCarloResult with all computed metrics.
    """
    if num_simulations is None:
        num_simulations = Config.MONTE_CARLO_NUM_SIMULATIONS
    if horizon_days is None:
        horizon_days = Config.MONTE_CARLO_HORIZON_DAYS
    if confidence_levels is None:
        confidence_levels = Config.MONTE_CARLO_CONFIDENCE_LEVELS

    result = MonteCarloResult(
        num_simulations=num_simulations,
        horizon_days=horizon_days,
        starting_equity=starting_equity,
    )

    returns = np.asarray(returns, dtype=float)
    if len(returns) < 2:
        return result

    # Generate paths
    paths = simulate_paths(returns, starting_equity, num_simulations,
                           horizon_days, seed)

    # Terminal values
    terminal_values = paths[:, -1]
    terminal_returns = (terminal_values - starting_equity) / starting_equity

    # Terminal wealth distribution
    result.terminal_mean = float(np.mean(terminal_values))
    result.terminal_median = float(np.median(terminal_values))
    result.terminal_std = float(np.std(terminal_values))
    result.terminal_percentiles = {
        "5%": float(np.percentile(terminal_values, 5)),
        "25%": float(np.percentile(terminal_values, 25)),
        "50%": float(np.percentile(terminal_values, 50)),
        "75%": float(np.percentile(terminal_values, 75)),
        "95%": float(np.percentile(terminal_values, 95)),
    }

    # VaR / CVaR
    result.var = compute_var(terminal_returns, confidence_levels)
    result.cvar = compute_cvar(terminal_returns, confidence_levels)

    # Drawdown distribution
    max_dds = compute_max_drawdowns(paths)
    result.max_drawdown_mean = float(np.mean(max_dds))
    result.max_drawdown_median = float(np.median(max_dds))
    result.max_drawdown_95th = float(np.percentile(max_dds, 95))
    result.max_drawdown_99th = float(np.percentile(max_dds, 99))

    # Probability estimates
    result.prob_loss = float(np.mean(terminal_values < starting_equity))
    result.prob_loss_10pct = float(np.mean(terminal_values < starting_equity * 0.9))
    result.prob_gain_20pct = float(np.mean(terminal_values > starting_equity * 1.2))

    # Optionally store paths
    if store_paths:
        result.paths = paths.tolist()

    return result


def run_from_equity_curve(equity_curve: list[float] | np.ndarray,
                          num_simulations: int | None = None,
                          horizon_days: int | None = None,
                          seed: int | None = None,
                          store_paths: bool = False) -> MonteCarloResult:
    """Convenience: run Monte Carlo from an equity curve.

    Extracts returns from the equity curve and uses the final equity
    value as the starting point for simulation.

    Args:
        equity_curve: Historical equity values.
        num_simulations: Number of paths.
        horizon_days: Simulation horizon.
        seed: Random seed.
        store_paths: Whether to store raw paths.

    Returns:
        MonteCarloResult.
    """
    eq = np.asarray(equity_curve, dtype=float)
    if len(eq) < 3:
        return MonteCarloResult(starting_equity=float(eq[-1]) if len(eq) > 0 else 0.0)

    # Compute returns
    returns = np.diff(eq) / eq[:-1]
    returns = returns[np.isfinite(returns)]

    if len(returns) < 2:
        return MonteCarloResult(starting_equity=float(eq[-1]))

    starting_equity = float(eq[-1])
    return run_simulation(returns, starting_equity, num_simulations,
                          horizon_days, seed=seed, store_paths=store_paths)


def format_report(result: MonteCarloResult) -> str:
    """Format Monte Carlo results as a human-readable report string."""
    lines = [
        f"Monte Carlo Simulation ({result.num_simulations:,} paths, "
        f"{result.horizon_days}-day horizon)",
        f"Starting equity: ${result.starting_equity:,.2f}",
        "",
        "Terminal Wealth Distribution:",
        f"  Mean:   ${result.terminal_mean:,.2f}",
        f"  Median: ${result.terminal_median:,.2f}",
        f"  Std:    ${result.terminal_std:,.2f}",
    ]

    if result.terminal_percentiles:
        lines.append("  Percentiles:")
        for pct, val in sorted(result.terminal_percentiles.items()):
            lines.append(f"    {pct:>4s}: ${val:,.2f}")

    lines.append("")
    lines.append("Value at Risk (VaR):")
    for cl, val in result.var.items():
        lines.append(f"  {cl} VaR:  {val:.2%}")

    lines.append("")
    lines.append("Conditional VaR (Expected Shortfall):")
    for cl, val in result.cvar.items():
        lines.append(f"  {cl} CVaR: {val:.2%}")

    lines.append("")
    lines.append("Max Drawdown Distribution:")
    lines.append(f"  Mean:   {result.max_drawdown_mean:.2%}")
    lines.append(f"  Median: {result.max_drawdown_median:.2%}")
    lines.append(f"  95th:   {result.max_drawdown_95th:.2%}")
    lines.append(f"  99th:   {result.max_drawdown_99th:.2%}")

    lines.append("")
    lines.append("Probability Estimates:")
    lines.append(f"  P(loss):       {result.prob_loss:.1%}")
    lines.append(f"  P(loss > 10%): {result.prob_loss_10pct:.1%}")
    lines.append(f"  P(gain > 20%): {result.prob_gain_20pct:.1%}")

    return "\n".join(lines)
