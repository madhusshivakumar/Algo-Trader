"""UserProfile — capital-tier-aware defaults for the engine.

The bot is designed for three tiers of user:

    Beginner  — $500 to $2,000. Brand-new to algo trading, can't afford meaningful losses.
    Hobbyist  — $2,000 to $10,000. Has traded before, knows basics, wants safer automation.
    Learner   — $10,000+ or explicitly set. Dev/student/enthusiast who wants full power.

**Every profile runs the full-powered system** — all 9 strategies, RL, LLM analyst, TWAP,
Monte Carlo, sentiment, MTF. Profiles only change numeric defaults and dashboard density,
never which features are enabled.

The profile is auto-detected from the Alpaca account balance at engine startup. Users can
manually override via the USER_PROFILE env var (always allowed to downgrade for safety;
upgrading requires editing config to acknowledge the risk).
"""

from dataclasses import dataclass
from typing import Literal

ProfileName = Literal["beginner", "hobbyist", "learner"]


@dataclass(frozen=True)
class UserProfile:
    """Capital-tier parameters. Never disables engine features, only scales them."""

    name: ProfileName

    # Position sizing
    base_position_pct_equity: float
    base_position_pct_crypto: float
    kelly_fraction: float
    max_single_position_pct: float

    # Daily limits
    max_daily_loss_pct: float  # % of equity (soft floor)
    max_daily_loss_usd_cap: float  # absolute $ floor — whichever is smaller triggers
    max_trades_per_day_equity: int
    max_trades_per_day_crypto: int
    max_positions_open: int  # 0 means unlimited

    # VaR contribution cap (per trade, as fraction of equity)
    max_var_contribution_pct: float

    # Dashboard UI default density
    dashboard_default_view: Literal["simple", "standard", "advanced"]

    def max_daily_loss_usd(self, equity: float) -> float:
        """Return the dollar-based daily loss halt threshold.

        Uses the smaller of (pct × equity, usd_cap) so that small accounts don't burn
        through "10% of $800 = $80" before the halt kicks in — a $50 hard cap is tighter.
        """
        pct_based = max(0.0, equity * self.max_daily_loss_pct)
        return min(pct_based, self.max_daily_loss_usd_cap)


# Tier definitions — single source of truth for per-profile defaults
BEGINNER = UserProfile(
    name="beginner",
    base_position_pct_equity=0.015,     # 1.5% per equity trade on a $1K account → $15
    base_position_pct_crypto=0.010,     # 1% per crypto trade
    kelly_fraction=0.10,                # heavily fractional Kelly — conservative
    max_single_position_pct=0.10,       # never more than 10% in a single name
    max_daily_loss_pct=0.03,            # 3% of equity per day
    max_daily_loss_usd_cap=50.0,        # hard $50 cap for tiny accounts
    max_trades_per_day_equity=3,
    max_trades_per_day_crypto=4,
    max_positions_open=5,
    max_var_contribution_pct=0.01,      # 1% VaR per trade
    dashboard_default_view="simple",
)

HOBBYIST = UserProfile(
    name="hobbyist",
    base_position_pct_equity=0.05,      # 5% per equity trade
    base_position_pct_crypto=0.04,      # 4% per crypto trade
    kelly_fraction=0.20,
    max_single_position_pct=0.15,
    max_daily_loss_pct=0.04,            # 4% of equity
    max_daily_loss_usd_cap=200.0,       # $200 cap for mid accounts
    max_trades_per_day_equity=4,
    max_trades_per_day_crypto=8,
    max_positions_open=10,
    max_var_contribution_pct=0.02,
    dashboard_default_view="standard",
)

LEARNER = UserProfile(
    name="learner",
    base_position_pct_equity=0.15,      # matches current pre-profile default
    base_position_pct_crypto=0.20,
    kelly_fraction=0.25,                # matches current KELLY_FRACTION default
    max_single_position_pct=0.25,       # matches current MAX_SINGLE_POSITION_PCT
    max_daily_loss_pct=0.05,            # 5% — broader latitude for bigger accounts
    max_daily_loss_usd_cap=1_000_000.0, # effectively unbounded — pct dominates
    max_trades_per_day_equity=8,
    max_trades_per_day_crypto=12,
    max_positions_open=0,               # 0 = unlimited
    max_var_contribution_pct=0.03,
    dashboard_default_view="advanced",
)

PROFILES: dict[str, UserProfile] = {
    "beginner": BEGINNER,
    "hobbyist": HOBBYIST,
    "learner": LEARNER,
}


def detect_profile(equity: float) -> UserProfile:
    """Auto-detect profile from account equity.

    Boundaries:
      < $2,000        → Beginner
      $2,000-$10,000  → Hobbyist
      >= $10,000      → Learner
    """
    if equity < 2_000:
        return BEGINNER
    if equity < 10_000:
        return HOBBYIST
    return LEARNER


def resolve_profile(equity: float, override: str | None = None) -> UserProfile:
    """Resolve the active profile.

    1. If `override` is a known profile name, use it (user explicit choice).
    2. Otherwise, auto-detect from equity.

    Allowed overrides: "beginner", "hobbyist", "learner" (case-insensitive).
    Unknown strings fall back to auto-detection with a printed warning.
    """
    if override:
        key = override.strip().lower()
        if key in PROFILES:
            return PROFILES[key]
        # Graceful fallback — don't crash because of a typo in .env
        print(f"  [UserProfile] Unknown override '{override}' — falling back to auto-detect")
    return detect_profile(equity)
