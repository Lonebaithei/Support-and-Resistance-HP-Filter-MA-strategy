"""
==============================================================================
config.py  —  Central Configuration
==============================================================================
Combined Strategy: HP Filter MA  +  Support & Resistance
Sources:
    • Chapter 8.1  – Moving Averages with HP Filter (Kakushadze & Serur, 2018)
    • Chapter 3.14 – Support and Resistance        (Kakushadze & Serur, 2018)
==============================================================================
"""

from dataclasses import dataclass, field
from typing import List


# ── MetaTrader 5 Connection ───────────────────────────────────────────────────
@dataclass
class MT5Config:
    login:    int   = 0              # ← Your MT5 account number
    password: str   = ""             # ← Your MT5 password
    server:   str   = ""             # ← Your broker server name
    path:     str   = ""             # ← Path to MT5 terminal (optional)
    timeout:  int   = 60_000         # Connection timeout (ms)


# ── Instrument Settings ───────────────────────────────────────────────────────
@dataclass
class InstrumentConfig:
    symbols:        List[str] = field(default_factory=lambda: [
        "EURUSD", "GBPUSD", "USDJPY", "AUDUSD"
    ])
    # MT5 timeframe constants are assigned at runtime (see mt5_connector.py)
    # Options: "M1","M5","M15","M30","H1","H4","D1","W1","MN1"
    hp_timeframe:   str  = "MN1"     # Monthly bars for HP Filter (paper default)
    sr_timeframe:   str  = "D1"      # Daily bars for S&R pivot points
    hp_lookback:    int  = 120        # Bars of history for HP computation (months)
    sr_lookback:    int  = 1          # Previous day bars for pivot calculation


# ── HP Filter Parameters (Chapter 8.1) ───────────────────────────────────────
@dataclass
class HPFilterConfig:
    # λ = 100 × n²  where n = observations per year
    # Monthly (MN1): λ = 100 × 12²  = 14,400   ← paper default
    # Weekly  (W1) : λ = 100 × 52²  = 270,400
    # Daily   (D1) : λ = 100 × 252² = 6,350,400
    lambda_monthly: float = 14_400
    lambda_weekly:  float = 270_400
    lambda_daily:   float = 6_350_400

    ma_short: int = 3    # T1 — short moving average window (months)
    ma_long:  int = 12   # T2 — long  moving average window (months)


# ── Support & Resistance Parameters (Chapter 3.14) ───────────────────────────
@dataclass
class SRConfig:
    # Classic pivot:  C = (PH + PL + PC) / 3
    # Resistance:     R = 2×C − PL
    # Support:        S = 2×C − PH
    # Additional S/R levels (optional extensions)
    use_extended_levels: bool  = True   # R2, R3, S2, S3
    sr_buffer_pips:      float = 2.0    # Buffer around S/R levels (pips)


# ── Combined Signal Logic ─────────────────────────────────────────────────────
@dataclass
class SignalConfig:
    # HP trend direction acts as the REGIME FILTER
    #   Trend rising  → only consider LONG  signals from S&R
    #   Trend falling → only consider SHORT signals from S&R
    use_hp_as_regime_filter: bool  = True

    # MA crossover on HP trend provides the DIRECTION signal
    #   MA(T1) > MA(T2) → bullish trend confirmation
    #   MA(T1) < MA(T2) → bearish trend confirmation
    require_ma_crossover: bool = True

    # S&R provides ENTRY / EXIT precision
    #   Entry when price > C (long) or < C (short)
    #   Exit  when price ≥ R (long) or ≤ S (short)
    use_sr_for_entry: bool = True
    use_sr_for_exit:  bool = True

    # Minimum signal strength (0.0 – 1.0)
    # 1.0 = all three components must agree
    # 0.5 = at least two of three must agree
    min_confluence: float = 1.0


# ── Risk Management ───────────────────────────────────────────────────────────
@dataclass
class RiskConfig:
    risk_per_trade_pct: float = 1.0    # % of account balance risked per trade
    max_open_trades:    int   = 4      # Maximum concurrent positions
    max_daily_loss_pct: float = 3.0    # Daily drawdown circuit breaker (%)
    stop_loss_pips:     float = 50.0   # Default SL in pips
    take_profit_pips:   float = 100.0  # Default TP in pips (2:1 R:R)
    use_sr_as_sl:       bool  = True   # Use S/R levels instead of fixed pips
    trailing_stop:      bool  = False  # Enable trailing stop


# ── Execution Settings ────────────────────────────────────────────────────────
@dataclass
class ExecutionConfig:
    magic_number:    int   = 20240101  # Unique ID for this bot's orders
    slippage:        int   = 3         # Max allowed slippage (points)
    order_comment:   str   = "HP_SR_Bot"
    live_trading:    bool  = False     # ← MUST SET True to place real orders
    poll_interval_s: int   = 60        # How often to check signals (seconds)


# ── Master Config ─────────────────────────────────────────────────────────────
@dataclass
class BotConfig:
    mt5:        MT5Config        = field(default_factory=MT5Config)
    instrument: InstrumentConfig = field(default_factory=InstrumentConfig)
    hp:         HPFilterConfig   = field(default_factory=HPFilterConfig)
    sr:         SRConfig         = field(default_factory=SRConfig)
    signal:     SignalConfig      = field(default_factory=SignalConfig)
    risk:       RiskConfig        = field(default_factory=RiskConfig)
    execution:  ExecutionConfig   = field(default_factory=ExecutionConfig)
    log_level:  str               = "INFO"
    log_dir:    str               = "logs"


# ── Default instance ──────────────────────────────────────────────────────────
CONFIG = BotConfig()
