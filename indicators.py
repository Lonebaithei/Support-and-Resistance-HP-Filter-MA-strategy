"""
==============================================================================
strategy/indicators.py  —  Technical Indicators
==============================================================================
Implements:
  • HP Filter          (Chapter 8.1  — Equations 437–439)
  • Moving Averages    (Chapter 8.1  — MA crossover on HP trend)
  • Pivot Points       (Chapter 3.14 — Equations 325–327)
  • Support/Resistance (Chapter 3.14 — Equation 328)
==============================================================================
"""

import numpy  as np
import pandas as pd
from dataclasses import dataclass
from typing      import Optional, Tuple
from scipy       import sparse
from scipy.sparse.linalg import spsolve


# =============================================================================
# DATA CONTAINERS
# =============================================================================

@dataclass
class HPResult:
    """Output of the HP Filter."""
    trend:  np.ndarray    # S*(t) — smoothed lower-frequency component
    cycle:  np.ndarray    # v(t)  — higher-frequency noise
    ma_short: np.ndarray  # MA(T1) computed on trend
    ma_long:  np.ndarray  # MA(T2) computed on trend
    signal:   int         # +1 (bullish) | -1 (bearish) | 0 (neutral)
    crossover: bool       # True if a fresh MA crossover just occurred


@dataclass
class PivotLevels:
    """
    Classic Pivot Point levels (Chapter 3.14, Equations 325–327).

    Primary:
        C  = (PH + PL + PC) / 3          ← Pivot centre
        R1 = 2×C − PL                    ← First resistance
        S1 = 2×C − PH                    ← First support

    Extended (optional R2/S2 and R3/S3):
        R2 = C + (PH − PL)
        S2 = C − (PH − PL)
        R3 = PH + 2×(C − PL)
        S3 = PL − 2×(PH − C)
    """
    pivot:  float    # C  — centre
    r1:     float    # First resistance
    s1:     float    # First support
    r2:     float = 0.0
    s2:     float = 0.0
    r3:     float = 0.0
    s3:     float = 0.0


@dataclass
class SRSignal:
    """Output of the S&R signal logic (Chapter 3.14, Equation 328)."""
    levels:    PivotLevels
    direction: int     # +1 = price above pivot (bullish zone)
                       # -1 = price below pivot (bearish zone)
                       #  0 = price at pivot (neutral)
    at_entry:  bool    # True if price just crossed C
    at_exit:   bool    # True if price hit R1 (long) or S1 (short)
    entry_px:  float   # Current price used for signal evaluation
    exit_px:   float   # The relevant exit level (R1 or S1)


# =============================================================================
# HP FILTER  (Chapter 8.1 — Equations 437–439)
# =============================================================================

def hp_filter(series: np.ndarray, lam: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Hodrick-Prescott Filter.

    Minimises:
        g = Σ[S(t) − S*(t)]²  +  λ × Σ[S*(t+1) − 2S*(t) + S*(t−1)]²

    Args:
        series : 1-D array of raw prices S(t)
        lam    : smoothing parameter λ
                 Monthly → 14,400  | Weekly → 270,400  | Daily → 6,350,400

    Returns:
        trend  : S*(t) — smooth trend component
        cycle  : v(t)  — noise / irregular component
    """
    T = len(series)
    if T < 4:
        return series.copy(), np.zeros(T)

    # Build second-difference matrix (T-2) × T
    e  = np.ones(T)
    D2 = sparse.diags([e, -2*e, e], [0, 1, 2], shape=(T-2, T), format="csc")

    # Solve (I + λ D2ᵀ D2) × trend = series
    A     = sparse.eye(T, format="csc") + lam * D2.T.dot(D2)
    trend = spsolve(A, series)
    cycle = series - trend
    return trend, cycle


# =============================================================================
# MOVING AVERAGES  (Chapter 8.1)
# =============================================================================

def simple_moving_average(arr: np.ndarray, window: int) -> np.ndarray:
    """Simple moving average with NaN for warm-up period."""
    s = pd.Series(arr)
    return s.rolling(window=window, min_periods=window).mean().values


def compute_hp_signal(prices: np.ndarray,
                      lam:      float,
                      t1:       int,
                      t2:       int) -> HPResult:
    """
    Full HP Filter pipeline:
      1. Apply HP filter to extract trend S*(t)
      2. Compute MA(T1) and MA(T2) on trend
      3. Determine MA crossover direction signal

    Args:
        prices : array of closing prices (monthly recommended)
        lam    : HP smoothing parameter
        t1     : short MA window
        t2     : long MA window

    Returns:
        HPResult with trend, MAs, and directional signal
    """
    trend, cycle = hp_filter(prices, lam)

    ma_short = simple_moving_average(trend, t1)
    ma_long  = simple_moving_average(trend, t2)

    # Latest valid values
    latest_short = ma_short[-1]
    latest_long  = ma_long[-1]
    prev_short   = ma_short[-2] if len(ma_short) > 1 else np.nan
    prev_long    = ma_long[-2]  if len(ma_long)  > 1 else np.nan

    # Direction signal: +1 bullish | -1 bearish | 0 neutral
    if np.isnan(latest_short) or np.isnan(latest_long):
        signal = 0
    elif latest_short > latest_long:
        signal = 1
    elif latest_short < latest_long:
        signal = -1
    else:
        signal = 0

    # Detect fresh crossover (signal flipped vs previous bar)
    if not (np.isnan(prev_short) or np.isnan(prev_long)):
        prev_signal = 1 if prev_short > prev_long else (-1 if prev_short < prev_long else 0)
        crossover = (signal != 0 and signal != prev_signal)
    else:
        crossover = False

    return HPResult(
        trend     = trend,
        cycle     = cycle,
        ma_short  = ma_short,
        ma_long   = ma_long,
        signal    = signal,
        crossover = crossover,
    )


# =============================================================================
# PIVOT POINTS & SUPPORT / RESISTANCE  (Chapter 3.14 — Equations 325–327)
# =============================================================================

def compute_pivot_levels(prev_high:  float,
                         prev_low:   float,
                         prev_close: float,
                         extended:   bool = True) -> PivotLevels:
    """
    Calculate classic pivot point levels.

    Equations 325–327 (Kakushadze & Serur, 2018):
        C  = (PH + PL + PC) / 3
        R1 = 2×C − PL
        S1 = 2×C − PH

    Extended levels:
        R2 = C + (PH − PL)
        S2 = C − (PH − PL)
        R3 = PH + 2×(C − PL)
        S3 = PL − 2×(PH − C)

    Args:
        prev_high  : PH — previous period's high
        prev_low   : PL — previous period's low
        prev_close : PC — previous period's close
        extended   : whether to compute R2/S2 and R3/S3

    Returns:
        PivotLevels dataclass
    """
    PH, PL, PC = prev_high, prev_low, prev_close

    C  = (PH + PL + PC) / 3.0    # Eq. 325
    R1 = 2.0 * C - PL             # Eq. 326
    S1 = 2.0 * C - PH             # Eq. 327

    if extended:
        rng = PH - PL
        R2  = C  + rng
        S2  = C  - rng
        R3  = PH + 2.0 * (C - PL)
        S3  = PL - 2.0 * (PH - C)
    else:
        R2 = S2 = R3 = S3 = 0.0

    return PivotLevels(pivot=C, r1=R1, s1=S1, r2=R2, s2=S2, r3=R3, s3=S3)


def compute_sr_signal(current_price: float,
                      levels:        PivotLevels,
                      pip_size:      float = 0.0001,
                      sr_buffer:     float = 2.0) -> SRSignal:
    """
    Generate S&R trading signal (Chapter 3.14, Equation 328):

        Establish long  if P > C   →  direction = +1
        Liquidate long  if P ≥ R1  →  at_exit   = True (long exit)
        Establish short if P < C   →  direction = -1
        Liquidate short if P ≤ S1  →  at_exit   = True (short exit)

    A buffer (in pips) is applied around C, R1, S1 to avoid false signals
    at exact level touches.

    Args:
        current_price : current market price P
        levels        : PivotLevels computed from previous day's OHLC
        pip_size      : pip value (e.g. 0.0001 for EURUSD)
        sr_buffer     : tolerance in pips for level touches

    Returns:
        SRSignal with direction, entry, and exit flags
    """
    buf = sr_buffer * pip_size

    P = current_price
    C, R1, S1 = levels.pivot, levels.r1, levels.s1

    # ── Direction: is price above or below pivot? ─────────────────────────────
    if P > C + buf:
        direction = 1
    elif P < C - buf:
        direction = -1
    else:
        direction = 0

    # ── Entry: has price just cleared the pivot with buffer? ──────────────────
    at_entry = (direction != 0)

    # ── Exit: has price reached the corresponding S/R boundary? ──────────────
    if direction == 1:
        at_exit  = P >= R1 - buf
        exit_px  = R1
    elif direction == -1:
        at_exit  = P <= S1 + buf
        exit_px  = S1
    else:
        at_exit  = False
        exit_px  = C

    return SRSignal(
        levels    = levels,
        direction = direction,
        at_entry  = at_entry,
        at_exit   = at_exit,
        entry_px  = P,
        exit_px   = exit_px,
    )


# =============================================================================
# UTILITY: Format levels for logging
# =============================================================================

def format_levels(symbol: str,
                  hp:     HPResult,
                  sr:     SRSignal,
                  digits: int = 5) -> str:
    d = digits
    lines = [
        f"── {symbol} ──────────────────────────────",
        f"  HP Trend:  {hp.trend[-1]:.{d}f}  "
        f"MA({len(hp.ma_short)}-short): {hp.ma_short[-1]:.{d}f}  "
        f"MA({len(hp.ma_long)}-long): {hp.ma_long[-1]:.{d}f}",
        f"  HP Signal: {'BULLISH ▲' if hp.signal==1 else 'BEARISH ▼' if hp.signal==-1 else 'NEUTRAL'}",
        f"  Pivot C:   {sr.levels.pivot:.{d}f}",
        f"  R1:        {sr.levels.r1:.{d}f}  |  S1: {sr.levels.s1:.{d}f}",
        f"  SR Dir:    {'ABOVE PIVOT ▲' if sr.direction==1 else 'BELOW PIVOT ▼' if sr.direction==-1 else 'AT PIVOT'}",
        f"  Exit Flag: {'YES — approaching {:.{}f}'.format(sr.exit_px, d) if sr.at_exit else 'No'}",
    ]
    return "\n".join(lines)
