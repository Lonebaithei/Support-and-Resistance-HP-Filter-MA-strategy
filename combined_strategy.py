"""
==============================================================================
strategy/combined_strategy.py  —  Combined Signal Engine
==============================================================================
Merges Chapter 8.1 (HP Filter MA) and Chapter 3.14 (Support & Resistance)
into a single three-layer signal framework:

  LAYER 1 — REGIME  (HP Trend direction)
    • Is the HP trend rising or falling?
    • Only take longs in a rising trend; only shorts in a falling trend.

  LAYER 2 — DIRECTION  (MA Crossover on HP trend)
    • Has MA(T1) crossed above MA(T2)? → Confirmed bullish.
    • Has MA(T1) crossed below MA(T2)? → Confirmed bearish.

  LAYER 3 — TIMING  (S&R Pivot Levels)
    • Is current price above the daily pivot C? → Bullish bias confirmed.
    • Has price reached R1 (long) or S1 (short)? → Exit signal.

Final Signal Logic:
    BUY  when: Regime=BULLISH  AND  MA=BULLISH  AND  SR=ABOVE PIVOT
    SELL when: Regime=BEARISH  AND  MA=BEARISH  AND  SR=BELOW PIVOT
    EXIT when: SR exit flag triggered (price reached R1 or S1)
==============================================================================
"""

import numpy  as np
import pandas as pd
from dataclasses import dataclass
from typing      import Optional, Dict, Tuple

from config                  import BotConfig
from strategy.indicators     import (
    HPResult, SRSignal, PivotLevels,
    compute_hp_signal,
    compute_pivot_levels,
    compute_sr_signal,
    format_levels,
)


# =============================================================================
# COMBINED SIGNAL OUTPUT
# =============================================================================

@dataclass
class CombinedSignal:
    """
    Full signal output from the three-layer strategy engine.
    """
    symbol:     str
    timestamp:  pd.Timestamp

    # ── Layer outputs ─────────────────────────────────────────────────────────
    hp:  HPResult       # HP Filter + MA crossover result
    sr:  SRSignal       # S&R pivot level result

    # ── Layer scores (+1 / 0 / -1 for each layer) ────────────────────────────
    regime_score:    int   # HP trend direction
    direction_score: int   # MA crossover direction
    sr_score:        int   # S&R pivot direction

    # ── Final decision ────────────────────────────────────────────────────────
    confluence: float  # How many layers agree (0.33 / 0.67 / 1.0)
    action:     str    # "BUY" | "SELL" | "EXIT_LONG" | "EXIT_SHORT" | "HOLD"

    # ── Recommended trade levels ──────────────────────────────────────────────
    entry_price: float = 0.0
    sl_price:    float = 0.0   # Stop-loss at S1 (long) or R1 (short)
    tp_price:    float = 0.0   # Take-profit at R1 (long) or S1 (short)
    rr_ratio:    float = 0.0   # Risk/Reward ratio


# =============================================================================
# STRATEGY ENGINE
# =============================================================================

class CombinedStrategy:
    """
    Three-layer signal engine combining HP Filter MA and S&R strategies.
    """

    def __init__(self, config: BotConfig):
        self.cfg     = config
        self.hp_cfg  = config.hp
        self.sr_cfg  = config.sr
        self.sig_cfg = config.signal
        self.risk_cfg = config.risk

    # =========================================================================
    # MAIN SIGNAL COMPUTATION
    # =========================================================================

    def compute_signal(self,
                       symbol:       str,
                       hp_prices:    np.ndarray,
                       daily_ohlcv:  pd.DataFrame,
                       current_price: float,
                       pip_size:     float = 0.0001) -> CombinedSignal:
        """
        Compute the full three-layer combined signal.

        Args:
            symbol        : e.g. "EURUSD"
            hp_prices     : array of monthly closes for HP filter (100+ bars)
            daily_ohlcv   : DataFrame with columns [open,high,low,close] daily
            current_price : current market price (mid)
            pip_size      : pip size for the symbol

        Returns:
            CombinedSignal with action and trade levels
        """
        timestamp = pd.Timestamp.now()

        # ── LAYER 1 & 2: HP Filter + MA Crossover ────────────────────────────
        hp = compute_hp_signal(
            prices = hp_prices,
            lam    = self.hp_cfg.lambda_monthly,
            t1     = self.hp_cfg.ma_short,
            t2     = self.hp_cfg.ma_long,
        )

        # ── LAYER 3: S&R Pivot Levels ─────────────────────────────────────────
        # Use previous day's OHLC to compute today's pivot
        prev_bar    = daily_ohlcv.iloc[-2]   # -1 is today (may be incomplete)
        pivot_lvls  = compute_pivot_levels(
            prev_high  = float(prev_bar["high"]),
            prev_low   = float(prev_bar["low"]),
            prev_close = float(prev_bar["close"]),
            extended   = self.sr_cfg.use_extended_levels,
        )
        sr = compute_sr_signal(
            current_price = current_price,
            levels        = pivot_lvls,
            pip_size      = pip_size,
            sr_buffer     = self.sr_cfg.sr_buffer_pips,
        )

        # ── Score each layer ──────────────────────────────────────────────────
        # Layer 1: HP trend regime
        regime_score    = hp.signal                    # +1 / -1 / 0

        # Layer 2: MA crossover direction (uses same hp.signal but crossover flag)
        direction_score = hp.signal if (
            not self.sig_cfg.require_ma_crossover or hp.crossover
        ) else hp.signal                               # Still directional even without fresh crossover

        # Layer 3: S&R pivot direction
        sr_score = sr.direction                        # +1 / -1 / 0

        # ── Confluence: how many non-zero layers agree ────────────────────────
        scores     = [regime_score, direction_score, sr_score]
        nonzero    = [s for s in scores if s != 0]
        if len(nonzero) == 0:
            confluence = 0.0
            net_direction = 0
        else:
            agreement     = sum(nonzero)
            net_direction = 1 if agreement > 0 else -1
            confluence    = abs(agreement) / (3 * len(nonzero) / len(nonzero))
            # Simplified: fraction of agreeing non-zero signals
            agreeing   = [s for s in nonzero if s == net_direction]
            confluence = len(agreeing) / 3.0

        # ── Determine action ──────────────────────────────────────────────────
        action = self._determine_action(
            hp            = hp,
            sr            = sr,
            regime_score  = regime_score,
            sr_score      = sr_score,
            confluence    = confluence,
            net_direction = net_direction,
        )

        # ── Trade levels from S&R ─────────────────────────────────────────────
        entry_price, sl_price, tp_price, rr = self._compute_trade_levels(
            action        = action,
            current_price = current_price,
            levels        = pivot_lvls,
            pip_size      = pip_size,
        )

        signal = CombinedSignal(
            symbol          = symbol,
            timestamp       = timestamp,
            hp              = hp,
            sr              = sr,
            regime_score    = regime_score,
            direction_score = direction_score,
            sr_score        = sr_score,
            confluence      = confluence,
            action          = action,
            entry_price     = entry_price,
            sl_price        = sl_price,
            tp_price        = tp_price,
            rr_ratio        = rr,
        )

        return signal

    # =========================================================================
    # ACTION LOGIC
    # =========================================================================

    def _determine_action(self,
                          hp:            HPResult,
                          sr:            SRSignal,
                          regime_score:  int,
                          sr_score:      int,
                          confluence:    float,
                          net_direction: int) -> str:
        """
        Map layer scores to a final trading action.

        Priority:
          1. EXIT signals always take precedence
          2. Entry requires minimum confluence
        """
        # ── EXIT signals (highest priority) ───────────────────────────────────
        if sr.at_exit:
            if sr.direction == 1:
                return "EXIT_LONG"
            elif sr.direction == -1:
                return "EXIT_SHORT"

        # ── Check minimum confluence ───────────────────────────────────────────
        if confluence < self.sig_cfg.min_confluence:
            return "HOLD"

        # ── Entry signals ─────────────────────────────────────────────────────
        if self.sig_cfg.use_hp_as_regime_filter:
            # HP trend must agree with SR direction for entry
            if regime_score == 1 and sr_score == 1 and net_direction == 1:
                return "BUY"
            elif regime_score == -1 and sr_score == -1 and net_direction == -1:
                return "SELL"
        else:
            if net_direction == 1:
                return "BUY"
            elif net_direction == -1:
                return "SELL"

        return "HOLD"

    # =========================================================================
    # TRADE LEVELS
    # =========================================================================

    def _compute_trade_levels(self,
                              action:        str,
                              current_price: float,
                              levels:        PivotLevels,
                              pip_size:      float
                              ) -> Tuple[float, float, float, float]:
        """
        Determine entry, stop-loss, and take-profit using S&R levels.

        For LONG:
            Entry : current market price (just above C)
            SL    : S1 (support level — below pivot)
            TP    : R1 (resistance level — exit target)

        For SHORT:
            Entry : current market price (just below C)
            SL    : R1 (resistance level — above pivot)
            TP    : S1 (support level — exit target)
        """
        if action not in ("BUY", "SELL"):
            return 0.0, 0.0, 0.0, 0.0

        if action == "BUY":
            entry = current_price
            if self.risk_cfg.use_sr_as_sl:
                sl = levels.s1
                tp = levels.r1
            else:
                sl = entry - self.risk_cfg.stop_loss_pips   * pip_size
                tp = entry + self.risk_cfg.take_profit_pips * pip_size

        else:  # SELL
            entry = current_price
            if self.risk_cfg.use_sr_as_sl:
                sl = levels.r1
                tp = levels.s1
            else:
                sl = entry + self.risk_cfg.stop_loss_pips   * pip_size
                tp = entry - self.risk_cfg.take_profit_pips * pip_size

        # Risk:Reward ratio
        risk   = abs(entry - sl)
        reward = abs(tp - entry)
        rr     = reward / risk if risk > 0 else 0.0

        return entry, sl, tp, rr

    # =========================================================================
    # SIGNAL SUMMARY (for logging)
    # =========================================================================

    def format_signal(self, signal: CombinedSignal) -> str:
        """Return a human-readable summary of the combined signal."""
        action_icons = {
            "BUY":        "🟢 BUY",
            "SELL":       "🔴 SELL",
            "EXIT_LONG":  "⬜ EXIT LONG",
            "EXIT_SHORT": "⬜ EXIT SHORT",
            "HOLD":       "⏸  HOLD",
        }
        icon   = action_icons.get(signal.action, signal.action)
        digits = 5

        lines = [
            f"\n{'='*55}",
            f"  {signal.symbol}  |  {signal.timestamp.strftime('%Y-%m-%d %H:%M')}",
            f"{'='*55}",
            f"  LAYER 1 — HP Regime:   {'▲ BULL' if signal.regime_score==1 else '▼ BEAR' if signal.regime_score==-1 else '— NEUT'}",
            f"  LAYER 2 — MA Cross:    {'▲ BULL' if signal.direction_score==1 else '▼ BEAR' if signal.direction_score==-1 else '— NEUT'}  "
            f"{'[FRESH CROSSOVER]' if signal.hp.crossover else ''}",
            f"  LAYER 3 — S/R Pivot:   {'▲ ABOVE C' if signal.sr_score==1 else '▼ BELOW C' if signal.sr_score==-1 else '— AT C'}",
            f"  {'─'*49}",
            f"  Confluence:  {signal.confluence:.0%}",
            f"  ACTION:      {icon}",
        ]
        if signal.action in ("BUY", "SELL"):
            lines += [
                f"  {'─'*49}",
                f"  Entry: {signal.entry_price:.{digits}f}  |  "
                f"SL: {signal.sl_price:.{digits}f}  |  "
                f"TP: {signal.tp_price:.{digits}f}",
                f"  R:R Ratio: {signal.rr_ratio:.2f}:1",
                f"  Pivot C: {signal.sr.levels.pivot:.{digits}f}  "
                f"R1: {signal.sr.levels.r1:.{digits}f}  "
                f"S1: {signal.sr.levels.s1:.{digits}f}",
            ]
        lines.append(f"{'='*55}\n")
        return "\n".join(lines)
