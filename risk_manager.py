"""
==============================================================================
execution/risk_manager.py  —  Risk & Portfolio Guard
==============================================================================
Enforces:
  • Per-trade risk limits
  • Maximum open positions
  • Daily loss circuit breaker
  • Duplicate position prevention
==============================================================================
"""

import pandas as pd
from typing  import Optional, Dict
from config  import BotConfig


class RiskManager:
    """Guards the bot against over-trading and runaway losses."""

    def __init__(self, config: BotConfig):
        self.cfg        = config.risk
        self.exec_cfg   = config.execution
        self._day_start_balance: Optional[float] = None
        self._daily_pnl: float  = 0.0
        self._today: Optional[pd.Timestamp] = None

    def reset_daily(self, current_balance: float) -> None:
        """Call at the start of each trading day."""
        today = pd.Timestamp.now().date()
        if self._today != today:
            self._day_start_balance = current_balance
            self._daily_pnl         = 0.0
            self._today             = today

    def is_daily_loss_breached(self, current_equity: float) -> bool:
        """Returns True if daily drawdown exceeds the configured limit."""
        if self._day_start_balance is None:
            return False
        daily_dd = (self._day_start_balance - current_equity) / \
                    self._day_start_balance * 100
        if daily_dd >= self.cfg.max_daily_loss_pct:
            print(f"[RISK] ⚠ Daily loss circuit breaker triggered: "
                  f"{daily_dd:.2f}% >= {self.cfg.max_daily_loss_pct:.2f}%")
            return True
        return False

    def can_open_trade(self,
                       symbol:          str,
                       open_positions:  list,
                       account_equity:  float) -> bool:
        """
        Returns True if a new trade is permitted.
        Checks:
          1. Maximum concurrent positions not reached
          2. Daily loss not breached
          3. No existing position in same symbol
        """
        # Max open trades check
        bot_positions = [p for p in open_positions
                         if p.magic == self.exec_cfg.magic_number]
        if len(bot_positions) >= self.cfg.max_open_trades:
            print(f"[RISK] Max open trades reached ({self.cfg.max_open_trades}). "
                  f"Skipping {symbol}.")
            return False

        # Daily loss circuit breaker
        if self.is_daily_loss_breached(account_equity):
            return False

        # Duplicate position check
        existing = [p for p in bot_positions if p.symbol == symbol]
        if existing:
            print(f"[RISK] Position already open for {symbol}. Skipping.")
            return False

        return True

    def validate_rr(self, rr_ratio: float, min_rr: float = 1.5) -> bool:
        """Reject trades with poor risk/reward."""
        if rr_ratio < min_rr:
            print(f"[RISK] R:R {rr_ratio:.2f}:1 below minimum {min_rr:.2f}:1. "
                  f"Skipping.")
            return False
        return True

    def portfolio_summary(self, open_positions: list) -> Dict:
        """Return a quick portfolio snapshot."""
        bot_pos = [p for p in open_positions
                   if p.magic == self.exec_cfg.magic_number]
        total_pnl = sum(p.profit for p in bot_pos)
        return {
            "open_positions": len(bot_pos),
            "unrealised_pnl": total_pnl,
            "symbols":        [p.symbol for p in bot_pos],
        }
