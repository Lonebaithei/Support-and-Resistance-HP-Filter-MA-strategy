"""
==============================================================================
bot.py  —  Main Trading Bot Orchestrator
==============================================================================
Combined Strategy: HP Filter MA  +  Support & Resistance
Sources:
    • Chapter 8.1  — Moving Averages with HP Filter (Kakushadze & Serur, 2018)
    • Chapter 3.14 — Support and Resistance         (Kakushadze & Serur, 2018)

Flow per poll cycle (default: every 60 seconds):
  ┌─────────────────────────────────────────────────┐
  │  For each symbol in config.instrument.symbols:  │
  │   1. Fetch monthly OHLCV  → HP Filter + MAs     │
  │   2. Fetch daily   OHLCV  → Pivot + S/R levels  │
  │   3. Get current price                          │
  │   4. Compute combined signal (3-layer)          │
  │   5. Risk checks                                │
  │   6. Execute: BUY / SELL / EXIT / HOLD          │
  └─────────────────────────────────────────────────┘

Usage:
    # 1. Fill in config.py with your MT5 credentials and settings
    # 2. Run:
    python bot.py

    # Dry-run (simulation, no real orders):
    python bot.py --dry-run

    # Single-pass (for cron / task scheduler):
    python bot.py --once
==============================================================================
"""

import sys
import time
import argparse
import numpy  as np
import pandas as pd
from typing import Optional

# ── Project imports ───────────────────────────────────────────────────────────
sys.path.insert(0, ".")
from config                          import CONFIG, BotConfig
from core.mt5_connector              import MT5Connector
from strategy.indicators             import format_levels
from strategy.combined_strategy      import CombinedStrategy, CombinedSignal
from execution.risk_manager          import RiskManager
from utils.logger                    import setup_logger


# =============================================================================
# BOT
# =============================================================================

class HPSRBot:
    """
    Orchestrates the HP Filter + S&R combined strategy via MetaTrader 5.
    """

    def __init__(self, config: BotConfig, dry_run: bool = False):
        self.cfg      = config
        self.dry_run  = dry_run

        # Override live trading flag in dry-run mode
        if dry_run:
            self.cfg.execution.live_trading = False

        self.logger   = setup_logger(
            log_dir   = config.log_dir,
            log_level = config.log_level,
        )
        self.mt5      = MT5Connector(config)
        self.strategy = CombinedStrategy(config)
        self.risk     = RiskManager(config)
        self._running = False

    # =========================================================================
    # STARTUP
    # =========================================================================

    def start(self, run_once: bool = False) -> None:
        """Connect to MT5 and begin the trading loop."""
        mode = "DRY-RUN" if self.dry_run else "LIVE"
        self.logger.info(f"Starting HP+SR Bot [{mode}]")
        self.logger.info(f"Symbols: {self.cfg.instrument.symbols}")
        self.logger.info(
            f"HP: λ={self.cfg.hp.lambda_monthly:,}  "
            f"MA({self.cfg.hp.ma_short}/{self.cfg.hp.ma_long})"
        )
        self.logger.info(
            f"S/R: buffer={self.cfg.sr.sr_buffer_pips} pips  "
            f"extended_levels={self.cfg.sr.use_extended_levels}"
        )

        self.mt5.connect()
        self._running = True

        try:
            if run_once:
                self._tick()
            else:
                self._loop()
        except KeyboardInterrupt:
            self.logger.info("Interrupted by user.")
        finally:
            self.mt5.disconnect()
            self.logger.info("Bot stopped.")

    def stop(self) -> None:
        self._running = False

    # =========================================================================
    # MAIN LOOP
    # =========================================================================

    def _loop(self) -> None:
        """Continuous polling loop."""
        poll = self.cfg.execution.poll_interval_s
        self.logger.info(f"Poll interval: {poll}s")

        while self._running:
            try:
                self._tick()
            except Exception as e:
                self.logger.error(f"Tick error: {e}", exc_info=True)
                if not self.mt5.is_connected():
                    self.logger.warning("Connection lost — attempting reconnect...")
                    self.mt5.reconnect()

            self.logger.debug(f"Sleeping {poll}s...")
            time.sleep(poll)

    def _tick(self) -> None:
        """Single evaluation cycle across all configured symbols."""
        acct = self.mt5.get_account_info()
        if acct is None:
            self.logger.error("Could not retrieve account info.")
            return

        self.risk.reset_daily(acct["balance"])

        # Emergency stop: daily loss breached
        if self.risk.is_daily_loss_breached(acct["equity"]):
            self.logger.warning("Daily loss limit hit — closing all positions.")
            self.mt5.close_all_positions()
            return

        open_positions = self.mt5.get_open_positions(
            magic_number=self.cfg.execution.magic_number
        )
        portfolio = self.risk.portfolio_summary(open_positions)
        self.logger.info(
            f"Account: balance={acct['balance']:.2f} {acct['currency']}  "
            f"equity={acct['equity']:.2f}  "
            f"open={portfolio['open_positions']}  "
            f"unrealised_PnL={portfolio['unrealised_pnl']:.2f}"
        )

        for symbol in self.cfg.instrument.symbols:
            self._process_symbol(symbol, acct, open_positions)

    # =========================================================================
    # PER-SYMBOL LOGIC
    # =========================================================================

    def _process_symbol(self,
                        symbol:         str,
                        acct:           dict,
                        open_positions: list) -> None:
        """Fetch data, compute signal, and execute for one symbol."""

        # ── 1. Fetch HP (monthly) data ────────────────────────────────────────
        hp_df = self.mt5.get_ohlcv(
            symbol    = symbol,
            timeframe = self.cfg.instrument.hp_timeframe,
            n_bars    = self.cfg.instrument.hp_lookback,
        )
        if hp_df is None or len(hp_df) < self.cfg.hp.ma_long + 2:
            self.logger.warning(f"{symbol}: Insufficient HP data — skipping.")
            return

        hp_prices = hp_df["close"].values

        # ── 2. Fetch S&R (daily) data ─────────────────────────────────────────
        sr_df = self.mt5.get_ohlcv(
            symbol    = symbol,
            timeframe = self.cfg.instrument.sr_timeframe,
            n_bars    = 5,    # Only need last couple of daily bars
        )
        if sr_df is None or len(sr_df) < 2:
            self.logger.warning(f"{symbol}: Insufficient daily data — skipping.")
            return

        # ── 3. Get current price ──────────────────────────────────────────────
        tick = self.mt5.get_current_price(symbol)
        if tick is None:
            self.logger.warning(f"{symbol}: No tick data — skipping.")
            return

        current_price = tick["mid"]
        pip_size      = self.mt5.get_pip_size(symbol)

        # ── 4. Compute combined signal ────────────────────────────────────────
        signal = self.strategy.compute_signal(
            symbol        = symbol,
            hp_prices     = hp_prices,
            daily_ohlcv   = sr_df,
            current_price = current_price,
            pip_size      = pip_size,
        )

        summary = self.strategy.format_signal(signal)
        self.logger.info(summary)

        # ── 5. Handle EXIT signals ────────────────────────────────────────────
        if signal.action in ("EXIT_LONG", "EXIT_SHORT"):
            self._handle_exit(symbol, signal, open_positions)
            return

        # ── 6. Handle ENTRY signals ───────────────────────────────────────────
        if signal.action in ("BUY", "SELL"):
            self._handle_entry(symbol, signal, acct, open_positions, pip_size)

    # =========================================================================
    # EXECUTION HANDLERS
    # =========================================================================

    def _handle_entry(self,
                      symbol:         str,
                      signal:         CombinedSignal,
                      acct:           dict,
                      open_positions: list,
                      pip_size:       float) -> None:
        """Execute a new trade if risk checks pass."""

        # Risk guard
        if not self.risk.can_open_trade(symbol, open_positions, acct["equity"]):
            return

        # R:R guard
        if not self.risk.validate_rr(signal.rr_ratio, min_rr=1.5):
            return

        direction = 1 if signal.action == "BUY" else -1

        result = self.mt5.place_market_order(
            symbol    = symbol,
            direction = direction,
            sl_price  = signal.sl_price,
            tp_price  = signal.tp_price,
            comment   = (
                f"{self.cfg.execution.order_comment}_"
                f"conf{signal.confluence:.0%}"
            ),
        )

        if result:
            self.logger.info(
                f"[ORDER] {signal.action} {symbol}  "
                f"entry={signal.entry_price:.5f}  "
                f"SL={signal.sl_price:.5f}  "
                f"TP={signal.tp_price:.5f}  "
                f"R:R={signal.rr_ratio:.2f}  "
                f"ticket=#{result.order}"
            )

    def _handle_exit(self,
                     symbol:         str,
                     signal:         CombinedSignal,
                     open_positions: list) -> None:
        """Close existing position when exit signal triggered."""
        for pos in open_positions:
            if pos.symbol != symbol:
                continue
            if pos.magic != self.cfg.execution.magic_number:
                continue

            # Match direction: EXIT_LONG closes a BUY, EXIT_SHORT closes a SELL
            is_long  = (pos.type == 0)   # 0 = ORDER_TYPE_BUY in MT5
            if signal.action == "EXIT_LONG"  and is_long:
                self.mt5.close_position(pos)
                self.logger.info(f"[EXIT] Closed LONG #{pos.ticket} {symbol}  "
                                 f"P&L={pos.profit:.2f}")
            elif signal.action == "EXIT_SHORT" and not is_long:
                self.mt5.close_position(pos)
                self.logger.info(f"[EXIT] Closed SHORT #{pos.ticket} {symbol}  "
                                 f"P&L={pos.profit:.2f}")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HP Filter + S&R FX Trading Bot")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run in simulation mode (no real orders)")
    parser.add_argument("--once",    action="store_true",
                        help="Run a single evaluation cycle then exit")
    args = parser.parse_args()

    bot = HPSRBot(config=CONFIG, dry_run=args.dry_run)
    bot.start(run_once=args.once)
