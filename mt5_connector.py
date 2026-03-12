"""
==============================================================================
core/mt5_connector.py  —  MetaTrader 5 Integration Layer
==============================================================================
Handles:
  • MT5 terminal connection / disconnection
  • Historical OHLCV data fetching
  • Account & position information
  • Order placement (market, pending, modify, close)

Requirements:
    pip install MetaTrader5 numpy pandas
==============================================================================
"""

import time
import numpy  as np
import pandas as pd
from datetime import datetime
from typing   import Optional, Dict, List, Tuple

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    print("[WARNING] MetaTrader5 package not installed. Run: pip install MetaTrader5")

from config import BotConfig


# ── Timeframe map ─────────────────────────────────────────────────────────────
TIMEFRAME_MAP: Dict[str, int] = {}
if MT5_AVAILABLE:
    TIMEFRAME_MAP = {
        "M1":  mt5.TIMEFRAME_M1,
        "M5":  mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1":  mt5.TIMEFRAME_H1,
        "H4":  mt5.TIMEFRAME_H4,
        "D1":  mt5.TIMEFRAME_D1,
        "W1":  mt5.TIMEFRAME_W1,
        "MN1": mt5.TIMEFRAME_MN1,
    }


class MT5Connector:
    """
    Wraps the MetaTrader5 Python API with connection management,
    data fetching, and order execution utilities.
    """

    def __init__(self, config: BotConfig):
        self.cfg      = config
        self.mt5_cfg  = config.mt5
        self.exec_cfg = config.execution
        self._connected = False

    # =========================================================================
    # CONNECTION
    # =========================================================================

    def connect(self) -> bool:
        """
        Initialise and log in to the MT5 terminal.
        Returns True on success, False on failure.
        """
        if not MT5_AVAILABLE:
            raise RuntimeError("MetaTrader5 package is not installed.")

        # Initialise terminal
        init_kwargs = {"timeout": self.mt5_cfg.timeout}
        if self.mt5_cfg.path:
            init_kwargs["path"] = self.mt5_cfg.path

        if not mt5.initialize(**init_kwargs):
            err = mt5.last_error()
            raise ConnectionError(f"MT5 initialize() failed: {err}")

        # Log in if credentials provided
        if self.mt5_cfg.login:
            ok = mt5.login(
                login    = self.mt5_cfg.login,
                password = self.mt5_cfg.password,
                server   = self.mt5_cfg.server,
            )
            if not ok:
                err = mt5.last_error()
                mt5.shutdown()
                raise ConnectionError(f"MT5 login() failed: {err}")

        info = mt5.account_info()
        print(f"[MT5] Connected — Account: {info.login} | "
              f"Balance: {info.balance:.2f} {info.currency} | "
              f"Server: {info.server}")
        self._connected = True
        return True

    def disconnect(self) -> None:
        """Cleanly shut down the MT5 connection."""
        if MT5_AVAILABLE and self._connected:
            mt5.shutdown()
            self._connected = False
            print("[MT5] Disconnected.")

    def is_connected(self) -> bool:
        if not MT5_AVAILABLE or not self._connected:
            return False
        return mt5.terminal_info() is not None

    def reconnect(self, retries: int = 3, delay: int = 5) -> bool:
        """Attempt reconnection with retries."""
        for attempt in range(1, retries + 1):
            print(f"[MT5] Reconnect attempt {attempt}/{retries}...")
            try:
                self.disconnect()
                time.sleep(delay)
                return self.connect()
            except Exception as e:
                print(f"[MT5] Reconnect failed: {e}")
        return False

    # =========================================================================
    # MARKET DATA
    # =========================================================================

    def get_ohlcv(self,
                  symbol:    str,
                  timeframe: str,
                  n_bars:    int) -> Optional[pd.DataFrame]:
        """
        Fetch the last n_bars of OHLCV data for a symbol.

        Returns a DataFrame with columns:
            time, open, high, low, close, tick_volume, spread, real_volume
        """
        tf = TIMEFRAME_MAP.get(timeframe)
        if tf is None:
            raise ValueError(f"Unknown timeframe: {timeframe}. "
                             f"Valid: {list(TIMEFRAME_MAP.keys())}")

        rates = mt5.copy_rates_from_pos(symbol, tf, 0, n_bars)
        if rates is None or len(rates) == 0:
            print(f"[MT5] No data for {symbol} {timeframe}: {mt5.last_error()}")
            return None

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)
        return df

    def get_current_price(self, symbol: str) -> Optional[Dict[str, float]]:
        """Return current bid/ask/mid for a symbol."""
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return None
        return {
            "bid": tick.bid,
            "ask": tick.ask,
            "mid": (tick.bid + tick.ask) / 2,
            "time": datetime.fromtimestamp(tick.time),
        }

    def get_symbol_info(self, symbol: str) -> Optional[object]:
        """Return MT5 symbol info (digits, point, volume_min, etc.)."""
        info = mt5.symbol_info(symbol)
        if info is None:
            print(f"[MT5] Symbol {symbol} not found: {mt5.last_error()}")
        return info

    def get_pip_size(self, symbol: str) -> float:
        """Return pip size (e.g. 0.0001 for EURUSD, 0.01 for USDJPY)."""
        info = self.get_symbol_info(symbol)
        if info is None:
            return 0.0001
        # For most FX pairs: pip = 10 × point
        return info.point * 10

    # =========================================================================
    # ACCOUNT & POSITIONS
    # =========================================================================

    def get_account_info(self) -> Optional[Dict]:
        """Return key account metrics."""
        info = mt5.account_info()
        if info is None:
            return None
        return {
            "balance":  info.balance,
            "equity":   info.equity,
            "margin":   info.margin,
            "free_margin": info.margin_free,
            "currency": info.currency,
            "leverage": info.leverage,
        }

    def get_open_positions(self,
                           symbol:       Optional[str] = None,
                           magic_number: Optional[int] = None) -> List:
        """Return list of open positions, optionally filtered."""
        positions = mt5.positions_get(symbol=symbol) or []
        if magic_number is not None:
            positions = [p for p in positions if p.magic == magic_number]
        return list(positions)

    def count_open_positions(self, symbol: Optional[str] = None) -> int:
        return len(self.get_open_positions(
            symbol=symbol,
            magic_number=self.exec_cfg.magic_number
        ))

    # =========================================================================
    # ORDER EXECUTION
    # =========================================================================

    def _calculate_lot_size(self, symbol: str, sl_pips: float) -> float:
        """
        Position sizing based on risk % of account balance.

        Formula:
            lots = (balance × risk_pct) / (sl_pips × pip_value × 10)
        """
        acct       = self.get_account_info()
        sym_info   = self.get_symbol_info(symbol)
        if acct is None or sym_info is None:
            return sym_info.volume_min if sym_info else 0.01

        risk_amount  = acct["balance"] * (self.cfg.risk.risk_per_trade_pct / 100)
        pip_size     = self.get_pip_size(symbol)
        pip_value    = sym_info.trade_contract_size * pip_size  # per lot

        lot = risk_amount / (sl_pips * pip_value)
        lot = max(sym_info.volume_min,
                  min(round(lot / sym_info.volume_step) * sym_info.volume_step,
                      sym_info.volume_max))
        return lot

    def place_market_order(self,
                           symbol:     str,
                           direction:  int,         # +1 = BUY, -1 = SELL
                           sl_price:   float = 0.0,
                           tp_price:   float = 0.0,
                           sl_pips:    Optional[float] = None,
                           tp_pips:    Optional[float] = None,
                           comment:    str = "") -> Optional[object]:
        """
        Place a market order.

        Args:
            symbol    : e.g. "EURUSD"
            direction : +1 for BUY, -1 for SELL
            sl_price  : explicit SL price (overrides sl_pips)
            tp_price  : explicit TP price (overrides tp_pips)
            sl_pips   : SL distance in pips (used if sl_price == 0)
            tp_pips   : TP distance in pips (used if tp_price == 0)
        """
        if not self.exec_cfg.live_trading:
            action = "BUY" if direction == 1 else "SELL"
            print(f"[SIM] {action} {symbol}  SL={sl_price:.5f}  TP={tp_price:.5f}")
            return None

        tick     = self.get_current_price(symbol)
        pip_size = self.get_pip_size(symbol)

        price = tick["ask"] if direction == 1 else tick["bid"]
        order_type = mt5.ORDER_TYPE_BUY if direction == 1 else mt5.ORDER_TYPE_SELL

        # Resolve SL / TP
        if sl_price == 0.0 and sl_pips:
            sl_pips = sl_pips or self.cfg.risk.stop_loss_pips
            sl_price = price - direction * sl_pips * pip_size
        if tp_price == 0.0 and tp_pips:
            tp_pips = tp_pips or self.cfg.risk.take_profit_pips
            tp_price = price + direction * tp_pips * pip_size

        # Lot size
        effective_sl_pips = abs(price - sl_price) / pip_size if sl_price else \
                            self.cfg.risk.stop_loss_pips
        lot = self._calculate_lot_size(symbol, effective_sl_pips)

        request = {
            "action":    mt5.TRADE_ACTION_DEAL,
            "symbol":    symbol,
            "volume":    lot,
            "type":      order_type,
            "price":     price,
            "sl":        round(sl_price, self.get_symbol_info(symbol).digits),
            "tp":        round(tp_price, self.get_symbol_info(symbol).digits),
            "deviation": self.exec_cfg.slippage,
            "magic":     self.exec_cfg.magic_number,
            "comment":   comment or self.exec_cfg.order_comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"[MT5] Order failed  {symbol}: "
                  f"retcode={result.retcode} | {result.comment}")
            return None

        action = "BUY" if direction == 1 else "SELL"
        print(f"[MT5] {action} {symbol}  lot={lot}  "
              f"price={result.price:.5f}  SL={sl_price:.5f}  TP={tp_price:.5f}  "
              f"ticket=#{result.order}")
        return result

    def close_position(self, position) -> bool:
        """Close an existing open position by ticket."""
        if not self.exec_cfg.live_trading:
            print(f"[SIM] CLOSE position #{position.ticket}  {position.symbol}")
            return True

        tick  = self.get_current_price(position.symbol)
        price = tick["bid"] if position.type == 0 else tick["ask"]   # 0=BUY, 1=SELL
        close_type = mt5.ORDER_TYPE_SELL if position.type == 0 else mt5.ORDER_TYPE_BUY

        request = {
            "action":    mt5.TRADE_ACTION_DEAL,
            "symbol":    position.symbol,
            "volume":    position.volume,
            "type":      close_type,
            "position":  position.ticket,
            "price":     price,
            "deviation": self.exec_cfg.slippage,
            "magic":     self.exec_cfg.magic_number,
            "comment":   f"CLOSE #{position.ticket}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"[MT5] Close failed #{position.ticket}: "
                  f"retcode={result.retcode}")
            return False

        print(f"[MT5] Closed #{position.ticket}  {position.symbol}  "
              f"P&L={position.profit:.2f}")
        return True

    def close_all_positions(self, symbol: Optional[str] = None) -> None:
        """Emergency close — close all open positions."""
        for pos in self.get_open_positions(
            symbol=symbol,
            magic_number=self.exec_cfg.magic_number
        ):
            self.close_position(pos)

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.disconnect()
