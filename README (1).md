# 📈 FX-HP-SR-Bot

> A quantitative FX trading bot combining the **Hodrick-Prescott Filter Moving Average** strategy with **Support & Resistance Pivot Levels**, integrated with **MetaTrader 5** via Python.

Based on *151 Trading Strategies* — Kakushadze & Serur (2018):
- **Chapter 8.1** — Moving Averages with HP Filter
- **Chapter 3.14** — Support and Resistance

---

## 🧠 Strategy Overview

Most moving average systems fail in FX because raw spot rate series are too noisy — they generate false crossover signals constantly. This bot solves that problem at the source by running the **Hodrick-Prescott (HP) filter** over the raw price series first, extracting the true underlying trend before any moving average is applied.

That denoised trend then feeds into a **three-layer confluence framework** that combines macro trend direction, momentum confirmation, and precise intraday entry/exit timing — all three must agree before a trade is placed.

```
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 1 — REGIME       HP Trend direction (monthly)            │
│                          Rising  → only longs considered        │
│                          Falling → only shorts considered       │
│                                 ↓                               │
│  LAYER 2 — DIRECTION    MA(T1) vs MA(T2) crossover on trend     │
│                          MA short > MA long → bullish confirmed  │
│                          MA short < MA long → bearish confirmed  │
│                                 ↓                               │
│  LAYER 3 — TIMING       Daily Pivot Point S&R levels            │
│                          Price > Pivot C  → long entry signal   │
│                          Price ≥ R1       → long exit signal    │
│                          Price < Pivot C  → short entry signal  │
│                          Price ≤ S1       → short exit signal   │
└─────────────────────────────────────────────────────────────────┘
```

| Signal | Condition |
|--------|-----------|
| 🟢 **BUY** | HP trend rising **AND** MA(T1) > MA(T2) **AND** price above daily pivot |
| 🔴 **SELL** | HP trend falling **AND** MA(T1) < MA(T2) **AND** price below daily pivot |
| ⬜ **EXIT** | Price reaches R1 (long exit) or S1 (short exit) |

---

## 🔢 The Mathematics

### HP Filter (Equations 437–439)
The filter decomposes the raw FX spot rate **S(t)** into a smooth trend **S*(t)** and a noise component **ν(t)**:

```
S(t) = S*(t) + ν(t)

Minimise: g = Σ[S(t) − S*(t)]²  +  λ × Σ[S*(t+1) − 2S*(t) + S*(t−1)]²

λ = 100 × n²   where n = observations per year
              Monthly data (default): λ = 100 × 12² = 14,400
```

### Pivot Point Levels (Equations 325–327)
Calculated each day from the **previous day's** high, low, and close:

```
C  = (PH + PL + PC) / 3       ← Pivot centre
R1 = 2×C − PL                 ← Resistance (exit for longs)
S1 = 2×C − PH                 ← Support    (exit for shorts)
```

---

## 🏗️ Project Structure

```
fx-hp-sr-bot/
│
├── config.py                    ← All parameters — edit this first
├── bot.py                       ← Main trading loop
├── backtest.py                  ← Standalone backtester (no MT5 needed)
├── requirements.txt
├── README.md
│
├── core/
│   └── mt5_connector.py         ← MT5 connection, OHLCV data, order execution
│
├── strategy/
│   ├── indicators.py            ← HP filter, moving averages, pivot levels, S&R signals
│   └── combined_strategy.py     ← Three-layer confluence signal engine
│
├── execution/
│   └── risk_manager.py          ← Position sizing, drawdown limits, trade guards
│
└── utils/
    └── logger.py                ← Structured file + console logging
```

---

## ⚙️ Installation

> **Windows only** — the MetaTrader5 Python package requires a Windows environment with MT5 terminal installed.

```bash
# Clone the repo
git clone https://github.com/your-username/fx-hp-sr-bot.git
cd fx-hp-sr-bot

# Install dependencies
pip install -r requirements.txt
```

**Requirements:**
- Python 3.10+
- MetaTrader 5 terminal (running)
- Windows OS

---

## 🚀 Quick Start

### Step 1 — Configure your MT5 account
Open `config.py` and fill in your broker credentials:

```python
mt5 = MT5Config(
    login    = 123456,
    password = "your_password",
    server   = "YourBroker-Live",
    path     = r"C:\Program Files\...\terminal64.exe"  # Optional
)
```

### Step 2 — Choose symbols and timeframes
```python
instrument = InstrumentConfig(
    symbols      = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"],
    hp_timeframe = "MN1",    # Monthly bars for HP filter (paper default)
    sr_timeframe = "D1",     # Daily bars for pivot point calculation
    hp_lookback  = 120,      # 10 years of monthly history
)
```

### Step 3 — Set risk parameters
```python
risk = RiskConfig(
    risk_per_trade_pct = 1.0,    # % of account balance risked per trade
    max_open_trades    = 4,      # Maximum concurrent positions
    max_daily_loss_pct = 3.0,    # Daily drawdown circuit breaker
    use_sr_as_sl       = True,   # Use S1/R1 as dynamic stop-loss levels
)
```

### Step 4 — Run

```bash
# Backtest on synthetic data (no MT5 required)
python backtest.py

# Dry run — connects to MT5, evaluates signals, places NO real orders
python bot.py --dry-run

# Single evaluation pass (useful for task schedulers)
python bot.py --dry-run --once

# Live trading (set live_trading=True in config.py first)
python bot.py
```

---

## 📊 Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lambda_monthly` | `14,400` | HP smoothing parameter for monthly data |
| `ma_short` (T1) | `3` | Short moving average window (months) |
| `ma_long` (T2) | `12` | Long moving average window (months) |
| `sr_buffer_pips` | `2.0` | Tolerance band around S/R levels in pips |
| `use_sr_as_sl` | `True` | Use S1/R1 as stop-loss instead of fixed pips |
| `risk_per_trade_pct` | `1.0` | Account percentage risked per trade |
| `max_open_trades` | `4` | Maximum simultaneous open positions |
| `max_daily_loss_pct` | `3.0` | Daily loss % that triggers emergency close-all |
| `poll_interval_s` | `60` | Seconds between signal evaluation cycles |

### HP Filter λ by Timeframe

| Timeframe | Formula | λ |
|-----------|---------|---|
| Monthly `MN1` | 100 × 12² | 14,400 |
| Weekly `W1` | 100 × 52² | 270,400 |
| Daily `D1` | 100 × 252² | 6,350,400 |

---

## 🛡️ Risk Management

The bot includes four built-in guards that run on every evaluation cycle:

- **Position limit** — will not open a new trade if `max_open_trades` is already reached
- **Duplicate prevention** — will not open a second position on a symbol already held
- **Daily loss circuit breaker** — closes all positions and halts trading if daily drawdown exceeds `max_daily_loss_pct`
- **R:R filter** — rejects any trade where the risk-to-reward ratio falls below 1.5:1

---

## 📚 References

- Kakushadze, Z. & Serur, J.A. (2018). *151 Trading Strategies*. Palgrave Macmillan. [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3247865)
- Hodrick, R.J. & Prescott, E.C. (1997). Postwar U.S. Business Cycles. *Journal of Money, Credit and Banking*, 29(1): 1–16.
- Harris, R. & Yilmaz, F. (2009). A Momentum Trading Strategy Based on the Low Frequency Component of the Exchange Rate. *Journal of Banking & Finance*.

---

## ⚠️ Disclaimer

This project is intended for **educational and research purposes only**. It is not financial advice. Always conduct thorough backtesting and paper trading before deploying any automated strategy with real capital. Trading foreign exchange carries significant risk of loss. Past performance does not guarantee future results.

---

## 📄 License

MIT License — free to use, modify, and distribute with attribution.
