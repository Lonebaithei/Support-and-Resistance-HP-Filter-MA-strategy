"""
==============================================================================
backtest.py  —  Standalone Backtester (No MT5 Required)
==============================================================================
Validates the combined HP Filter MA + S&R strategy on synthetic data.

Usage:
    python backtest.py
    python backtest.py --t1 3 --t2 12 --lam 14400
==============================================================================
"""

import sys, argparse
import numpy  as np
import pandas as pd
import matplotlib.pyplot   as plt
import matplotlib.dates    as mdates
import matplotlib.gridspec as gridspec

sys.path.insert(0, ".")
from strategy.indicators import (
    compute_hp_signal, compute_pivot_levels, compute_sr_signal,
)


# =============================================================================
# DATA
# =============================================================================

def generate_data(n: int = 240, seed: int = 42):
    rng = np.random.default_rng(seed)
    dates   = pd.date_range("2005-01-01", periods=n, freq="MS")
    closes  = np.empty(n);  closes[0] = 1.20
    for t in range(1, n):
        shock    = rng.choice([0, 0.02, -0.02], p=[0.96, 0.02, 0.02])
        closes[t] = closes[t-1] * np.exp(0.0005 + 0.022*rng.standard_normal() + shock)
    rng2   = np.random.default_rng(seed+1)
    rng_hi = np.abs(rng2.normal(0.018, 0.006, n))
    highs  = closes * (1 + rng_hi * rng2.uniform(0.4, 0.7, n))
    lows   = closes * (1 - rng_hi * rng2.uniform(0.3, 0.6, n))
    df = pd.DataFrame({"close": closes, "high": highs, "low": lows}, index=dates)
    return dates, closes, df


# =============================================================================
# BACKTEST
# =============================================================================

def run_backtest(dates, closes, ohlc_df, lam=14400, t1=3, t2=12,
                 pip_size=0.0001, tc_bps=2.0):
    tc      = tc_bps / 10_000
    warmup  = t2 + 2
    records = []

    for i in range(warmup, len(closes)):
        # Layer 1 & 2: HP trend + MA signal
        hp    = compute_hp_signal(closes[:i+1], lam=lam, t1=t1, t2=t2)
        # Layer 3: S&R pivot from previous bar
        prev  = ohlc_df.iloc[i-1]
        pivot = compute_pivot_levels(float(prev.high), float(prev.low),
                                     float(prev.close))
        sr    = compute_sr_signal(closes[i], pivot, pip_size=pip_size)

        # Combined: require HP and SR to agree
        if hp.signal == 1 and sr.direction == 1:
            desired = 1
        elif hp.signal == -1 and sr.direction == -1:
            desired = -1
        else:
            desired = 0

        records.append({
            "date":      dates[i],
            "price":     closes[i],
            "hp_signal": hp.signal,
            "sr_dir":    sr.direction,
            "desired":   desired,
            "hp_trend":  hp.trend[-1],
            "ma_short":  hp.ma_short[-1],
            "ma_long":   hp.ma_long[-1],
            "pivot":     pivot.pivot,
            "r1":        pivot.r1,
            "s1":        pivot.s1,
        })

    df  = pd.DataFrame(records).set_index("date")
    px  = df["price"].values
    des = df["desired"].values

    pos, ep          = 0, 0.0
    positions, raw_r, strat_r, actions_out = [], [], [], []

    for i in range(len(px)):
        raw_ret   = (px[i]/px[i-1] - 1) if i > 0 else 0.0
        strat_ret = 0.0
        act       = "HOLD"

        target = des[i]

        # Exit if signal flipped or neutralised
        if pos == 1 and target != 1:
            strat_ret = raw_ret - tc
            act = "EXIT_LONG";  pos = 0; ep = 0.0
        elif pos == -1 and target != -1:
            strat_ret = -raw_ret - tc
            act = "EXIT_SHORT"; pos = 0; ep = 0.0

        # Enter if no position and signal present
        if pos == 0 and target != 0:
            pos = target; ep = px[i]
            strat_ret -= tc
            act = "BUY" if target == 1 else "SELL"

        # Carry
        if act == "HOLD":
            if pos == 1:  strat_ret = raw_ret
            elif pos ==-1: strat_ret = -raw_ret

        positions.append(pos)
        raw_r.append(raw_ret)
        strat_r.append(strat_ret)
        actions_out.append(act)

    df["position"]        = positions
    df["action"]          = actions_out
    df["raw_return"]      = raw_r
    df["strategy_return"] = strat_r
    df["cum_bh"]          = (1 + df["raw_return"]).cumprod()
    df["cum_strategy"]    = (1 + df["strategy_return"]).cumprod()
    return df


# =============================================================================
# METRICS
# =============================================================================

def print_metrics(df, ppy=12):
    def ann_ret(r): return (1+r.mean())**ppy - 1
    def ann_vol(r): return r.std()*np.sqrt(ppy)
    def sharpe(r):
        v = ann_vol(r); return ann_ret(r)/v if v else 0
    def mdd(r):
        c = (1+r).cumprod(); return ((c-c.cummax())/c.cummax()).min()

    s = df["strategy_return"]; b = df["raw_return"]
    entries = (df["action"]=="BUY").sum() + (df["action"]=="SELL").sum()
    exits   = (df["action"]=="EXIT_LONG").sum() + (df["action"]=="EXIT_SHORT").sum()

    print("\n" + "="*55)
    print("  COMBINED STRATEGY BACKTEST RESULTS")
    print("="*55)
    print(f"  {'Metric':<33} {'Strategy':>10} {'Buy&Hold':>10}")
    print("  " + "-"*53)
    print(f"  {'Annualised Return':<33} {ann_ret(s):>10.2%} {ann_ret(b):>10.2%}")
    print(f"  {'Annualised Volatility':<33} {ann_vol(s):>10.2%} {ann_vol(b):>10.2%}")
    print(f"  {'Sharpe Ratio':<33} {sharpe(s):>10.2f} {sharpe(b):>10.2f}")
    print(f"  {'Max Drawdown':<33} {mdd(s):>10.2%} {mdd(b):>10.2%}")
    print(f"  {'Total Return':<33} {df['cum_strategy'].iloc[-1]-1:>10.2%} "
          f"{df['cum_bh'].iloc[-1]-1:>10.2%}")
    print(f"\n  Entries: {entries}  |  Exits: {exits}")
    print("="*55 + "\n")


# =============================================================================
# PLOT
# =============================================================================

def plot_backtest(df, title="HP Filter + S&R Combined Strategy"):
    fig = plt.figure(figsize=(15, 13))
    gs  = gridspec.GridSpec(4, 1, height_ratios=[3,1.5,1,2], hspace=0.35)
    dates = df.index

    ax1 = fig.add_subplot(gs[0])
    ax1.plot(dates, df["price"],    color="#B0BEC5", lw=0.8, label="Price", alpha=0.9)
    ax1.plot(dates, df["hp_trend"], color="#2196F3", lw=1.8, label="HP Trend S*(t)")
    ax1.plot(dates, df["ma_short"], color="#FF9800", lw=1.2, linestyle="--", label="MA Short")
    ax1.plot(dates, df["ma_long"],  color="#E91E63", lw=1.2, linestyle="--", label="MA Long")
    ax1.plot(dates, df["pivot"],    color="#9C27B0", lw=0.8, linestyle=":",  label="Pivot C")
    ax1.plot(dates, df["r1"],       color="#4CAF50", lw=0.8, linestyle=":",  label="R1")
    ax1.plot(dates, df["s1"],       color="#F44336", lw=0.8, linestyle=":",  label="S1")

    buys  = df[df["action"]=="BUY"];   sells = df[df["action"]=="SELL"]
    el    = df[df["action"]=="EXIT_LONG"]; es = df[df["action"]=="EXIT_SHORT"]
    ax1.scatter(buys.index,  buys["price"],  marker="^", color="#4CAF50", s=90, zorder=5, label="Buy")
    ax1.scatter(sells.index, sells["price"], marker="v", color="#F44336", s=90, zorder=5, label="Sell")
    ax1.scatter(el.index,    el["price"],    marker="x", color="#FF9800", s=80, zorder=5, label="Exit L")
    ax1.scatter(es.index,    es["price"],    marker="x", color="#9C27B0", s=80, zorder=5, label="Exit S")
    ax1.set_title(title, fontsize=14, fontweight="bold", pad=10)
    ax1.set_ylabel("Price"); ax1.legend(loc="upper left", fontsize=7, ncol=4)
    ax1.grid(True, alpha=0.25)

    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.step(dates, df["hp_signal"], color="#2196F3", lw=1.2, label="HP Signal", where="post")
    ax2.step(dates, df["sr_dir"],    color="#FF9800", lw=1.2, label="S/R Dir",   where="post", linestyle="--")
    ax2.fill_between(dates, df["hp_signal"], 0, where=(df["hp_signal"]>0), color="#2196F3", alpha=0.12)
    ax2.fill_between(dates, df["hp_signal"], 0, where=(df["hp_signal"]<0), color="#F44336", alpha=0.12)
    ax2.axhline(0, color="black", lw=0.5, linestyle="--")
    ax2.set_ylabel("Signal"); ax2.set_yticks([-1,0,1])
    ax2.legend(loc="upper left", fontsize=8); ax2.grid(True, alpha=0.25)

    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.step(dates, df["position"], color="#673AB7", lw=1.2, where="post")
    ax3.fill_between(dates, df["position"], 0, where=(df["position"]>0),  color="#4CAF50", alpha=0.3, label="Long")
    ax3.fill_between(dates, df["position"], 0, where=(df["position"]<0),  color="#F44336", alpha=0.3, label="Short")
    ax3.set_ylabel("Position"); ax3.set_yticks([-1,0,1])
    ax3.legend(loc="upper left", fontsize=8); ax3.grid(True, alpha=0.25)

    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    ax4.plot(dates, df["cum_strategy"], color="#4CAF50", lw=2.0, label="Combined Strategy")
    ax4.plot(dates, df["cum_bh"],       color="#2196F3", lw=1.5, linestyle="--", label="Buy & Hold")
    ax4.axhline(1.0, color="black", lw=0.5, linestyle="--")
    ax4.fill_between(dates, df["cum_strategy"], 1, where=(df["cum_strategy"]>=1), color="#4CAF50", alpha=0.1)
    ax4.fill_between(dates, df["cum_strategy"], 1, where=(df["cum_strategy"]< 1), color="#F44336", alpha=0.1)
    ax4.set_ylabel("Cumulative Return"); ax4.set_xlabel("Date")
    ax4.legend(loc="upper left", fontsize=9); ax4.grid(True, alpha=0.25)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax4.xaxis.set_major_locator(mdates.YearLocator(2))

    plt.savefig("/mnt/user-data/outputs/backtest_combined_strategy.png", dpi=150, bbox_inches="tight")
    print("Chart saved → backtest_combined_strategy.png")
    plt.show()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="EURUSD")
    parser.add_argument("--t1",  type=int,   default=3)
    parser.add_argument("--t2",  type=int,   default=12)
    parser.add_argument("--lam", type=float, default=14_400)
    parser.add_argument("--tc",  type=float, default=2.0)
    args = parser.parse_args()

    print(f"\nBacktesting: {args.symbol}")
    print(f"HP: λ={args.lam:,.0f}  MA({args.t1}/{args.t2})  TC={args.tc}bps\n")

    dates, closes, ohlc_df = generate_data(n=240)
    df = run_backtest(dates, closes, ohlc_df,
                      lam=args.lam, t1=args.t1, t2=args.t2, tc_bps=args.tc)
    print_metrics(df)
    plot_backtest(df, title=f"HP Filter + S&R  |  {args.symbol}  |  "
                            f"λ={args.lam:,.0f}  MA({args.t1}/{args.t2})")
