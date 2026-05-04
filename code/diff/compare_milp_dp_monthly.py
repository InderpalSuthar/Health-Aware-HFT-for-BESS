"""
MILP vs DP head-to-head, extended to the new monthly synthetic order books.

This script generalises `comparison_strategies.py` (a single MILP-vs-DP run on
the toy `orderbook_feb_01_14.csv`) to all four monthly periods in
code/strategy_comparison_data/syn_data/.

For tractability, MILP is run on the FIRST 14 DAYS of each month (this matches
the policy already documented in the paper's Section 6 caption); DP-101 is run
on the same 14-day window so the comparison is apples-to-apples on identical
data.

Outputs (written to code/strategy_comparison_data/, the paper's graphicspath):
  - tab_milp_vs_dp.csv             : tidy table for inclusion in main.tex
  - milp_vs_dp_panels.png          : 3-panel figure (price+signals, SoC, time)
                                     for the first month, mirroring
                                     `comparison_strategies.py`'s
                                     `comparison_analysis.png` layout.

A combined per-month summary is also printed to stdout.
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
SYN_ROOT = os.path.join(CODE_DIR, "strategy_comparison_data", "syn_data")
PAPER_FIG_DIR = os.path.join(CODE_DIR, "strategy_comparison_data")

sys.path.insert(0, CODE_DIR)
from MILP import (
    BatteryParams,
    MarketParams,
    DegradationModel,
    OrderBookLoader,
    RollingIntrinsicBacktest,
)
from DP import DPRollingIntrinsicBacktest


# Limit MILP to the first 14 days for tractability.
MILP_DAYS = 14
INITIAL_SOC = 6.25
HORIZON_HOURS = 24  # paper uses T_max = 24h
DP_GRID_SIZE = 101

PERIODS = [
    ("feb", os.path.join(SYN_ROOT, "feb", "orderbook_feb_01_29.csv")),
    ("apr", os.path.join(SYN_ROOT, "apr", "orderbook_apr_01_30.csv")),
    ("jul", os.path.join(SYN_ROOT, "jul", "orderbook_jul_01_31.csv")),
    ("oct", os.path.join(SYN_ROOT, "oct", "orderbook_oct_01_31.csv")),
]


def load_first_n_days(csv_path, n_days):
    """Load the order book and trim to the first `n_days` of delivery."""
    loader = OrderBookLoader(csv_path)
    first_start = loader.data["start"].min()
    cutoff = first_start + pd.Timedelta(days=n_days)
    loader.data = loader.data[loader.data["start"] < cutoff].reset_index(drop=True)
    print(f"  Trimmed to first {n_days} days -> {len(loader.data):,} orders, "
          f"delivery {loader.data['start'].min()} to {loader.data['start'].max()}")
    return loader


def run_one_period(label, csv_path):
    print(f"\n{'='*72}\n[{label.upper()}]  loading {os.path.basename(csv_path)}\n{'='*72}")

    battery = BatteryParams()
    market = MarketParams()
    deg = DegradationModel(battery)

    loader = load_first_n_days(csv_path, MILP_DAYS)

    print(f"\n[1/2] MILP rolling-intrinsic, T_max = {HORIZON_HOURS}h ...")
    t0 = time.time()
    milp = RollingIntrinsicBacktest(battery, market, deg, phi=0.0)
    milp_res = milp.run(
        loader, initial_soc=INITIAL_SOC, max_horizon_hours=HORIZON_HOURS,
        update_frequency="hourly",
    )
    milp_wall = time.time() - t0
    print(f"  MILP done in {milp_wall:.1f}s — net €{milp_res['net_profit']:,.0f}, "
          f"deg €{milp_res['degradation_cost']:,.0f}")

    print(f"\n[2/2] DP-{DP_GRID_SIZE} rolling-intrinsic, T_max = {HORIZON_HOURS}h ...")
    t0 = time.time()
    dp = DPRollingIntrinsicBacktest(
        battery, market, deg, phi=0.0, dp_grid_size=DP_GRID_SIZE
    )
    dp_res = dp.run(
        loader, initial_soc=INITIAL_SOC, max_horizon_hours=HORIZON_HOURS,
        update_frequency="hourly",
    )
    dp_wall = time.time() - t0
    print(f"  DP done in {dp_wall:.1f}s — net €{dp_res['net_profit']:,.0f}, "
          f"deg €{dp_res['degradation_cost']:,.0f}")

    return {
        "label": label,
        "milp": milp_res,
        "dp": dp_res,
        "milp_wall_s": milp_wall,
        "dp_wall_s": dp_wall,
    }


def build_table(all_results):
    rows = []
    for r in all_results:
        m, d = r["milp"], r["dp"]
        rows.append(dict(
            period=r["label"].upper(),
            method="MILP",
            net_profit=round(m["net_profit"], 0),
            gross_revenue=round(m["gross_revenue"], 0),
            degradation_cost=round(m["degradation_cost"], 0),
            avg_solve_ms=round(m["avg_solve_time_ms"], 1),
            total_solve_s=round(m["total_solve_time_s"], 1),
            wall_clock_s=round(r["milp_wall_s"], 1),
        ))
        rows.append(dict(
            period=r["label"].upper(),
            method=f"DP-{DP_GRID_SIZE}",
            net_profit=round(d["net_profit"], 0),
            gross_revenue=round(d["gross_revenue"], 0),
            degradation_cost=round(d["degradation_cost"], 0),
            avg_solve_ms=round(d["avg_solve_time_ms"], 1),
            total_solve_s=round(d["total_solve_time_s"], 1),
            wall_clock_s=round(r["dp_wall_s"], 1),
        ))
        # Speedup row
        speedup = (
            m["avg_solve_time_ms"] / d["avg_solve_time_ms"]
            if d["avg_solve_time_ms"] > 0 else float("nan")
        )
        rows.append(dict(
            period=r["label"].upper(),
            method="speedup (MILP/DP)",
            net_profit=None,
            gross_revenue=None,
            degradation_cost=None,
            avg_solve_ms=round(speedup, 1),
            total_solve_s=None,
            wall_clock_s=None,
        ))
    return pd.DataFrame(rows)


def plot_panels(first_result, out_path):
    """3-panel plot mirroring comparison_strategies.py for the first period."""
    milp = first_result["milp"]
    dp = first_result["dp"]
    label = first_result["label"]

    # Timeline
    timeline = [x["time"] for x in milp["soc_history"]]

    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)

    # Panel 1: mid price + DP buy/sell signals
    if dp.get("price_history"):
        ph = dp["price_history"]
        ax_t = [p["time"] for p in ph]
        mids = [(p["best_bid"] + p["best_ask"]) / 2 for p in ph]
        axes[0].plot(ax_t, mids, color="gray", alpha=0.45, linewidth=0.9, label="Mid price")
    dp_trades = dp.get("trade_history", [])
    buy_t = [t["time"] for t in dp_trades if t["is_buy"]]
    buy_p = [t["price"] for t in dp_trades if t["is_buy"]]
    sell_t = [t["time"] for t in dp_trades if not t["is_buy"]]
    sell_p = [t["price"] for t in dp_trades if not t["is_buy"]]
    axes[0].scatter(buy_t, buy_p, marker="^", color="green", s=40, label="DP buy", zorder=5)
    axes[0].scatter(sell_t, sell_p, marker="v", color="red", s=40, label="DP sell", zorder=5)
    axes[0].set_ylabel("Price (€/MWh)")
    axes[0].set_title(f"{label.upper()} 14d — DP execution against mid price")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Panel 2: SoC overlay
    milp_soc = [x["soc"] for x in milp["soc_history"]]
    dp_soc = [x["soc"] for x in dp["soc_history"]]
    axes[1].plot(timeline, milp_soc, label="MILP", linewidth=1.3, alpha=0.85)
    axes[1].plot(timeline, dp_soc, label=f"DP-{DP_GRID_SIZE}",
                 linestyle="--", linewidth=1.3, color="orange")
    axes[1].set_ylabel("SoC (MWh)")
    axes[1].set_title("Inventory trajectory")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    # Panel 3: solve time, log scale
    if "solve_time_history" in milp and "solve_time_history" in dp:
        n = min(len(milp["solve_time_history"]),
                len(dp["solve_time_history"]),
                len(timeline))
        axes[2].plot(timeline[:n], milp["solve_time_history"][:n],
                     label="MILP", alpha=0.75)
        axes[2].plot(timeline[:n], dp["solve_time_history"][:n],
                     label=f"DP-{DP_GRID_SIZE}", alpha=0.85)
        axes[2].set_yscale("log")
        axes[2].set_ylabel("Per-step solve time (s, log)")
        axes[2].set_title("Computational efficiency")
        axes[2].legend(fontsize=9)
        axes[2].grid(True, alpha=0.3)
    axes[2].set_xlabel("Time")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")


def main():
    print("=" * 72)
    print("MILP vs DP — extending comparison_strategies.py to monthly data")
    print("=" * 72)
    print(f"  MILP window: first {MILP_DAYS} days  |  DP grid: {DP_GRID_SIZE}  |  T_max: {HORIZON_HOURS}h")

    all_results = []
    for label, csv in PERIODS:
        if not os.path.exists(csv):
            print(f"\n  WARNING: missing {csv}, skipping")
            continue
        all_results.append(run_one_period(label, csv))

    if not all_results:
        print("No periods ran; aborting.")
        return

    table = build_table(all_results)
    csv_out = os.path.join(PAPER_FIG_DIR, "tab_milp_vs_dp.csv")
    table.to_csv(csv_out, index=False)
    print(f"\n[saved] {csv_out}\n")
    print(table.to_string(index=False))

    fig_out = os.path.join(PAPER_FIG_DIR, "milp_vs_dp_panels.png")
    plot_panels(all_results[0], fig_out)


if __name__ == "__main__":
    main()
