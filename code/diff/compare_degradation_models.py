"""
Compare the bitepy linear battery degradation model against our
piecewise-linear convex (cycle-depth-aware) model from
"Health-Aware High-Frequency Trading for BESS" on the same synthetic
order-book data.

Two things are compared:

1. Cost-curve comparison: for a given starting SoC, plot degradation cost
   as a function of discharge depth W under both models. The bitepy model
   is linear in |W|; our model is convex and SoC-dependent.

2. Trajectory-level comparison: replay a simple intrinsic-style strategy
   on the synthetic order book and apply BOTH degradation cost models to
   the resulting discharge events to show the dollar gap each accumulates.

Outputs go to code/diff/output/.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
PROJECT_DIR = os.path.abspath(os.path.join(CODE_DIR, ".."))
OUTPUT_DIR = os.path.join(THIS_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

sys.path.insert(0, CODE_DIR)
from Cdeg_calc import DegradationCostCalculator


# ---------------------------------------------------------------------------
# 1. Bitepy-style linear degradation cost
# ---------------------------------------------------------------------------
class LinearDegradationCost:
    """
    Mirrors bitepy's `lin_deg_cost` parameter:
    cost = lin_deg_cost * |discharge_energy_MWh|

    Bitepy default is 4.0 €/MWh (see bitepy.Simulation `lin_deg_cost`).
    """

    def __init__(self, lin_deg_cost: float = 4.0, eta_discharge: float = 0.95):
        self.lin_deg_cost = lin_deg_cost
        self.eta_dis = eta_discharge

    def compute_cost(self, s_t: float, W: float) -> float:
        # SoC is irrelevant — that's the whole point of the linear model.
        return self.lin_deg_cost * W


# ---------------------------------------------------------------------------
# 2. Cost-curve comparison (cost as a function of discharge depth W)
# ---------------------------------------------------------------------------
def plot_cost_curves(convex_calc, linear_calc, energy_capacity):
    """
    For a few starting SoCs, sweep W from 0 -> available energy and
    compare degradation cost under each model.
    """
    starting_socs = [
        0.25 * energy_capacity,
        0.50 * energy_capacity,
        0.75 * energy_capacity,
        1.00 * energy_capacity,
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left panel: total cost vs discharge volume
    for s0 in starting_socs:
        Ws = np.linspace(0, s0, 60)
        convex_costs = [convex_calc.compute_cost(s0, w) for w in Ws]
        linear_costs = [linear_calc.compute_cost(s0, w) for w in Ws]

        axes[0].plot(
            Ws,
            convex_costs,
            label=f"Convex (SoC={s0:.1f} MWh)",
            linewidth=1.8,
        )
        axes[0].plot(
            Ws,
            linear_costs,
            linestyle="--",
            alpha=0.6,
            label=f"Linear bitepy (SoC={s0:.1f} MWh)",
        )

    axes[0].set_xlabel("Discharge volume W (MWh)")
    axes[0].set_ylabel("Degradation cost (€)")
    axes[0].set_title("Cost vs discharge depth: convex vs linear")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Right panel: marginal cost (€/MWh) vs SoC at fixed small discharge
    # Highlights that the convex model penalises deep-cycle discharge.
    s_grid = np.linspace(0.05 * energy_capacity, energy_capacity, 80)
    delta_W = 0.1  # 0.1 MWh probe = bitepy lot size
    convex_marginal = [
        convex_calc.compute_cost(s, delta_W) / delta_W for s in s_grid
    ]
    linear_marginal = [
        linear_calc.compute_cost(s, delta_W) / delta_W for s in s_grid
    ]

    axes[1].plot(s_grid, convex_marginal, label="Convex (paper)", linewidth=1.8)
    axes[1].plot(
        s_grid,
        linear_marginal,
        linestyle="--",
        label=f"Linear bitepy ({linear_calc.lin_deg_cost} €/MWh)",
    )
    axes[1].set_xlabel("State of charge (MWh)")
    axes[1].set_ylabel("Marginal degradation cost (€/MWh)")
    axes[1].set_title(
        f"Marginal cost of a {delta_W:.2f} MWh discharge at varying SoC"
    )
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "cost_curves.png")
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[saved] {out_path}")


# ---------------------------------------------------------------------------
# 3. Trajectory-level comparison on the synthetic order book
# ---------------------------------------------------------------------------
def run_trajectory_comparison(
    csv_path,
    convex_calc,
    linear_calc,
    energy_capacity=12.5,
    energy_min=1.875,
    energy_max=11.875,
    eta_charge=0.95,
    eta_discharge=0.95,
    trading_fee=0.09,
    initial_soc=6.25,
):
    """
    Greedy intrinsic strategy on the synthetic LOB:
      - For each delivery hour the synthetic LOB has one BUY and one SELL.
      - Compute mid-spread; if BUY price - SELL price > 2 * trading_fee
        and we have headroom on both sides, do a unit charge then discharge.
      - For each discharge action, evaluate degradation cost under BOTH
        models and accumulate. The trades themselves are identical, only
        the cost models differ.
    """
    df = pd.read_csv(csv_path)
    df["start"] = pd.to_datetime(df["start"], utc=True)
    df["transaction"] = pd.to_datetime(df["transaction"], utc=True)

    # Best price per (delivery_hour, side): cheapest SELL (best ask we can lift),
    # highest BUY (best bid we can hit).
    sells = (
        df[df["side"] == "SELL"]
        .groupby("start")
        .agg(price=("price", "min"), quantity=("quantity", "sum"))
    )
    buys = (
        df[df["side"] == "BUY"]
        .groupby("start")
        .agg(price=("price", "max"), quantity=("quantity", "sum"))
    )

    # Build a unified per-hour view; some hours have only buys, some only sells.
    all_hours = sorted(set(sells.index).union(buys.index))

    soc = initial_soc
    revenue = 0.0
    convex_deg = 0.0
    linear_deg = 0.0
    soc_trace = [(all_hours[0], soc)]
    convex_trace = [0.0]
    linear_trace = [0.0]
    trades = []

    # bitepy's default per-action volume cap (5 MW * 1 h = 5 MWh).
    max_action = 5.0

    for h in all_hours:
        # SELL liquidity present at this hour => we can BUY (charge battery)
        if h in sells.index and soc < energy_max:
            sell_price = float(sells.loc[h, "price"])
            sell_qty = float(sells.loc[h, "quantity"])
            # Cheap-enough threshold: only charge when ask is materially below
            # the median observed bid (so we have a realistic resale upside).
            ask_threshold = float(buys["price"].median()) - 2 * trading_fee
            if sell_price < ask_threshold:
                max_charge = min(
                    (energy_max - soc) / eta_charge,
                    sell_qty,
                    max_action,
                )
                if max_charge > 1e-3:
                    cost = (sell_price + trading_fee) * max_charge
                    revenue -= cost
                    soc_before = soc
                    soc = min(soc + max_charge * eta_charge, energy_max)
                    trades.append(
                        dict(
                            time=h,
                            side="charge",
                            volume=max_charge,
                            price=sell_price,
                            soc_before=soc_before,
                            soc_after=soc,
                            cost_convex=0.0,
                            cost_linear=0.0,
                        )
                    )

        # BUY liquidity present at this hour => we can SELL (discharge battery)
        if h in buys.index and soc > energy_min:
            buy_price = float(buys.loc[h, "price"])
            buy_qty = float(buys.loc[h, "quantity"])
            max_discharge = min(soc - energy_min, buy_qty, max_action)
            if max_discharge > 1e-3:
                gross = (buy_price - trading_fee) * max_discharge * eta_discharge
                cost_convex = convex_calc.compute_cost(soc, max_discharge)
                cost_linear = linear_calc.compute_cost(soc, max_discharge)
                # Trigger on convex (more conservative). Same trade is then
                # also evaluated under the linear model.
                if gross - cost_convex > 0:
                    revenue += gross
                    convex_deg += cost_convex
                    linear_deg += cost_linear
                    soc_before = soc
                    soc -= max_discharge
                    trades.append(
                        dict(
                            time=h,
                            side="discharge",
                            volume=max_discharge,
                            price=buy_price,
                            soc_before=soc_before,
                            soc_after=soc,
                            cost_convex=cost_convex,
                            cost_linear=cost_linear,
                        )
                    )

        soc_trace.append((h, soc))
        convex_trace.append(convex_deg)
        linear_trace.append(linear_deg)

    summary = pd.DataFrame(
        {
            "Metric": [
                "Gross revenue (€)",
                "Convex deg cost (€)",
                "Linear bitepy deg cost (€)",
                "Net profit — convex (€)",
                "Net profit — linear (€)",
                "Total discharge volume (MWh)",
                "Number of trades",
            ],
            "Value": [
                revenue,
                convex_deg,
                linear_deg,
                revenue - convex_deg,
                revenue - linear_deg,
                sum(t["volume"] for t in trades if t["side"] == "discharge"),
                len(trades),
            ],
        }
    )

    print("\n=== Trajectory comparison ===")
    print(summary.to_string(index=False))
    summary.to_csv(os.path.join(OUTPUT_DIR, "trajectory_summary.csv"), index=False)

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df.to_csv(
            os.path.join(OUTPUT_DIR, "trades.csv"), index=False
        )

    # Plot cumulative degradation cost under both models
    times = [t for t, _ in soc_trace]
    socs = [s for _, s in soc_trace]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    axes[0].plot(times, socs, color="steelblue", linewidth=1.5, label="SoC")
    axes[0].axhline(energy_min, color="red", linestyle=":", alpha=0.5)
    axes[0].axhline(energy_max, color="red", linestyle=":", alpha=0.5)
    axes[0].set_ylabel("SoC (MWh)")
    axes[0].set_title("Shared SoC trajectory (same trades, different cost models)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(times, convex_trace, label="Convex (paper)", linewidth=1.8)
    axes[1].plot(
        times,
        linear_trace,
        label=f"Linear bitepy ({linear_calc.lin_deg_cost} €/MWh)",
        linewidth=1.8,
        linestyle="--",
    )
    axes[1].set_ylabel("Cumulative degradation cost (€)")
    axes[1].set_xlabel("Delivery hour")
    axes[1].set_title("Cumulative degradation cost: convex vs bitepy linear")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "trajectory_comparison.png")
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[saved] {out_path}")

    return summary


# ---------------------------------------------------------------------------
# 4. Optional: run the actual bitepy engine if it is importable.
# Bitepy expects EPEX-style continuous LOB data with id/validity columns,
# so this only runs if we can find such data; otherwise it's skipped.
# ---------------------------------------------------------------------------
def try_run_bitepy(csv_path):
    try:
        import bitepy  # noqa: F401
    except ImportError:
        print("\n[bitepy] not installed in this interpreter — skipping engine run.")
        print("        Install via:  pip install bitepy   (Python 3.13 wheel exists)")
        return

    df = pd.read_csv(csv_path)
    if "id" not in df.columns or df["validity"].isna().all():
        print(
            "\n[bitepy] synthetic order book lacks `id` and `validity` "
            "columns required by the engine — skipping engine run.\n"
            "        Use bitepy.Data on raw EPEX SFTP files to generate "
            "compatible binaries."
        )
        return

    print(
        "\n[bitepy] order book looks compatible. A full engine run is left as a "
        "follow-up; this script focuses on the degradation-cost comparison."
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    csv_path = os.path.join(
        CODE_DIR, "strategy_comparison_data", "orderbook_feb_01_14.csv"
    )
    if not os.path.exists(csv_path):
        print(f"ERROR: synthetic order book not found at {csv_path}")
        sys.exit(1)

    energy_capacity = 12.5
    convex_calc = DegradationCostCalculator(
        energy_capacity=energy_capacity,
        num_segments=16,
        replacement_cost=300_000.0,
        eta_discharge=0.95,
    )
    linear_calc = LinearDegradationCost(lin_deg_cost=4.0, eta_discharge=0.95)

    print("=" * 72)
    print("Comparing bitepy linear degradation vs paper convex degradation")
    print("=" * 72)
    print(f"Convex segment costs (€/MWh) by depth segment 1..J:")
    print("  ", np.round(convex_calc.c_j, 4).tolist())
    print(f"Linear bitepy cost: {linear_calc.lin_deg_cost} €/MWh (constant)")

    plot_cost_curves(convex_calc, linear_calc, energy_capacity)
    run_trajectory_comparison(csv_path, convex_calc, linear_calc)
    try_run_bitepy(csv_path)


if __name__ == "__main__":
    main()
