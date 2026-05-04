"""
Run bitepy on each of the four monthly synthetic order books in
code/strategy_comparison_data/syn_data/ and re-evaluate every discharge
under our convex piecewise-linear model from code/Cdeg_calc.py.

For each month, produce:
  - cumulative-cost time series under both models
  - per-month summary row (gross reward, linear deg, convex deg, ratios)

A combined plot (2x2 subplots, one per month) and a tidy CSV table are
written to code/diff/output_monthly/.

Run:
    /Users/inderpal/Documents/NGU_Project/bitepy_venv/bin/python \
        code/diff/compare_models_monthly.py
"""

import os
import shutil
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
SYN_ROOT = os.path.join(CODE_DIR, "strategy_comparison_data", "syn_data")
OUTPUT_DIR = os.path.join(THIS_DIR, "output_monthly")
# The paper LaTeX `\graphicspath` resolves figures from this folder, so we
# also write the three referenced figures (model_curves.png, cumulative_gap.png,
# trajectory_oct.png) into strategy_comparison_data/.
PAPER_FIG_DIR = os.path.join(CODE_DIR, "strategy_comparison_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

sys.path.insert(0, CODE_DIR)
from Cdeg_calc import DegradationCostCalculator


# ---------------------------------------------------------------------------
# Battery / market constants — matched to the paper's defaults.
# ---------------------------------------------------------------------------
STORAGE_MAX = 12.5         # MWh
SOC_MIN_FRAC = 0.15
SOC_MAX_FRAC = 0.95
ETA_IN = 0.95
ETA_OUT = 0.95
INJECT_MAX = 5.0           # MW
WITHDRAW_MAX = 5.0         # MW
LIN_DEG_COST = 4.0         # €/MWh (bitepy default)
INITIAL_SOC = 6.25         # MWh
CONVEX_SEGMENTS = 16
REPLACEMENT_COST = 300_000.0  # €/MWh

# Maps short month label to (year, month, last_day_for_simulation).
# October is trimmed to before the 2024 DST fall-back (Oct 27) — the bitepy
# C++ engine segfaults across that boundary in our environment.
PERIODS = {
    "feb": (2024, 2, 29),
    "apr": (2024, 4, 30),
    "jul": (2024, 7, 31),
    "oct": (2024, 10, 26),
}


def _zip_dir(period):
    return os.path.join(SYN_ROOT, period, "daily_zips")


def _bin_dir(period):
    return os.path.join(OUTPUT_DIR, "bins", period)


def run_bitepy_for_period(period, year, month, last_day):
    """Run bitepy on one month and return a DataFrame of executed orders."""
    import bitepy as bp

    zip_dir = _zip_dir(period)
    bin_dir = _bin_dir(period)
    if os.path.isdir(bin_dir):
        shutil.rmtree(bin_dir)
    os.makedirs(bin_dir, exist_ok=True)

    csv_files = sorted(
        os.path.join(zip_dir, f)
        for f in os.listdir(zip_dir)
        if f.endswith(".csv.zip")
    )

    parser = bp.Data()
    parser.create_bins_from_csv(csv_files, bin_dir, verbose=False)

    # DST handling: bitepy reads the day before sim_start in Berlin time.
    # On a DST-transition month (Mar/Oct in Europe), the Berlin-minus-1-day
    # of the start instant lands TWO calendar days back, so we synthesise an
    # empty placeholder by copying the existing prefix-1 day's binary.
    bins = sorted(f for f in os.listdir(bin_dir) if f.endswith(".bin"))
    if bins:
        first_bin_date = bins[0].replace("orderbook_", "").replace(".bin", "")
        prefix2_date = (
            pd.Timestamp(first_bin_date) - pd.Timedelta(days=1)
        ).strftime("%Y-%m-%d")
        prefix2_path = os.path.join(bin_dir, f"orderbook_{prefix2_date}.bin")
        if not os.path.exists(prefix2_path):
            shutil.copy(os.path.join(bin_dir, bins[0]), prefix2_path)

    sim_start = pd.Timestamp(year=year, month=month, day=1, tz="Europe/Berlin")
    # End on the last day of the month at 23:00 — going to next-day midnight
    # would make bitepy look for a binary file that doesn't exist.
    sim_end = pd.Timestamp(
        year=year, month=month, day=last_day, hour=23, minute=0, tz="Europe/Berlin"
    )
    sim = bp.Simulation(
        start_date=sim_start,
        end_date=sim_end,
        storage_max=STORAGE_MAX,
        lin_deg_cost=LIN_DEG_COST,
        loss_in=ETA_IN,
        loss_out=ETA_OUT,
        inject_max=INJECT_MAX,
        withdraw_max=WITHDRAW_MAX,
        log_transactions=False,
    )
    sim.run(bin_dir, verbose=False)
    logs = sim.get_logs()
    return logs["executed_orders"], sim_start, sim_end


def replay_with_convex(executed_df):
    """Reconstruct SoC from bitepy's executed orders and re-cost discharges."""
    convex_calc = DegradationCostCalculator(
        energy_capacity=STORAGE_MAX,
        num_segments=CONVEX_SEGMENTS,
        replacement_cost=REPLACEMENT_COST,
        eta_discharge=ETA_OUT,
    )
    energy_min = SOC_MIN_FRAC * STORAGE_MAX
    energy_max = SOC_MAX_FRAC * STORAGE_MAX

    df = executed_df.sort_values("time").reset_index(drop=True)
    soc = INITIAL_SOC
    rows = []
    cum_lin = 0.0
    cum_cvx = 0.0
    for _, r in df.iterrows():
        vol = float(r["volume"])
        otype = str(r.get("type", "")).lower()
        if "sell" in otype:
            internal = min(vol / ETA_OUT, max(0.0, soc - energy_min))
            cl = LIN_DEG_COST * internal
            cc = convex_calc.compute_cost(soc, internal)
            cum_lin += cl
            cum_cvx += cc
            soc -= internal
            rows.append(
                dict(time=r["time"], side="discharge", internal=internal,
                     soc=soc, cum_linear=cum_lin, cum_convex=cum_cvx)
            )
        elif "buy" in otype:
            added = min(vol * ETA_IN, max(0.0, energy_max - soc))
            soc += added
            rows.append(
                dict(time=r["time"], side="charge", internal=added,
                     soc=soc, cum_linear=cum_lin, cum_convex=cum_cvx)
            )
    return pd.DataFrame(rows), cum_lin, cum_cvx


def main():
    print("=" * 72)
    print("Monthly bitepy engine + convex re-evaluation")
    print("=" * 72)

    summary_rows = []
    per_month_traces = {}

    for period, (year, month, last_day) in PERIODS.items():
        print(f"\n[{period}] {year}-{month:02d} (running bitepy ...)")
        executed, sim_start, sim_end = run_bitepy_for_period(
            period, year, month, last_day
        )
        if executed.empty:
            print("  (bitepy produced no trades — check data density)")
            continue

        gross = float(executed["reward"].sum())
        bitepy_net = float(executed["reward_incl_deg_costs"].sum())
        bitepy_implied_deg = gross - bitepy_net

        rebuilt, cum_lin, cum_cvx = replay_with_convex(executed)
        n_disch = int((rebuilt["side"] == "discharge").sum())
        total_disch = float(
            rebuilt.loc[rebuilt["side"] == "discharge", "internal"].sum()
        )

        summary_rows.append(
            dict(
                month=period,
                executed_orders=len(executed),
                discharges=n_disch,
                total_discharge_MWh=round(total_disch, 2),
                gross_reward=round(gross, 2),
                bitepy_implied_deg=round(bitepy_implied_deg, 2),
                discharge_only_linear=round(cum_lin, 2),
                discharge_only_convex=round(cum_cvx, 2),
                convex_minus_linear=round(cum_cvx - cum_lin, 2),
                ratio_convex_over_linear=(
                    round(cum_cvx / cum_lin, 3) if cum_lin > 0 else float("nan")
                ),
                net_profit_linear=round(gross - cum_lin, 2),
                net_profit_convex=round(gross - cum_cvx, 2),
            )
        )
        per_month_traces[period] = rebuilt
        print(
            f"  trades={len(executed)}  discharges={n_disch}  "
            f"gross=€{gross:,.0f}  linear=€{cum_lin:,.0f}  convex=€{cum_cvx:,.0f}  "
            f"ratio={cum_cvx / cum_lin:.3f}" if cum_lin > 0 else f"  trades={len(executed)}"
        )

    if not summary_rows:
        print("No data to summarise. Exiting.")
        return

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(OUTPUT_DIR, "monthly_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n[saved] {summary_path}")
    print("\n" + summary_df.to_string(index=False))

    # ---- 2x2 plot: cumulative degradation under both models, one panel per month
    n = len(per_month_traces)
    if n == 0:
        return
    rows = 2 if n > 2 else 1
    cols = 2 if n > 1 else 1
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4 * rows), sharex=False)
    axes = np.atleast_1d(axes).ravel()
    for ax, (period, trace) in zip(axes, per_month_traces.items()):
        ax.plot(trace["time"], trace["cum_convex"], label="Convex (paper)", linewidth=1.6)
        ax.plot(
            trace["time"],
            trace["cum_linear"],
            linestyle="--",
            label=f"Linear bitepy ({LIN_DEG_COST} €/MWh)",
            linewidth=1.6,
        )
        ax.set_title(f"{period.upper()} 2024")
        ax.set_ylabel("Cumulative deg. cost (€)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(
        "Discharge degradation cost: bitepy linear vs paper convex (per month)",
        y=1.02,
    )
    fig.tight_layout()
    cum_path = os.path.join(OUTPUT_DIR, "monthly_cumulative.png")
    fig.savefig(cum_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {cum_path}")

    # ---- Bar chart: per-month cost under each model
    fig, ax = plt.subplots(figsize=(10, 5))
    months = summary_df["month"].tolist()
    x = np.arange(len(months))
    w = 0.38
    ax.bar(x - w / 2, summary_df["discharge_only_linear"], w, label="Linear bitepy")
    ax.bar(x + w / 2, summary_df["discharge_only_convex"], w, label="Convex (paper)")
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in months])
    ax.set_ylabel("Cumulative discharge degradation cost (€)")
    ax.set_title("Linear vs convex degradation cost on bitepy's trade sequence")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    bar_path = os.path.join(OUTPUT_DIR, "monthly_bar.png")
    fig.savefig(bar_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {bar_path}")

    # =====================================================================
    # Three figures consumed by paper/main.tex (filenames are referenced
    # by `\includegraphics{...}` and resolved through `\graphicspath`).
    # =====================================================================
    convex_calc = DegradationCostCalculator(
        energy_capacity=STORAGE_MAX,
        num_segments=CONVEX_SEGMENTS,
        replacement_cost=REPLACEMENT_COST,
        eta_discharge=ETA_OUT,
    )

    # --- model_curves.png : static cost-curve & marginal-cost comparison ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    starting_socs = [
        0.25 * STORAGE_MAX, 0.50 * STORAGE_MAX,
        0.75 * STORAGE_MAX, 1.00 * STORAGE_MAX,
    ]
    for s0 in starting_socs:
        Ws = np.linspace(0, max(0.01, s0), 60)
        cvx = [convex_calc.compute_cost(s0, w) for w in Ws]
        lin = [LIN_DEG_COST * w for w in Ws]
        axes[0].plot(Ws, cvx, linewidth=1.8, label=f"Convex (SoC={s0:.1f})")
        axes[0].plot(Ws, lin, linewidth=1.4, linestyle="--", alpha=0.6,
                     label=f"Linear (SoC={s0:.1f})")
    axes[0].set_xlabel("Discharge volume W (MWh)")
    axes[0].set_ylabel("Degradation cost (€)")
    axes[0].set_title("Total cost vs discharge volume")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    s_grid = np.linspace(0.05 * STORAGE_MAX, STORAGE_MAX, 80)
    probe = 0.1
    cvx_marg = [convex_calc.compute_cost(s, probe) / probe for s in s_grid]
    lin_marg = [LIN_DEG_COST for _ in s_grid]
    axes[1].plot(s_grid, cvx_marg, linewidth=1.8, label="Convex (paper)")
    axes[1].plot(s_grid, lin_marg, linewidth=1.6, linestyle="--",
                 label=f"Linear bitepy ({LIN_DEG_COST} €/MWh)")
    axes[1].set_xlabel("State of charge (MWh)")
    axes[1].set_ylabel("Marginal cost (€/MWh)")
    axes[1].set_title("Marginal cost of a 0.1 MWh discharge probe")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    mc_path = os.path.join(PAPER_FIG_DIR, "model_curves.png")
    fig.savefig(mc_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {mc_path}")

    # --- cumulative_gap.png : cumulative (convex - linear) over time, all months ---
    fig, ax = plt.subplots(figsize=(12, 5))
    for period, trace in per_month_traces.items():
        gap = trace["cum_convex"] - trace["cum_linear"]
        ax.plot(trace["time"], gap, linewidth=1.5, label=f"{period.upper()} 2024")
    ax.set_xlabel("Time")
    ax.set_ylabel("Cumulative cost gap, convex – linear (€)")
    ax.set_title("Cumulative degradation-cost gap over time, per month")
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    gap_path = os.path.join(PAPER_FIG_DIR, "cumulative_gap.png")
    fig.savefig(gap_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {gap_path}")

    # --- trajectory_oct.png : SoC and cumulative cost on Oct trace ---
    if "oct" in per_month_traces:
        oct_trace = per_month_traces["oct"]
        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        e_min = SOC_MIN_FRAC * STORAGE_MAX
        e_max = SOC_MAX_FRAC * STORAGE_MAX
        axes[0].plot(oct_trace["time"], oct_trace["soc"],
                     color="steelblue", linewidth=1.2)
        axes[0].axhline(e_min, color="red", linestyle=":", alpha=0.5)
        axes[0].axhline(e_max, color="red", linestyle=":", alpha=0.5)
        axes[0].set_ylabel("SoC (MWh)")
        axes[0].set_title("Oct 2024 SoC trajectory under bitepy")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(oct_trace["time"], oct_trace["cum_convex"],
                     linewidth=1.8, label="Convex (paper)")
        axes[1].plot(oct_trace["time"], oct_trace["cum_linear"],
                     linewidth=1.6, linestyle="--",
                     label=f"Linear bitepy ({LIN_DEG_COST} €/MWh)")
        axes[1].set_xlabel("Time")
        axes[1].set_ylabel("Cumulative degradation cost (€)")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        fig.tight_layout()
        oct_path = os.path.join(PAPER_FIG_DIR, "trajectory_oct.png")
        fig.savefig(oct_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[saved] {oct_path}")


if __name__ == "__main__":
    main()
