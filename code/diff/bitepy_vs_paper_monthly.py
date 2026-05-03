"""
Run the bitepy intraday-trading engine (linear degradation model) on each of
the four monthly synthetic order books we generated under
`code/strategy_comparison_data/syn_data/<period>/daily_zips/`, then re-cost
every executed trade under the paper's near-quadratic piecewise-linear
stress function.

The script produces, per period and combined:

  * `monthly_summary.csv`            — engine totals + recosted convex deg
  * `monthly_summary.tex`            — same table in LaTeX form
  * `cost_curves_<period>.png`       — convex vs linear cost vs discharge depth
  * `trajectory_<period>.png`        — SoC + cumulative cost (both models)
  * `cumulative_gap.png`             — €-gap (convex – linear) over time, all months
  * `model_curves.png`               — model-only comparison (no engine needed)

Run with:
    /Users/inderpal/Documents/NGU_Project/bitepy_venv/bin/python \
        code/diff/bitepy_vs_paper_monthly.py
"""

from __future__ import annotations

import os
import sys
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
PROJECT_DIR = os.path.abspath(os.path.join(CODE_DIR, ".."))
SYN_ROOT = os.path.join(CODE_DIR, "strategy_comparison_data", "syn_data")
OUTPUT_DIR = os.path.join(THIS_DIR, "monthly_engine_output")
BIN_ROOT = os.path.join(OUTPUT_DIR, "binaries")

sys.path.insert(0, CODE_DIR)
from Cdeg_calc import DegradationCostCalculator  # convex (paper) model

# ---------------------------------------------------------------------------
# Battery / market parameters – aligned with run_strategy_comparison.py
# ---------------------------------------------------------------------------
STORAGE_MAX = 12.5          # MWh (E_rate)
SOC_MIN = 1.875             # MWh
SOC_MAX = 11.875            # MWh
ETA_IN = 0.95
ETA_OUT = 0.95
INJECT_MAX = 5.0            # MW
WITHDRAW_MAX = 5.0          # MW
TRADING_FEE = 0.09          # €/MWh
INITIAL_SOC = 6.25
LIN_DEG_COST = 4.0          # €/MWh — bitepy default linear degradation cost
CONVEX_SEGMENTS = 16
REPLACEMENT_COST = 300_000.0

PERIODS = {
    "feb": "2024-02",
    "apr": "2024-04",
    "jul": "2024-07",
    "oct": "2024-10",
}


# ---------------------------------------------------------------------------
# 1. Bitepy linear degradation cost wrapper
# ---------------------------------------------------------------------------
class LinearDeg:
    def __init__(self, lin_deg_cost: float):
        self.c = lin_deg_cost

    def compute_cost(self, soc: float, w: float) -> float:  # noqa: ARG002
        return self.c * w


# ---------------------------------------------------------------------------
# 2. Pure-model curve plot (no engine needed)
# ---------------------------------------------------------------------------
def plot_model_curves(convex: DegradationCostCalculator, linear: LinearDeg, out_dir: str):
    socs = [0.25 * STORAGE_MAX, 0.50 * STORAGE_MAX, 0.75 * STORAGE_MAX, 1.00 * STORAGE_MAX]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for s0 in socs:
        Ws = np.linspace(0, s0, 80)
        axes[0].plot(Ws, [convex.compute_cost(s0, w) for w in Ws],
                     label=f"Convex SoC={s0:.1f} MWh", linewidth=1.6)
        axes[0].plot(Ws, [linear.compute_cost(s0, w) for w in Ws],
                     linestyle="--", alpha=0.6,
                     label=f"Linear SoC={s0:.1f} MWh")
    axes[0].set_xlabel("Discharge volume W (MWh)")
    axes[0].set_ylabel("Degradation cost (€)")
    axes[0].set_title("Cost vs discharge volume")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    s_grid = np.linspace(0.05 * STORAGE_MAX, STORAGE_MAX, 100)
    delta_W = 0.1
    axes[1].plot(s_grid,
                 [convex.compute_cost(s, delta_W) / delta_W for s in s_grid],
                 label="Convex (paper)", linewidth=1.8)
    axes[1].plot(s_grid,
                 [linear.compute_cost(s, delta_W) / delta_W for s in s_grid],
                 linestyle="--",
                 label=f"Linear bitepy ({linear.c} €/MWh)")
    axes[1].set_xlabel("State of charge (MWh)")
    axes[1].set_ylabel("Marginal cost (€/MWh)")
    axes[1].set_title(f"Marginal cost of a {delta_W:.2f} MWh discharge")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "model_curves.png")
    plt.savefig(path, dpi=200)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 3. Per-month engine run + convex re-cost
# ---------------------------------------------------------------------------
def run_one_period(period: str, year_month: str, convex: DegradationCostCalculator,
                   linear: LinearDeg, out_dir: str):
    import bitepy as bp

    daily_zip_dir = os.path.join(SYN_ROOT, period, "daily_zips")
    if not os.path.isdir(daily_zip_dir):
        raise FileNotFoundError(f"missing daily zips: {daily_zip_dir}")

    bin_dir = os.path.join(BIN_ROOT, period)
    if os.path.isdir(bin_dir):
        shutil.rmtree(bin_dir)
    os.makedirs(bin_dir)

    # Bitepy's run() reads one binary per day from (start_date - 1 day) through
    # end_date inclusive. We always have the in-month days and the day before
    # the first (the prefix day generated by generate_synthetic_data.py).
    # If the user's sim_end falls on the last day of the month, no extra files
    # are needed beyond what daily_zips already contains.
    csv_files = sorted(
        os.path.join(daily_zip_dir, f)
        for f in os.listdir(daily_zip_dir)
        if f.endswith(".csv.zip")
    )
    parser = bp.Data()
    parser.create_bins_from_csv(csv_files, bin_dir, verbose=False)

    year, month = year_month.split("-")
    # Use 02:00 Berlin so that (start − 1 day) safely lands on the previous
    # day in UTC regardless of CET/CEST offset (1h/2h). Otherwise bitepy
    # would request `orderbook_<prev_day - 1>.bin`, which we don't have.
    sim_start = pd.Timestamp(f"{year_month}-01 02:00:00", tz="Europe/Berlin")
    # End at 23:59 on the last day of the month (no need for a next-month .bin).
    last_day = pd.Period(year_month, freq="M").end_time.day
    sim_end = pd.Timestamp(f"{year_month}-{last_day:02d} 23:59:00",
                           tz="Europe/Berlin")

    sim = bp.Simulation(
        start_date=sim_start,
        end_date=sim_end,
        storage_max=STORAGE_MAX,
        lin_deg_cost=LIN_DEG_COST,
        loss_in=ETA_IN,
        loss_out=ETA_OUT,
        trading_fee=TRADING_FEE,
        inject_max=INJECT_MAX,
        withdraw_max=WITHDRAW_MAX,
        log_transactions=False,
    )
    print(f"\n[{period}] running bitepy from {sim_start} to {sim_end} ...", flush=True)
    sim.run(bin_dir, verbose=False)
    print(f"  [{period}] bitepy run completed", flush=True)

    logs = sim.get_logs()
    executed = logs["executed_orders"]
    print(f"  [{period}] executed_orders rows: {len(executed)}", flush=True)
    if executed.empty:
        print(f"  [{period}] WARNING: bitepy produced no executed orders.")
        return None

    df = executed.sort_values("time").reset_index(drop=True)

    # Recompute SoC & both costs from scratch.
    soc = INITIAL_SOC
    soc_trace = []
    cum_lin, cum_cvx = 0.0, 0.0
    lin_trace, cvx_trace = [], []
    discharge_volumes = []

    for _, row in df.iterrows():
        vol = float(row["volume"])
        order_type = str(row.get("type", "")).lower()
        is_discharge = "sell" in order_type
        is_charge = "buy" in order_type

        if is_discharge:
            internal = min(vol / ETA_OUT, max(0.0, soc - SOC_MIN))
            cum_lin += linear.compute_cost(soc, internal)
            cum_cvx += convex.compute_cost(soc, internal)
            soc -= internal
            discharge_volumes.append(internal)
        elif is_charge:
            added = min(vol * ETA_IN, max(0.0, SOC_MAX - soc))
            soc += added

        soc_trace.append((row["time"], soc))
        lin_trace.append(cum_lin)
        cvx_trace.append(cum_cvx)

    bitepy_reward = float(executed["reward"].sum())
    bitepy_reward_incl_deg = float(executed["reward_incl_deg_costs"].sum())
    bitepy_implied_deg = bitepy_reward - bitepy_reward_incl_deg

    summary = {
        "period": period,
        "executed_orders": int(len(executed)),
        "discharges": int(len(discharge_volumes)),
        "total_internal_discharge_MWh": round(sum(discharge_volumes), 2),
        "bitepy_gross_eur": round(bitepy_reward, 2),
        "bitepy_reward_incl_lin_deg_eur": round(bitepy_reward_incl_deg, 2),
        "bitepy_implied_lin_deg_eur": round(bitepy_implied_deg, 2),
        "discharge_only_lin_eur": round(cum_lin, 2),
        "discharge_only_convex_eur": round(cum_cvx, 2),
        "convex_minus_lin_gap_eur": round(cum_cvx - cum_lin, 2),
        "convex_over_lin_ratio": round(cum_cvx / cum_lin, 3) if cum_lin > 0 else float("nan"),
    }

    # ---------- Plots ----------
    times = [t for t, _ in soc_trace]
    socs = [s for _, s in soc_trace]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    axes[0].plot(times, socs, color="steelblue", linewidth=1.2)
    axes[0].axhline(SOC_MIN, color="red", linestyle=":", alpha=0.5, label="SoC bounds")
    axes[0].axhline(SOC_MAX, color="red", linestyle=":", alpha=0.5)
    axes[0].set_ylabel("SoC (MWh)")
    axes[0].set_title(f"Bitepy SoC trajectory — {period.upper()} {year_month}")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(times, cvx_trace, label="Convex (paper)", linewidth=1.8)
    axes[1].plot(times, lin_trace, linestyle="--",
                 label=f"Linear bitepy ({linear.c} €/MWh)", linewidth=1.8)
    axes[1].set_ylabel("Cumulative degradation cost (€)")
    axes[1].set_title("Same trade sequence, two cost models")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, f"trajectory_{period}.png")
    plt.savefig(path, dpi=200)
    plt.close(fig)

    return summary, list(zip(times, lin_trace, cvx_trace))


# ---------------------------------------------------------------------------
# 4. Cross-month combined plot
# ---------------------------------------------------------------------------
def plot_combined_gap(traces_by_period, out_dir: str):
    fig, ax = plt.subplots(figsize=(14, 6))
    for period, trace in traces_by_period.items():
        if not trace:
            continue
        times = [t for t, _, _ in trace]
        gap = [c - l for _, l, c in trace]
        ax.plot(times, gap, label=period.upper(), linewidth=1.4)
    ax.set_ylabel("Convex − Linear cost gap (€)")
    ax.set_xlabel("Time")
    ax.set_title("Cumulative degradation-cost gap on bitepy trades (convex − linear)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "cumulative_gap.png")
    plt.savefig(path, dpi=200)
    plt.close(fig)
    return path


def write_latex_table(summaries, out_path):
    """Compact LaTeX table for inclusion in the paper."""
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Period & Discharges & Discharge MWh & Linear €/MWh deg & Convex deg & Gap (€) \\",
        r"\midrule",
    ]
    for s in summaries:
        lines.append(
            f"{s['period'].capitalize()} & {s['discharges']} & {s['total_internal_discharge_MWh']:.1f} & "
            f"{s['discharge_only_lin_eur']:.0f} & {s['discharge_only_convex_eur']:.0f} & "
            f"{s['convex_minus_lin_gap_eur']:+.0f} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(BIN_ROOT, exist_ok=True)

    convex = DegradationCostCalculator(
        energy_capacity=STORAGE_MAX,
        num_segments=CONVEX_SEGMENTS,
        replacement_cost=REPLACEMENT_COST,
        eta_discharge=ETA_OUT,
    )
    linear = LinearDeg(LIN_DEG_COST)

    # Always emit the model-only comparison curves first — they don't require
    # the engine and are useful even if a period fails.
    print("=" * 72)
    print("Bitepy linear deg vs paper convex deg — monthly comparison")
    print("=" * 72)
    print("Convex segment costs (€/MWh) j=1..J:")
    print("  ", np.round(convex.c_j, 4).tolist())
    print(f"Linear bitepy cost: {linear.c} €/MWh (constant)")
    p = plot_model_curves(convex, linear, OUTPUT_DIR)
    print(f"[saved] {p}")

    summaries = []
    traces_by_period = {}
    for period, ym in PERIODS.items():
        try:
            res = run_one_period(period, ym, convex, linear, OUTPUT_DIR)
        except Exception as e:
            import traceback
            print(f"  [{period}] FAILED: {e}")
            traceback.print_exc()
            res = None
        if res is None:
            continue
        summary, trace = res
        summaries.append(summary)
        traces_by_period[period] = trace

    if not summaries:
        print("\nNo periods produced trades. Check syn_data/<period>/daily_zips/.")
        return

    df_summary = pd.DataFrame(summaries)
    csv_path = os.path.join(OUTPUT_DIR, "monthly_summary.csv")
    df_summary.to_csv(csv_path, index=False)
    print("\n=== Monthly summary ===")
    print(df_summary.to_string(index=False))
    print(f"[saved] {csv_path}")

    tex_path = os.path.join(OUTPUT_DIR, "monthly_summary.tex")
    write_latex_table(summaries, tex_path)
    print(f"[saved] {tex_path}")

    gap_path = plot_combined_gap(traces_by_period, OUTPUT_DIR)
    print(f"[saved] {gap_path}")


if __name__ == "__main__":
    main()
