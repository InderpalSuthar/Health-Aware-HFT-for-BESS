"""
End-to-end comparison: bitepy linear-degradation engine vs paper convex model.

Pipeline:
  1. Generate a short window of synthetic EPEX-format order books in
     bitepy's required CSV layout (initial, side, start, transaction,
     validity, price, quantity).
  2. Convert the zipped CSVs to .bin via `bitepy.Data.create_bins_from_csv`.
  3. Run bitepy's `Simulation` with its linear degradation cost.
  4. From bitepy's executed-order log, reconstruct the SoC trajectory and
     re-evaluate every discharge under our convex (cycle-depth-aware) model.
  5. Save plots and CSVs to code/diff/output_engine/.

Run with the venv that has bitepy installed:
    /Users/inderpal/Documents/NGU_Project/bitepy_venv/bin/python \
        code/diff/compare_with_bitepy_engine.py
"""

import os
import sys
import shutil
import zipfile
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
PROJECT_DIR = os.path.abspath(os.path.join(CODE_DIR, ".."))
OUTPUT_DIR = os.path.join(THIS_DIR, "output_engine")
SYNTH_DIR = os.path.join(OUTPUT_DIR, "synthetic_csvs")
BIN_DIR = os.path.join(OUTPUT_DIR, "binaries")

sys.path.insert(0, CODE_DIR)
from Cdeg_calc import DegradationCostCalculator  # convex model


# ---------------------------------------------------------------------------
# 1. Synthetic data generator (mirrors bitepy/example_data/generate_synthetic_data.py)
# ---------------------------------------------------------------------------
START_DATE = "2024-01-01"
NUM_DAYS = 7
ORDERS_PER_HOUR = 80
BASE_PRICE = 50.0
PRICE_VOLATILITY = 25.0
MIN_QUANTITY = 0.1
MAX_QUANTITY = 20.0
VALIDITY_RATE = 0.6


def _hourly_prices(rng, num_hours=24):
    hours = np.arange(num_hours)
    pattern = (
        10 * np.sin(np.pi * (hours - 6) / 12)
        + 5 * np.sin(np.pi * (hours - 18) / 6)
        + 3 * np.sin(2 * np.pi * hours / 24)
    )
    daily_offset = rng.normal(0, 10)
    walk = np.cumsum(rng.normal(0, 2, num_hours))
    noise = rng.normal(0, PRICE_VOLATILITY / 4, num_hours)
    return BASE_PRICE + pattern + daily_offset + walk + noise


def _orders_for_hour(rng, date, hour, mid_price, order_id_start, index_start):
    n_orders = max(10, int(rng.normal(ORDERS_PER_HOUR, ORDERS_PER_HOUR / 3)))
    delivery_start = datetime(date.year, date.month, date.day, hour, 0, 0)
    earliest = datetime(date.year, date.month, date.day, 0, 0, 0)
    latest = delivery_start - timedelta(minutes=5)
    if latest < earliest:
        earliest = delivery_start - timedelta(hours=12)
    window = (latest - earliest).total_seconds()

    out = []
    for i in range(n_orders):
        ts = earliest + timedelta(seconds=rng.uniform(0, window))
        ms = rng.integers(0, 1000)
        trans = ts.strftime("%Y-%m-%dT%H:%M:%S") + f".{ms:03d}Z"

        side = "BUY" if rng.random() < 0.5 else "SELL"
        spread = rng.exponential(2)
        price = mid_price + (spread + rng.normal(0, 1)) * (1 if side == "SELL" else -1)
        price = round(price, 2)
        quantity = round(rng.uniform(MIN_QUANTITY, MAX_QUANTITY), 1)

        if rng.random() < VALIDITY_RATE:
            v_secs = rng.uniform(60, (delivery_start - ts).total_seconds())
            v_time = ts + timedelta(seconds=v_secs)
            v_ms = rng.integers(0, 1000)
            validity = v_time.strftime("%Y-%m-%dT%H:%M:%S") + f".{v_ms:03d}Z"
        else:
            validity = ""

        out.append(
            dict(
                index=index_start + i,
                initial=order_id_start + i,
                side=side,
                start=delivery_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                transaction=trans,
                validity=validity,
                price=price,
                quantity=quantity,
            )
        )
    return out


def _generate_day(rng, date):
    rows = []
    base_oid = int(rng.integers(10_000_000_000, 99_999_999_999))
    base_idx = int(rng.integers(100_000, 999_999))

    # Today's products
    prices_today = _hourly_prices(rng)
    for h in range(24):
        rows.extend(
            _orders_for_hour(rng, date, h, prices_today[h], base_oid, base_idx)
        )
        base_oid += len(rows[-1:]) or 0  # keep IDs unique-ish
        base_oid += 64
        base_idx += 64

    # Tomorrow's products, submitted today (fewer orders)
    next_date = date + timedelta(days=1)
    prices_next = _hourly_prices(rng)
    for h in range(24):
        n_save = ORDERS_PER_HOUR
        # locally reduce density for next-day products
        globals()["ORDERS_PER_HOUR"] = max(5, ORDERS_PER_HOUR // 3)
        try:
            os = _orders_for_hour(rng, next_date, h, prices_next[h], base_oid, base_idx)
        finally:
            globals()["ORDERS_PER_HOUR"] = n_save
        # rebase transaction timestamps to today
        for o in os:
            t = datetime.strptime(o["transaction"][:19], "%Y-%m-%dT%H:%M:%S")
            t = t.replace(year=date.year, month=date.month, day=date.day)
            o["transaction"] = t.strftime("%Y-%m-%dT%H:%M:%S") + o["transaction"][19:]
            if o["validity"]:
                v = datetime.strptime(o["validity"][:19], "%Y-%m-%dT%H:%M:%S")
                v = v.replace(year=date.year, month=date.month, day=date.day)
                o["validity"] = v.strftime("%Y-%m-%dT%H:%M:%S") + o["validity"][19:]
        rows.extend(os)
        base_oid += len(os) + 64
        base_idx += len(os) + 64

    df = pd.DataFrame(rows).sort_values("transaction").reset_index(drop=True)
    df = df[["initial", "side", "start", "transaction", "validity", "price", "quantity"]]
    df.index = df.index + int(rng.integers(100_000, 999_999))
    return df


def generate_synthetic(out_dir, start_date_str=START_DATE, num_days=NUM_DAYS, seed=42):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    start = datetime.strptime(start_date_str, "%Y-%m-%d").date()
    files = []
    for i in range(num_days):
        d = start + timedelta(days=i)
        df = _generate_day(rng, d)
        zip_path = os.path.join(out_dir, f"orderbook_{d.isoformat()}.csv.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(f"orderbook_{d.isoformat()}.csv", df.to_csv())
        files.append(zip_path)
        print(f"  wrote {os.path.basename(zip_path)} ({len(df)} orders)")
    return files


# ---------------------------------------------------------------------------
# 2. Bitepy run + 3. Convex re-evaluation
# ---------------------------------------------------------------------------
def run_bitepy_and_recost(
    csv_files,
    bin_dir,
    sim_start,
    sim_end,
    storage_max=12.5,
    soc_min_frac=0.15,
    soc_max_frac=0.95,
    lin_deg_cost=4.0,
    inject_max=5.0,
    withdraw_max=5.0,
    eta_in=0.95,
    eta_out=0.95,
    initial_soc=6.25,
    convex_segments=16,
    replacement_cost=300_000.0,
):
    import bitepy as bp

    if os.path.isdir(bin_dir):
        shutil.rmtree(bin_dir)
    os.makedirs(bin_dir)
    parser = bp.Data()
    parser.create_bins_from_csv(csv_files, bin_dir, verbose=True)

    sim = bp.Simulation(
        start_date=sim_start,
        end_date=sim_end,
        storage_max=storage_max,
        lin_deg_cost=lin_deg_cost,
        loss_in=eta_in,
        loss_out=eta_out,
        inject_max=inject_max,
        withdraw_max=withdraw_max,
        log_transactions=False,
    )
    sim.print_parameters()
    sim.run(bin_dir, verbose=True)

    logs = sim.get_logs()
    executed = logs["executed_orders"]

    if executed.empty:
        print("WARNING: bitepy produced no executed orders for this window.")
        return None

    # bitepy's executed_orders frame already includes 'reward', 'reward_incl_deg_costs',
    # 'volume', and 'type'. We rebuild SoC from scratch and recompute degradation.
    energy_min = soc_min_frac * storage_max
    energy_max = soc_max_frac * storage_max

    convex_calc = DegradationCostCalculator(
        energy_capacity=storage_max,
        num_segments=convex_segments,
        replacement_cost=replacement_cost,
        eta_discharge=eta_out,
    )

    df = executed.sort_values("time").reset_index(drop=True)

    soc = initial_soc
    soc_trace = []
    convex_costs = []
    linear_costs = []
    cum_convex = 0.0
    cum_linear = 0.0
    rebuilt = []

    for _, row in df.iterrows():
        vol = float(row["volume"])
        order_type = str(row.get("type", "")).lower()
        # bitepy 'type' is typically 'sell'/'buy' from the BATTERY's perspective.
        # SELL = battery sells = discharge ; BUY = battery buys = charge.
        is_discharge = "sell" in order_type
        is_charge = "buy" in order_type

        if is_discharge:
            # bitepy's `volume` is the energy delivered TO the grid (post-loss).
            # Energy drawn from battery internal store = vol / eta_out.
            internal = vol / eta_out
            internal = min(internal, max(0.0, soc - energy_min))
            cost_linear = lin_deg_cost * internal
            cost_convex = convex_calc.compute_cost(soc, internal)
            cum_linear += cost_linear
            cum_convex += cost_convex
            soc -= internal
            rebuilt.append(
                dict(
                    time=row["time"],
                    side="discharge",
                    volume_internal=internal,
                    volume_grid=vol,
                    soc_after=soc,
                    cost_linear=cost_linear,
                    cost_convex=cost_convex,
                )
            )
        elif is_charge:
            # vol = energy purchased from grid; SoC adds vol * eta_in
            added = vol * eta_in
            added = min(added, max(0.0, energy_max - soc))
            soc += added
            rebuilt.append(
                dict(
                    time=row["time"],
                    side="charge",
                    volume_internal=added,
                    volume_grid=vol,
                    soc_after=soc,
                    cost_linear=0.0,
                    cost_convex=0.0,
                )
            )
        soc_trace.append((row["time"], soc))
        convex_costs.append(cum_convex)
        linear_costs.append(cum_linear)

    rebuilt_df = pd.DataFrame(rebuilt)

    # Bitepy's own bookkeeping for cross-check.
    bitepy_reward = float(executed["reward"].sum())
    bitepy_reward_incl_deg = float(executed["reward_incl_deg_costs"].sum())
    bitepy_implied_deg = bitepy_reward - bitepy_reward_incl_deg

    n_discharges = (
        int((rebuilt_df["side"] == "discharge").sum()) if not rebuilt_df.empty else 0
    )
    total_discharge_internal = (
        rebuilt_df.loc[rebuilt_df["side"] == "discharge", "volume_internal"].sum()
        if not rebuilt_df.empty
        else 0.0
    )

    summary = pd.DataFrame(
        {
            "Metric": [
                "Executed orders (bitepy)",
                "Number of discharges",
                "Total internal discharge (MWh)",
                # bitepy's own bookkeeping (linear deg model, full lifecycle)
                "Bitepy gross reward (€)",
                "Bitepy reward incl. linear deg (€)",
                "Bitepy implied total deg cost (€)",
                # Discharge-only re-evaluation (apples-to-apples between models)
                f"Discharge-only linear cost ({lin_deg_cost} €/MWh) (€)",
                "Discharge-only convex cost (paper) (€)",
                "Convex – Linear gap on discharges (€)",
                "Convex / Linear ratio on discharges",
            ],
            "Value": [
                len(executed),
                n_discharges,
                round(total_discharge_internal, 2),
                round(bitepy_reward, 2),
                round(bitepy_reward_incl_deg, 2),
                round(bitepy_implied_deg, 2),
                round(cum_linear, 2),
                round(cum_convex, 2),
                round(cum_convex - cum_linear, 2),
                round(cum_convex / cum_linear, 3) if cum_linear > 0 else float("nan"),
            ],
        }
    )

    print("\n=== Bitepy engine + convex re-evaluation ===")
    print(summary.to_string(index=False))
    summary.to_csv(os.path.join(OUTPUT_DIR, "engine_summary.csv"), index=False)
    rebuilt_df.to_csv(os.path.join(OUTPUT_DIR, "engine_trades.csv"), index=False)

    # ---- Plots ----
    times = [t for t, _ in soc_trace]
    socs = [s for _, s in soc_trace]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    axes[0].plot(times, socs, color="steelblue", linewidth=1.4)
    axes[0].axhline(energy_min, color="red", linestyle=":", alpha=0.5, label="SoC limits")
    axes[0].axhline(energy_max, color="red", linestyle=":", alpha=0.5)
    axes[0].set_ylabel("SoC (MWh)")
    axes[0].set_title("Bitepy-decided SoC trajectory (synthetic order book)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(times, convex_costs, label="Convex (paper)", linewidth=1.8)
    axes[1].plot(
        times,
        linear_costs,
        linestyle="--",
        label=f"Linear bitepy ({lin_deg_cost} €/MWh)",
        linewidth=1.8,
    )
    axes[1].set_ylabel("Cumulative degradation cost (€)")
    axes[1].set_xlabel("Time")
    axes[1].set_title("Same trade sequence, two cost models")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "engine_comparison.png")
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[saved] {out_path}")

    return summary


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 72)
    print("End-to-end bitepy engine vs convex re-evaluation")
    print("=" * 72)

    # bitepy reads the day *before* sim_start to capture orders submitted
    # earlier with delivery on the first trading day, so we generate one
    # extra prefix day.
    prefix_date = (
        datetime.strptime(START_DATE, "%Y-%m-%d").date() - timedelta(days=1)
    ).isoformat()
    total_days = NUM_DAYS + 1
    print(
        f"\n[1/3] Generating {total_days} days of synthetic order books "
        f"(starting {prefix_date}) in {SYNTH_DIR} ..."
    )
    csv_files = generate_synthetic(SYNTH_DIR, prefix_date, total_days)

    sim_start = pd.Timestamp(f"{START_DATE} 00:00:00", tz="Europe/Berlin")
    sim_end = sim_start + pd.Timedelta(days=NUM_DAYS - 1)

    print(f"\n[2/3] Running bitepy from {sim_start} to {sim_end} ...")
    summary = run_bitepy_and_recost(
        csv_files=csv_files,
        bin_dir=BIN_DIR,
        sim_start=sim_start,
        sim_end=sim_end,
    )

    if summary is None:
        print("\nNo trades produced — try increasing NUM_DAYS or order density.")
        return

    print("\n[3/3] Done. Outputs:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        full = os.path.join(OUTPUT_DIR, f)
        if os.path.isfile(full):
            print(f"  - {full}")


if __name__ == "__main__":
    main()
