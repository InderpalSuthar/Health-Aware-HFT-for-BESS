"""
Synthetic intraday market data generator.

Produces order books in the bitepy / EPEX SPOT continuous-intraday format
(`initial, side, start, transaction, validity, price, quantity`) but emits
them with the `strategy_comparison_data` naming convention so that MILP.py,
DP.py, and the existing run_strategy_comparison.py can ingest them.

Two output modes are produced for each period:

  1. Combined CSV (one file per month, e.g. `orderbook_feb.csv`) — the
     format expected by strategy_comparison_data's `OrderBookLoader`.
  2. Daily zipped CSVs (`orderbook_YYYY-MM-DD.csv.zip`) — the format
     expected by `bitepy.Data.create_bins_from_csv` so the same data can
     be fed straight into the bitepy engine.

By default we generate four full months (Feb / Apr / Jul / Oct of 2024)
matching the existing comparison study.

Output location: code/strategy_comparison_data/syn_data/<period>/

Run:
    /Users/inderpal/Documents/NGU_Project/bitepy_venv/bin/python \
        code/generate_synthetic_data.py
"""

import os
import sys
import zipfile
import calendar
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ORDERS_PER_HOUR = 50           # average orders per delivery hour
BASE_PRICE = 50.0              # €/MWh
PRICE_VOLATILITY = 25.0        # €/MWh
MIN_QUANTITY = 0.1             # MWh
MAX_QUANTITY = 20.0            # MWh
VALIDITY_RATE = 0.6            # fraction of orders with explicit validity
RANDOM_SEED = 42

# Periods mirror code/strategy_comparison_data/generate_orderbooks.py.
# Key = short period label used in the filename; value = (year, month).
DEFAULT_PERIODS = {
    "feb": (2024, 2),
    "apr": (2024, 4),
    "jul": (2024, 7),
    "oct": (2024, 10),
}

OUTPUT_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "strategy_comparison_data",
    "syn_data",
)


# ---------------------------------------------------------------------------
# Price curve and order generation (bitepy-style)
# ---------------------------------------------------------------------------
def _hourly_price_curve(rng, num_hours=24):
    hours = np.arange(num_hours)
    pattern = (
        10 * np.sin(np.pi * (hours - 6) / 12)        # morning peak
        + 5 * np.sin(np.pi * (hours - 18) / 6)       # evening peak
        + 3 * np.sin(2 * np.pi * hours / 24)         # daily cycle
    )
    daily_offset = rng.normal(0, 10)
    walk = np.cumsum(rng.normal(0, 2, num_hours))
    noise = rng.normal(0, PRICE_VOLATILITY / 4, num_hours)
    return BASE_PRICE + pattern + daily_offset + walk + noise


def _orders_for_hour(rng, date, hour, mid_price, base_oid, base_idx, density):
    n_orders = max(10, int(rng.normal(density, density / 3)))
    delivery_start = datetime(date.year, date.month, date.day, hour, 0, 0)
    earliest = datetime(date.year, date.month, date.day, 0, 0, 0)
    latest = delivery_start - timedelta(minutes=5)
    if latest < earliest:
        earliest = delivery_start - timedelta(hours=12)
    window = (latest - earliest).total_seconds()

    rows = []
    for i in range(n_orders):
        ts = earliest + timedelta(seconds=rng.uniform(0, window))
        ms = int(rng.integers(0, 1000))
        trans = ts.strftime("%Y-%m-%dT%H:%M:%S") + f".{ms:03d}Z"

        side = "BUY" if rng.random() < 0.5 else "SELL"
        spread = rng.exponential(2)
        bias = (spread + rng.normal(0, 1)) * (1 if side == "SELL" else -1)
        price = round(mid_price + bias, 2)
        quantity = round(rng.uniform(MIN_QUANTITY, MAX_QUANTITY), 1)

        if rng.random() < VALIDITY_RATE:
            v_secs = rng.uniform(60, max(120.0, (delivery_start - ts).total_seconds()))
            v_time = ts + timedelta(seconds=v_secs)
            v_ms = int(rng.integers(0, 1000))
            validity = v_time.strftime("%Y-%m-%dT%H:%M:%S") + f".{v_ms:03d}Z"
        else:
            validity = ""

        rows.append(
            dict(
                initial=base_oid + i,
                side=side,
                start=delivery_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                transaction=trans,
                validity=validity,
                price=price,
                quantity=quantity,
            )
        )
    return rows


def _generate_day(rng, date, density=ORDERS_PER_HOUR):
    """Generate a single day of orders: today's products + next-day products
    that were submitted today (mirroring real EPEX submission patterns)."""
    rows = []
    base_oid = int(rng.integers(10_000_000_000, 99_999_999_999))
    base_idx = int(rng.integers(100_000, 999_999))

    today_curve = _hourly_price_curve(rng)
    for h in range(24):
        rows.extend(
            _orders_for_hour(rng, date, h, today_curve[h], base_oid, base_idx, density)
        )
        base_oid += density + 64
        base_idx += density + 64

    next_date = date + timedelta(days=1)
    next_curve = _hourly_price_curve(rng)
    next_density = max(5, density // 3)
    for h in range(24):
        os_rows = _orders_for_hour(
            rng, next_date, h, next_curve[h], base_oid, base_idx, next_density
        )
        for o in os_rows:
            t = datetime.strptime(o["transaction"][:19], "%Y-%m-%dT%H:%M:%S")
            t = t.replace(year=date.year, month=date.month, day=date.day)
            o["transaction"] = t.strftime("%Y-%m-%dT%H:%M:%S") + o["transaction"][19:]
            if o["validity"]:
                v = datetime.strptime(o["validity"][:19], "%Y-%m-%dT%H:%M:%S")
                v = v.replace(year=date.year, month=date.month, day=date.day)
                o["validity"] = v.strftime("%Y-%m-%dT%H:%M:%S") + o["validity"][19:]
        rows.extend(os_rows)
        base_oid += next_density + 64
        base_idx += next_density + 64

    return rows


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------
def _df_from_rows(rows):
    df = pd.DataFrame(rows).sort_values("transaction").reset_index(drop=True)
    df = df[["initial", "side", "start", "transaction", "validity", "price", "quantity"]]
    return df


def _write_combined_csv(df, period_label, year, month, out_dir):
    """strategy_comparison_data-style: single CSV with `Unnamed: 0` index."""
    last_day = calendar.monthrange(year, month)[1]
    fname = f"orderbook_{period_label}_01_{last_day:02d}.csv"
    path = os.path.join(out_dir, fname)
    # write with the leading unnamed index column the loader expects
    df.to_csv(path, index=True, index_label="Unnamed: 0")
    return path


def _write_daily_zips(df, out_dir):
    """bitepy-style: one zipped CSV per day, named orderbook_YYYY-MM-DD.csv.zip.

    The bitepy `Data._load_csv` reads `compression="zip"` and renames the
    first unnamed column to `id`, so we keep an integer index column.
    Files are split by transaction date (the date the order was submitted),
    matching how the bitepy engine consumes per-day binaries.
    """
    daily_dir = os.path.join(out_dir, "daily_zips")
    os.makedirs(daily_dir, exist_ok=True)
    df = df.copy()
    df["_trans_date"] = df["transaction"].str[:10]
    written = []
    for d, day_df in df.groupby("_trans_date"):
        day_df = day_df.drop(columns="_trans_date")
        # reset index so each per-day CSV has its own contiguous integer index
        day_df = day_df.reset_index(drop=True)
        csv_name = f"orderbook_{d}.csv"
        zip_path = os.path.join(daily_dir, f"orderbook_{d}.csv.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(csv_name, day_df.to_csv())
        written.append(zip_path)
    return daily_dir, written


# ---------------------------------------------------------------------------
# Per-period generation
# ---------------------------------------------------------------------------
def generate_period(
    period_label,
    year,
    month,
    out_root=OUTPUT_ROOT,
    density=ORDERS_PER_HOUR,
    seed=RANDOM_SEED,
    write_daily_zips=True,
):
    """Generate one calendar month of synthetic order book data."""
    rng = np.random.default_rng(seed + hash(period_label) % 100_000)
    last_day = calendar.monthrange(year, month)[1]
    period_dir = os.path.join(out_root, period_label)
    os.makedirs(period_dir, exist_ok=True)

    print(
        f"\n[{period_label}] generating {year}-{month:02d} "
        f"({last_day} days, ~{density} orders/hour) ..."
    )

    all_rows = []
    for day_offset in range(last_day):
        d = datetime(year, month, 1).date() + timedelta(days=day_offset)
        all_rows.extend(_generate_day(rng, d, density=density))

    df = _df_from_rows(all_rows)

    combined_path = _write_combined_csv(df, period_label, year, month, period_dir)
    print(f"  combined CSV : {combined_path}  ({len(df):,} orders)")

    if write_daily_zips:
        daily_dir, written = _write_daily_zips(df, period_dir)
        print(f"  daily zips   : {daily_dir}  ({len(written)} files)")

    return df


def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    print("=" * 72)
    print("Synthetic monthly order books -> code/strategy_comparison_data/syn_data/")
    print("=" * 72)

    for label, (year, month) in DEFAULT_PERIODS.items():
        generate_period(label, year, month)

    print("\nDone. Output root:", OUTPUT_ROOT)


if __name__ == "__main__":
    main()
