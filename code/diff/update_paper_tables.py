"""
Patch the numeric tables in paper/main.tex with the freshly generated
results. Idempotent.

Approach:
  1. Locate every uncommented `\\begin{table}...\\end{table}` block in
     the paper and pair it with its `\\label{...}`.
  2. For each known label key, replace exactly that block with a freshly
     rendered LaTeX table built from the corresponding CSV.

Updates:
  * tab:dp_milp_comparison        from comparison_results.csv
  * tab:efficiency_sensitivity    from table_efficiency.csv
  * tab:capacity_sensitivity      from table_capacity_*h_battery.csv
"""

from __future__ import annotations

import os
import re

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
SCD = os.path.join(ROOT, "code", "strategy_comparison_data")
PAPER = os.path.join(ROOT, "paper", "main.tex")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _euro(x: float) -> str:
    if x is None or pd.isna(x):
        return "--"
    return f"{int(round(x)):,}".replace(",", "{,}")


# ---------------------------------------------------------------------------
# table renderers
# ---------------------------------------------------------------------------
def render_dp_vs_milp(csv_path: str) -> str:
    df = pd.read_csv(csv_path)
    body_lines = []
    last_period = None
    pending_period_rows = []

    def flush_period(period: str, rows: list[str]):
        if not rows:
            return
        out = [f"\\multirow{{{len(rows)}}}{{*}}{{{period}}} & " + rows[0]]
        for r in rows[1:]:
            out.append("& " + r)
        body_lines.extend(out)
        body_lines.append(r"\midrule")

    for _, row in df.iterrows():
        period = row["period"]
        line = (
            f"{row['strategy']} & {_euro(row['reward_eur'])} & "
            f"{row['cycles_per_day']:.1f} & {int(row['solves'])} & "
            f"{_euro(row['traded_vol_mwh'])} \\\\"
        )
        if period != last_period:
            flush_period(last_period, pending_period_rows)
            pending_period_rows = []
            last_period = period
        pending_period_rows.append(line)
    flush_period(last_period, pending_period_rows)

    if body_lines and body_lines[-1] == r"\midrule":
        body_lines = body_lines[:-1]
    inner = "\n".join(body_lines)

    return rf"""\begin{{table}}[htb!]
\centering
\caption{{Performance Comparison: DP vs.\ MILP on monthly synthetic order books (2024). MILP is restricted to the first 14 days of each month for tractability; DP is evaluated on the full month. Cycles/Day are normalized by the strategy's effective window length.}}
\label{{tab:dp_milp_comparison}}
\resizebox{{\columnwidth}}{{!}}{{
\begin{{tabular}}{{llrrrr}}
\toprule
\textbf{{Period}} & \textbf{{Method}} & \textbf{{Reward (\euro{{}})}} & \textbf{{Cycles/Day}} & \textbf{{Solves}} & \textbf{{Vol. (MWh)}} \\
\midrule
{inner}
\bottomrule
\end{{tabular}}
}}
\end{{table}}"""


def render_efficiency(csv_path: str) -> str:
    df = pd.read_csv(csv_path)
    eta_cols = [c for c in df.columns if c not in ("Reward [€]", "Reward/s̄")]
    rows = []
    for _, row in df.iterrows():
        strat = row.iloc[0]
        vals = " & ".join(_euro(row[c]) for c in eta_cols)
        rows.append(f"{strat} & {vals} \\\\")
    inner = "\n".join(rows)
    headers = " & ".join(rf"$\boldsymbol{{\eta={c}}}$" for c in eta_cols)
    return rf"""\begin{{table}}[htb!]
\centering
\caption{{Reward Sensitivity to Round-Trip Efficiency $\eta^+ = \eta^-$ (Feb 2024, first 14 days).}}
\label{{tab:efficiency_sensitivity}}
\resizebox{{\columnwidth}}{{!}}{{
\begin{{tabular}}{{l{'r' * len(eta_cols)}}}
\toprule
\textbf{{Strategy}} & {headers} \\
\midrule
{inner}
\bottomrule
\end{{tabular}}
}}
\end{{table}}"""


def render_capacity() -> str:
    df1 = pd.read_csv(os.path.join(SCD, "table_capacity_1h_battery.csv"))
    df2 = pd.read_csv(os.path.join(SCD, "table_capacity_2h_battery.csv"))
    cap_cols = [c for c in df1.columns if c not in ("Reward [€]", "Reward/s̄")]
    rows = []
    for (_, r1), (_, r2) in zip(df1.iterrows(), df2.iterrows()):
        strat = r1.iloc[0]
        v1 = " & ".join(_euro(r1[c]) for c in cap_cols)
        v2 = " & ".join(_euro(r2[c]) for c in cap_cols)
        rows.append(f"{strat} & {v1} & {v2} \\\\")
    inner = "\n".join(rows)
    h_cap = " & ".join(rf"\textbf{{{c}}}" for c in cap_cols)
    return rf"""\begin{{table}}[htb!]
\centering
\caption{{Specific Reward per MWh of Capacity [\euro{{}}/MWh] vs.\ Storage Size (Feb 2024, first 14 days).}}
\label{{tab:capacity_sensitivity}}
\resizebox{{\columnwidth}}{{!}}{{
\begin{{tabular}}{{l|{'r' * len(cap_cols)}|{'r' * len(cap_cols)}}}
\toprule
& \multicolumn{{{len(cap_cols)}}}{{c|}}{{\textbf{{1h Battery}} ($P = \bar{{s}}$)}} & \multicolumn{{{len(cap_cols)}}}{{c}}{{\textbf{{2h Battery}} ($P = \bar{{s}}/2$)}} \\
\textbf{{Strategy}} & {h_cap} & {h_cap} \\
\midrule
{inner}
\bottomrule
\end{{tabular}}
}}
\end{{table}}"""


# ---------------------------------------------------------------------------
# block-aware paper rewrite
# ---------------------------------------------------------------------------
def find_uncommented_table_blocks(tex: str):
    """Return list of (start, end, label) for every \\begin{table}...\\end{table}
    block that is NOT inside a comment. We skip lines whose first non-whitespace
    char is '%' when looking for \\begin{table}."""
    blocks = []
    pattern = re.compile(r"\\begin\{table\}", re.M)
    for m in pattern.finditer(tex):
        # Check that the line containing this match is not commented.
        line_start = tex.rfind("\n", 0, m.start()) + 1
        line_prefix = tex[line_start : m.start()]
        if line_prefix.lstrip().startswith("%"):
            continue
        end_match = re.search(r"\\end\{table\}", tex[m.end():])
        if not end_match:
            continue
        end = m.end() + end_match.end()
        block = tex[m.start():end]
        label_m = re.search(r"\\label\{(tab:[^}]+)\}", block)
        label = label_m.group(1) if label_m else None
        blocks.append((m.start(), end, label))
    return blocks


def replace_block_by_label(tex: str, label: str, new_block: str) -> tuple[str, bool]:
    blocks = find_uncommented_table_blocks(tex)
    for start, end, lbl in blocks:
        if lbl == label:
            return tex[:start] + new_block + tex[end:], True
    return tex, False


def main():
    with open(PAPER) as f:
        tex = f.read()

    edits = []

    res_csv = os.path.join(SCD, "comparison_results.csv")
    if os.path.exists(res_csv):
        tex, ok = replace_block_by_label(
            tex, "tab:dp_milp_comparison", render_dp_vs_milp(res_csv)
        )
        if ok:
            edits.append("tab:dp_milp_comparison")
        else:
            print("[skip] tab:dp_milp_comparison not found in paper")

    eff_csv = os.path.join(SCD, "table_efficiency.csv")
    if os.path.exists(eff_csv):
        tex, ok = replace_block_by_label(
            tex, "tab:efficiency_sensitivity", render_efficiency(eff_csv)
        )
        if ok:
            edits.append("tab:efficiency_sensitivity")
        else:
            print("[skip] tab:efficiency_sensitivity not found in paper")

    cap1 = os.path.join(SCD, "table_capacity_1h_battery.csv")
    cap2 = os.path.join(SCD, "table_capacity_2h_battery.csv")
    if os.path.exists(cap1) and os.path.exists(cap2):
        tex, ok = replace_block_by_label(
            tex, "tab:capacity_sensitivity", render_capacity()
        )
        if ok:
            edits.append("tab:capacity_sensitivity")
        else:
            print("[skip] tab:capacity_sensitivity not found in paper")

    with open(PAPER, "w") as f:
        f.write(tex)

    print("Replaced:", edits if edits else "(no changes made)")


if __name__ == "__main__":
    main()
