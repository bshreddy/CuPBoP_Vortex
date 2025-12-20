#!/usr/bin/env python3
import argparse
import sys
from typing import Dict, List, Tuple

import pandas as pd


# -----------------------------
# Expected table embedded in code
# -----------------------------
def build_expected_df() -> pd.DataFrame:
    # Columns are flattened for simpler exact matching.
    rows = [
        {"benchmark": "backprop",
         "L1+Localmem_instrs": 1267828, "L1+Localmem_cycles": 3105979,
         "L1+L2+Localmem_instrs": 1267828, "L1+L2+Localmem_cycles": 2749718,
         "L1_instrs": 1308660, "L1_cycles": 3194916,
         "L1+L2_instrs": 1308661, "L1+L2_cycles": 2792165},

        {"benchmark": "bfs",
         "L1+Localmem_instrs": 8207781, "L1+Localmem_cycles": 27238966,
         "L1+L2+Localmem_instrs": 8207795, "L1+L2+Localmem_cycles": 23498043,
         "L1_instrs": 8207781, "L1_cycles": 27238966,
         "L1+L2_instrs": 8207795, "L1+L2_cycles": 23498043},

        {"benchmark": "btree",
         "L1+Localmem_instrs": 801096985, "L1+Localmem_cycles": 2138431250,
         "L1+L2+Localmem_instrs": 801096985, "L1+L2+Localmem_cycles": 1396980086,
         "L1_instrs": 801096985, "L1_cycles": 2138431250,
         "L1+L2_instrs": 801096985, "L1+L2_cycles": 1396980086},

        {"benchmark": "conv3",
         "L1+Localmem_instrs": 5829181, "L1+Localmem_cycles": 15085285,
         "L1+L2+Localmem_instrs": 5829180, "L1+L2+Localmem_cycles": 11360147,
         "L1_instrs": 5829181, "L1_cycles": 15085285,
         "L1+L2_instrs": 5829180, "L1+L2_cycles": 11360147},

        {"benchmark": "dotproduct",
         "L1+Localmem_instrs": 22445113, "L1+Localmem_cycles": 39351230,
         "L1+L2+Localmem_instrs": 22445113, "L1+L2+Localmem_cycles": 13919836,
         "L1_instrs": 23051062, "L1_cycles": 40741404,
         "L1+L2_instrs": 23051062, "L1+L2_cycles": 17301387},

        {"benchmark": "nn",
         "L1+Localmem_instrs": 2881194, "L1+Localmem_cycles": 10434009,
         "L1+L2+Localmem_instrs": 2881194, "L1+L2+Localmem_cycles": 8350216,
         "L1_instrs": 2881194, "L1_cycles": 10434009,
         "L1+L2_instrs": 2881194, "L1+L2_cycles": 8350216},

        {"benchmark": "pathfinder",
         "L1+Localmem_instrs": 1271552, "L1+Localmem_cycles": 2436926,
         "L1+L2+Localmem_instrs": 1271501, "L1+L2+Localmem_cycles": 2060861,
         "L1_instrs": 1373772, "L1_cycles": 2483289,
         "L1+L2_instrs": 1373724, "L1+L2_cycles": 2102681},

        {"benchmark": "psum",
         "L1+Localmem_instrs": 20860728, "L1+Localmem_cycles": 36953130,
         "L1+L2+Localmem_instrs": 20860730, "L1+L2+Localmem_cycles": 14877778,
         "L1_instrs": 40119351, "L1_cycles": 103217850,
         "L1+L2_instrs": 40119353, "L1+L2_cycles": 84116076},

        {"benchmark": "saxpy",
         "L1+Localmem_instrs": 3338810, "L1+Localmem_cycles": 12399543,
         "L1+L2+Localmem_instrs": 3338810, "L1+L2+Localmem_cycles": 9429299,
         "L1_instrs": 3338810, "L1_cycles": 12399543,
         "L1+L2_instrs": 3338810, "L1+L2_cycles": 9429299},

        {"benchmark": "sgemm",
         "L1+Localmem_instrs": 113373752, "L1+Localmem_cycles": 22496419,
         "L1+L2+Localmem_instrs": 113373752, "L1+L2+Localmem_cycles": 13791103,
         "L1_instrs": 113373752, "L1_cycles": 22496419,
         "L1+L2_instrs": 113373752, "L1+L2_cycles": 13791103},

        {"benchmark": "stencil",
         "L1+Localmem_instrs": 1587734, "L1+Localmem_cycles": 5703353,
         "L1+L2+Localmem_instrs": 1587736, "L1+L2+Localmem_cycles": 4479180,
         "L1_instrs": 1587734, "L1_cycles": 5703353,
         "L1+L2_instrs": 1587736, "L1+L2_cycles": 4479180},

        {"benchmark": "transpose",
         "L1+Localmem_instrs": 57963064, "L1+Localmem_cycles": 226444216,
         "L1+L2+Localmem_instrs": 57963064, "L1+L2+Localmem_cycles": 172078525,
         "L1_instrs": 57963064, "L1_cycles": 226444216,
         "L1+L2_instrs": 57963064, "L1+L2_cycles": 172078525},

        {"benchmark": "vecadd",
         "L1+Localmem_instrs": 487998, "L1+Localmem_cycles": 1584678,
         "L1+L2+Localmem_instrs": 487998, "L1+L2+Localmem_cycles": 1391675,
         "L1_instrs": 487998, "L1_cycles": 1584678,
         "L1+L2_instrs": 487998, "L1+L2_cycles": 1391675},
         
    ]
    return pd.DataFrame(rows)


# -----------------------------
# Actual table from one CSV
# -----------------------------
def build_actual_df_from_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required = [
        "benchmark",
        "Lmem_off_L2_off_instrs", "Lmem_off_L2_off_cycles",
        "Lmem_off_L2_on_instrs",  "Lmem_off_L2_on_cycles",
        "Lmem_on_L2_off_instrs",  "Lmem_on_L2_off_cycles",
        "Lmem_on_L2_on_instrs",   "Lmem_on_L2_on_cycles",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns in CSV: {missing}")

    # Mapping into your grouped table semantics:
    # - L1 + Localmem         <- Lmem_on_L2_off
    # - L1 + L2 + Localmem    <- Lmem_on_L2_on
    # - L1                   <- Lmem_off_L2_off
    # - L1 + L2              <- Lmem_off_L2_on
    out = pd.DataFrame({
        "benchmark": df["benchmark"],
        "L1+Localmem_instrs": df["Lmem_on_L2_off_instrs"],
        "L1+Localmem_cycles": df["Lmem_on_L2_off_cycles"],
        "L1+L2+Localmem_instrs": df["Lmem_on_L2_on_instrs"],
        "L1+L2+Localmem_cycles": df["Lmem_on_L2_on_cycles"],
        "L1_instrs": df["Lmem_off_L2_off_instrs"],
        "L1_cycles": df["Lmem_off_L2_off_cycles"],
        "L1+L2_instrs": df["Lmem_off_L2_on_instrs"],
        "L1+L2_cycles": df["Lmem_off_L2_on_cycles"],
    })
    return out


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize for robust comparisons:
    # - benchmark as string
    # - numeric columns as floats with NaN for blanks/None
    df = df.copy()
    df["benchmark"] = df["benchmark"].astype(str).str.strip()
    for c in df.columns:
        if c == "benchmark":
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def values_equal(a, b) -> bool:
    # Treat NaN == NaN
    if pd.isna(a) and pd.isna(b):
        return True
    if pd.isna(a) != pd.isna(b):
        return False
    return int(a) == int(b)


def diff_columns(exp_row: pd.Series, act_row: pd.Series) -> List[str]:
    diffs = []
    for c in exp_row.index:
        if c == "benchmark":
            continue
        if not values_equal(exp_row[c], act_row[c]):
            diffs.append(c)
    return diffs


def print_status_table(status_rows: List[Tuple[str, str, str]]) -> None:
    # status_rows: (benchmark, status, details)
    bench_w = max(9, max((len(b) for b, _, _ in status_rows), default=9))
    status_w = max(6, max((len(s) for _, s, _ in status_rows), default=6))

    print("\n=== Comparison Summary ===")
    print(f"{'benchmark':<{bench_w}}  {'status':<{status_w}}  details")
    print(f"{'-'*bench_w}  {'-'*status_w}  {'-'*40}")

    for b, s, d in status_rows:
        print(f"{b:<{bench_w}}  {s:<{status_w}}  {d}")

    # Counts
    counts: Dict[str, int] = {}
    for _, s, _ in status_rows:
        counts[s] = counts.get(s, 0) + 1

    print("\nCounts:")
    for k in ["passed", "cycle different", "benchmark doesn't exist"]:
        if k in counts:
            print(f"  {k}: {counts[k]}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Compare one CSV against an embedded expected results table and print per-benchmark status."
    )
    ap.add_argument("csv", help="Input CSV path (only one file)")
    args = ap.parse_args()

    expected = normalize(build_expected_df())
    actual = normalize(build_actual_df_from_csv(args.csv))

    # Index by benchmark for quick lookup
    exp_map = {r["benchmark"]: r for _, r in expected.iterrows()}
    act_map = {r["benchmark"]: r for _, r in actual.iterrows()}

    all_benchmarks = sorted(set(exp_map.keys()) | set(act_map.keys()))

    status_rows: List[Tuple[str, str, str]] = []
    overall_ok = True

    for b in all_benchmarks:
        in_exp = b in exp_map
        in_act = b in act_map

        if not in_exp or not in_act:
            overall_ok = False
            where = []
            if in_exp and not in_act:
                where.append("missing in CSV")
            if in_act and not in_exp:
                where.append("extra in CSV")
            status_rows.append((b, "benchmark doesn't exist", ", ".join(where)))
            continue

        diffs = diff_columns(exp_map[b], act_map[b])
        if not diffs:
            status_rows.append((b, "passed", ""))
        else:
            overall_ok = False
            # User requested category name "cycle different".
            # We still list which columns differ for debugging.
            status_rows.append((b, "cycle different", "diff: " + ", ".join(diffs)))

    print_status_table(status_rows)

    if overall_ok:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())