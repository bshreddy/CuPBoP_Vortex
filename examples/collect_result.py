#!/usr/bin/env python3
import re
import csv
from pathlib import Path
from typing import Optional, Dict, Tuple
import argparse

# Benchmarks to be checked
benchmarks = [
    "backprop",
    "bfs",
    "btree",
    "conv3",
    "dotproduct",
    "nn",
    "pathfinder",
    "psum",
    "saxpy",
    "sgemm",
    "stencil",
    "transpose",
    "vecadd",
    "psort",
]

LINE_RE = re.compile(
    r'^PERF:\s*instrs=(?P<instrs>\d+),\s*cycles=(?P<cycles>\d+),\s*IPC=',
    re.IGNORECASE
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Example argparse script"
    )

    # suffix
    parser.add_argument("-s", "--suffix", default="test_tmp",
                        help="suffix string to lookup files")

    return parser.parse_args()

def sum_global_perf(fp: Path) -> Tuple[int, int]:
    instrs_sum = 0
    cycles_sum = 0
    with fp.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            # Skip per-core reports, only sum global lines
            if "core" in line:
                continue
            m = LINE_RE.match(line.strip())
            if m:
                instrs_sum += int(m.group("instrs"))
                cycles_sum += int(m.group("cycles"))
    return instrs_sum, cycles_sum

def classify_candidate_files(folder: Path, suffix: str) -> Dict[str, Optional[Path]]:
    """
    Classify candidate files into 4 categories:
      - off_off:  LOCAL_MEM_0 & no 'L2'
      - off_on:   LOCAL_MEM_0 & 'L2'
      - on_off:   LOCAL_MEM_1 & no 'L2'
      - on_on:    LOCAL_MEM_1 & 'L2'
    """
    results: Dict[str, Optional[Path]] = {
        "off_off": None,
        "off_on":  None,
        "on_off":  None,
        "on_on":   None,
    }

    patterns = [
        "Perf_counter_4C_16W_32T_SCHE_2_*",
    ]



    seen = []
    for pat in patterns:
        for p in folder.glob(pat):
            
            #print(p.name.lower())
            # accept *.txt and also variants like *.L2txt
            if p.suffix.lower() != ".txt" and not p.name.lower().endswith(suffix + ".txt"):
                continue
            if p in seen or not p.is_file():
                continue
            seen.append(p)
            name_lower = p.name.lower()
            l2_on = ("l2on" in name_lower)
            if "lmoff" in name_lower:
                key = "off_on" if l2_on else "off_off"
            elif "lmon" in name_lower:
                key = "on_on" if l2_on else "on_off"
            else:
                continue

            if results[key] is None:
                results[key] = p

    print(results)
    return results

def sums_from_file(p: Optional[Path]) -> Optional[Tuple[int, int]]:
    if p is None:
        return None
    instrs, cycles = sum_global_perf(p)
    return (instrs, cycles)

def main():
    root = Path.cwd()
    out_csv = root / "perf_summary_localmem_l2_sums.csv"

    args = parse_args()

    # Columns: benchmark + (instrs, cycles) x 4 conditions
    headers = [
        "benchmark",
        "Lmem_off_L2_off_instrs", "Lmem_off_L2_off_cycles",
        "Lmem_off_L2_on_instrs",  "Lmem_off_L2_on_cycles",
        "Lmem_on_L2_off_instrs",  "Lmem_on_L2_off_cycles",
        "Lmem_on_L2_on_instrs",   "Lmem_on_L2_on_cycles",
    ]

    rows = []
    for bench in benchmarks:
        folder = root / bench
        if not folder.is_dir():
            print(f"[WARN] {bench} folder does not exist, skipping")
            continue

        files = classify_candidate_files(folder, args.suffix)

        off_off = sums_from_file(files["off_off"])
        off_on  = sums_from_file(files["off_on"])
        on_off  = sums_from_file(files["on_off"])
        on_on   = sums_from_file(files["on_on"])

        def fmt_pair(v: Optional[Tuple[int, int]]):
            if v is None:
                return ["", ""]
            instrs, cycles = v
            return [str(instrs), str(cycles)]

        row = [bench]
        row += fmt_pair(off_off)
        row += fmt_pair(off_on)
        row += fmt_pair(on_off)
        row += fmt_pair(on_on)

        rows.append(row)

    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    print(f"[OK] Wrote {out_csv} ({len(rows)} benchmarks)")

if __name__ == "__main__":
    main()