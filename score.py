#!/usr/bin/env python3
"""
Summarise OJBench's judged.jsonl. Run this on the judge machine.

    # the CLR run
    python score.py judged_clr.jsonl

    # pass@1 baseline: average across the 8 single-rollout runs
    python score.py judged_base_*.jsonl

The paper's 38.6 is a pass@1 mean over independent rollouts, so compare it with
the averaged baseline, and report the CLR number separately -- exactly the way
Table 2 lists "VibeThinker-3B" and "+ CLR" as two rows.
"""
import sys
from collections import defaultdict
from statistics import mean, pstdev

from common import read_jsonl


def summarise(path):
    rows = read_jsonl(path)
    if not rows:
        return None
    buckets = defaultdict(list)
    for r in rows:
        ok = bool(r.get("is_passed", r.get("verdict") == "AC"))
        buckets["overall"].append(ok)
        for f in ("dataset", "language", "difficulty"):
            if r.get(f) is not None:
                buckets[f"{f}={r[f]}"].append(ok)
    return {k: 100.0 * sum(v) / len(v) for k, v in buckets.items()}, len(rows)


def main():
    paths = sys.argv[1:]
    if not paths:
        sys.exit("usage: python score.py judged.jsonl [more.jsonl ...]")

    runs = []
    for p in paths:
        s = summarise(p)
        if s is None:
            print(f"[warn] {p} is empty, skipping")
            continue
        scores, n = s
        runs.append(scores)
        print(f"{p}  ({n} rows)")
        for k in sorted(scores):
            print(f"    {k:<22} {scores[k]:6.2f}")
        print()

    if len(runs) > 1:
        print("=" * 46)
        print(f"mean over {len(runs)} runs")
        keys = set().union(*(r.keys() for r in runs))
        for k in sorted(keys):
            vals = [r[k] for r in runs if k in r]
            sd = f" +/- {pstdev(vals):.2f}" if len(vals) > 1 else ""
            print(f"    {k:<22} {mean(vals):6.2f}{sd}")


if __name__ == "__main__":
    main()
