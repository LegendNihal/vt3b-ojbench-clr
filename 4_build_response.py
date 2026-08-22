#!/usr/bin/env python3
"""
Stage 4 -- write model_response.jsonl for OJBench.

Every row keeps all the original fields from full.jsonl (id, prompt, dataset,
language, difficulty) and adds `content`. Row count and order match full.jsonl,
so nothing goes missing in the judge.

    # the CLR submission
    python 4_build_response.py --mode clr --out model_response_clr.jsonl

    # the pass@1 baseline to compare it against -- one file per rollout
    for i in 0 1 2 3 4 5 6 7; do
      python 4_build_response.py --mode single --take-idx $i --out base_$i.jsonl
    done

--content-mode code (default) writes a single clean fenced block, so OJBench's
extractor cannot pick up a throwaway snippet from the middle of the reasoning.
--content-mode raw writes the untouched generation, which needs --keep-full-text
back in stage 1.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from common import read_jsonl, task_key, wrap_as_content, write_jsonl

PLACEHOLDER = "The model did not produce a usable program for this task."

# what the OJBench README documents for a model response line
OJBENCH_FIELDS = {"id", "prompt", "dataset", "language", "difficulty", "content"}
OJBENCH_REQUIRED = {"id", "content"}      # the judge only needs these two


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prompts", default="OJBench_testdata/prompts/full.jsonl")
    p.add_argument("--candidates", default="work/candidates.jsonl")
    p.add_argument("--selection", default="work/selection.jsonl")
    p.add_argument("--out", default="model_response_clr.jsonl")
    p.add_argument("--mode", choices=["clr", "single"], default="clr")
    p.add_argument("--take-idx", type=int, default=0, help="rollout index for --mode single")
    p.add_argument("--content-mode", choices=["code", "raw"], default="code")
    p.add_argument("--schema-from", default=None,
                   help="optional: another model_response.jsonl to compare keys against")
    p.add_argument("--keep-fields", default=None,
                   help="comma-separated field whitelist (default: everything in full.jsonl)")
    return p.parse_args()


def main():
    a = build_args()
    tasks = read_jsonl(a.prompts)
    cands = {(c["key"], c["idx"]): c for c in read_jsonl(a.candidates)}

    chosen = {}
    if a.mode == "clr":
        for s in read_jsonl(a.selection):
            chosen[s["key"]] = s.get("chosen_idx")

    keep = a.keep_fields.split(",") if a.keep_fields else None
    rows, stat = [], Counter()
    by_group = Counter()

    for t in tasks:
        k = task_key(t)
        idx = chosen.get(k) if a.mode == "clr" else a.take_idx
        c = cands.get((k, idx)) if idx is not None else None

        if c and c.get("code"):
            if a.content_mode == "raw":
                content = c.get("text")
                if content is None:
                    sys.exit("[fatal] --content-mode raw needs stage 1 run with --keep-full-text")
            else:
                content = wrap_as_content(c["code"], t["language"])
            stat["ok"] += 1
            by_group[(str(t.get("dataset")), str(t.get("language")), "ok")] += 1
        else:
            content = PLACEHOLDER
            stat["missing"] += 1
            by_group[(str(t.get("dataset")), str(t.get("language")), "missing")] += 1

        row = {kk: vv for kk, vv in t.items() if (keep is None or kk in keep)}
        row["content"] = content
        rows.append(row)

    write_jsonl(a.out, rows)

    print(f"[done] {a.out}")
    print(f"  rows: {len(rows)} (full.jsonl has {len(tasks)})")
    print(f"  with a program: {stat['ok']} | placeholder: {stat['missing']}")
    groups = sorted({(d, l) for d, l, _ in by_group})
    for d, l in groups:
        ok = by_group[(d, l, "ok")]
        tot = ok + by_group[(d, l, "missing")]
        flag = "   <- ALL MISSING, check stage 1 filters" if ok == 0 else ""
        print(f"    {d}/{l}: {ok}/{tot}{flag}")
    print(f"  fields: {sorted(rows[0].keys()) if rows else '-'}")

    if rows:
        have = set(rows[0].keys())
        missing_req = OJBENCH_REQUIRED - have
        missing_doc = OJBENCH_FIELDS - have
        if missing_req:
            print(f"\n[schema] FAIL: the judge needs {sorted(missing_req)}")
        elif missing_doc:
            print(f"\n[schema] ok for judging, but the README also lists "
                  f"{sorted(missing_doc)}")
        else:
            print("\n[schema] matches the OJBench README: "
                  f"{sorted(OJBENCH_FIELDS)}")
        extra = have - OJBENCH_FIELDS
        if extra:
            print(f"[schema] extra fields carried over from full.jsonl "
                  f"(harmless): {sorted(extra)}")

    if a.schema_from and not Path(a.schema_from).exists():
        print(f"[schema] --schema-from {a.schema_from} does not exist, skipping")
    elif a.schema_from:
        ref = read_jsonl(a.schema_from)
        if ref:
            rk, mk = set(ref[0].keys()), set(rows[0].keys())
            print(f"\n[schema] your working file: {sorted(rk)}")
            if rk - mk:
                print(f"[schema] MISSING here: {sorted(rk - mk)}  <-- fix before judging")
            if mk - rk:
                print(f"[schema] extra here (usually harmless): {sorted(mk - rk)}")
            if rk == mk:
                print("[schema] exact match")

    print("\nNext: copy this to the machine with the judge and run\n"
          "  python -c \"import ojbench;from pathlib import Path;"
          "ojbench.init(problem_dirs=[Path('OJBench_testdata/NOI'),Path('OJBench_testdata/ICPC')]);"
          f"ojbench.judge_jsonl('{Path(a.out).name}','judged.jsonl',num_workers=8)\"")


if __name__ == "__main__":
    main()