#!/usr/bin/env python3
"""
Stage 2 -- execution gate. CPU only, no GPU needed.

For every candidate we (a) compile it and (b) run it on the sample tests printed
in the problem statement, then record a behavioural signature (a hash of what it
printed on every input we ran).

    python 2_verify_samples.py --prompts OJBench_testdata/prompts/full.jsonl \
                               --candidates work/candidates.jsonl \
                               --out work/verify.jsonl --workers 8

IMPORTANT: only statement samples are used. OJBench's hidden testdata under
NOI/ and ICPC/ is never read here -- selecting on hidden tests would invalidate
the benchmark result.
"""
import argparse
import os
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from common import (append_jsonl, code_hash, find_pypy3, parse_samples, read_jsonl,
                    run_candidate_on_tests, task_key, write_jsonl)


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prompts", default="OJBench_testdata/prompts/full.jsonl")
    p.add_argument("--candidates", default="work/candidates.jsonl")
    p.add_argument("--out", default="work/verify.jsonl")
    p.add_argument("--samples-out", default="work/samples.jsonl")
    p.add_argument("--samples-override", default=None,
                   help="jsonl of {id, samples:[{input,output}]} to use instead of parsing")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--timeout", type=float, default=10.0, help="seconds per sample test")
    p.add_argument("--mem-mb", type=int, default=1024)
    p.add_argument("--tmp", default=None, help="scratch dir (default: system temp)")
    p.add_argument("--py-cmd", default=None, help="force a python interpreter (default: pypy3)")
    p.add_argument("--fresh", action="store_true")
    return p.parse_args()


def main():
    a = build_args()
    tasks = {task_key(t): t for t in read_jsonl(a.prompts)}
    cands = read_jsonl(a.candidates)

    # ---- 1. sample tests ---------------------------------------------------
    override = {}
    if a.samples_override:
        override = {str(r["id"]): r["samples"] for r in read_jsonl(a.samples_override)}

    samples, cov = {}, Counter()
    for k, t in tasks.items():
        s = override.get(str(t["id"])) or parse_samples(t.get("prompt", ""))
        samples[k] = s
        cov["with" if s else "without"] += 1
    write_jsonl(a.samples_out, [{"key": k, "id": tasks[k]["id"], "n": len(v), "samples": v}
                                for k, v in samples.items()])
    total = cov["with"] + cov["without"]
    print(f"[samples] parsed tests for {cov['with']}/{total} tasks "
          f"({100*cov['with']/max(total,1):.0f}%). Dumped to {a.samples_out} -- "
          f"spot-check a few before trusting the gate.")
    if cov["with"] < 0.5 * total:
        print("[warn] low coverage. Tasks without samples fall back to a weaker CLR-only "
              "gate; see the README for how to supply --samples-override.")

    py = a.py_cmd or find_pypy3()
    print(f"[runtime] python -> {py or sys.executable}"
          f"{'' if py else '  (pypy3 NOT found; install it, OJBench judges with pypy3)'}")
    print(f"[runtime] c++    -> {shutil.which('g++') or 'MISSING -- apt install g++'}")

    # ---- 2. resume ---------------------------------------------------------
    out_path = Path(a.out)
    if a.fresh and out_path.exists():
        out_path.unlink()
    done = {(r["key"], r["idx"]) for r in read_jsonl(out_path)} if out_path.exists() else set()
    todo = [c for c in cands if (c["key"], c["idx"]) not in done]
    print(f"[plan] {len(todo)} candidates to run ({len(done)} cached)")

    root = Path(a.tmp or tempfile.mkdtemp(prefix="ojclr_"))
    root.mkdir(parents=True, exist_ok=True)

    # compile/run once per distinct program, then fan the result back out
    by_prog = defaultdict(list)
    for c in todo:
        by_prog[(c["key"], code_hash(c["code"]) if c["code"] else "NOCODE")].append(c)
    print(f"[plan] {len(by_prog)} distinct programs after de-duplication")

    def work(item):
        (key, h), group = item
        if h == "NOCODE":
            res = {"compile_error": "no code block in response", "results": [],
                   "n_pass": 0, "n_total": 0, "signature": "NOCODE"}
        else:
            tests = samples.get(key, [])
            c = group[0]
            wd = root / f"{key.replace('::','_')}_{h}"
            try:
                res = run_candidate_on_tests(
                    c["code"], c["language"], tests, wd,
                    timeout_s=a.timeout, mem_mb=a.mem_mb, py_cmd=py)
            finally:
                shutil.rmtree(wd, ignore_errors=True)
        rows = []
        for c in group:
            rows.append({
                "key": key, "idx": c["idx"], "id": c["id"], "language": c["language"],
                "code_hash": h,
                "compile_error": res["compile_error"],
                "n_pass": res["n_pass"], "n_total": res["n_total"],
                "all_pass": res["n_total"] > 0 and res["n_pass"] == res["n_total"],
                "signature": res["signature"],
                "first_fail": next(({"i": i, "status": r["status"], "got": r["got"][:600]}
                                    for i, r in enumerate(res["results"]) if not r["passed"]), None),
            })
        return rows

    n_done = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(work, it) for it in by_prog.items()]
        buf = []
        for f in as_completed(futs):
            buf.extend(f.result())
            n_done += 1
            if len(buf) >= 200:
                append_jsonl(out_path, buf)
                buf = []
            if n_done % 50 == 0:
                print(f"  ... {n_done}/{len(by_prog)} programs", flush=True)
        if buf:
            append_jsonl(out_path, buf)

    # ---- 3. report ---------------------------------------------------------
    rows = read_jsonl(out_path)
    per_task = defaultdict(list)
    for r in rows:
        per_task[r["key"]].append(r)
    solved = sum(1 for v in per_task.values() if any(x["all_pass"] for x in v))
    nocode = sum(1 for r in rows if r["code_hash"] == "NOCODE")
    ce = sum(1 for r in rows if r["compile_error"] and r["code_hash"] != "NOCODE")
    print(f"\n[done] {out_path}")
    print(f"  tasks with >=1 candidate passing all statement samples: {solved}/{len(per_task)}")
    print(f"  candidates with no code block: {nocode} | compile errors: {ce}")
    print("  (this is an upper-bound sanity signal, NOT the OJBench score)")
    if not a.tmp:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
