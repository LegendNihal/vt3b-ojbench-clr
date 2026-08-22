#!/usr/bin/env python3
"""
Prove the execution gate actually fires, without spending a minute of GPU time.

The gate has never been observed working, because every smoke run so far failed
to produce a compilable program for a task that had samples. That confounds two
questions: "is the gate wired correctly?" and "can the model write code?".

This answers the first one alone. It builds synthetic candidates for tasks whose
samples were parsed -- one that prints the known-correct output, one that prints
junk -- runs the real stage 2 and stage 3, and checks that the correct one is
gated in and the wrong one is gated out.

    python test_gate.py --samples work/samples_fixed.jsonl --prompts full.jsonl

This is a plumbing test. It says nothing about benchmark performance.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from common import read_jsonl, task_key, write_jsonl

HERE = Path(__file__).resolve().parent


def py_correct(samples):
    table = {s["input"].strip(): s["output"] for s in samples}
    return ("import sys\n"
            f"table = {table!r}\n"
            "data = sys.stdin.read().strip()\n"
            "sys.stdout.write(table.get(data, 'UNKNOWN-INPUT'))\n")


def py_wrong():
    return "import sys\nsys.stdin.read()\nsys.stdout.write('definitely-not-the-answer\\n')\n"


def py_broken():
    return "def main(:\n    this is not python\n"


def cpp_correct(samples):
    out = samples[0]["output"]
    return ('#include <cstdio>\nint main(){ fputs(R"VTG(' + out + ')VTG", stdout); return 0; }\n')


def cpp_wrong():
    return '#include <cstdio>\nint main(){ fputs("definitely-not-the-answer\\n", stdout); return 0; }\n'


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--samples", default="work/samples.jsonl")
    p.add_argument("--prompts", default="full.jsonl")
    p.add_argument("--work", default="work/gatetest")
    p.add_argument("--n-tasks", type=int, default=6)
    a = p.parse_args()

    W = Path(a.work)
    W.mkdir(parents=True, exist_ok=True)

    tasks = {task_key(t): t for t in read_jsonl(a.prompts)}
    have = [r for r in read_jsonl(a.samples) if r.get("samples")]
    if not have:
        sys.exit("no parsed samples to test with")

    picked, cands = [], []
    for r in have:
        key = r["key"]
        t = tasks.get(key)
        if not t:
            continue
        lang = t["language"]
        if lang == "cpp" and len(r["samples"]) != 1:
            continue                      # the trivial cpp solver only handles one case
        variants = ([py_correct(r["samples"]), py_wrong(), py_broken()] if lang == "python"
                    else [cpp_correct(r["samples"]), cpp_wrong()])
        for i, code in enumerate(variants):
            cands.append({"key": key, "id": t["id"], "language": lang,
                          "dataset": t.get("dataset"), "difficulty": t.get("difficulty"),
                          "idx": i, "code": code, "tail": "synthetic",
                          "finish": "stop", "n_tokens": 100})
        picked.append((key, lang, len(variants)))
        if len(picked) >= a.n_tasks:
            break

    if not picked:
        sys.exit("could not build any synthetic candidates")
    write_jsonl(W / "cand.jsonl", cands)
    print(f"[setup] {len(picked)} tasks, {len(cands)} synthetic candidates "
          f"(index 0 = correct, the rest are wrong or broken)\n")

    def run(script, *args):
        r = subprocess.run([sys.executable, str(HERE / script), *map(str, args)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-3000:]); print(r.stderr[-3000:])
            sys.exit(f"{script} failed")
        return r.stdout

    print(run("2_verify_samples.py", "--prompts", a.prompts,
              "--candidates", W / "cand.jsonl", "--out", W / "verify.jsonl",
              "--samples-out", W / "s.jsonl", "--fresh")[-700:])
    run("3_clr_rank.py", "--prompts", a.prompts, "--candidates", W / "cand.jsonl",
        "--verify", W / "verify.jsonl", "--out", W / "sel.jsonl", "--no-llm")

    ver = {(r["key"], r["idx"]): r for r in read_jsonl(W / "verify.jsonl")}
    sel = {s["key"]: s for s in read_jsonl(W / "sel.jsonl")}

    print("=" * 68)
    print(f"{'task':<22}{'correct':>9}{'wrong':>9}{'mode':>12}{'chose':>8}")
    print("-" * 68)
    ok = True
    for key, lang, n in picked:
        c = ver.get((key, 0), {})
        w = ver.get((key, 1), {})
        s = sel.get(key, {})
        c_pass, w_pass = c.get("all_pass"), w.get("all_pass")
        mode, chosen = s.get("mode"), s.get("chosen_idx")
        good = (c_pass is True and w_pass is False and mode == "gated" and chosen == 0)
        ok &= good
        print(f"{key[:21]:<22}{str(c_pass):>9}{str(w_pass):>9}{str(mode):>12}"
              f"{str(chosen):>8}{'' if good else '   <-- FAIL'}")
    print("-" * 68)

    if ok:
        print("PASS. The gate runs candidates, passes the correct one, rejects the wrong\n"
              "one, reports mode=gated, and selects index 0. Any 'gated: 0' you see in a\n"
              "real run is the model failing to produce compilable code, not the gate.")
    else:
        print("FAIL. Expected every row: correct=True, wrong=False, mode=gated, chose=0.\n"
              "  correct=False -> parsed expected output does not match what the program\n"
              "                   prints; run check_samples.py and inspect that task\n"
              "  wrong=True    -> outputs_match is too lenient\n"
              "  mode!=gated   -> stage 3 is not routing sample-passing tasks correctly")
        sys.exit(1)


if __name__ == "__main__":
    main()