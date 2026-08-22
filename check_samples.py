#!/usr/bin/env python3
"""
Check whether the sample tests parsed out of statements are trustworthy.

The execution gate rejects any candidate that fails these, so a wrong expected
output is worse than no sample at all: it throws away correct programs and
quietly drops the task into `degraded`. This runs no model and takes seconds.

    python check_samples.py --samples work/smoke/samples.jsonl --prompts full.jsonl

Read the flag counts, then eyeball the printed pairs against the real statements.
"""
import argparse
import re
from collections import Counter

from common import read_jsonl

CJK = re.compile(r"[\u4e00-\u9fff]")
CODE = re.compile(r"(def\s+main\s*\(|if\s+__name__|#include\s*<|int\s+main\s*\(|"
                  r"<\s*Your code is here\s*>|public\s+static\s+void)")
PLACEHOLDER = re.compile(r"(<\s*insert[^>]*>|\[\s*Example\s+(Input|Output)\s*\]|"
                         r"\[\s*your code here\s*\])", re.I)
MARKUP = re.compile(r"(^|\n)\s*(#{1,6}\s|\*\*|\|\s*-{2,}|```|\$\$)")
ENGLISH = re.compile(r"\b(the|is|are|of|and|for|with|that|this|which|should|must|"
                     r"means|where|each|than|then|note|explanation|example)\b", re.I)


def flags_for(inp, out):
    f = []
    if not inp.strip() or not out.strip():
        f.append("empty")
    # the failure that slipped past the first version of this script: the parser
    # paired a real sample input with the ```python skeleton from the prompt
    if CODE.search(out):
        f.append("EXPECTED_OUTPUT_IS_CODE")
    if PLACEHOLDER.search(inp) or PLACEHOLDER.search(out):
        f.append("PLACEHOLDER_NOT_A_REAL_SAMPLE")
    for name, txt in (("input", inp), ("output", out)):
        if txt.strip() and not re.search(r"[A-Za-z0-9]", txt):
            f.append(f"{name}_is_markup_only")
    if len(out) > 5000:
        f.append("output_huge")
    if len(inp) > 20000:
        f.append("input_huge")
    if MARKUP.search(out) or MARKUP.search(inp):
        f.append("contains_markup")
    for name, txt in (("input", inp), ("output", out)):
        for line in txt.split("\n"):
            words = line.split()
            if len(words) > 8 and len(ENGLISH.findall(line)) >= 3:
                f.append(f"{name}_looks_like_prose")
                break
    if CJK.search(out):
        f.append("output_has_cjk")
    if re.search(r"[A-Za-z]{2,}.*[.。]\s*$", out.strip()) and len(out.split()) > 6:
        f.append("output_ends_like_sentence")
    return f


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--samples", default="work/samples.jsonl")
    p.add_argument("--prompts", default="full.jsonl")
    p.add_argument("--show", type=int, default=4, help="pairs to print for eyeballing")
    p.add_argument("--show-suspicious", type=int, default=3)
    a = p.parse_args()

    rows = [r for r in read_jsonl(a.samples) if r.get("samples")]
    prompts = {str(t["id"]): t.get("prompt", "") for t in read_jsonl(a.prompts)}
    if not rows:
        print("no parsed samples at all -- the gate is inactive everywhere")
        return

    counts, suspicious, clean = Counter(), [], []
    n_cases = 0
    for r in rows:
        bad = []
        for s in r["samples"]:
            n_cases += 1
            f = flags_for(s.get("input", ""), s.get("output", ""))
            for x in f:
                counts[x] += 1
            if f:
                bad.append((s, f))
        (suspicious if bad else clean).append((r, bad))

    severe = {"EXPECTED_OUTPUT_IS_CODE", "PLACEHOLDER_NOT_A_REAL_SAMPLE"}
    print(f"tasks with parsed samples : {len(rows)}")
    print(f"sample cases total        : {n_cases}")
    print(f"tasks with no flags       : {len(clean)}")
    print(f"tasks with >=1 flag       : {len(suspicious)}")
    if counts:
        print("\nflags raised:")
        for k, v in counts.most_common():
            print(f"  {v:5d}  {k}")
    else:
        print("\nno heuristic flags raised")

    print("\n" + "=" * 72)
    print("EYEBALL THESE. Heuristics cannot tell you an expected output is *correct*,")
    print("only that it does not look like prose. Compare against the real statement.")
    print("=" * 72)

    ranked = sorted(suspicious, key=lambda rb: -sum(
        1 for _, f in rb[1] for x in f if x in severe))
    suspicious = ranked

    for r, _ in clean[: a.show]:
        s = r["samples"][0]
        print(f"\n--- id={r['id']}  (no flags, {len(r['samples'])} case(s)) ---")
        print(f"INPUT   {s['input'].strip()[:300]!r}")
        print(f"EXPECT  {s['output'].strip()[:300]!r}")

    for r, bad in suspicious[: a.show_suspicious]:
        s, f = bad[0]
        print(f"\n--- id={r['id']}  FLAGS: {f} ---")
        print(f"INPUT   {s['input'].strip()[:300]!r}")
        print(f"EXPECT  {s['output'].strip()[:300]!r}")
        stmt = prompts.get(str(r["id"]), "")
        idx = stmt.find(s["output"].strip()[:40]) if s["output"].strip() else -1
        if idx > 0:
            print(f"CONTEXT ...{stmt[max(0, idx-200):idx+200]}...")

    hard = sum(counts[k] for k in severe if k in counts)
    if hard:
        print(f"\n!! {hard} case(s) are definitely wrong, not merely suspicious.")
        print("!! Do NOT run with these: the gate would reject correct programs.")

    print("\n" + "=" * 72)
    print("If the EXPECT values look like real program output (numbers, short tokens),")
    print("the gate is safe. If any is a sentence, a heading, or LaTeX, fix the parser")
    print("in common.py or pass --samples-override to stage 2.")


if __name__ == "__main__":
    main()