#!/usr/bin/env python3
"""
Diagnose the two things that go wrong first: responses with no code block, and
statements whose sample tests would not parse.

    python diagnose.py --candidates work/smoke_cand.jsonl --prompts full.jsonl
"""
import argparse
import collections
import json
import re

from common import code_looks_valid, parse_samples, read_jsonl, task_key


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prompts", default="full.jsonl")
    p.add_argument("--candidates", default=None)
    p.add_argument("--show", type=int, default=2, help="how many failing examples to print")
    p.add_argument("--tail-chars", type=int, default=1500)
    return p.parse_args()


def diagnose_candidates(path, show, tail_chars):
    rows = read_jsonl(path)
    if not rows:
        print("no candidates found")
        return
    print("=" * 72)
    print("A. GENERATION")
    print("=" * 72)

    fin = collections.Counter(r.get("finish") for r in rows)
    both = collections.Counter((r.get("finish"), r.get("code") is not None) for r in rows)
    toks = sorted(r.get("n_tokens", 0) for r in rows)

    nvalid = sum(1 for r in rows
                 if r.get("valid", code_looks_valid(r.get("code"), r.get("language", "python"))))
    ncode = sum(1 for r in rows if r.get("code"))
    print(f"  {len(rows)} samples")
    print(f"  USABLE programs       : {nvalid}/{len(rows)} ({100*nvalid/len(rows):.0f}%)"
          f"   <- the number that matters")
    print(f"  had any code block    : {ncode}/{len(rows)}"
          f"   (the gap is fragments salvaged from truncated fences)")
    print(f"  finish reasons        : {dict(fin)}")
    print(f"  (finish, has_code)    : {dict(both)}")
    print(f"  tokens min/med/max    : {toks[0]} / {toks[len(toks)//2]} / {toks[-1]}")
    n_cap = sum(1 for r in rows if r.get("finish") == "length")
    print(f"  hit the token cap     : {n_cap}/{len(rows)} "
          f"({100*n_cap/len(rows):.0f}%)")

    done = sorted(r.get("n_tokens", 0) for r in rows
                  if r.get("finish") == "stop" and r.get("code"))
    if done:
        def pct(q):
            return done[min(len(done) - 1, int(q * len(done)))]
        print(f"\n  tokens used by rollouts that FINISHED with code ({len(done)} of them):")
        print(f"    p50 {pct(.5)}   p75 {pct(.75)}   p90 {pct(.9)}   max {done[-1]}")
        print("""
  This is the number that sets your token budget. A cap a little above p90 keeps
  almost every rollout that was ever going to succeed, and every token beyond it
  is spent on rollouts that will be thrown away. Raising the cap past this point
  buys very little and costs proportionally.""")
        waste = sum(r.get("n_tokens", 0) for r in rows if r.get("finish") == "length")
        total = sum(r.get("n_tokens", 0) for r in rows)
        print(f"  tokens spent on truncated rollouts: {waste}/{total} "
              f"({100*waste/max(total,1):.0f}% of all compute)")

        cap_now = max(r.get("n_tokens", 0) for r in rows)
        print(f"""
  Yield curve. What WOULD have happened at a smaller cap, computed exactly from
  these rollouts: a rollout that finished at 20k yields nothing under a 16k cap
  but still costs 16k. Pick the cap with the lowest tokens/usable -- that is the
  cheapest way to buy a candidate. If the minimum sits at the current cap, the
  cap is still binding and trimming will not help.""")
        print(f"\n  {'cap':>8}{'usable':>9}{'Mtokens':>10}{'tokens/usable':>16}")
        best = None
        for frac in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
            C = int(cap_now * frac)
            ok = sum(1 for r in rows if r.get("finish") == "stop" and r.get("code")
                     and r.get("n_tokens", 0) <= C)
            cost = sum(min(r.get("n_tokens", 0), C) for r in rows)
            per = cost / ok if ok else float("inf")
            mark = ""
            if ok and (best is None or per < best[1]):
                best, mark = (C, per), ""
            print(f"  {C:>8}{ok:>9}{cost/1e6:>10.2f}{(per if ok else 0):>16.0f}"
                  f"{'  <- cheapest' if ok and per == min(per, best[1]) else ''}")
        if best:
            print(f"\n  cheapest cap in this range: {best[0]} at {best[1]:.0f} tokens/usable")
            if best[0] >= cap_now:
                print("  -> the minimum is at your current cap, so the cap is still binding. "
                      "Raise it if you have context headroom; do not trim.")

    print("""
  How to read this:
    finish='length' + no code  -> the cap cut it off mid-thought. Raise
                                  --max-new-tokens (and --max-model-len).
    finish='stop'   + no code  -> it finished but never emitted a fence. Look at
                                  the tail below: either it answered in prose, or
                                  it used a format extract_code does not match.""")

    bad = [r for r in rows if r.get("code") is None]
    for r in bad[:show]:
        print("\n" + "-" * 72)
        print(f"NO CODE: {r['key']} idx={r['idx']} finish={r.get('finish')} "
              f"tokens={r.get('n_tokens')}")
        print("-" * 72)
        tail = r.get("tail") or ""
        print(f"...{tail[-tail_chars:]}")
        print("-" * 72)
        print(f"backticks anywhere in the tail: {tail.count('`')}")


def diagnose_samples(path, show):
    tasks = read_jsonl(path)
    print("\n" + "=" * 72)
    print("B. SAMPLE TEST PARSING")
    print("=" * 72)

    by_id, cov = {}, collections.Counter()
    for t in tasks:
        if t["id"] in by_id:
            continue
        by_id[t["id"]] = t
        s = parse_samples(t.get("prompt", ""))
        cov[(t.get("dataset"), bool(s))] += 1

    for ds in sorted({t.get("dataset") for t in tasks}):
        ok, no = cov[(ds, True)], cov[(ds, False)]
        print(f"  {ds}: {ok}/{ok+no} problems parsed ({100*ok/max(ok+no,1):.0f}%)")

    # what do the headings actually look like?
    heads = collections.Counter()
    for t in by_id.values():
        for line in (t.get("prompt") or "").split("\n"):
            ln = line.strip()
            if not ln or len(ln) > 60:
                continue
            if re.match(r"^(#{1,6}\s|\*\*|【|\[)", ln) or re.match(r"^[A-Z][A-Za-z ]{2,30}:?$", ln):
                heads[re.sub(r"\d+", "N", ln)] += 1
    print("\n  most common heading lines across all statements:")
    for h, n in heads.most_common(25):
        print(f"    {n:5d}  {h}")

    failing = [t for t in by_id.values() if not parse_samples(t.get("prompt", ""))]
    print(f"\n  {len(failing)} problems failed to parse. Tail of the first {show}:")
    for t in failing[:show]:
        print("\n" + "-" * 72)
        print(f"id={t['id']} dataset={t.get('dataset')}")
        print("-" * 72)
        print("..." + (t.get("prompt") or "")[-1800:])


def main():
    a = build_args()
    if a.candidates:
        diagnose_candidates(a.candidates, a.show, a.tail_chars)
    diagnose_samples(a.prompts, a.show)
    print("\n" + "=" * 72)
    print("Paste section A's counters and one NO CODE tail, plus section B's "
          "heading list, if you want me to look at it.")


if __name__ == "__main__":
    main()