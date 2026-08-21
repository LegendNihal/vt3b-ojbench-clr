#!/usr/bin/env python3
"""
Stage 3 -- Claim-Level Reliability Assessment, adapted to code.

The paper applies CLR to answer-verifiable maths: sample K traces, pull M=5
decision-relevant claims out of each, have the model try to falsify them, score
each trace r_k = (mean verdict)^M, cluster answers by equivalence, and take the
cluster with the largest summed reliability.

Code needs three changes:

  * "answer equivalence" is behavioural, not textual. Two programs belong to the
    same cluster when they print the same thing on every input we ran (stage 2's
    signature), not when their source matches.
  * code comes with a free, sound verifier the maths setting lacks: the sample
    tests in the statement. A candidate that fails those is dead regardless of
    how confident the model is about it, so execution gates the claim score
    rather than competing with it.
  * the M=5 claims are pinned to the five axes that actually decide OJ verdicts
    -- complexity vs. limits, the central invariant, boundary cases, the I/O
    contract, and implementation limits -- instead of being free-form. A model
    asked "is this right?" says yes; asked "does O(n^2) fit n <= 2e5?" it can
    check.

    python 3_clr_rank.py --prompts OJBench_testdata/prompts/full.jsonl \
                         --candidates work/candidates.jsonl \
                         --verify work/verify.jsonl \
                         --out work/selection.jsonl
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from common import (load_tokenizer, parse_last_json, parse_verdicts, read_jsonl,
                    render_chat, task_key, write_jsonl)

M_CLAIMS = 5

AXES = """A1 COMPLEXITY  -- the time and memory complexity of this code, and why it fits the stated limits at the maximum input size.
A2 ALGORITHM    -- the central invariant, recurrence, or exchange argument the code relies on, and why it is valid.
A3 BOUNDARIES   -- how the code behaves on the extreme inputs the constraints allow (minimum sizes, all-equal values, empty cases, maximum values).
A4 I/O CONTRACT -- exactly how it reads input and formats output: multi-test handling, modulus, precision, separators, trailing output.
A5 LIMITS       -- overflow and integer width, recursion depth, memory footprint, and other language-specific limits."""

EXTRACT_TMPL = """You are reviewing a candidate solution to a competitive programming problem.

<problem>
{problem}
</problem>

<candidate language="{lang}">
{code}
</candidate>

Write exactly five claims that this candidate's correctness depends on, one per axis:

{axes}

Each claim must be specific and checkable against THIS code and THIS problem: name the actual complexity, the actual limits from the statement, the actual variables or formulas. "The algorithm is correct" is useless. Do not hedge, do not add caveats -- state each claim as a flat assertion so it can be tested.

Reply with ONLY this JSON object and nothing else:
{{"claims": ["A1 ...", "A2 ...", "A3 ...", "A4 ...", "A5 ..."]}}"""

VERIFY_TMPL = """You are stress-testing a candidate solution to a competitive programming problem. Your job is to BREAK it.

<problem>
{problem}
</problem>

<candidate language="{lang}">
{code}
</candidate>

<claims>
{claims}
</claims>

Take each claim in turn and try hard to falsify it. Look for a concrete counterexample input allowed by the constraints, an off-by-one, an overflow, a recursion depth or memory blow-up, an output format mismatch, or a complexity that is too slow at the maximum input size. Work out the actual numbers -- if the limit is n <= 200000 and the loop is quadratic, that is 4*10^10 operations and the claim is false.

Mark a claim 1 only if you genuinely could not break it. Mark it 0 if you found a concrete failure. When in doubt, mark 0.

Reply with ONLY this JSON object and nothing else:
{{"verdicts": [0 or 1, 0 or 1, 0 or 1, 0 or 1, 0 or 1], "worst_issue": "one sentence, or empty if none"}}"""


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prompts", default="OJBench_testdata/prompts/full.jsonl")
    p.add_argument("--candidates", default="work/candidates.jsonl")
    p.add_argument("--verify", default="work/verify.jsonl")
    p.add_argument("--out", default="work/selection.jsonl")
    p.add_argument("--trace-out", default="work/clr_trace.jsonl")
    p.add_argument("--model", default="WeiboAI/VibeThinker-3B")

    p.add_argument("--top-n", type=int, default=6,
                   help="distinct programs per task that get the full CLR treatment")
    p.add_argument("--problem-chars", type=int, default=9000, help="statement budget in the CLR prompt")
    p.add_argument("--code-chars", type=int, default=7000)
    p.add_argument("--extract-tokens", type=int, default=8192)
    p.add_argument("--verify-tokens", type=int, default=12288)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=-1,
                   help="-1 = off (the paper's setting). Set this if ablate.py says so")
    p.add_argument("--min-p", type=float, default=0.0)
    p.add_argument("--repetition-penalty", type=float, default=1.0)

    p.add_argument("--seed", type=int, default=7)

    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--gpu-mem-util", type=float, default=0.90)
    p.add_argument("--max-num-seqs", type=int, default=64)
    p.add_argument("--kv-cache-dtype", default="auto")
    p.add_argument("--no-prefix-caching", action="store_true")
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--cuda-graphs", action="store_true",
                   help="re-enable CUDA graphs (~40%% faster). OFF by default: graph replay "
                        "corrupts long generations on vllm 0.6.3 + Ada. See ablate.py")
    p.add_argument("--no-async-output", action="store_true",
                   help="disable async output processing. Try --cuda-graphs together with "
                        "this if ablate.py says async output was the broken half")


    # scoring knobs
    p.add_argument("--gate-pass", type=float, default=1.0, help="weight when all samples pass")
    p.add_argument("--gate-nosamples", type=float, default=0.6, help="weight when no samples exist")
    p.add_argument("--gate-fail", type=float, default=0.12,
                   help="weight for sample-failing candidates (only used if nothing passes)")
    p.add_argument("--stress-floor", type=float, default=0.30,
                   help="gate multiplier for a candidate that crashes on every valid "
                        "generated input (1.0 = ignore generated-input evidence)")
    p.add_argument("--alpha", type=float, default=0.25, help="self-consistency bonus per cluster")
    p.add_argument("--parse-fail-score", type=float, default=0.30)
    p.add_argument("--no-llm", action="store_true",
                   help="skip the model entirely: execution gate + self-consistency only")
    return p.parse_args()


def clip(s, n):
    s = s or ""
    if len(s) <= n:
        return s
    return s[: n * 2 // 3] + "\n\n...[statement truncated]...\n\n" + s[-n // 3:]


def main():
    a = build_args()
    tasks = {task_key(t): t for t in read_jsonl(a.prompts)}
    cands = read_jsonl(a.candidates)
    vmap = {(r["key"], r["idx"]): r for r in read_jsonl(a.verify)}

    by_task = defaultdict(list)
    for c in cands:
        by_task[c["key"]].append(c)

    # ------------------------------------------------------------------
    # 1. build the pool per task: gate on execution, dedupe, keep top-N
    # ------------------------------------------------------------------
    plans = {}
    for key, group in by_task.items():
        alive = []
        for c in group:
            v = vmap.get((key, c["idx"]))
            if not c.get("code") or not v or v["code_hash"] == "NOCODE":
                continue
            if v["compile_error"]:
                continue
            alive.append((c, v))
        if not alive:
            plans[key] = {"pool": [], "mode": "empty", "clusters": {}}
            continue

        has_samples = any(v["n_total"] > 0 for _, v in alive)
        passing = [(c, v) for c, v in alive if v["all_pass"]]
        if passing:
            pool, mode, gate = passing, "gated", a.gate_pass
        elif not has_samples:
            pool, mode, gate = alive, "nosamples", a.gate_nosamples
        else:
            pool, mode, gate = alive, "degraded", a.gate_fail

        # A generated input that almost everyone crashes on is a bad input, not a
        # bad program, so only inputs the majority survives count as evidence.
        n_stress = max((v.get("n_stress", 0) for _, v in pool), default=0)
        stress_ratio = {}
        if n_stress:
            need = max(1, len(pool) // 2)
            valid = [j for j in range(n_stress)
                     if sum(1 for _, v in pool
                            if (v.get("stress_status") or [None] * n_stress)[j] == "ok") >= need]
            for c, v in pool:
                st = v.get("stress_status") or []
                if not valid:
                    stress_ratio[c["idx"]] = 1.0
                else:
                    ok = sum(1 for j in valid if j < len(st) and st[j] == "ok")
                    stress_ratio[c["idx"]] = ok / len(valid)

        clusters = defaultdict(list)          # behavioural signature -> [idx...]
        for c, v in pool:
            clusters[v["signature"]].append(c["idx"])

        # one representative per behaviour, biggest clusters first, shortest code wins ties
        by_hash = {}
        for c, v in pool:
            h = v["code_hash"]
            if h not in by_hash or len(c["code"]) < len(by_hash[h][0]["code"]):
                by_hash[h] = (c, v)
        reps = sorted(by_hash.values(),
                      key=lambda cv: (-len(clusters[cv[1]["signature"]]), len(cv[0]["code"])))
        plans[key] = {"pool": reps[: a.top_n], "mode": mode, "gate": gate,
                      "clusters": dict(clusters), "pool_size": len(pool),
                      "stress_ratio": stress_ratio, "n_stress": n_stress}

    n_jobs = sum(len(p["pool"]) for p in plans.values())
    print(f"[plan] {len(plans)} tasks, {n_jobs} programs to assess "
          f"(<= {a.top_n} per task, 2 model calls each)")
    for m in ("gated", "nosamples", "degraded", "empty"):
        n = sum(1 for p in plans.values() if p["mode"] == m)
        print(f"        {m:>10}: {n}")

    # ------------------------------------------------------------------
    # 2. CLR: extract claims, then try to falsify them
    # ------------------------------------------------------------------
    reliability = {}
    traces = []

    if not a.no_llm and n_jobs:
        from vllm import LLM, SamplingParams

        tok = load_tokenizer(a.model)
        llm = LLM(model=a.model, dtype="bfloat16", max_model_len=a.max_model_len,
                  gpu_memory_utilization=a.gpu_mem_util, max_num_seqs=a.max_num_seqs,
                  swap_space=4, enable_prefix_caching=not a.no_prefix_caching,
                  enforce_eager=not a.cuda_graphs,
                  disable_async_output_proc=a.no_async_output,
                  kv_cache_dtype=a.kv_cache_dtype, trust_remote_code=True, seed=a.seed)

        def chat(prompts, max_tokens):
            texts = [render_chat(tok, p) for p in prompts]
            sps = []
            for t in texts:
                room = a.max_model_len - len(tok(t)["input_ids"]) - 16
                sps.append(SamplingParams(n=1, temperature=a.temperature, top_p=a.top_p,
                                          top_k=a.top_k, min_p=a.min_p,
                                          repetition_penalty=a.repetition_penalty,
                                          max_tokens=max(256, min(max_tokens, room)),
                                          seed=a.seed))
            return [o.outputs[0].text for o in llm.generate(texts, sps)]

        jobs = [(key, c, v) for key, p in plans.items() for c, v in p["pool"]]

        print("\n[clr] pass 1/2 -- extracting claims")
        p1 = [EXTRACT_TMPL.format(problem=clip(tasks[k]["prompt"], a.problem_chars),
                                  lang=c["language"], code=clip(c["code"], a.code_chars),
                                  axes=AXES) for k, c, v in jobs]
        r1 = chat(p1, a.extract_tokens)

        claim_sets = []
        for txt in r1:
            js = parse_last_json(txt) or {}
            cl = js.get("claims")
            if not isinstance(cl, list):
                cl = []
            cl = [str(x) for x in cl][:M_CLAIMS]
            claim_sets.append(cl)
        print(f"      parsed claims for {sum(1 for c in claim_sets if c)}/{len(claim_sets)}")

        print("\n[clr] pass 2/2 -- falsification")
        p2 = []
        for (k, c, v), cl in zip(jobs, claim_sets):
            body = "\n".join(f"{i+1}. {x}" for i, x in enumerate(cl)) if cl else \
                   "1. A1 the complexity fits the limits\n2. A2 the algorithm is correct\n" \
                   "3. A3 boundary cases are handled\n4. A4 the I/O format matches\n" \
                   "5. A5 no overflow or depth limit is exceeded"
            p2.append(VERIFY_TMPL.format(problem=clip(tasks[k]["prompt"], a.problem_chars),
                                         lang=c["language"], code=clip(c["code"], a.code_chars),
                                         claims=body))
        r2 = chat(p2, a.verify_tokens)

        n_parsed = 0
        for (k, c, v), cl, txt in zip(jobs, claim_sets, r2):
            vs, worst = parse_verdicts(txt, M_CLAIMS)
            if vs is not None:
                # Eq. 5: r = (mean v)^M -- one broken claim collapses the trace
                r = (sum(vs) / M_CLAIMS) ** M_CLAIMS
                n_parsed += 1
                ok = True
            else:
                vs, r, ok = [], a.parse_fail_score, False
            reliability[(k, c["idx"])] = r
            traces.append({"key": k, "idx": c["idx"], "claims": cl, "verdicts": vs,
                           "parsed": ok, "reliability": round(r, 4),
                           "worst_issue": worst})
        print(f"      parsed verdicts for {n_parsed}/{len(r2)}")
        write_jsonl(a.trace_out, traces)
    else:
        for key, p in plans.items():
            for c, v in p["pool"]:
                reliability[(key, c["idx"])] = 1.0

    # ------------------------------------------------------------------
    # 3. reliability-weighted cluster vote
    # ------------------------------------------------------------------
    cand_by = {(c["key"], c["idx"]): c for c in cands}
    selections, stats = [], defaultdict(int)

    for key, p in plans.items():
        stats[p["mode"]] += 1
        if p["mode"] == "empty":
            fallback = next((c for c in by_task[key] if c.get("code")), None)
            if fallback is None:
                fallback = by_task[key][0] if by_task[key] else None
            selections.append({"key": key, "id": tasks[key]["id"],
                               "language": tasks[key]["language"],
                               "chosen_idx": fallback["idx"] if fallback else None,
                               "mode": "empty", "score": 0.0, "reliability": 0.0,
                               "n_candidates": len(by_task[key])})
            continue

        gate = p["gate"]
        scored = defaultdict(float)
        for c, v in p["pool"]:
            sr = p["stress_ratio"].get(c["idx"], 1.0) if p["n_stress"] else 1.0
            g = gate * (a.stress_floor + (1.0 - a.stress_floor) * sr)
            r = g * reliability.get((key, c["idx"]), 0.0)
            scored[v["signature"]] += r
        k_total = max(len(by_task[key]), 1)
        for sig, members in p["clusters"].items():
            scored[sig] += a.alpha * len(members) / k_total   # self-consistency bonus

        best_sig = max(scored, key=lambda s: scored[s])
        in_best = [(c, v) for c, v in p["pool"] if v["signature"] == best_sig]
        if not in_best:                       # cluster won on consistency alone
            best_idx = p["clusters"][best_sig][0]
            best_r = 0.0
        else:
            c, v = max(in_best, key=lambda cv: (reliability.get((key, cv[0]["idx"]), 0.0),
                                                -len(cv[0]["code"])))
            best_idx, best_r = c["idx"], reliability.get((key, c["idx"]), 0.0)

        selections.append({
            "key": key, "id": tasks[key]["id"], "language": tasks[key]["language"],
            "chosen_idx": best_idx, "mode": p["mode"],
            "score": round(scored[best_sig], 4), "reliability": round(best_r, 4),
            "cluster_size": len(p["clusters"].get(best_sig, [])),
            "n_clusters": len(p["clusters"]), "n_candidates": len(by_task[key]),
            "assessed": len(p["pool"]), "n_stress": p["n_stress"],
            "stress_ratio": round(p["stress_ratio"].get(best_idx, 1.0), 3)
                            if p["n_stress"] else None,
        })

    write_jsonl(a.out, selections)
    print(f"\n[done] {a.out}  ({len(selections)} tasks)")
    for m, n in sorted(stats.items()):
        print(f"   {m:>10}: {n}")
    print("   mode meanings: gated = a candidate passed the statement samples; "
          "nosamples = no samples parsed, CLR only; degraded = none passed, best effort; "
          "empty = nothing compiled")


if __name__ == "__main__":
    main()