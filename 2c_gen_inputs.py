#!/usr/bin/env python3
"""
Stage 2c -- generate valid inputs for differential testing.

OJBench strips the sample tests out of the NOI statements (159 of the 232
problems). Without them a Python candidate is never executed at all, and every
candidate hashes to the same behavioural signature, so clustering does nothing.

This asks the model for a small input *generator* per problem, runs it under a
few seeds, and hands the resulting inputs to stage 2. They have no expected
output -- they exist so that candidates can be compared against each other. Two
programs that disagree on a valid input cannot both be right, and the larger
agreeing group is the better bet.

Inputs are deliberately tiny (dimensions <= 8): the point is to expose logic
differences, not to measure speed.

    python 2c_gen_inputs.py --model /workspace/VibeThinker-3B \
           --prompts full.jsonl --out work/stress.jsonl

    python 2_verify_samples.py --prompts full.jsonl \
           --candidates work/candidates.jsonl --out work/verify.jsonl \
           --stress-file work/stress.jsonl --fresh
"""
import argparse
import shutil
import sys
import tempfile
from pathlib import Path

from common import (extract_code, find_pypy3, load_tokenizer, parse_samples, read_jsonl,
                    render_chat, run_program, write_jsonl)

PROMPT = """Below is a competitive programming problem.

<statement>
{statement}
</statement>

Write a Python 3 script that prints ONE randomly generated **valid** input for this problem to stdout.

Requirements:
- Read an integer seed from `sys.argv[1]` and call `random.seed(seed)` first.
- Obey every constraint in the statement (ordering, ranges, sums, connectivity, uniqueness, whatever is required). An input that violates a constraint is useless.
- Keep the instance TINY: no dimension above 8 and no value above 20, even if the statement allows far more. This is for comparing solutions against each other, not for stress testing speed.
- Print exactly the input format the problem specifies and nothing else -- no labels, no prompts, no trailing commentary.
- Self-contained, reads nothing from stdin.

Reply with the script in a single ```python code block."""


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prompts", default="full.jsonl")
    p.add_argument("--out", default="work/stress.jsonl")
    p.add_argument("--gen-out", default="work/generators.jsonl")
    p.add_argument("--model", default="/workspace/VibeThinker-3B")
    p.add_argument("--n-inputs", type=int, default=6, help="seeds per problem")
    p.add_argument("--only-missing", action="store_true",
                   help="only problems with no parseable statement samples")
    p.add_argument("--statement-chars", type=int, default=12000)
    p.add_argument("--max-tokens", type=int, default=6144)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=-1,
                   help="-1 = off (the paper's setting). Set this if ablate.py says so")
    p.add_argument("--min-p", type=float, default=0.0)
    p.add_argument("--repetition-penalty", type=float, default=1.0)

    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--gpu-mem-util", type=float, default=0.93)
    p.add_argument("--max-num-seqs", type=int, default=64)
    p.add_argument("--kv-cache-dtype", default="auto")
    p.add_argument("--cuda-graphs", action="store_true",
                   help="re-enable CUDA graphs (~40%% faster). OFF by default: graph replay "
                        "corrupts long generations on vllm 0.6.3 + Ada. See ablate.py")
    p.add_argument("--no-async-output", action="store_true",
                   help="disable async output processing. Try --cuda-graphs together with "
                        "this if ablate.py says async output was the broken half")

    p.add_argument("--gen-timeout", type=float, default=10.0)
    p.add_argument("--max-input-bytes", type=int, default=65536)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--seed", type=int, default=23)
    return p.parse_args()


def main():
    a = build_args()
    from vllm import LLM, SamplingParams

    tasks = read_jsonl(a.prompts)
    by_id = {}
    for t in tasks:
        by_id.setdefault(t["id"], t)

    targets = [t for t in by_id.values()
               if not (a.only_missing and parse_samples(t.get("prompt", "")))]
    if a.limit:
        targets = targets[: a.limit]
    print(f"[plan] {len(targets)} problems need a generator (of {len(by_id)})")
    if not targets:
        return

    tok = load_tokenizer(a.model)
    llm = LLM(model=a.model, dtype="bfloat16", max_model_len=a.max_model_len,
              gpu_memory_utilization=a.gpu_mem_util, max_num_seqs=a.max_num_seqs,
              swap_space=4, enable_prefix_caching=True, trust_remote_code=True,
              enforce_eager=not a.cuda_graphs, kv_cache_dtype=a.kv_cache_dtype,
              disable_async_output_proc=a.no_async_output, seed=a.seed)

    texts, sps = [], []
    for t in targets:
        stmt = t["prompt"][: a.statement_chars]
        rendered = render_chat(tok, PROMPT.format(statement=stmt))
        room = a.max_model_len - len(tok(rendered)["input_ids"]) - 16
        texts.append(rendered)
        sps.append(SamplingParams(n=1, temperature=a.temperature, top_p=0.95, top_k=a.top_k,
                                  min_p=a.min_p, repetition_penalty=a.repetition_penalty,
                                  max_tokens=max(512, min(a.max_tokens, room)), seed=a.seed))

    outs = llm.generate(texts, sps)

    interp = find_pypy3() or sys.executable
    root = Path(tempfile.mkdtemp(prefix="ojgen_"))
    rows, gens = [], []
    n_nocode = n_dead = n_ok = 0

    for t, o in zip(targets, outs):
        code = extract_code(o.outputs[0].text, "python")
        gens.append({"id": t["id"], "generator": code})
        if not code:
            n_nocode += 1
            continue

        wd = root / str(t["id"])
        wd.mkdir(parents=True, exist_ok=True)
        src = wd / "gen.py"
        src.write_text(code, encoding="utf-8")

        inputs, seen = [], set()
        for seed in range(1, a.n_inputs + 1):
            r = run_program([interp, str(src), str(seed)], "", timeout_s=a.gen_timeout,
                            mem_mb=1024, cwd=str(wd),
                            apply_as="pypy" not in interp)
            if r["status"] != "ok":
                continue
            data = r["stdout"]
            if not data.strip() or len(data) > a.max_input_bytes:
                continue
            if data in seen:                 # generator ignores the seed
                continue
            seen.add(data)
            inputs.append(data if data.endswith("\n") else data + "\n")
        shutil.rmtree(wd, ignore_errors=True)

        if inputs:
            rows.append({"id": t["id"], "inputs": inputs})
            n_ok += 1
        else:
            n_dead += 1

    shutil.rmtree(root, ignore_errors=True)
    write_jsonl(a.out, rows)
    write_jsonl(a.gen_out, gens)

    total_in = sum(len(r["inputs"]) for r in rows)
    print(f"\n[done] {a.out}")
    print(f"  problems with usable inputs : {n_ok}/{len(targets)}")
    print(f"  generator had no code block : {n_nocode}")
    print(f"  generator produced nothing  : {n_dead}")
    print(f"  distinct inputs total       : {total_in} "
          f"({total_in/max(n_ok,1):.1f} per problem)")
    print(f"  generator sources kept in {a.gen_out} -- spot-check a couple against "
          "the statement's input format")
    print(f"\nNow: python 2_verify_samples.py --stress-file {a.out} --fresh ...")


if __name__ == "__main__":
    main()