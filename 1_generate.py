#!/usr/bin/env python3
"""
Stage 1 -- generate K candidate solutions per OJBench task with VibeThinker-3B.

The OJBench prompt already tells the model how to format its answer, so we pass
it through untouched and only wrap it in the chat template.

  python 1_generate.py --prompts OJBench_testdata/prompts/full.jsonl \
                       --out work/candidates.jsonl --k 16

Writes one row per (task, sample):
  {key, id, language, dataset, difficulty, idx, code, tail, finish, n_tokens}
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

from common import (append_jsonl, extract_code, load_tokenizer, read_jsonl, render_chat,
                    shard_filter, task_key)

FORMAT_NUDGE = (
    "\n\nBefore you finish: re-read the constraints and confirm your complexity fits "
    "the time limit at the maximum input size. Put your complete, self-contained final "
    "program in a single {lang} code block at the very end of your reply. It must read "
    "from standard input and write to standard output."
)


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prompts", default="OJBench_testdata/prompts/full.jsonl")
    p.add_argument("--out", default="work/candidates.jsonl")
    p.add_argument("--model", default="WeiboAI/VibeThinker-3B")

    p.add_argument("--k", type=int, default=16, help="candidates per task")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-new-tokens", type=int, default=24576)
    p.add_argument("--seed", type=int, default=1234)

    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--gpu-mem-util", type=float, default=0.90)
    p.add_argument("--max-num-seqs", type=int, default=48)
    p.add_argument("--swap-space", type=int, default=4)
    p.add_argument("--no-prefix-caching", action="store_true")
    p.add_argument("--kv-cache-dtype", default="auto",
                   help="'fp8' roughly doubles KV capacity but needs a supported backend")

    p.add_argument("--languages", default="python,cpp")
    p.add_argument("--datasets", default="NOI,ICPC")
    p.add_argument("--difficulties", default="easy,medium,hard")
    p.add_argument("--limit", type=int, default=0, help="first N tasks only (smoke tests)")
    p.add_argument("--shard", default="0/1")

    p.add_argument("--chunk", type=int, default=12, help="tasks per flush to disk")
    p.add_argument("--nudge", action="store_true",
                   help="append a short complexity/format reminder to the OJBench prompt. "
                        "Off by default so the generation protocol matches the paper's and "
                        "any gain is attributable to CLR alone.")
    p.add_argument("--keep-full-text", action="store_true", help="store raw generations too (big)")
    p.add_argument("--fresh", action="store_true", help="ignore existing output and start over")
    return p.parse_args()


def main():
    a = build_args()
    from vllm import LLM, SamplingParams

    tasks = read_jsonl(a.prompts)
    langs = set(a.languages.split(","))
    dsets = set(a.datasets.split(","))
    diffs = set(a.difficulties.split(","))
    tasks = [t for t in tasks
             if t.get("language") in langs
             and t.get("dataset") in dsets
             and str(t.get("difficulty", "")).lower() in diffs]
    tasks = shard_filter(tasks, a.shard)
    if a.limit:
        tasks = tasks[: a.limit]

    out_path = Path(a.out)
    if a.fresh and out_path.exists():
        out_path.unlink()
    done = set()
    if out_path.exists():
        done = {r["key"] for r in read_jsonl(out_path)}
        print(f"[resume] {len(done)} tasks already generated, skipping them")
    todo = [t for t in tasks if task_key(t) not in done]

    print(f"[plan] {len(todo)} tasks x {a.k} samples "
          f"({len(tasks)} selected, {len(tasks) - len(todo)} already done)")
    if not todo:
        return

    tok = load_tokenizer(a.model)

    def render(t):
        user = t["prompt"]
        if a.nudge:
            user += FORMAT_NUDGE.format(lang="Python" if t["language"] == "python" else "C++")
        return render_chat(tok, user)

    print("\n" + "=" * 70 + "\n[check] rendered prompt for the first task (verify this looks sane):\n"
          + "=" * 70)
    print(render(todo[0])[:1200] + "\n... [truncated]\n" + "=" * 70 + "\n")

    llm = LLM(
        model=a.model,
        dtype="bfloat16",
        max_model_len=a.max_model_len,
        gpu_memory_utilization=a.gpu_mem_util,
        max_num_seqs=a.max_num_seqs,
        swap_space=a.swap_space,
        enable_prefix_caching=not a.no_prefix_caching,
        kv_cache_dtype=a.kv_cache_dtype,
        trust_remote_code=True,
        seed=a.seed,
    )

    t_start = time.time()
    for c0 in range(0, len(todo), a.chunk):
        chunk = todo[c0: c0 + a.chunk]
        prompts, sps = [], []
        for t in chunk:
            text = render(t)
            n_prompt = len(tok(text)["input_ids"])
            budget = a.max_model_len - n_prompt - 16
            if budget < 2048:
                print(f"[warn] {task_key(t)}: prompt is {n_prompt} tokens, "
                      f"only {budget} left to think in. Raise --max-model-len.")
                budget = max(budget, 512)
            prompts.append(text)
            sps.append(SamplingParams(
                n=a.k, temperature=a.temperature, top_p=a.top_p, top_k=-1,
                max_tokens=min(a.max_new_tokens, budget), seed=a.seed + c0,
            ))

        outs = llm.generate(prompts, sps)

        rows, n_code = [], 0
        for t, o in zip(chunk, outs):
            for i, comp in enumerate(o.outputs):
                code = extract_code(comp.text, t["language"])
                n_code += code is not None
                last = comp.text.rfind("```")
                open_fence = comp.text.rfind("```", 0, last) if last != -1 else -1
                head = comp.text[:open_fence] if open_fence != -1 else comp.text
                rec = {
                    "key": task_key(t), "id": t["id"], "language": t["language"],
                    "dataset": t.get("dataset"), "difficulty": t.get("difficulty"),
                    "idx": i, "code": code,
                    "tail": head[-2500:],                      # context for the CLR stage
                    "finish": comp.finish_reason,
                    "n_tokens": len(comp.token_ids),
                }
                if a.keep_full_text:
                    rec["text"] = comp.text
                rows.append(rec)
        append_jsonl(out_path, rows)

        seen = c0 + len(chunk)
        el = time.time() - t_start
        eta = el / max(seen, 1) * (len(todo) - seen)
        print(f"[{seen}/{len(todo)}] +{len(rows)} samples, "
              f"{n_code}/{len(rows)} had a code block | "
              f"elapsed {el/60:.1f}m, eta {eta/60:.1f}m", flush=True)

    print(f"\n[done] wrote {out_path}")


if __name__ == "__main__":
    main()
