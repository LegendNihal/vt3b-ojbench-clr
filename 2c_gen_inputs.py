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
from collections import defaultdict
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
    p.add_argument("--top-k", type=int, default=-1,
                   help="-1 = off (the paper's setting). Set this if ablate.py says so")
    p.add_argument("--min-p", type=float, default=0.0)
    p.add_argument("--repetition-penalty", type=float, default=1.0)

    p.add_argument("--seed", type=int, default=1234)

    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--gpu-mem-util", type=float, default=0.93,
                   help="eager mode frees the 1-3 GiB CUDA graphs would use, "
                        "so this can go higher than the usual 0.90")
    p.add_argument("--max-num-seqs", type=int, default=48,
                   help="auto-lowered at startup to whatever the KV cache "
                        "actually holds")
    p.add_argument("--swap-space", type=int, default=16,
                   help="GiB of CPU swap. Only used if preemption still happens")
    p.add_argument("--fork-n", action="store_true",
                   help="issue one n=K request per task instead of K n=1 requests. "
                        "Slightly cheaper, but forces vLLM into SWAP preemption, "
                        "which aborts the run when CPU swap fills. Not recommended")
    p.add_argument("--no-prefix-caching", action="store_true")
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--cuda-graphs", action="store_true",
                   help="re-enable CUDA graphs (~40%% faster). OFF by default: graph replay "
                        "corrupts long generations on vllm 0.6.3 + Ada. See ablate.py")
    p.add_argument("--no-async-output", action="store_true",
                   help="disable async output processing. Try --cuda-graphs together with "
                        "this if ablate.py says async output was the broken half")

    p.add_argument("--kv-cache-dtype", default="auto",
                   help="'fp8' roughly doubles KV capacity but needs a supported backend")

    # These default to EMPTY = keep everything. A non-empty value is matched
    # case-insensitively: OJBench's own full.jsonl mixes "NOI" with "icpc", and a
    # case-sensitive filter here silently drops a third of the benchmark.
    p.add_argument("--languages", default="", help="e.g. python,cpp. Empty = all")
    p.add_argument("--datasets", default="", help="e.g. noi,icpc. Empty = all")
    p.add_argument("--difficulties", default="", help="e.g. easy,medium. Empty = all")
    p.add_argument("--limit", type=int, default=0, help="first N tasks only (smoke tests)")
    p.add_argument("--shuffle", action="store_true",
                   help="shuffle (fixed seed) before --limit, so a smoke set spans both "
                        "datasets instead of taking the first N rows of one of them")
    p.add_argument("--shard", default="0/1")

    p.add_argument("--chunk", type=int, default=12, help="tasks per flush to disk")
    p.add_argument("--nudge", action="store_true",
                   help="append a short complexity/format reminder to the OJBench prompt. "
                        "Off by default so the generation protocol matches the paper's and "
                        "any gain is attributable to CLR alone.")
    p.add_argument("--keep-full-text", action="store_true", help="store raw generations too (big)")
    p.add_argument("--fresh", action="store_true", help="ignore existing output and start over")
    p.add_argument("--only-missing-from", default=None,
                   help="a candidates jsonl from an earlier pass. Only generate for tasks that "
                        "came out of it with NO usable code. Lets a cheap wide pass run at a low "
                        "token cap and a second deep pass spend a big cap only where it is needed")
    p.add_argument("--idx-offset", type=int, default=0,
                   help="added to every sample index, so a second pass can be concatenated "
                        "with the first without idx collisions")
    return p.parse_args()


def main():
    a = build_args()
    from vllm import LLM, SamplingParams

    all_tasks = read_jsonl(a.prompts)

    # full.jsonl mixes "NOI" with "icpc". Two defences: an empty filter keeps
    # everything, and a non-empty one is matched case-insensitively. Getting this
    # wrong drops a third of the benchmark and stage 4 quietly fills the gap with
    # placeholders, so the run looks like it worked.
    def norm_set(v):
        vals = {x.strip().lower() for x in (v or "").split(",") if x.strip()}
        return vals or None          # None means "no filter on this field"

    langs, dsets, diffs = (norm_set(a.languages), norm_set(a.datasets),
                           norm_set(a.difficulties))

    def keep(t):
        for want, field in ((langs, "language"), (dsets, "dataset"), (diffs, "difficulty")):
            if want and str(t.get(field, "")).lower() not in want:
                return False
        return True

    tasks = [t for t in all_tasks if keep(t)]

    from collections import Counter
    have = Counter(str(t.get("dataset")) for t in all_tasks)
    got = Counter(str(t.get("dataset")) for t in tasks)
    print(f"[select] {len(tasks)}/{len(all_tasks)} rows | "
          + ", ".join(f"{k}={got.get(k, 0)}/{v}" for k, v in sorted(have.items())))
    for k in sorted(have):
        if got.get(k, 0) == 0:
            print(f"[WARN] dataset {k!r} is entirely excluded by your filters. Every one "
                  f"of those rows becomes a placeholder in model_response.jsonl.")
    if not tasks:
        sys.exit("[fatal] no tasks selected -- check --languages / --datasets / --difficulties")

    if a.shuffle:
        import random
        random.Random(12345).shuffle(tasks)
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

    if a.only_missing_from:
        prev = read_jsonl(a.only_missing_from)
        solved = {r["key"] for r in prev if r.get("code")}
        attempted = {r["key"] for r in prev}
        before = len(todo)
        todo = [t for t in todo if task_key(t) not in solved]
        print(f"[pass2] {a.only_missing_from}: {len(attempted)} tasks attempted, "
              f"{len(solved)} produced code. Regenerating the {before - len(todo)} "
              f"skipped + {len(todo)} still empty")

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

    plens = sorted(len(tok(render(t))["input_ids"]) for t in todo[:64])
    if plens:
        print(f"[prompts] tokens p50 {plens[len(plens)//2]} "
              f"p90 {plens[min(len(plens)-1, int(.9*len(plens)))]} max {plens[-1]} "
              f"| thinking room left at max: {a.max_model_len - plens[-1]}")

    print("\n" + "=" * 70 + "\n[check] rendered prompt for the first task (verify this looks sane):\n"
          + "=" * 70)
    print(render(todo[0])[:1200] + "\n... [truncated]\n" + "=" * 70 + "\n")

    if not a.cuda_graphs:
        print("[engine] CUDA graphs disabled (default). Graph replay corrupts long "
              "generations on vllm 0.6.3 + Ada; measured cost ~25% throughput. "
              "Pass --cuda-graphs to override (do not, unless ablate.py says otherwise).")
    llm = LLM(
        model=a.model,
        dtype="bfloat16",
        max_model_len=a.max_model_len,
        gpu_memory_utilization=a.gpu_mem_util,
        max_num_seqs=a.max_num_seqs,
        swap_space=a.swap_space,
        enable_prefix_caching=not a.no_prefix_caching,
        enforce_eager=not a.cuda_graphs,
        disable_async_output_proc=a.no_async_output,
        kv_cache_dtype=a.kv_cache_dtype,
        trust_remote_code=True,
        seed=a.seed,
    )

    # vLLM can only preempt by RECOMPUTE when a sequence group has a single
    # running sequence. With n=K it is forced into SWAP, and SWAP dies with
    # "lack of CPU swap space" the moment the batch outgrows the KV cache. So
    # issue K independent n=1 requests instead; prefix caching still shares the
    # prompt blocks between them, and preemption becomes survivable.
    try:
        cc = llm.llm_engine.cache_config
        kv_tokens = (cc.num_gpu_blocks or 0) * cc.block_size
        est_per_seq = min(a.max_model_len, a.max_new_tokens + 4096)
        safe = max(4, int(0.9 * kv_tokens / max(est_per_seq, 1)))
        cur = llm.llm_engine.scheduler_config.max_num_seqs
        print(f"[kv] {kv_tokens} tokens of cache, ~{est_per_seq} per sequence "
              f"-> room for ~{safe} concurrent")
        if safe < cur:
            llm.llm_engine.scheduler_config.max_num_seqs = safe
            print(f"[kv] lowering --max-num-seqs {cur} -> {safe} to stay off the "
                  f"preemption path (raise --gpu-mem-util or use --kv-cache-dtype "
                  f"fp8 for more room)")
    except Exception as e:
        print(f"[kv] could not auto-tune max_num_seqs ({e}); watch for preemption warnings")

    t_start = time.time()
    req_seed = a.seed
    for c0 in range(0, len(todo), a.chunk):
        chunk = todo[c0: c0 + a.chunk]
        prompts, sps, owners = [], [], []
        for ti, t in enumerate(chunk):
            text = render(t)
            n_prompt = len(tok(text)["input_ids"])
            budget = a.max_model_len - n_prompt - 16
            if budget < 2048:
                print(f"[warn] {task_key(t)}: prompt is {n_prompt} tokens, "
                      f"only {budget} left to think in. Raise --max-model-len.")
                budget = max(budget, 512)
            mt = min(a.max_new_tokens, budget)
            common = dict(temperature=a.temperature, top_p=a.top_p, top_k=a.top_k,
                          min_p=a.min_p, repetition_penalty=a.repetition_penalty,
                          max_tokens=mt)
            if a.fork_n:
                prompts.append(text)
                sps.append(SamplingParams(n=a.k, seed=req_seed, **common))
                owners.append(ti)
                req_seed += 1
            else:
                for _ in range(a.k):
                    prompts.append(text)
                    sps.append(SamplingParams(n=1, seed=req_seed, **common))
                    owners.append(ti)
                    req_seed += 1

        outs = llm.generate(prompts, sps)

        by_task = defaultdict(list)
        for ti, o in zip(owners, outs):
            by_task[ti].extend(o.outputs)

        rows, n_code = [], 0
        for ti, t in enumerate(chunk):
            for i, comp in enumerate(by_task.get(ti, [])):
                code = extract_code(comp.text, t["language"])
                n_code += code is not None
                last = comp.text.rfind("```")
                open_fence = comp.text.rfind("```", 0, last) if last != -1 else -1
                head = comp.text[:open_fence] if open_fence != -1 else comp.text
                rec = {
                    "key": task_key(t), "id": t["id"], "language": t["language"],
                    "dataset": t.get("dataset"), "difficulty": t.get("difficulty"),
                    "idx": i + a.idx_offset, "code": code,
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