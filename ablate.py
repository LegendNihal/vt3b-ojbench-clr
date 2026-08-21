#!/usr/bin/env python3
"""
Find out why generations come out as word salad.

Runs the same real prompt under several inference configurations, each in its own
subprocess (vLLM does not free GPU memory cleanly when re-instantiated), and
scores the outputs for degeneracy.

    python ablate.py --model /workspace/VibeThinker-3B --prompts full.jsonl

Read the table it prints bottom-up: the first config with a low `degen` score is
your fix. If `hf_baseline` is also degenerate, the problem is the weights, the
prompt or the sampling params rather than vLLM.
"""
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from common import extract_code, load_tokenizer, read_jsonl, render_chat

CJK = re.compile(r"[\u4e00-\u9fff]")

CONFIGS = {
    # name              kwargs passed to LLM(...)                      sampling overrides
    "hf_baseline":      (None, {}),                     # plain transformers, no vLLM at all
    "vllm_default":     ({}, {}),                       # what 1_generate.py does today
    "no_prefix_cache":  ({"enable_prefix_caching": False}, {}),
    "no_async_out":     ({"disable_async_output_proc": True}, {}),
    "eager":            ({"enforce_eager": True}, {}),
    "no_pc_eager":      ({"enable_prefix_caching": False, "enforce_eager": True}, {}),
    "greedy":           ({}, {"temperature": 0.0, "top_p": 1.0}),
    "temp_0.6":         ({}, {"temperature": 0.6}),
}


def degeneracy(text):
    """0 = clean prose, 1 = word salad. Garbage from a broken decode is mostly
    very short lines and stray rare tokens, not repetition."""
    lines = [l for l in (text or "").split("\n") if l.strip()]
    if not lines:
        return 1.0
    short = sum(1 for l in lines if len(l.split()) < 4) / len(lines)
    words = (text or "").split()
    if not words:
        return 1.0
    cjk = sum(1 for w in words if CJK.search(w)) / len(words)
    return round(min(1.0, 0.8 * short + 4.0 * cjk), 3)


def run_child(args):
    name = args.config
    llm_kwargs, sp_kwargs = CONFIGS[name]

    tasks = read_jsonl(args.prompts)[: args.n_prompts]
    tok = load_tokenizer(args.model)
    prompts = [render_chat(tok, t["prompt"]) for t in tasks]

    if llm_kwargs is None:                      # transformers control
        import torch
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map="cuda",
            low_cpu_mem_usage=True)
        texts = []
        for p in prompts:
            ids = tok([p], return_tensors="pt").to(model.device)
            out = model.generate(**ids, max_new_tokens=args.max_tokens, do_sample=True,
                                 temperature=sp_kwargs.get("temperature", 1.0),
                                 top_p=0.95, top_k=None)
            texts.append(tok.decode(out[0][ids["input_ids"].shape[-1]:],
                                    skip_special_tokens=True))
        langs = [t["language"] for t in tasks]
    else:
        from vllm import LLM, SamplingParams
        base = dict(model=args.model, dtype="bfloat16", max_model_len=args.max_model_len,
                    gpu_memory_utilization=0.90, max_num_seqs=32, swap_space=4,
                    enable_prefix_caching=True, trust_remote_code=True, seed=args.seed)
        base.update(llm_kwargs)
        llm = LLM(**base)
        sp = dict(n=args.n_samples, temperature=1.0, top_p=0.95, top_k=-1,
                  max_tokens=args.max_tokens, seed=args.seed)
        sp.update(sp_kwargs)
        if sp["temperature"] == 0.0:
            sp["n"] = 1
        outs = llm.generate(prompts, SamplingParams(**sp))
        texts, langs = [], []
        for t, o in zip(tasks, outs):
            for c in o.outputs:
                texts.append(c.text)
                langs.append(t["language"])

    scores = [degeneracy(x) for x in texts]
    fenced = [extract_code(x, l) is not None for x, l in zip(texts, langs)]
    result = {
        "config": name,
        "n": len(texts),
        "degen_mean": round(sum(scores) / len(scores), 3),
        "n_bad": sum(1 for s in scores if s > 0.5),
        "n_fenced": sum(fenced),
        "sample": texts[0][-500:] if texts else "",
    }
    Path(args.child_out).write_text(json.dumps(result), encoding="utf-8")
    print(f"[{name}] degen={result['degen_mean']} bad={result['n_bad']}/{result['n']} "
          f"fenced={result['n_fenced']}/{result['n']}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/workspace/VibeThinker-3B")
    p.add_argument("--prompts", default="full.jsonl")
    p.add_argument("--configs", default=",".join(CONFIGS))
    p.add_argument("--n-prompts", type=int, default=2)
    p.add_argument("--n-samples", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=3000)
    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out", default="work/ablation.json")
    # internal
    p.add_argument("--config", default=None, help=argparse.SUPPRESS)
    p.add_argument("--child-out", default=None, help=argparse.SUPPRESS)
    a = p.parse_args()

    if a.config:
        run_child(a)
        return

    Path("work").mkdir(exist_ok=True)
    results = []
    for name in a.configs.split(","):
        if name not in CONFIGS:
            print(f"[skip] unknown config {name}")
            continue
        print(f"\n{'='*60}\n  {name}\n{'='*60}", flush=True)
        child = f"work/_ablate_{name}.json"
        cmd = [sys.executable, __file__, "--model", a.model, "--prompts", a.prompts,
               "--config", name, "--child-out", child,
               "--n-prompts", str(a.n_prompts), "--n-samples", str(a.n_samples),
               "--max-tokens", str(a.max_tokens), "--max-model-len", str(a.max_model_len),
               "--seed", str(a.seed)]
        r = subprocess.run(cmd)
        if r.returncode != 0 or not Path(child).exists():
            results.append({"config": name, "degen_mean": None, "note": "crashed"})
            continue
        results.append(json.loads(Path(child).read_text()))

    Path(a.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\n" + "=" * 60)
    print(f"{'config':<18}{'degen':>8}{'bad':>8}{'fenced':>9}")
    print("-" * 60)
    for r in results:
        if r.get("degen_mean") is None:
            print(f"{r['config']:<18}{'CRASH':>8}")
            continue
        print(f"{r['config']:<18}{r['degen_mean']:>8}"
              f"{r['n_bad']}/{r['n']:>5}{r['n_fenced']}/{r['n']:>6}")
    print("-" * 60)
    print("degen < 0.3 is healthy, > 0.5 is word salad.")
    print(f"full output in {a.out}")


if __name__ == "__main__":
    main()
