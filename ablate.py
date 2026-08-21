#!/usr/bin/env python3
"""
Find out why long generations turn into word salad.

The first version of this script ran 3000 tokens and found every config clean --
which only proved the *opening* of a generation is fine. Degeneration shows up
several thousand tokens in, so this runs long and reports where it starts.

    python ablate.py --model /workspace/VibeThinker-3B --prompts full.jsonl

Two numbers matter:
  deg_tail   degeneracy of the last quarter of the output. This is the one to read.
  1st_bad    which tenth of the output first goes bad (- = never).

If 1st_bad lands at roughly the same position in every config, something
positional is wrong (rope, cache, context handling). If it moves around and only
the low-temperature rows are clean, it is sampling instability and the fix is a
sampling parameter.
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from common import extract_code, load_tokenizer, read_jsonl, render_chat

CJK = re.compile(r"[\u4e00-\u9fff]")

# name -> (LLM kwargs, or None for the transformers control | SamplingParams overrides)
CONFIGS = {
    "hf_baseline":      (None, {}),
    "vllm_default":     ({}, {}),
    "no_prefix_cache":  ({"enable_prefix_caching": False}, {}),
    "eager":            ({"enforce_eager": True}, {}),
    # keeps CUDA graphs, drops only async output processing. enforce_eager kills
    # both at once, so this is what tells you which of the two is actually broken.
    "no_async_out":     ({"disable_async_output_proc": True}, {}),
    "graphs_no_async":  ({"disable_async_output_proc": True, "enable_prefix_caching": False}, {}),
    "temp_0.6":         ({}, {"temperature": 0.6}),
    "temp_0.8":         ({}, {"temperature": 0.8}),
    "topk_50":          ({}, {"top_k": 50}),
    "topp_0.9":         ({}, {"top_p": 0.90}),
    "min_p_0.05":       ({}, {"min_p": 0.05}),
    "rep_pen_1.05":     ({}, {"repetition_penalty": 1.05}),
    "greedy":           ({}, {"temperature": 0.0, "top_p": 1.0}),
    # throughput candidates. fp8 roughly doubles KV capacity, hence concurrency,
    # but vLLM falls back to a scaling factor of 1.0 when the checkpoint ships no
    # calibration, which can quietly degrade quality -- so measure, do not assume.
    "eager_fp8":        ({"enforce_eager": True, "kv_cache_dtype": "fp8"}, {}),
    "eager_mem95":      ({"enforce_eager": True, "gpu_memory_utilization": 0.95}, {}),
}
FAST = "vllm_default,temp_0.6,topk_50,min_p_0.05,rep_pen_1.05"
ENGINE = "vllm_default,eager,no_async_out,graphs_no_async"
SPEED = "eager,eager_fp8,eager_mem95"


def degeneracy(text):
    """0 = clean prose, 1 = word salad. A broken decode is mostly very short
    lines and stray rare tokens, not clean repetition."""
    lines = [l for l in (text or "").split("\n") if l.strip()]
    if not lines:
        return 1.0
    short = sum(1 for l in lines if len(l.split()) < 4) / len(lines)
    words = (text or "").split()
    if not words:
        return 1.0
    cjk = sum(1 for w in words if CJK.search(w)) / len(words)
    return round(min(1.0, 0.8 * short + 4.0 * cjk), 3)


def profile(text, n=10):
    """Degeneracy of each tenth of the output, so we can see where it breaks."""
    if not text:
        return [1.0] * n
    step = max(1, len(text) // n)
    return [degeneracy(text[i * step:(i + 1) * step]) for i in range(n)]


def first_bad(prof, thresh=0.5):
    for i, v in enumerate(prof):
        if v > thresh:
            return i
    return -1


def run_child(args):
    name = args.config
    llm_kwargs, sp_kwargs = CONFIGS[name]

    tasks = read_jsonl(args.prompts)[: args.n_prompts]
    tok = load_tokenizer(args.model)
    prompts = [render_chat(tok, t["prompt"]) for t in tasks]
    t0 = __import__("time").time()

    if llm_kwargs is None:
        import torch
        from transformers import AutoModelForCausalLM
        try:
            model = AutoModelForCausalLM.from_pretrained(
                args.model, torch_dtype=torch.bfloat16, device_map="cuda",
                low_cpu_mem_usage=True)
        except ImportError:          # accelerate missing -- load the plain way
            print("[hf] accelerate not installed, loading without device_map "
                  "(slower, needs ~13 GB host RAM)")
            model = AutoModelForCausalLM.from_pretrained(
                args.model, torch_dtype=torch.bfloat16).to("cuda")
        model.eval()
        texts, langs = [], []
        for p, t in zip(prompts, tasks):
            ids = tok([p], return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**ids, max_new_tokens=args.max_tokens, do_sample=True,
                                     temperature=sp_kwargs.get("temperature", 1.0),
                                     top_p=0.95, top_k=None)
            texts.append(tok.decode(out[0][ids["input_ids"].shape[-1]:],
                                    skip_special_tokens=True))
            langs.append(t["language"])
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

    elapsed = __import__("time").time() - t0
    profs = [profile(x) for x in texts]
    tails = [degeneracy(x[-max(len(x) // 4, 1):]) if x else 1.0 for x in texts]
    fenced = [extract_code(x, l) is not None for x, l in zip(texts, langs)]
    firsts = [first_bad(p) for p in profs]
    seen = [f for f in firsts if f >= 0]
    worst = max(range(len(texts)), key=lambda i: tails[i]) if texts else 0

    result = {
        "config": name,
        "n": len(texts),
        "degen_all": round(sum(degeneracy(x) for x in texts) / max(len(texts), 1), 3),
        "degen_tail": round(sum(tails) / max(len(tails), 1), 3),
        "n_bad_tail": sum(1 for t in tails if t > 0.5),
        "n_fenced": sum(fenced),
        "first_bad_mean": round(sum(seen) / len(seen), 1) if seen else None,
        "first_bad_all": firsts,
        "profile_worst": profs[worst] if profs else [],
        "chars": [len(x) for x in texts],
        "worst_tail_sample": texts[worst][-700:] if texts else "",
        "seconds": round(elapsed, 1),
        "tok_per_s": round(sum(len(x) for x in texts) / 4.0 / max(elapsed, 1e-6), 1),
    }
    Path(args.child_out).write_text(json.dumps(result), encoding="utf-8")
    print(f"[{name}] deg_tail={result['degen_tail']} "
          f"bad={result['n_bad_tail']}/{result['n']} "
          f"fenced={result['n_fenced']}/{result['n']} "
          f"1st_bad={result['first_bad_mean']}")
    print(f"  worst profile (10 slices): {result['profile_worst']}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/workspace/VibeThinker-3B")
    p.add_argument("--prompts", default="full.jsonl")
    p.add_argument("--configs", default=",".join(CONFIGS),
                   help=f"comma separated, or a preset: 'fast'={FAST}, "
                        f"'engine'={ENGINE}, 'speed'={SPEED}")
    p.add_argument("--n-prompts", type=int, default=2)
    p.add_argument("--n-samples", type=int, default=3)
    p.add_argument("--max-tokens", type=int, default=14000,
                   help="must be long enough to reach the degenerate zone; 3000 is not")
    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out", default="work/ablation.json")
    p.add_argument("--config", default=None, help=argparse.SUPPRESS)
    p.add_argument("--child-out", default=None, help=argparse.SUPPRESS)
    a = p.parse_args()

    if a.config:
        run_child(a)
        return

    preset = {"fast": FAST, "engine": ENGINE, "speed": SPEED}.get(a.configs)
    names = (preset or a.configs).split(",")
    Path("work").mkdir(exist_ok=True)
    results = []
    for name in names:
        if name not in CONFIGS:
            print(f"[skip] unknown config {name}")
            continue
        print(f"\n{'='*64}\n  {name}\n{'='*64}", flush=True)
        child = f"work/_ablate_{name}.json"
        cmd = [sys.executable, __file__, "--model", a.model, "--prompts", a.prompts,
               "--config", name, "--child-out", child,
               "--n-prompts", str(a.n_prompts), "--n-samples", str(a.n_samples),
               "--max-tokens", str(a.max_tokens), "--max-model-len", str(a.max_model_len),
               "--seed", str(a.seed)]
        r = subprocess.run(cmd)
        if r.returncode != 0 or not Path(child).exists():
            results.append({"config": name, "degen_tail": None})
            continue
        results.append(json.loads(Path(child).read_text()))

    Path(a.out).write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n" + "=" * 64)
    print(f"{'config':<16}{'deg_all':>9}{'deg_tail':>10}{'bad':>8}"
          f"{'fenced':>9}{'1st_bad':>9}{'tok/s':>9}")
    print("-" * 64)
    for r in results:
        if r.get("degen_tail") is None:
            print(f"{r['config']:<16}{'CRASH':>9}")
            continue
        bad = f"{r['n_bad_tail']}/{r['n']}"
        fen = f"{r['n_fenced']}/{r['n']}"
        fb = "-" if r["first_bad_mean"] is None else str(r["first_bad_mean"])
        print(f"{r['config']:<16}{r['degen_all']:>9}{r['degen_tail']:>10}"
              f"{bad:>8}{fen:>9}{fb:>9}{r.get('tok_per_s', '-'):>9}")
    print("-" * 64)
    print("deg_tail < 0.3 healthy, > 0.5 word salad. 1st_bad = which tenth of the")
    print("output first went bad (- = never). How to read the pattern:")
    print("  one engine flag clean, every sampling variant broken -> engine bug")
    print("  greedy among the worst                               -> NOT sampling")
    print("  identical numbers across configs                     -> that knob is irrelevant")
    print("  hf_baseline broken too                               -> not vLLM at all")
    print(f"\nfull output, including a sample of the worst tail, in {a.out}")


if __name__ == "__main__":
    main()