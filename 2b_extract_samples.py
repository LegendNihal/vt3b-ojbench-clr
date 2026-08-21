#!/usr/bin/env python3
"""
Stage 2b (optional) -- have the model read the sample tests out of statements the
regexes cannot parse, then feed the result to stage 2 as --samples-override.

Samples belong to a problem, not to a language, so this runs once per `id`
(232 problems, not 464 rows) and both the python and cpp rows get the result.

    python 2b_extract_samples.py --model /workspace/VibeThinker-3B \
           --prompts full.jsonl --only-missing --out work/samples_llm.jsonl

    python 2_verify_samples.py --prompts full.jsonl \
           --candidates work/candidates.jsonl --out work/verify.jsonl \
           --samples-override work/samples_llm.jsonl --fresh

Every extracted case is checked against the statement: if the numbers the model
produced do not actually appear there, it is dropped. A hallucinated sample would
poison the execution gate, so the bar for accepting one is that it be verbatim.
"""
import argparse
import re
from pathlib import Path

from common import (load_tokenizer, parse_last_json, parse_samples, read_jsonl,
                    render_chat, write_jsonl)

PROMPT = """Below is a competitive programming problem statement.

<statement>
{statement}
</statement>

Copy out the example test cases exactly as they appear in the statement. Do not solve the problem, do not invent extra cases, do not reformat the numbers. Preserve line breaks inside each case with \\n. If the statement has no example test cases, return an empty list.

Reply with ONLY this JSON object and nothing else:
{{"samples": [{{"input": "...", "output": "..."}}]}}"""


def norm_tokens(s):
    return " ".join((s or "").split())


def verbatim(part, statement_tokens):
    """Accept a sample only if its content really appears in the statement."""
    t = norm_tokens(part)
    if not t:
        return False
    if len(t) > 4000:                 # a huge "sample" is a sign it copied the whole section
        return False
    return t in statement_tokens


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prompts", default="full.jsonl")
    p.add_argument("--out", default="work/samples_llm.jsonl")
    p.add_argument("--model", default="/workspace/VibeThinker-3B")
    p.add_argument("--only-missing", action="store_true",
                   help="skip problems the regex already handled (recommended)")
    p.add_argument("--statement-chars", type=int, default=14000)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--gpu-mem-util", type=float, default=0.90)
    p.add_argument("--max-num-seqs", type=int, default=64)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--no-verbatim-check", action="store_true",
                   help="keep unverified samples too (not recommended)")
    return p.parse_args()


def main():
    a = build_args()
    from vllm import LLM, SamplingParams

    tasks = read_jsonl(a.prompts)
    by_id = {}
    for t in tasks:
        by_id.setdefault(t["id"], t)

    targets = []
    for pid, t in by_id.items():
        if a.only_missing and parse_samples(t.get("prompt", "")):
            continue
        targets.append(t)
    if a.limit:
        targets = targets[: a.limit]
    print(f"[plan] {len(targets)} problems to extract "
          f"(out of {len(by_id)} total)")
    if not targets:
        print("nothing to do")
        return

    tok = load_tokenizer(a.model)
    llm = LLM(model=a.model, dtype="bfloat16", max_model_len=a.max_model_len,
              gpu_memory_utilization=a.gpu_mem_util, max_num_seqs=a.max_num_seqs,
              swap_space=4, enable_prefix_caching=True, trust_remote_code=True, seed=11)

    texts, sps = [], []
    for t in targets:
        stmt = t["prompt"]
        if len(stmt) > a.statement_chars:      # samples live at the end, keep that
            stmt = stmt[: a.statement_chars // 4] + "\n...[cut]...\n" + stmt[-3 * a.statement_chars // 4:]
        rendered = render_chat(tok, PROMPT.format(statement=stmt))
        room = a.max_model_len - len(tok(rendered)["input_ids"]) - 16
        texts.append(rendered)
        sps.append(SamplingParams(n=1, temperature=a.temperature, top_p=0.95, top_k=-1,
                                  max_tokens=max(256, min(a.max_tokens, room)), seed=11))

    outs = llm.generate(texts, sps)

    rows, n_ok, n_drop, n_none = [], 0, 0, 0
    for t, o in zip(targets, outs):
        js = parse_last_json(o.outputs[0].text) or {}
        raw = js.get("samples")
        if not isinstance(raw, list):
            n_none += 1
            continue
        stmt_tokens = norm_tokens(t["prompt"])
        keep = []
        for s in raw[:5]:
            if not isinstance(s, dict):
                continue
            i, out = str(s.get("input", "")), str(s.get("output", ""))
            if not i.strip() or not out.strip():
                continue
            if a.no_verbatim_check or (verbatim(i, stmt_tokens) and verbatim(out, stmt_tokens)):
                keep.append({"input": i.rstrip() + "\n", "output": out.rstrip() + "\n",
                             "source": "llm"})
            else:
                n_drop += 1
        if keep:
            rows.append({"id": t["id"], "samples": keep})
            n_ok += 1
        else:
            n_none += 1

    write_jsonl(a.out, rows)
    print(f"\n[done] {a.out}")
    print(f"  problems with usable samples : {n_ok}/{len(targets)}")
    print(f"  cases dropped as not verbatim: {n_drop}")
    print(f"  problems still with nothing  : {n_none}")
    print("\nNow re-run stage 2 with --samples-override "
          f"{a.out} --fresh")


if __name__ == "__main__":
    main()
