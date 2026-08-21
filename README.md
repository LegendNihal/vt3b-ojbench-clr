# VibeThinker-3B + CLR on OJBench

Generates `model_response.jsonl` for OJBench by sampling many candidate programs from
VibeThinker-3B and picking one with a code-adapted version of the paper's Claim-Level
Reliability Assessment. The paper reports **38.6** on OJBench without CLR; the paper only
ever applied CLR to answer-verifiable maths and GPQA, so the coding version below is an
adaptation, not a reproduction.

---

## The idea

CLR in the paper (§3.1): sample K traces, pull M=5 decision-relevant claims out of each,
have the model try to falsify them, score each trace `r_k = (mean verdict)^M`, cluster
answers by equivalence, and return the cluster with the largest summed reliability.

Three things have to change for code:

**Equivalence becomes behavioural.** Two programs are almost never textually equal, so the
cluster key is a hash of what a program *printed* on every input we ran it against. Same
outputs, same cluster.

**Execution gates the claim score.** Maths has no free verifier; code does. The sample tests
printed in the statement are exactly what a human contestant sees before submitting, and a
program that fails them is dead no matter how confident the model sounds. So execution is a
gate in front of CLR, not a competitor to it.

> The hidden testdata under `OJBench_testdata/NOI/` and `ICPC/` is **never** read by this
> pipeline. Selecting on hidden tests would make the resulting number meaningless.

**The five claims are pinned to axes.** Free-form "is this correct?" gets you "yes". The five
claims are fixed to complexity-vs-limits, the central invariant, boundary cases, the I/O
contract, and implementation limits — the things that actually decide OJ verdicts. A model
asked *"does an O(n²) loop fit n ≤ 2·10⁵?"* can do that arithmetic; asked *"is my solution
good?"* it cannot.

Final score per behavioural cluster:

```
r_k             = gate_k · (mean verdict_k)^5
Score(cluster)  = Σ r_k  +  α · (cluster size / K)
```

`gate` is 1.0 when the candidate passes every statement sample, 0.6 when no samples could be
parsed, 0.12 in the fallback case where nothing passed. The `α` term is a self-consistency
bonus, since a behaviour that 9 of 16 rollouts agree on is more likely right than a singleton.

---

## Where each piece runs

```
vast.ai 4090 (24 GB)                          your WSL2 box
─────────────────────                          ─────────────
1_generate.py     GPU, the long one            OJBench judge server
2_verify_samples  CPU, needs g++ + pypy3       (already working)
3_clr_rank.py     GPU
4_build_response  CPU
        │
        └── model_response_clr.jsonl ──scp──▶ judge_jsonl() ──▶ judged.jsonl
```

You need `OJBench_testdata/prompts/full.jsonl` on the vast.ai box (just that one file, a few
MB — not the test data). The `content` strings this writes are single clean fenced code
blocks, which is the format OJBench's extractor handles, and the same shape your synthetic
file already uses.

---

## Setup on vast.ai

```bash
git clone <this folder>  # or scp it over
cd vt3b-ojbench-clr
bash setup_vastai.sh
```

That installs `vllm==0.6.3.post1`, `transformers==4.45.2`, `numpy<2` (0.6.3 predates the
numpy 2 ABI break), plus `g++` and `pypy3`, and downloads the model to
`/workspace/VibeThinker-3B`. It puts `HF_HOME` on `/workspace` because vast.ai images
usually have a small `/`.

**pypy3 matters.** OJBench judges Python submissions with pypy3, not CPython. If you verify
locally with CPython you get a different picture of what is fast enough. `setup_vastai.sh`
installs it; stage 2 prints which interpreter it found.

Then copy the prompts over from WSL2:

```bash
scp -P <port> OJBench_testdata/prompts/full.jsonl root@<host>:~/vt3b-ojbench-clr/full.jsonl
```

`full.jsonl` has 464 rows: 232 problems × {python, cpp}. Every row needs a `content`.

---

## Smoke test first (~10 minutes)

Never start an 8-hour run without this.

```bash
python 1_generate.py --model /workspace/VibeThinker-3B --prompts full.jsonl \
       --out work/smoke_cand.jsonl --k 2 --limit 4 --fresh

python 2_verify_samples.py --prompts full.jsonl --candidates work/smoke_cand.jsonl \
       --out work/smoke_verify.jsonl --fresh

python 3_clr_rank.py --model /workspace/VibeThinker-3B --prompts full.jsonl \
       --candidates work/smoke_cand.jsonl --verify work/smoke_verify.jsonl \
       --out work/smoke_sel.jsonl --top-n 2

python 4_build_response.py --prompts full.jsonl --candidates work/smoke_cand.jsonl \
       --selection work/smoke_sel.jsonl --out smoke.jsonl \
       --schema-from /path/to/your/working_synthetic.jsonl
```

Four things to check:

1. Stage 1 prints the rendered chat prompt. Confirm it has `<|im_start|>user` around the
   problem and ends with `<|im_start|>assistant`.
2. Stage 2 prints sample-test coverage. Open `work/samples.jsonl` and eyeball two or three
   against the real statements — this is the single highest-leverage thing to verify.
3. Stage 3 prints how many claim sets and verdict sets parsed. Below ~70% means the model is
   running out of tokens before it emits the JSON; raise `--extract-tokens` / `--verify-tokens`.
4. Stage 4's `--schema-from` compares field names against your known-good file and tells you
   about any mismatch.

Then judge `smoke.jsonl` on WSL2 to confirm the round trip works before committing GPU hours.

---

## Full run

```bash
MODEL=/workspace/VibeThinker-3B PROMPTS=full.jsonl K=16 bash run_all.sh
```

Or stage by stage, which is what I'd do — stage 1 resumes cleanly if it dies, so run it under
`tmux`:

```bash
tmux new -s gen
python 1_generate.py --model /workspace/VibeThinker-3B --prompts full.jsonl \
       --out work/candidates.jsonl --k 16 --max-model-len 32768 --max-new-tokens 24576
```

Rough budget on one 4090 for all 464 tasks:

| stage | K=8 | K=16 |
|---|---|---|
| 1 generate | 3–5 h | 6–10 h |
| 2 verify (CPU) | 20–40 min | 30–60 min |
| 3 CLR | 1–1.5 h | 1.5–2.5 h |

Stage 1 prints a live ETA after the first chunk — trust that over this table. Two rented GPUs
halve it: run `--shard 0/2` and `--shard 1/2` and `cat` the two candidate files together.

**VRAM.** Weights are 6.2 GB. The KV cache costs ~36 KB/token (36 layers × 2 KV heads ×
128 dim × 2 for K and V × 2 bytes), so at `--gpu-mem-util 0.90` you get roughly 14 GB of
cache ≈ 390k tokens ≈ a dozen concurrent 32K sequences. That is comfortable. If you hit OOM,
drop `--max-num-seqs` to 24 before touching anything else.

**Disk.** Candidates are stored as extracted code plus a 2.5 KB reasoning tail, not full
transcripts, so `work/` stays around 100–200 MB at K=16. `--keep-full-text` stores everything
and will run you 5–10 GB — only needed for `--content-mode raw`.

---

## Judging on WSL2

```bash
scp -P <port> root@<host>:~/vt3b-ojbench-clr/'model_response_*.jsonl' .

python - <<'EOF'
import ojbench
from pathlib import Path
ojbench.init(problem_dirs=[Path('OJBench_testdata/NOI'), Path('OJBench_testdata/ICPC')])
ojbench.judge_jsonl('model_response_clr.jsonl', 'judged_clr.jsonl', num_workers=8)
for i in range(8):
    ojbench.judge_jsonl(f'model_response_base_{i}.jsonl', f'judged_base_{i}.jsonl', num_workers=8)
EOF

python score.py judged_clr.jsonl
python score.py judged_base_*.jsonl     # averages them, with a std dev
```

---

## Reporting it honestly

The paper's 38.6 is a pass@1 mean over 8 independent rollouts. Best-of-16-with-a-verifier is
a different protocol and will score higher almost mechanically — that is not cheating, it is
what test-time scaling *is*, and it is exactly why Table 2 lists `VibeThinker-3B` and `+ CLR`
as two separate rows. Report it the same way:

| | OJBench |
|---|---|
| VibeThinker-3B (pass@1, 8 rollouts) | your `score.py judged_base_*.jsonl` mean |
| + CLR (K=16) | your `score.py judged_clr.jsonl` |

The baseline files come from the same generations as the CLR file, so the delta between the
two rows is attributable to selection alone. That is the number worth quoting. Quoting only
the CLR figure against the paper's 38.6 compares two different protocols.

`--nudge` is off by default for the same reason: the prompt goes to the model exactly as
OJBench wrote it, so nothing in the delta comes from prompt engineering. Turn it on if you
want the extra point, but then say so.

---

## Knobs worth turning

| flag | default | effect |
|---|---|---|
| `--k` (stage 1) | 16 | more candidates → better ceiling, linear cost. 32 if you have the hours |
| `--top-n` (stage 3) | 6 | distinct programs given the full CLR treatment. 2 model calls each |
| `--max-new-tokens` | 24576 | truncated answers have no code block; see the diagnostic below |
| `--gate-fail` | 0.12 | how much to trust a candidate that failed the samples |
| `--alpha` | 0.25 | weight on self-consistency vs. claim reliability |
| `--no-llm` (stage 3) | off | skip CLR entirely: execution gate + majority vote. Useful ablation |

Check how often the model ran out of room to think:

```bash
python -c "
import json,collections
c=collections.Counter(json.loads(l)['finish'] for l in open('work/candidates.jsonl'))
print(c)"
```

More than ~10% `length` means raise `--max-new-tokens` or `--max-model-len`.

---

## Troubleshooting

**Context beyond 32K.** Qwen2.5-Coder-3B declares `max_position_embeddings: 32768`, and the
paper's RL used a 64K window. Check what the checkpoint actually ships:

```bash
python -c "import json;c=json.load(open('/workspace/VibeThinker-3B/config.json'));print(c['max_position_embeddings'], c.get('rope_scaling'))"
```

If it says 32768 and you want more, add YaRN to `config.json`:

```json
"rope_scaling": {"type": "yarn", "factor": 2.0, "original_max_position_embeddings": 32768}
```

then pass `--max-model-len 65536`. It costs some short-context quality and doubles KV memory,
and most OJ solutions fit in 32K, so I would leave it alone unless the `finish` counter says
otherwise.

**Chat template fails to render.** The model card asks for `transformers>=4.54` but 4.45.2
renders Qwen2.5 templates fine in practice. If it does throw, the loader falls back to plain
ChatML automatically and says so. To force a specific template:
`VT_CHAT_TEMPLATE=/path/to/template.jinja python 1_generate.py ...`. You can read the real one
out of `/workspace/VibeThinker-3B/tokenizer_config.json`.

**`enable_prefix_caching` errors on 0.6.3** — pass `--no-prefix-caching`. You lose some speed
on the K-way sampling, nothing else.

**`kv_cache_dtype=fp8`** roughly doubles cache capacity on Ada but needs a backend that
supports it (`VLLM_ATTENTION_BACKEND=FLASHINFER`). Skip unless you are memory-bound.

**Low sample coverage in stage 2.** If far fewer than half the tasks get parsed samples, the
statement format differs from what the regexes expect. Look at a raw prompt, then either
extend `_IN_WORDS`/`_OUT_WORDS` in `common.py` or supply your own via
`--samples-override file.jsonl` with rows `{"id": 1000, "samples": [{"input": "...", "output": "..."}]}`.
Tasks with no samples are not lost — they fall into `nosamples` mode and are ranked by CLR alone.

**Zero candidates passing on a task you expect to be easy** usually means the samples were
misparsed, not that all 16 rollouts were wrong. Stage 3 handles it (`degraded` mode scales
every candidate equally and lets CLR decide) but check `work/samples.jsonl` for that id.

**Safety.** Stage 2 executes model-written code with rlimits and a timeout but no real
sandbox. Run it on the disposable vast.ai VM, never on your own machine.

---

## If the score still isn't where you want it

In rough order of expected value per GPU-hour:

1. **Self-repair round.** For every task where nothing passed the samples, feed the failing
   input, expected output and actual output back and sample 8 more. OJBench's own paper found
   models use execution feedback well, and these are exactly the tasks you are losing. This is
   the biggest single lever and it isn't in this pipeline yet.
2. **Generated stress inputs.** Ask the model for a small generator that emits one random valid
   input, run every candidate on 8 of them, and cluster on those outputs. Clustering is weak
   right now because everything that survives the gate agrees on the samples by construction —
   that is what the `α` term is voting on. This needs a small change in `common.py`
   (`run_candidate_on_tests` currently folds pass/fail and signature into one pass; you want
   the signature computed over extra inputs that have no expected output).
3. **Per-claim verification.** The paper verifies each of the M claims in its own pass; this
   does all five in one call to save tokens. Splitting them is 5× the cost for a sharper signal.
4. **Raise K to 32.** Reliable but purely linear.
