#!/usr/bin/env bash
# End-to-end validation on a handful of problems, ~30-40 min.
# Run this before the full job. It exercises every stage and then prints the
# three numbers that decide whether the full run is worth starting.
set -euo pipefail

MODEL=${MODEL:-/workspace/VibeThinker-3B}
PROMPTS=${PROMPTS:-full.jsonl}
NTASKS=${NTASKS:-8}          # rows of full.jsonl = NTASKS/2 problems
DATASETS=${DATASETS:-}       # e.g. icpc  -- restrict the smoke set to one half
K=${K:-4}
MAXLEN=${MAXLEN:-32768}
NEWTOK=${NEWTOK:-26000}   # see "Budgeting the run"; 16000 truncated 87% of rollouts
SAMPLING=${SAMPLING:-}
ENGINE=${ENGINE:-}           # empty = CUDA graphs off, the config ablate.py validated
KVCACHE=${KVCACHE:-fp8}      # validated clean by ablate.py --configs speed

mkdir -p work logs
W=work/smoke

echo "### 1/4 generate  (${NTASKS} tasks x ${K})"
python 1_generate.py --model "$MODEL" --prompts "$PROMPTS" --out $W/cand.jsonl \
  --k "$K" --limit "$NTASKS" --shuffle --max-model-len "$MAXLEN" --max-new-tokens "$NEWTOK" \
  --kv-cache-dtype "$KVCACHE" $SAMPLING $ENGINE --fresh 2>&1 | tee logs/smoke_1.log

echo "### 2/4 input generators"
python 2c_gen_inputs.py --model "$MODEL" --prompts "$PROMPTS" --out $W/stress.jsonl \
  --gen-out $W/generators.jsonl --limit "$NTASKS" --max-model-len "$MAXLEN" \
  --kv-cache-dtype "$KVCACHE" $SAMPLING $ENGINE 2>&1 | tee logs/smoke_2c.log

echo "### 3/4 verify"
python 2_verify_samples.py --prompts "$PROMPTS" --candidates $W/cand.jsonl \
  --out $W/verify.jsonl --samples-out $W/samples.jsonl --stress-file $W/stress.jsonl \
  --fresh 2>&1 | tee logs/smoke_2.log

echo "### 4/4 CLR + build"
python 3_clr_rank.py --model "$MODEL" --prompts "$PROMPTS" --candidates $W/cand.jsonl \
  --verify $W/verify.jsonl --out $W/sel.jsonl --trace-out $W/trace.jsonl \
  --top-n 4 --max-model-len "$MAXLEN" --kv-cache-dtype "$KVCACHE" \
  $SAMPLING $ENGINE 2>&1 | tee logs/smoke_3.log

python 4_build_response.py --prompts "$PROMPTS" --candidates $W/cand.jsonl \
  --selection $W/sel.jsonl --out smoke_response.jsonl

echo
echo "================ HEALTH CHECK ================"
python - <<'PY'
import json, collections, pathlib
W = pathlib.Path("work/smoke")
rd = lambda p: [json.loads(l) for l in open(p) if l.strip()]

cand = rd(W/"cand.jsonl")
have = sum(1 for c in cand if c.get("valid"))
anycode = sum(1 for c in cand if c.get("code"))
fin = collections.Counter(c.get("finish") for c in cand)
rate = 100*have/max(len(cand),1)
print(f"1. USABLE program rate    : {have}/{len(cand)}  ({rate:.0f}%)   want > 40%")
print(f"   (had some code block   : {anycode}/{len(cand)} -- the gap is truncated fragments)")
print(f"   finish reasons         : {dict(fin)}")
if fin.get("length",0) > 0.15*len(cand):
    print("   -> many hit the cap; raise --max-new-tokens / --max-model-len")
if rate < 80:
    print("   -> still losing rollouts. Re-run diagnose.py on work/smoke/cand.jsonl")

ver = rd(W/"verify.jsonl")
bytask = collections.defaultdict(list)
for r in ver: bytask[r["key"]].append(r)
multi = sum(1 for v in bytask.values()
            if len({x["signature"] for x in v if x["code_hash"]!="NOCODE"}) > 1)
print(f"\n2. tasks splitting into >1 behaviour: {multi}/{len(bytask)}   want most of them")
if multi == 0:
    print("   -> clustering is doing nothing. Check work/smoke/stress.jsonl is non-empty")

sel = rd(W/"sel.jsonl")
modes = collections.Counter(s["mode"] for s in sel)
print(f"\n3. selection modes        : {dict(modes)}")
print("   gated = samples passed | nosamples = no samples, CLR+clustering")
print("   degraded = samples exist but NOTHING passed | empty = nothing usable")
if modes.get("empty",0) > 0.2*len(sel):
    print("   -> too many empty; that traces back to check 1")

# the gate is the core of the design; if samples exist it must actually fire
smp = {r["key"]: r["n"] for r in rd(W/"samples.jsonl")}
with_s = [s for s in sel if smp.get(s["key"], 0) > 0]
if with_s:
    g = sum(1 for s in with_s if s["mode"] == "gated")
    print(f"\n3b. tasks WITH samples that reached 'gated': {g}/{len(with_s)}")
    if g == 0:
        print("   -> RED FLAG. Samples parsed but no candidate ever passed them.")
        print("      Either every rollout is wrong, or the parsed expected outputs are")
        print("      wrong. Check work/smoke/samples.jsonl against a real statement")
        print("      before trusting the gate on a full run.")
else:
    print("\n3b. no task in this smoke set had samples -- the execution gate was never")
    print("    exercised. Re-run with DATASETS=icpc to test it.")

tr = W/"trace.jsonl"
if tr.exists():
    t = rd(tr)
    ok = sum(1 for x in t if x.get("parsed"))
    print(f"\n4. CLR verdicts parsed    : {ok}/{len(t)}   want > 80%")
    if ok < 0.8*len(t):
        print("   -> raise --verify-tokens in stage 3")
PY
echo "=============================================="
echo
echo "smoke_response.jsonl is a full-length file (every row of full.jsonl);"
echo "only the ${NTASKS} generated rows carry a real program. Judge it on WSL2 to"
echo "confirm the round trip, then start the full run."