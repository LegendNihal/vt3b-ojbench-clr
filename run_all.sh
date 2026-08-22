#!/usr/bin/env bash
# Full pipeline on the vast.ai box. Edit the knobs, then: bash run_all.sh
set -euo pipefail

MODEL=${MODEL:-/workspace/VibeThinker-3B}
PROMPTS=${PROMPTS:-OJBench_testdata/prompts/full.jsonl}
K=${K:-8}                      # candidates per task
NEWTOK=${NEWTOK:-26000}        # see "Budgeting the run"
TOPN=${TOPN:-6}                # programs per task that get the CLR treatment
MAXLEN=${MAXLEN:-32768}
WORKERS=${WORKERS:-8}          # CPU threads for the sandbox
BASELINE_ROLLOUTS=${BASELINE_ROLLOUTS:-8}
N_INPUTS=${N_INPUTS:-6}      # generated inputs per problem
# whatever ablate.py found, e.g. SAMPLING="--top-k 50" or "--min-p 0.05" or "--temperature 0.6"
SAMPLING=${SAMPLING:-}
# engine flags. Default (empty) = CUDA graphs OFF, which is the safe config.
# If ablate.py --configs engine shows no_async_out is clean, use:
#   ENGINE="--cuda-graphs --no-async-output"   for ~40% more throughput
ENGINE=${ENGINE:-}
KVCACHE=${KVCACHE:-fp8}      # doubles KV capacity; validated by ablate.py --configs speed

mkdir -p work logs

echo "### stage 1: generating ${K} candidates per task"
python 1_generate.py \
  --model "$MODEL" --prompts "$PROMPTS" --out work/candidates.jsonl \
  --k "$K" --max-model-len "$MAXLEN" --max-new-tokens "$NEWTOK" --kv-cache-dtype "$KVCACHE" $SAMPLING $ENGINE \
  2>&1 | tee -a logs/1_generate.log

echo "### stage 2c: input generators (OJBench strips NOI samples, so this is"
echo "###            the only execution signal on 159 of the 232 problems)"
python 2c_gen_inputs.py \
  --model "$MODEL" --prompts "$PROMPTS" --out work/stress.jsonl \
  --n-inputs "$N_INPUTS" --max-model-len "$MAXLEN" --kv-cache-dtype "$KVCACHE" $SAMPLING $ENGINE \
  2>&1 | tee -a logs/2c_gen.log

echo "### stage 2: execution gate + behavioural signatures"
python 2_verify_samples.py \
  --prompts "$PROMPTS" --candidates work/candidates.jsonl \
  --out work/verify.jsonl --workers "$WORKERS" --stress-file work/stress.jsonl \
  2>&1 | tee -a logs/2_verify.log

echo "### stage 3: claim-level reliability assessment"
python 3_clr_rank.py \
  --model "$MODEL" --prompts "$PROMPTS" \
  --candidates work/candidates.jsonl --verify work/verify.jsonl \
  --out work/selection.jsonl --top-n "$TOPN" --max-model-len "$MAXLEN" --kv-cache-dtype "$KVCACHE" $SAMPLING $ENGINE \
  2>&1 | tee -a logs/3_clr.log

echo "### stage 4: building submissions"
python 4_build_response.py --prompts "$PROMPTS" --mode clr \
  --out model_response_clr.jsonl

for i in $(seq 0 $((BASELINE_ROLLOUTS - 1))); do
  python 4_build_response.py --prompts "$PROMPTS" --mode single --take-idx "$i" \
    --out "model_response_base_${i}.jsonl" >/dev/null
done
echo "baseline files: model_response_base_0..$((BASELINE_ROLLOUTS - 1)).jsonl"

echo
echo "Done. Copy these to the judge machine:"
ls -la model_response_clr.jsonl model_response_base_*.jsonl