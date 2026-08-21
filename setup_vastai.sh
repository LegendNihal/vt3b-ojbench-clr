#!/usr/bin/env bash
# Run once on the vast.ai 4090 box. Assumes Ubuntu 22.04/24.04 with CUDA drivers.
set -euo pipefail

# vast.ai images often have a small /, so keep the HF cache on the big volume
export HF_HOME=${HF_HOME:-/workspace/hf}
mkdir -p "$HF_HOME"
echo "export HF_HOME=$HF_HOME" >> ~/.bashrc

echo "== system packages =="
apt-get update -qq
# g++ compiles the C++ candidates; pypy3 is what OJBench itself uses for Python,
# so verifying with plain CPython would give you the wrong TLE picture
apt-get install -y -qq build-essential g++ pypy3 git curl

echo "== python packages =="
# these two versions are known to work together; vllm 0.6.3 pulls torch 2.4.0
pip install -q "vllm==0.6.3.post1"
pip install -q "transformers==4.45.2"
# vllm 0.6.3 predates the numpy 2 ABI break
pip install -q "numpy<2"
# transformers needs jinja2>=3.1 for chat templates; some images ship 3.0.x
pip install -q "jinja2>=3.1.4"
pip install -q "huggingface_hub[cli]"
# needed by ablate.py's transformers control run
pip install -q "accelerate>=0.26.0"

echo "== download the model (~6.2 GB) =="
huggingface-cli download WeiboAI/VibeThinker-3B --local-dir /workspace/VibeThinker-3B

echo
echo "== versions =="
python -c "import torch,transformers,vllm;print('torch',torch.__version__);print('transformers',transformers.__version__);print('vllm',vllm.__version__);print('cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
g++ --version | head -1
pypy3 --version | head -1

cat <<'EOF'

== chat template check ==
The model card asks for transformers>=4.54 but 4.45.2 works as long as the chat
template renders. Verify it now:
EOF
python - <<'PY'
from transformers import AutoTokenizer
try:
    t = AutoTokenizer.from_pretrained("/workspace/VibeThinker-3B", trust_remote_code=True)
    if t.chat_template is None:
        print("!! no chat_template in the repo -- the pipeline will use its ChatML fallback")
    else:
        print(t.apply_chat_template([{"role": "user", "content": "hi"}],
                                    tokenize=False, add_generation_prompt=True))
except ImportError as e:
    print(f"!! {e}")
    print("!! fix with:  pip install -U 'jinja2>=3.1.4'   then re-run this check")
except Exception as e:
    print(f"!! chat template check failed: {e}")
    print("!! the pipeline falls back to ChatML, but investigate before a long run")
PY

cat <<'EOF'

Also check the context window the checkpoint declares:
  python -c "import json;c=json.load(open('/workspace/VibeThinker-3B/config.json'));print('max_position_embeddings',c['max_position_embeddings']);print('rope_scaling',c.get('rope_scaling'))"

If it says 32768 and you want the 64K the paper's RL used, see the README.
Setup done.
EOF