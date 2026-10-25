#!/usr/bin/env bash
# Set up the Python environment on the GPU VM. Run from the project root:
#   bash setup_env.sh            # first person: resolve versions and write requirements.lock.txt
#   bash setup_env.sh --locked   # everyone else: install exactly the committed lock file
set -euo pipefail
cd "$(dirname "$0")"

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# uv fetches its own Python, so we don't depend on the image's system Python.
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

uv venv --python 3.11 .venv
source .venv/bin/activate

if [[ "${1:-}" == "--locked" ]]; then
  uv pip install -r requirements.lock.txt
else
  uv pip install -r requirements.txt
  uv pip freeze > requirements.lock.txt
  echo "Wrote requirements.lock.txt. Commit it so the team installs identical versions."
fi

python - <<'EOF'
import torch, transformers, kvpress
print("torch", torch.__version__, "| CUDA", torch.version.cuda, "| GPU", torch.cuda.get_device_name(0))
print("transformers", transformers.__version__, "| kvpress", kvpress.__version__ if hasattr(kvpress, "__version__") else "?")
EOF

echo
echo "Next: log in to Hugging Face (Llama 3.1 is gated; accept the licence on its model page first):"
echo "  source .venv/bin/activate && hf auth login"
