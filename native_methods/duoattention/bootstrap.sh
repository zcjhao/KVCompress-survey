#!/usr/bin/env bash
# Safe stage-1 bootstrap. Does not modify shared scripts, baselines, or the existing .venv.
set -euo pipefail
THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$THIS_DIR/../.." && pwd)"
WORKSPACE="$(dirname "$ROOT")"
UPSTREAM="$WORKSPACE/duoattention-upstream"
PINNED_COMMIT="fe93c314ae87306ef6629dc16713250b4718ffe7"

if [[ ! -f "$ROOT/scripts/eval_ruler.py" || ! -f "$ROOT/requirements.lock.txt" ]]; then
  echo 'ERROR: expected to run inside KVCompress-survey/native_methods/duoattention' >&2
  exit 1
fi
if ! command -v git >/dev/null 2>&1; then
  echo 'ERROR: git not installed.' >&2
  exit 1
fi
printf 'Team root: %s\nUpstream: %s\n' "$ROOT" "$UPSTREAM"
mkdir -p "$WORKSPACE/native-cache" "$WORKSPACE/native-results"
if [[ ! -e "$UPSTREAM" ]]; then
  git clone https://github.com/mit-han-lab/duo-attention.git "$UPSTREAM"
  git -C "$UPSTREAM" checkout --detach "$PINNED_COMMIT"
elif [[ ! -d "$UPSTREAM/.git" ]]; then
  echo "ERROR: $UPSTREAM already exists but is not a Git checkout" >&2
  exit 1
else
  echo 'Reusing existing official checkout; no automatic git pull.'
fi
printf '\nOfficial commit (record this):\n'
git -C "$UPSTREAM" rev-parse HEAD
if [[ "$(git -C "$UPSTREAM" rev-parse HEAD)" != "$PINNED_COMMIT" ]]; then
  echo "WARNING: checkout is not at the pinned pattern commit; do not run reproducibility tests yet." >&2
fi
printf '\nOfficial patch symbols:\n'
grep -R -n -m 1 'def enable_duo_attention_eval' "$UPSTREAM/duo_attn" || true
printf '\nFrozen head pattern SHA256:\n'
sha256sum "$ROOT/assets/duoattention_pattern/full_attention_heads.tsv"
printf '\nDisk:\n'
df -h "$WORKSPACE"
printf '\nGPU:\n'
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
printf '\nStage 1 complete. No packages installed, no model re-downloaded, and no team code changed.\n'
