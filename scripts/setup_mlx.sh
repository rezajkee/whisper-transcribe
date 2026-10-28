#!/bin/bash
# Explicit setup/download command. Ordinary worker runs are offline.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${MLX_VENV:-$HOME/whisper-mlx-env}"
PREFETCH=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefetch) PREFETCH=1; shift ;;
    --venv) VENV="$2"; shift 2 ;;
    -h|--help) echo 'Usage: scripts/setup_mlx.sh [--venv PATH] [--prefetch]'; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
  echo 'MLX requires Apple Silicon macOS.' >&2; exit 1
fi
RESOLVED_VENV="$("${PYTHON_MLX:-python3.12}" -c 'import os,sys; print(os.path.realpath(os.path.expanduser(sys.argv[1])))' "$VENV")"
RESOLVED_WHISPER="$("${PYTHON_MLX:-python3.12}" -c 'import os; print(os.path.realpath(os.path.expanduser("~/whisper-env")))')"
if [[ "$RESOLVED_VENV" == "$RESOLVED_WHISPER" ]]; then
  echo 'Refusing to modify the existing WhisperX environment.' >&2; exit 1
fi
if [[ ! -x "$VENV/bin/python" ]]; then
  "${PYTHON_MLX:-python3.12}" -m venv "$VENV"
fi
"$VENV/bin/python" -m pip install -r "$ROOT/requirements-mlx.txt"
if [[ "$PREFETCH" == 1 ]]; then
  cd "$ROOT"
  "$VENV/bin/python" - <<'PY'
from huggingface_hub import snapshot_download
from whisper_transcribe.asr_worker import MODELS
from whisper_transcribe.reconcile import MODEL, REVISION
for engine, (model, revision) in MODELS.items():
    path = snapshot_download(model, revision=revision)
    print(f'{engine}: {revision} cached at {path}')
path = snapshot_download(MODEL, revision=REVISION)
print(f'reconciliation: {REVISION} cached at {path}')
PY
fi
printf 'MLX environment ready: %s\n' "$VENV"
