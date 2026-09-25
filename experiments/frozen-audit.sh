#!/usr/bin/env bash
# Run from an installed rtrl checkout. All inputs/results stay outside Git.
# Usage: frozen-audit-recipe.sh OUTPUT_DIR HISTORY_DIR [recorded|collect|audit]
# recorded: CPU synthetic probe + analysis of the preserved banks (default).
# collect: explicitly regenerate the pinned MPS calibration/main banks, then analyze.
# audit: analyze preserved banks and explicitly run the MPS parameter-block audit.
set -euo pipefail
if [[ ${1:-} == --help || $# -lt 2 ]]; then
  sed -n '2,6p' "$0"
  exit 0
fi
RTRL_OUTPUT=$1
RTRL_HISTORY=$2
RTRL_MODE=${3:-recorded}
RTRL_PYTHON=${RTRL_PYTHON:-python3}
case "$RTRL_MODE" in recorded|collect|audit) ;; *) echo "Unknown mode: $RTRL_MODE" >&2; exit 2 ;; esac
mkdir -p "$RTRL_OUTPUT"
"$RTRL_PYTHON" -m rtrl probe --seed 17 --replicates 2000 --output "$RTRL_OUTPUT/synthetic-probe.jsonl"
RTRL_CALIBRATION="$RTRL_HISTORY/calibration-v2.jsonl"
RTRL_MAIN="$RTRL_HISTORY/main.jsonl"
if [[ $RTRL_MODE == collect ]]; then
  # Recorded runtime: torch2.14.0, transformers5.17.0; FP32 MPS is fixed by 'local'.
  "$RTRL_PYTHON" -c 'import torch, transformers; assert torch.__version__.split("+")[0] == "2.14.0"; assert transformers.__version__ == "5.17.0"; assert torch.backends.mps.is_available()'
  RTRL_CALIBRATION="$RTRL_OUTPUT/calibration-v2.jsonl"
  RTRL_MAIN="$RTRL_OUTPUT/main.jsonl"
  RTRL_MODEL=Qwen/Qwen2.5-0.5B-Instruct
  RTRL_REVISION=7ae557604adf67be50417f59c2c2f167def9a775
  "$RTRL_PYTHON" -m rtrl local --prompts "$RTRL_HISTORY/calibration-prompts.jsonl" --model "$RTRL_MODEL" --revision "$RTRL_REVISION" --reward rtrl.rewards:arithmetic --group-size 4 --trials 2 --attempts 1 --max-tokens 512 --seed 19000 --output "$RTRL_CALIBRATION"
  "$RTRL_PYTHON" -m rtrl local --prompts "$RTRL_HISTORY/main-prompts.jsonl" --model "$RTRL_MODEL" --revision "$RTRL_REVISION" --reward rtrl.rewards:arithmetic --group-size 4 --trials 8 --attempts 1 --max-tokens 512 --seed 27000 --output "$RTRL_MAIN"
fi
"$RTRL_PYTHON" -m rtrl diagnose --trace "$RTRL_MAIN" --calibration "$RTRL_CALIBRATION" --output "$RTRL_OUTPUT/frozen-diagnostic.jsonl"
"$RTRL_PYTHON" -m rtrl replay --trace "$RTRL_MAIN" --clock generation_s --output "$RTRL_OUTPUT/complete-replay.jsonl"
if [[ $RTRL_MODE == audit ]]; then
  "$RTRL_PYTHON" -m rtrl audit --trace "$RTRL_CALIBRATION" --device mps --parameters model.layers.23.mlp.down_proj.weight --output "$RTRL_OUTPUT/numerical-audit.jsonl"
fi
