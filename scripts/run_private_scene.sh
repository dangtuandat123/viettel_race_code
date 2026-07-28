#!/usr/bin/env bash
# Round-2 flow for ONE scene:
#   prepare -> fold-0 train (for evaluation) -> full train -> final render
# Result: <WORK>/<SCENE>/pred_test_final
#   bash scripts/run_private_scene.sh bonsai
set -euo pipefail
cd "$(dirname "$0")/.."

SCENE=${1:?usage: run_private_scene.sh <SCENE>}
DATA=${DATA:-/workspace/VAI_NVS_DATA_ROUND2}
WORK=${WORK:-/workspace/work}
STRAT=${STRAT:-mcmc}
STEPS=${STEPS:-45000}
FOLD0=${FOLD0:-1}          # FOLD0=1: run fold-0 for validation metrics

# Defaults: MCMC with 3M cap, SH degree 2; all other params use train_gs.py
# code defaults (init-opacity 0.1, init-scale 1.0, lpips-weight 0.1, etc.).
TRAIN_FLAGS=${TRAIN_FLAGS:---cap-max 1500000 --sh-degree 3}
EXTRA_ARGS=${EXTRA_ARGS:---grow-grad2d 0.0005 --lpips-weight 0.1 --ssim-lambda 0.25}
RUN_FULL="${STRAT}_f-1_s$((STEPS/1000))k"
RUN_FOLD="${STRAT}_f0_s$((STEPS/1000))k"

python -m vai_nvs.prepare --data "$DATA/$SCENE" --work "$WORK"

# --- fold-0 train (validation) ---
if [ "$FOLD0" = "1" ]; then
  python -m vai_nvs.train_gs --work "$WORK" --scene "$SCENE" --fold 0 \
    --strategy "$STRAT" --max-steps "$STEPS" $TRAIN_FLAGS $EXTRA_ARGS
fi

# --- full training (all images) ---
python -m vai_nvs.train_gs --work "$WORK" --scene "$SCENE" --fold -1 \
  --strategy "$STRAT" --max-steps "$STEPS" $TRAIN_FLAGS $EXTRA_ARGS

# --- final renders (GS only) ---
python -m vai_nvs.render_test --work "$WORK" --scene "$SCENE" --run "$RUN_FULL" \
  --which last --out "$WORK/$SCENE/pred_test_final"

echo "DONE $SCENE"
if [ "$FOLD0" = "1" ]; then
  echo "  fold-0 GS metrics : $WORK/$SCENE/runs/$RUN_FOLD/eval.jsonl"
fi
echo "  submission source : $WORK/$SCENE/pred_test_final"
