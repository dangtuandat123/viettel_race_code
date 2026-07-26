#!/usr/bin/env bash
# Full flow for ONE private scene:
#   prepare -> fold-0 train (pseudo-val) -> full train -> render test poses
#   -> [if RIFE installed] VFI gate on fold + apply to test (pred_test_final)
#   bash scripts/run_private_scene.sh HCM0249
set -euo pipefail
cd "$(dirname "$0")/.."

SCENE=${1:?usage: run_private_scene.sh <SCENE>}
DATA=${DATA:-/workspace/VAI_NVS_DATA/phase1/private_set1}
WORK=${WORK:-/workspace/work}
STRAT=${STRAT:-mcmc}
STEPS=${STEPS:-40000}
FOLD0=${FOLD0:-1}          # FOLD0=0: skip the fold-0 training run (saves ~40 min/scene;
                           # VFI gating then reuses the newest existing fold-0 checkpoint)
EXTRA_ARGS=${EXTRA_ARGS:-} # extra train_gs flags, e.g. "--lpips-weight 0.05"
RIFE_DIR=${RIFE_DIR:-/workspace/Practical-RIFE}
RUN_FULL="${STRAT}_f-1_s$((STEPS/1000))k"
RUN_FOLD="${STRAT}_f0_s$((STEPS/1000))k"

python -m vai_nvs.prepare --data "$DATA/$SCENE" --work "$WORK"
if [ "$FOLD0" = "1" ]; then
  python -m vai_nvs.train_gs --work "$WORK" --scene "$SCENE" --fold 0 --strategy "$STRAT" --max-steps "$STEPS" $EXTRA_ARGS
fi
python -m vai_nvs.train_gs --work "$WORK" --scene "$SCENE" --fold -1 --strategy "$STRAT" --max-steps "$STEPS" $EXTRA_ARGS
python -m vai_nvs.render_test --work "$WORK" --scene "$SCENE" --run "$RUN_FULL"

if [ -f "$RIFE_DIR/train_log/flownet.pkl" ]; then
  echo "--- VFI (RIFE) branch: fold gating + test apply ---"
  # pick a fold-0 checkpoint for honest (out-of-fold) gating: this run's if it
  # exists, else the newest older fold-0 run, else fall back to the full ckpt
  if [ -f "$WORK/$SCENE/runs/$RUN_FOLD/ckpt_last.pt" ]; then
    VFI_VAL_RUN="$RUN_FOLD"
  else
    ALT=$(ls -dt "$WORK/$SCENE"/runs/*_f0_* 2>/dev/null | head -1 | xargs -r basename || true)
    VFI_VAL_RUN=${ALT:-$RUN_FULL}
  fi
  echo "VFI gating uses run: $VFI_VAL_RUN"
  python -m vai_nvs.vfi --work "$WORK" --scene "$SCENE" --run "$VFI_VAL_RUN" \
    --mode val --fold 0 --rife-dir "$RIFE_DIR" 2>&1 | tee "$WORK/$SCENE/vfi_val.log"
  python -m vai_nvs.vfi --work "$WORK" --scene "$SCENE" --run "$RUN_FULL" \
    --mode test --rife-dir "$RIFE_DIR"
else
  echo "NOTE: RIFE not found at $RIFE_DIR (run: bash setup_rife.sh) — skipping VFI, "
  echo "      submission will use plain GS renders (pred_test)."
fi

echo "DONE $SCENE — fold metrics: $WORK/$SCENE/runs/$RUN_FOLD/eval.jsonl"
