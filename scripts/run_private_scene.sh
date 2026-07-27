#!/usr/bin/env bash
# Round-2 flow for ONE scene:
#   prepare -> fold-0 train -> warp_fuse gate (see scores early)
#   -> full train -> render baseline -> warp_fuse test -> final render
# Result: <WORK>/<SCENE>/pred_test_final (== baseline GS when fusion is gated off)
#   bash scripts/run_private_scene.sh bonsai
set -euo pipefail
cd "$(dirname "$0")/.."

SCENE=${1:?usage: run_private_scene.sh <SCENE>}
DATA=${DATA:-/workspace/VAI_NVS_DATA_ROUND2}
WORK=${WORK:-/workspace/work}
STRAT=${STRAT:-mcmc}
STEPS=${STEPS:-60000}
FOLD0=${FOLD0:-1}          # FOLD0=0: skip the fold-0 training run (fusion gate
                           # then needs an existing fold-0 checkpoint)
# Round-2 defaults: per-image appearance (video AE drift), random background
# (floaters), canonical MCMC init (keep all SfM points), stronger LPIPS loss.
TRAIN_FLAGS=${TRAIN_FLAGS:---appearance --random-bkgd \
  --init-opacity 0.5 --init-scale 0.1 --init-max-error 100 \
  --lpips-weight 0.2 --lpips-start-frac 0.25 --lpips-crop 640}
EXTRA_ARGS=${EXTRA_ARGS:-} # appended to train_gs, e.g. "--cap-max 1500000 --sh-degree 2"
FUSE_ARGS=${FUSE_ARGS:-}   # extra warp_fuse flags for both val and test
RUN_FULL="${STRAT}_f-1_s$((STEPS/1000))k"
RUN_FOLD="${STRAT}_f0_s$((STEPS/1000))k"

python -m vai_nvs.prepare --data "$DATA/$SCENE" --work "$WORK"

# --- fold-0 train (pseudo-validation) ---
if [ "$FOLD0" = "1" ]; then
  python -m vai_nvs.train_gs --work "$WORK" --scene "$SCENE" --fold 0 \
    --strategy "$STRAT" --max-steps "$STEPS" $TRAIN_FLAGS $EXTRA_ARGS
fi

# --- warp_fuse gating on fold-0 (scores visible BEFORE full training) ---
GATE_RUN=""
if [ -f "$WORK/$SCENE/runs/$RUN_FOLD/ckpt_last.pt" ]; then
  GATE_RUN="$RUN_FOLD"
else
  ALT=$(ls -dt "$WORK/$SCENE"/runs/*_f0_* 2>/dev/null | head -1 | xargs -r basename || true)
  GATE_RUN=${ALT:-}
fi
if [ -n "$GATE_RUN" ]; then
  echo "--- warp_fuse gating on fold-0 run: $GATE_RUN ---"
  python -m vai_nvs.warp_fuse --work "$WORK" --scene "$SCENE" --run "$GATE_RUN" \
    --mode val --fold 0 $FUSE_ARGS 2>&1 | tee "$WORK/$SCENE/fusion_val.log"
else
  echo "NOTE: no fold-0 checkpoint found — skipping fusion gate (pred_test_final = GS only)."
fi

# --- full training (all images) ---
python -m vai_nvs.train_gs --work "$WORK" --scene "$SCENE" --fold -1 \
  --strategy "$STRAT" --max-steps "$STEPS" $TRAIN_FLAGS $EXTRA_ARGS

# baseline GS renders (always produced; also the fallback inside the final dir)
python -m vai_nvs.render_test --work "$WORK" --scene "$SCENE" --run "$RUN_FULL"

# --- warp fusion on test (uses full checkpoint; skipped if gate disabled) ---
if [ -n "$GATE_RUN" ]; then
  python -m vai_nvs.warp_fuse --work "$WORK" --scene "$SCENE" --run "$RUN_FULL" \
    --mode test $FUSE_ARGS
fi

# final renders: fused .npy used where present, per-image GS fallback otherwise
python -m vai_nvs.render_test --work "$WORK" --scene "$SCENE" --run "$RUN_FULL" \
  --fused-dir "$WORK/$SCENE/fused_test" --out "$WORK/$SCENE/pred_test_final"

echo "DONE $SCENE"
echo "  fold-0 GS metrics : $WORK/$SCENE/runs/$RUN_FOLD/eval.jsonl"
echo "  fusion decision   : $WORK/$SCENE/fusion_decision.json"
echo "  submission source : $WORK/$SCENE/pred_test_final"
