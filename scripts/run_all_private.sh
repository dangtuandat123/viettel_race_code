#!/usr/bin/env bash
# All 8 private scenes sequentially, then package the submission.
# Uses pred_test_final (GS + gated VFI) when RIFE is installed, else pred_test.
#   bash scripts/run_all_private.sh
set -uo pipefail
cd "$(dirname "$0")/.."

DATA=${DATA:-/workspace/VAI_NVS_DATA/phase1/private_set1}
WORK=${WORK:-/workspace/work}
RIFE_DIR=${RIFE_DIR:-/workspace/Practical-RIFE}

python -m vai_nvs.audit --data "$DATA" --out "$WORK/audit_private.json"

for S in $(ls -d "$DATA"/*/ | xargs -n1 basename); do
  echo "================ PRIVATE $S ================"
  bash scripts/run_private_scene.sh "$S"
done

if [ -f "$RIFE_DIR/train_log/flownet.pkl" ]; then
  PRED=pred_test_final
  echo "--- VFI decisions ---"
  grep -h '"mode"' "$WORK"/*/vfi_decision.json || true
else
  PRED=pred_test
fi
python -m vai_nvs.make_submission --data "$DATA" --work "$WORK" \
  --pred-dirname "$PRED" --out "$WORK/submission_round1.zip"
echo "Packaged from: $PRED"
