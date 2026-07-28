#!/usr/bin/env bash
# All round-2 scenes sequentially, then package the submission from
# pred_test_final (GS).
#   bash scripts/run_all_private.sh
set -uo pipefail
cd "$(dirname "$0")/.."

DATA=${DATA:-/workspace/VAI_NVS_DATA_ROUND2}
WORK=${WORK:-/workspace/work}

python -m vai_nvs.audit --data "$DATA" --out "$WORK/audit_round2.json"

for S in $(ls -d "$DATA"/*/ | xargs -n1 basename); do
  echo "================ SCENE $S ================"
  bash scripts/run_private_scene.sh "$S"
done

python -m vai_nvs.make_submission --data "$DATA" --work "$WORK" \
  --pred-dirname pred_test_final --out "$WORK/submission_round2.zip"
echo "Packaged from: pred_test_final"
