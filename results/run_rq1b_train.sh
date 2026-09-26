#!/usr/bin/env bash
# RQ1b -- train RiskProp-full for seeds 42, 43, 44 one after another with the authors'
# tools/train.py. Safe to re-run: finished seeds are skipped, an interrupted seed resumes
# from its last epoch checkpoint.
#
# Usage (inside tmux):
#   bash /workspace/CPV301/riskprop_full/run_rq1b_train.sh            # fp32 (needs ~80 GB GPU)
#   AMP=1 bash /workspace/CPV301/riskprop_full/run_rq1b_train.sh      # mixed precision (deviation)
#   SEEDS="42" bash ...                                               # only some seeds
set -uo pipefail
WS=/workspace
CPV=$WS/CPV301
RP=$WS/RiskProp
# shellcheck disable=SC1091
source "$WS/rq1b_env/bin/activate"
export PYTHONPATH="$CPV/riskprop_full:${PYTHONPATH:-}"
cd "$RP" || exit 1
mkdir -p logs_rq1b

AMP_FLAG=""
if [ "${AMP:-0}" = "1" ]; then AMP_FLAG="--amp"; fi

for SEED in ${SEEDS:-42 43 44}; do
    WD=work_dirs/rq1b_seed$SEED
    if [ -f "$WD/DONE" ]; then echo "[skip] seed $SEED already finished"; continue; fi
    # IMPORTANT: the authors' tools/train.py sets load_from=None when --resume is given.
    # So --resume is passed ONLY when this seed already has checkpoints; a fresh run must
    # start without it, otherwise the Kinetics-710 weights (load_from) would be skipped.
    RESUME_FLAG=""
    if [ -f "$WD/last_checkpoint" ]; then RESUME_FLAG="--resume"; fi
    echo "[$(date '+%F %T')] START seed $SEED ${AMP_FLAG} ${RESUME_FLAG:-(fresh start, loads Kinetics-710)}"
    python -u tools/train.py configs/riskprop_full_nexar.py --work-dir "$WD" $RESUME_FLAG $AMP_FLAG \
        --cfg-options randomness.seed=$SEED 2>&1 | tee -a "logs_rq1b/seed$SEED.log"
    if [ "${PIPESTATUS[0]}" -eq 0 ] && [ -f "$WD/epoch_50.pth" ]; then
        touch "$WD/DONE"
        echo "[$(date '+%F %T')] DONE seed $SEED"
    else
        echo "[$(date '+%F %T')] FAILED seed $SEED -- see logs_rq1b/seed$SEED.log; fix, then re-run this script (it resumes)."
        exit 1
    fi
done
echo "[$(date '+%F %T')] all seeds finished"
ls -la work_dirs/rq1b_seed*/epoch_50.pth
