#!/usr/bin/env bash
# run_rq3_sens.sh -- RQ3 sensitivity: FixedLag tau 0.5 / 1.5 s, seeds 42/43/44,
# via reeval/re09_train_rq3_sensitivity.py. Seed-major order (tau0.5 s42,
# tau1.5 s42, tau0.5 s43, ...) so stopping after the first 2 runs still gives
# one complete seed. Re-run any time: finished runs are skipped, an
# interrupted run resumes from latest_*.pth.
cd /workspace/CPV301 2>/dev/null || cd "$(dirname "$0")/.."
[ -f /venv/main/bin/activate ] && source /venv/main/bin/activate
LOG_DIR=outputs_riskprop/rq3_sens_logs
mkdir -p "$LOG_DIR"
for SEED in 42 43 44; do
  for TAU in 0.5 1.5; do
    DONE_MARK="$LOG_DIR/done_seed${SEED}_tau${TAU}"
    if [ -f "$DONE_MARK" ]; then echo "[skip] seed=$SEED tau=$TAU (da xong)"; continue; fi
    if [ "${DRY_RUN:-0}" = "1" ]; then
      echo "[dry] RISKPROP_SEED=$SEED RISKPROP_TAU=$TAU python -u reeval/re09_train_rq3_sensitivity.py"; continue
    fi
    echo "[$(date '+%F %T')] START seed=$SEED tau=$TAU"
    RISKPROP_SEED=$SEED RISKPROP_TAU=$TAU \
      python -u reeval/re09_train_rq3_sensitivity.py 2>&1 | tee -a "$LOG_DIR/seed${SEED}_tau${TAU}.log"
    if [ "${PIPESTATUS[0]}" -eq 0 ]; then
      touch "$DONE_MARK"; echo "[$(date '+%F %T')] DONE  seed=$SEED tau=$TAU"
    else
      echo "[$(date '+%F %T')] FAILED seed=$SEED tau=$TAU -- xem $LOG_DIR/seed${SEED}_tau${TAU}.log"; exit 1
    fi
  done
done
echo "[$(date '+%F %T')] Xong ca 6 run RQ3 sensitivity."
