#!/usr/bin/env bash
# run_rq2_all.sh -- chay lien tiep ca 9 run RQ2 (A/B/C x seed 42/43/44)
# bang reeval/re06_train_rq2_ablation.py. Khong sua file nao khac.
# Thu tu theo SEED truoc (A42,B42,C42 -> A43,... -> A44,...).
# Chay lai bat ky luc nao: run xong (co done_*) -> bo qua; run dang do -> re06 tu resume.
cd /workspace/CPV301 2>/dev/null || cd "$(dirname "$0")/.."
# vast.ai: python/torch nam trong venv /venv/main, shell tmux moi khong tu bat
[ -f /venv/main/bin/activate ] && source /venv/main/bin/activate
LOG_DIR=outputs_riskprop/rq2_logs
mkdir -p "$LOG_DIR"
for SEED in 42 43 44; do
  for COND in "0 0 neither" "1 0 ffronly" "0 1 amconly"; do
    set -- $COND
    FFR=$1; AMC=$2; TAG=$3
    DONE_MARK="$LOG_DIR/done_seed${SEED}_${TAG}"
    if [ -f "$DONE_MARK" ]; then echo "[skip] seed=$SEED cond=$TAG (da xong)"; continue; fi
    CMD="RISKPROP_SEED=$SEED RISKPROP_USE_FFR=$FFR RISKPROP_USE_AMC=$AMC python -u reeval/re06_train_rq2_ablation.py"
    if [ "${DRY_RUN:-0}" = "1" ]; then echo "[dry] $CMD"; continue; fi
    echo "[$(date '+%F %T')] START seed=$SEED cond=$TAG"
    RISKPROP_SEED=$SEED RISKPROP_USE_FFR=$FFR RISKPROP_USE_AMC=$AMC \
      python -u reeval/re06_train_rq2_ablation.py 2>&1 | tee -a "$LOG_DIR/seed${SEED}_${TAG}.log"
    if [ "${PIPESTATUS[0]}" -eq 0 ]; then
      touch "$DONE_MARK"; echo "[$(date '+%F %T')] DONE  seed=$SEED cond=$TAG"
    else
      echo "[$(date '+%F %T')] FAILED seed=$SEED cond=$TAG -- xem $LOG_DIR/seed${SEED}_${TAG}.log, sua loi roi chay lai script (se resume)."
      exit 1
    fi
  done
done
echo "[$(date '+%F %T')] Xong ca 9 run RQ2."
ls -la outputs_riskprop/*_ffronly.pth outputs_riskprop/*_amconly.pth outputs_riskprop/*_neither.pth 2>/dev/null || true
