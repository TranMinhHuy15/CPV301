#!/usr/bin/env bash
# run_progress_rq3.sh -- RQ3 (new): Progressive RiskProp = RiskProp + PRE-ACT-inspired
# Continuous Progress Supervision (CPS). Stage-by-stage, re-runnable; finished
# steps are skipped (done markers), interrupted training resumes from latest.pth.
#
#   bash reeval/run_progress_rq3.sh setup    # deps, folder links, HF token check
#   bash reeval/run_progress_rq3.sh data     # HF download (cell09) + RiskProp cache (cell30) + val cache (re01)
#   bash reeval/run_progress_rq3.sh check    # unit tests + audit (re11) + sidecar (re12)      [CPU]
#   bash reeval/run_progress_rq3.sh smoke    # B/P/S, 2 optimizer steps each, real model         [GPU, ~5 min]
#   bash reeval/run_progress_rq3.sh mini     # B/P/S seed 42, 40 videos x 2 epochs + eval          [GPU, ~30 min]
#   LOCK_CONFIRMED=1 bash reeval/run_progress_rq3.sh lock   # write outputs_progress/LOCK.json
#   bash reeval/run_progress_rq3.sh full     # B/P/S x seeds 42/43/44 = 9 runs, 50 epochs      [GPU, ~12-16 h]
#   bash reeval/run_progress_rq3.sh eval     # validation eval + bootstrap (re15)               [GPU, ~1 h]
#   bash reeval/run_progress_rq3.sh sens     # optional: P with H = 1.5 s, seed 42 + eval vs B   [GPU, ~2 h]
#   bash reeval/run_progress_rq3.sh pack     # copy small result files to results/rq3_progress/ for GitHub
#
# Nothing here edits existing repository files. Official test is NOT run.
set -o pipefail
REPO=${REPO:-/workspace/CPV301}
cd "$REPO" || exit 1
[ -f /venv/main/bin/activate ] && source /venv/main/bin/activate
DATA=$REPO/data
OUT=${OUT:-$REPO/outputs_progress}          # /workspace = persistent volume on vast.ai
MAN=${MAN:-$REPO/results/repro/split_manifest_seed42.json}
MODEL=${MODEL:-riskprop}         # "dummy" only for a CPU dry run of smoke/mini (full refuses it)
mkdir -p "$OUT/logs"

# ---------------- DESIGN TO LOCK (team decision -- edit BEFORE `lock`) ----------------
PAIRING=${PAIRING:-fixed}      # RQ2 (pairing) result; fixed 1.0 s = cell32 v2 default
H=${H:-2.0}                    # main horizon (s)
ALPHA=${ALPHA:-3.0}            # PRE-ACT exp_above alpha
LAMBDA=${LAMBDA:-1.0}          # single pre-registered value, no sweep (NOT PRE-ACT's 10)
SEEDS=${SEEDS:-"42 43 44"}
CONDS=${CONDS:-"B P S"}        # add "F FP" only if time allows
PRIMARY=${PRIMARY:-"AP@1.5s"}  # early-anticipation primary metric (P - B)
NI_METRIC=${NI_METRIC:-mAP}
NI_MARGIN=${NI_MARGIN:-0.02}   # P must not lose more than this mAP vs B (bootstrap CI lower bound)
SENS_H=1.5
# ----------------------------------------------------------------------------------------

lam_arg() { [ "$1" = "B" ] || [ "$1" = "F" ] && echo "" || echo "--lambda-prog $LAMBDA"; }
COMMON="--pairing-mode $PAIRING --alpha $ALPHA --cache-root $DATA/nexar_cache_riskprop \
 --cache-5f $DATA/nexar_cache_5f --sidecar $OUT/sidecar/sidecar.json --split-manifest $MAN \
 --output-root $OUT"
DEV="--model $MODEL"
EVAL_COMMON="--pairing-mode $PAIRING --alpha $ALPHA --lambda-prog $LAMBDA \
 --data-dir $DATA/nexar_kaggle_style --cache-5f $DATA/nexar_cache_5f --dense-cache $DATA/nexar_cache_dense"

stage=${1:-help}
case "$stage" in
setup)
  pip install -q decord datasets huggingface_hub scikit-learn pandas pyflakes 2>&1 | tail -1
  python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"
  # cell10/cell30 write to /kaggle/working/data -> point it at the persistent repo data folder
  mkdir -p "$DATA" /kaggle/working
  [ -e /kaggle/working/data ] || ln -s "$DATA" /kaggle/working/data
  ls -la /kaggle/working/data
  [ -n "$HF_TOKEN" ] && echo "HF_TOKEN is set (not printed)" || echo "WARNING: export HF_TOKEN=... (read token, dataset terms accepted)"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
  ;;
data)
  [ -f "$DATA/nexar_kaggle_style/train.csv" ] || (echo | python pipeline/cell09_prepare_hf_data.py)
  [ -f "$DATA/nexar_cache_riskprop/train_index.json" ] || python pipeline/cell30_precache_riskprop.py
  [ -f "$DATA/nexar_cache_5f/val_index.json" ] || python reeval/re01_precache_val.py
  python - "$MAN" "$DATA/nexar_cache_5f/split_manifest_seed42.json" <<'EOF'
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
ok = all(a[k] == b[k] for k in ("train_csv_sha256", "train_ids", "val_ids"))
print("split manifest identical to results/repro:", ok)
sys.exit(0 if ok else 2)
EOF
  ;;
check)
  python reeval/re16_test_progress_protocol.py || exit 1
  python reeval/re11_audit_progress_data.py --data-dir "$DATA/nexar_kaggle_style" --split-manifest "$MAN" \
    --cache-root "$DATA/nexar_cache_riskprop" --cache-5f "$DATA/nexar_cache_5f" \
    --id-map results/repro/id_to_hf_source.csv --probe-videos --scan-cache \
    --output-dir "$OUT/audit" 2>&1 | tee "$OUT/logs/audit.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "AUDIT FAILED -> read $OUT/audit/audit_report.txt; stop."; exit 2; }
  python reeval/re12_build_progress_sidecar.py --audit "$OUT/audit/audit_summary.json" \
    --data-dir "$DATA/nexar_kaggle_style" --split-manifest "$MAN" --cache-root "$DATA/nexar_cache_riskprop" \
    --output-dir "$OUT/sidecar_sample" --limit-videos 8 --verify-pixels 8 || exit 2
  python reeval/re12_build_progress_sidecar.py --audit "$OUT/audit/audit_summary.json" \
    --data-dir "$DATA/nexar_kaggle_style" --split-manifest "$MAN" --cache-root "$DATA/nexar_cache_riskprop" \
    --output-dir "$OUT/sidecar" --verify-pixels 24 ${ACCEPT_CACHE_ZEROS:+--accept-cache-zeros} \
    2>&1 | tee "$OUT/logs/sidecar.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || exit 2
  python -c "import json,sys;s=json.load(open('$OUT/sidecar/sidecar_summary.json'));u=s['usable_for_full_run'];print('SIDECAR usable_for_full_run:',u,s.get('not_usable_because',[]));print('zero snippets:',s['zero_snippets_all_videos'],'| PTS offset videos:',s['videos_with_pts_offset'],'| pixel check:',s['pixel_verification']['n_identical'],'/',s['pixel_verification']['n_verified'])"
  ;;
smoke)
  for C in $CONDS; do
    python reeval/re14_train_progress_riskprop.py --condition $C --seed 42 --horizon-sec $H $(lam_arg $C) \
      $COMMON $DEV --max-steps 2 --limit-val 16 --tag smoke 2>&1 | tee "$OUT/logs/smoke_$C.log" | grep -E "epoch|REFUSE|Error|done"
    [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "SMOKE FAILED ($C)"; exit 1; }
  done
  ;;
mini)
  for C in $CONDS; do
    python reeval/re14_train_progress_riskprop.py --condition $C --seed 42 --horizon-sec $H $(lam_arg $C) \
      $COMMON $DEV --limit-videos 40 --max-epochs 2 --tag mini 2>&1 | tee "$OUT/logs/mini_$C.log" | grep -E "epoch|done|Error|REFUSE"
    [ "${PIPESTATUS[0]}" -eq 0 ] || exit 1
  done
  python reeval/re15_eval_progress_riskprop.py --runs-root "$OUT/dev" --conditions "$(echo $CONDS | tr ' ' ',')" \
    --seeds 42 --horizon-sec $H --tag mini $EVAL_COMMON $DEV --out-dir "$OUT/eval_mini" --n-boot 200 \
    2>&1 | tee "$OUT/logs/eval_mini.log" | grep -E "STALE|REFUSE|^\| |Written"
  echo ">>> mini = pipeline check only, NOT a research result; do not pick lambda from it."
  ;;
lock)
  [ "${LOCK_CONFIRMED:-0}" = "1" ] || { echo "Set LOCK_CONFIRMED=1 after the team agreed on: pairing=$PAIRING H=$H alpha=$ALPHA lambda=$LAMBDA seeds=[$SEEDS] conds=[$CONDS] primary=$PRIMARY NI=$NI_METRIC margin $NI_MARGIN"; exit 1; }
  [ -f "$OUT/LOCK.json" ] && { echo "LOCK.json already exists (not overwritten):"; cat "$OUT/LOCK.json"; exit 0; }
  python - "$OUT/LOCK.json" <<EOF
import json, sys, time, subprocess
lock = {"confirmed": True, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_sha": subprocess.run(["git","rev-parse","HEAD"],capture_output=True,text=True).stdout.strip(),
        "pairing_mode": "$PAIRING", "horizon_sec": float("$H"), "alpha": float("$ALPHA"),
        "lambda_prog": float("$LAMBDA"), "seeds": [int(s) for s in "$SEEDS".split()],
        "conditions": "$CONDS".split(), "sensitivity_horizons": [float("$SENS_H")],
        "sensitivity_conditions": ["P"], "primary_metric": "$PRIMARY",
        "secondary_metrics": ["mAP", "mAUC01", "PVR", "ADS", "RCJ", "amplitude"],
        "ni_metric": "$NI_METRIC", "ni_margin": float("$NI_MARGIN"),
        "checkpoint_rule": "per_run (re03 locked): higher mean val AP over 0.5/1.0/1.5 s among best/latest",
        "ci": "paired label-stratified bootstrap B=2000 seed 12345, validation split",
        "test_policy": "official test scored once, after validation results are frozen, fresh solution.csv"}
json.dump(lock, open(sys.argv[1], "w"), indent=2)
print(json.dumps(lock, indent=2))
EOF
  ;;
full)
  [ -f "$OUT/LOCK.json" ] || { echo "Run the lock stage first."; exit 1; }
  for SEED in $SEEDS; do for C in $CONDS; do
    TAG="seed${SEED}_${C}_H${H}"
    [ -f "$OUT/logs/done_$TAG" ] && { echo "[skip] $TAG"; continue; }
    echo "[$(date '+%F %T')] START $TAG"
    python -u reeval/re14_train_progress_riskprop.py --condition $C --seed $SEED --horizon-sec $H $(lam_arg $C) \
      $COMMON --audit "$OUT/audit/audit_summary.json" --lock-file "$OUT/LOCK.json" --full-run \
      2>&1 | tee -a "$OUT/logs/full_$TAG.log"
    if [ "${PIPESTATUS[0]}" -eq 0 ]; then touch "$OUT/logs/done_$TAG"; echo "[$(date '+%F %T')] DONE $TAG"
    else echo "FAILED $TAG -- see $OUT/logs/full_$TAG.log; rerun this stage to resume"; exit 1; fi
  done; done
  ;;
eval)
  python reeval/re15_eval_progress_riskprop.py --runs-root "$OUT/runs" --conditions "$(echo $CONDS | tr ' ' ',')" \
    --seeds "$(echo $SEEDS | tr ' ' ',')" --horizon-sec $H $EVAL_COMMON --lock-file "$OUT/LOCK.json" \
    --out-dir "$OUT/eval_val" 2>&1 | tee "$OUT/logs/eval_val.log"
  ;;
sens)
  TAG="seed42_P_H${SENS_H}"
  if [ ! -f "$OUT/logs/done_$TAG" ]; then
    python -u reeval/re14_train_progress_riskprop.py --condition P --seed 42 --horizon-sec $SENS_H \
      --lambda-prog $LAMBDA $COMMON --audit "$OUT/audit/audit_summary.json" --lock-file "$OUT/LOCK.json" \
      --full-run 2>&1 | tee -a "$OUT/logs/full_$TAG.log"
    [ "${PIPESTATUS[0]}" -eq 0 ] && touch "$OUT/logs/done_$TAG" || exit 1
  fi
  python reeval/re15_eval_progress_riskprop.py --runs-root "$OUT/runs" --conditions B,P --seeds 42 \
    --horizon-sec $SENS_H $EVAL_COMMON --lock-file "$OUT/LOCK.json" --out-dir "$OUT/eval_val_sens_H${SENS_H}" \
    2>&1 | tee "$OUT/logs/eval_sens.log"
  ;;
pack)
  DST=$REPO/results/rq3_progress
  mkdir -p "$DST"
  cp -v "$OUT"/LOCK.json "$DST"/ 2>/dev/null
  for d in audit sidecar sidecar_sample; do
    [ -d "$OUT/$d" ] && mkdir -p "$DST/$d" && cp -v "$OUT/$d"/*.json "$OUT/$d"/*.txt "$OUT/$d"/*.csv "$DST/$d"/ 2>/dev/null
  done
  rm -f "$DST/sidecar/sidecar.json"            # large; summary + snippets csv are kept
  for r in "$OUT"/runs/*/; do
    [ -d "$r" ] || continue; n=$(basename "$r"); mkdir -p "$DST/runs/$n"; cp -v "$r"config.json "$r"train_log.csv "$DST/runs/$n"/
  done
  for e in "$OUT"/eval_val*; do
    [ -d "$e" ] || continue; n=$(basename "$e"); mkdir -p "$DST/$n"
    cp -v "$e"/*.md "$e"/*.csv "$e"/*.json "$DST/$n"/ 2>/dev/null
    cp -r "$e"/preds_val "$e"/dense_val "$DST/$n"/ 2>/dev/null
  done
  cp -v "$OUT"/logs/*.log "$DST"/ 2>/dev/null
  du -sh "$DST"
  grep -rl "hf_[A-Za-z0-9]\{20,\}" "$DST" && { echo "TOKEN-LIKE STRING FOUND -- do not commit"; exit 1; }
  echo "Ready: git add reeval/re1[1-6]_*.py reeval/run_progress_rq3.sh reeval/README_PROGRESS_RQ3.md results/rq3_progress"
  ;;
*)
  sed -n '2,20p' "$0"
  ;;
esac
