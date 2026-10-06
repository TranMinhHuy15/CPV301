#!/usr/bin/env bash
# progress_cps_v4/run_all.sh (v4) -- RQ3 Progressive RiskProp, stage by stage, FAIL-FAST.
# Every stage stops on the first non-zero exit (python, pip, tee pipelines included);
# full logs are kept under $OUT/logs; summaries are printed from the logs afterwards and
# can never hide an error. Finished runs are skipped ONLY when train_bps.py --check-complete
# confirms the same fingerprint (config + code + data + init + LOCK) and checkpoint hashes.
#
#   bash reeval/progress_cps_v4/run_all.sh setup       # pinned small deps, folder links, HF token check
#   bash reeval/progress_cps_v4/run_all.sh preflight   # preflight.py: versions, CUDA, backbone weights, disk   [G3 part]
#   bash reeval/progress_cps_v4/run_all.sh prepare     # prepare_data_pinned.py: pinned HF train/**, id map, train.csv sha [G1]
#   bash reeval/progress_cps_v4/run_all.sh cache       # riskprop_precache_snippet_sequence.py train cache + precache_val.py val cache (after G1)
#   bash reeval/progress_cps_v4/run_all.sh check       # unit_tests_v3.py tests + audit_data.py audit + build_sidecar.py sidecar           [G2]
#   bash reeval/progress_cps_v4/run_all.sh smoke       # B/P/S, 2 optimizer steps each, real model, CUDA   [G3]
#   bash reeval/progress_cps_v4/run_all.sh mini        # B/P/S 40 videos x 2 epochs + resume test + eval  [G4]
#   LOCK_CONFIRMED=1 bash reeval/progress_cps_v4/run_all.sh lock   # lock_config.py: immutable LOCK.json             [G5]
#   bash reeval/progress_cps_v4/run_all.sh full        # B/P/S x seeds 42/43/44, 50 epochs                 [G6]
#   bash reeval/progress_cps_v4/run_all.sh eval        # eval_val.py --research on all 300 val videos             [G7]
#   bash reeval/progress_cps_v4/run_all.sh sens        # optional: P, H = 1.5 s, seed 42 + eval vs B
#   bash reeval/progress_cps_v4/run_all.sh test-freeze | test-infer | test-score   # official_test_protocol.py: official test  [G8]
#   bash reeval/progress_cps_v4/run_all.sh pack        # eval/test at root; provenance and logs under results/progress_cps_v4/log/
set -euo pipefail
REPO=${REPO:-/workspace/CPV301}
cd "$REPO"
if [ -f /venv/main/bin/activate ]; then set +u; source /venv/main/bin/activate; set -u; fi
DATA=${DATA:-$REPO/data}
OUT=${OUT:-$REPO/outputs_progress}
MAN=${MAN:-$REPO/results/repro/split_manifest_seed42.json}
IDMAP=${IDMAP:-$REPO/results/repro/id_to_hf_source.csv}
MODEL=${MODEL:-riskprop}       # "dummy" only for a CPU dry run of smoke/mini (full refuses it)
NUM_WORKERS=${NUM_WORKERS:-4}
mkdir -p "$OUT/logs"

# ---------------- locked design (team decision 2026-10-05; also written by lock_config into LOCK.json) ----------------
PAIRING=fixed; H=2.0; ALPHA=3.0; LAMBDA=1.0; SEEDS="42 43 44"; CONDS="B P S"; SENS_H=1.5
# ------------------------------------------------------------------------------------------------------------------

die() { echo "[FAIL] $*" >&2; exit 1; }
# run_logged LOGFILE cmd... : full output to the log and the terminal; exit with the command's own status
run_logged() {
  local log="$1"; shift
  echo "+ $*" | tee -a "$log"
  set +e
  "$@" 2>&1 | tee -a "$log"
  local rc=${PIPESTATUS[0]}
  set -e
  [ "$rc" -eq 0 ] || die "rc=$rc: $* (full log: $log)"
}
lam_arg() { case "$1" in B|F) echo "" ;; *) echo "--lambda-prog $LAMBDA" ;; esac; }
COMMON=(--pairing-mode "$PAIRING" --alpha "$ALPHA" --cache-root "$DATA/nexar_cache_riskprop"
        --cache-5f "$DATA/nexar_cache_5f" --sidecar "$OUT/sidecar/sidecar.json" --split-manifest "$MAN"
        --output-root "$OUT" --num-workers "$NUM_WORKERS")
EVAL_COMMON=(--pairing-mode "$PAIRING" --alpha "$ALPHA" --lambda-prog "$LAMBDA" --data-dir "$DATA/nexar_kaggle_style"
             --cache-5f "$DATA/nexar_cache_5f" --dense-cache "$DATA/nexar_cache_dense" --split-manifest "$MAN"
             --val-timing "$OUT/audit/val_timing.csv")
need() { [ -e "$1" ] || die "missing $1 -- run the '$2' stage first"; }

stage=${1:-help}
case "$stage" in
setup)
  run_logged "$OUT/logs/setup.log" pip install -r reeval/progress_cps_v4/requirements.txt
  mkdir -p "$DATA" /kaggle/working
  [ -e /kaggle/working/data ] || ln -s "$DATA" /kaggle/working/data     # riskprop_precache_snippet_sequence writes to /kaggle/working/data
  [ "$(readlink -f /kaggle/working/data)" = "$(readlink -f "$DATA")" ] || die "/kaggle/working/data does not point to $DATA"
  if [ -n "${HF_TOKEN:-}" ]; then echo "HF_TOKEN is set (not printed)"; else echo "WARNING: export HF_TOKEN=... before 'prepare'"; fi
  ;;
preflight)
  run_logged "$OUT/logs/preflight.log" python reeval/progress_cps_v4/preflight.py --out-dir "$OUT/preflight" --require-cuda
  ;;
prepare)
  SKIP=(); [ "${SKIP_DOWNLOAD:-0}" = 1 ] && SKIP=(--skip-download)
  run_logged "$OUT/logs/prepare.log" python reeval/progress_cps_v4/prepare_data_pinned.py --raw-dir "$DATA/nexar_hf_raw" \
    --out-dir "$DATA/nexar_kaggle_style" --report-dir "$OUT/prepare" --id-map "$IDMAP" --split-manifest "$MAN" "${SKIP[@]}"
  ;;
cache)
  [ "$REPO" = /workspace/CPV301 ] || die "riskprop_precache_snippet_sequence has /workspace/CPV301 hard-coded; clone the repo there"
  need "$OUT/prepare/prepare_report.json" prepare
  python -c "import json,sys; sys.exit(0 if json.load(open('$OUT/prepare/prepare_report.json'))['pass'] else 1)" \
    || die "prepare_report did not pass"
  [ -f "$DATA/nexar_cache_riskprop/train_index.json" ] || run_logged "$OUT/logs/cache_riskprop_precache.log" python pipeline/riskprop_precache_snippet_sequence.py
  [ -f "$DATA/nexar_cache_5f/val_index.json" ] || run_logged "$OUT/logs/cache_re01.log" python reeval/precache_val.py --data-dir "$DATA/nexar_kaggle_style" --cache-dir "$DATA/nexar_cache_5f"
  ;;
check)
  run_logged "$OUT/logs/unit_tests.log" python reeval/progress_cps_v4/unit_tests_v3.py
  run_logged "$OUT/logs/unit_tests.log" python reeval/progress_cps_v4/regression_tests_v4.py
  run_logged "$OUT/logs/audit.log" python reeval/progress_cps_v4/audit_data.py --data-dir "$DATA/nexar_kaggle_style" \
    --split-manifest "$MAN" --cache-root "$DATA/nexar_cache_riskprop" --cache-5f "$DATA/nexar_cache_5f" \
    --id-map "$IDMAP" --prepare-report "$OUT/prepare/prepare_report.json" --output-dir "$OUT/audit"
  run_logged "$OUT/logs/sidecar_sample.log" python reeval/progress_cps_v4/build_sidecar.py --audit "$OUT/audit/audit_summary.json" \
    --data-dir "$DATA/nexar_kaggle_style" --split-manifest "$MAN" --cache-root "$DATA/nexar_cache_riskprop" \
    --output-dir "$OUT/sidecar_sample" --limit-videos 8 --verify-pixels 8
  run_logged "$OUT/logs/sidecar.log" python reeval/progress_cps_v4/build_sidecar.py --audit "$OUT/audit/audit_summary.json" \
    --data-dir "$DATA/nexar_kaggle_style" --split-manifest "$MAN" --cache-root "$DATA/nexar_cache_riskprop" \
    --output-dir "$OUT/sidecar" --verify-pixels 24
  python - "$OUT/sidecar/sidecar_summary.json" <<'EOF'
import json, sys
s = json.load(open(sys.argv[1]))
print("SIDECAR usable_for_full_run:", s["usable_for_full_run"], s.get("not_usable_because", []))
print("zero snippets:", s["zero_snippets_all_videos"], "| PTS offset videos:", s["videos_with_pts_offset"],
      "| pixel check:", s["pixel_verification"]["n_identical"], "/", s["pixel_verification"]["n_verified"])
print("CPS-valid snippets per positive:", s["valid_snippets_per_positive"])
sys.exit(0 if s["usable_for_full_run"] else 1)
EOF
  ;;
smoke)
  [ "$MODEL" = dummy ] || run_logged "$OUT/logs/preflight_smoke.log" python reeval/progress_cps_v4/preflight.py --out-dir "$OUT/preflight" --require-cuda
  for C in $CONDS; do
    # shellcheck disable=SC2046
    run_logged "$OUT/logs/smoke_$C.log" python reeval/progress_cps_v4/train_bps.py --condition "$C" --seed 42 \
      --horizon-sec "$H" $(lam_arg "$C") "${COMMON[@]}" --model "$MODEL" --max-steps 2 --limit-val 16 --tag smoke
  done
  grep -h "epoch " "$OUT"/logs/smoke_*.log || true
  ;;
mini)
  for C in $CONDS; do
    # shellcheck disable=SC2046
    run_logged "$OUT/logs/mini_$C.log" python reeval/progress_cps_v4/train_bps.py --condition "$C" --seed 42 \
      --horizon-sec "$H" $(lam_arg "$C") "${COMMON[@]}" --model "$MODEL" --limit-videos 40 --max-epochs 2 --tag mini
  done
  # resume test: interrupt P after epoch 1, then resume to epoch 2 (CUDA RNG/optimizer/scaler restore)
  run_logged "$OUT/logs/mini_resume.log" python reeval/progress_cps_v4/train_bps.py --condition P --seed 42 \
    --horizon-sec "$H" --lambda-prog "$LAMBDA" "${COMMON[@]}" --model "$MODEL" --limit-videos 40 --max-epochs 2 \
    --tag mini_resume --stop-after-epochs 1
  run_logged "$OUT/logs/mini_resume.log" python reeval/progress_cps_v4/train_bps.py --condition P --seed 42 \
    --horizon-sec "$H" --lambda-prog "$LAMBDA" "${COMMON[@]}" --model "$MODEL" --limit-videos 40 --max-epochs 2 \
    --tag mini_resume
  grep -q "Resumed P_" "$OUT/logs/mini_resume.log" || die "resume test did not resume"
  run_logged "$OUT/logs/eval_mini.log" python reeval/progress_cps_v4/eval_val.py --runs-root "$OUT/dev" \
    --conditions "$(echo $CONDS | tr ' ' ',')" --seeds 42 --horizon-sec "$H" --tag mini "${EVAL_COMMON[@]}" \
    --model "$MODEL" --out-dir "$OUT/eval_mini" --n-boot 200
  echo ">>> mini = feasibility check only (dev, not a research result); do not pick lambda/H from it."
  ;;
lock)
  run_logged "$OUT/logs/lock.log" python reeval/progress_cps_v4/lock_config.py --out "$OUT/LOCK.json" \
    --prepare "$OUT/prepare/prepare_report.json" --audit "$OUT/audit/audit_summary.json" \
    --sidecar "$OUT/sidecar/sidecar.json" --preflight "$OUT/preflight/preflight.json" \
    --mini-root "$OUT/dev" --cache-5f "$DATA/nexar_cache_5f" --pairing "$PAIRING" --horizon "$H" \
    --alpha "$ALPHA" --lam "$LAMBDA" --seeds "$(echo $SEEDS | tr ' ' ',')" --conditions "$(echo $CONDS | tr ' ' ',')"
  ;;
full)
  need "$OUT/LOCK.json" lock
  run_logged "$OUT/logs/preflight_full.log" python reeval/progress_cps_v4/preflight.py --out-dir "$OUT/preflight_full" --require-cuda
  for SEED in $SEEDS; do for C in $CONDS; do
    TAG="seed${SEED}_${C}_H${H}"
    # shellcheck disable=SC2046
    ARGS=(--condition "$C" --seed "$SEED" --horizon-sec "$H" $(lam_arg "$C") "${COMMON[@]}"
          --audit "$OUT/audit/audit_summary.json" --lock-file "$OUT/LOCK.json" --full-run)
    if python reeval/progress_cps_v4/train_bps.py "${ARGS[@]}" --check-complete >> "$OUT/logs/full_$TAG.log" 2>&1; then
      echo "[skip] $TAG verified complete (same fingerprint)"; continue
    fi
    echo "[$(date '+%F %T')] START $TAG"
    run_logged "$OUT/logs/full_$TAG.log" python -u reeval/progress_cps_v4/train_bps.py "${ARGS[@]}"
    python reeval/progress_cps_v4/train_bps.py "${ARGS[@]}" --check-complete >> "$OUT/logs/full_$TAG.log" 2>&1 \
      || die "$TAG finished but COMPLETE.json does not verify -- see $OUT/logs/full_$TAG.log"
    echo "[$(date '+%F %T')] DONE  $TAG"
  done; done
  ;;
eval)
  run_logged "$OUT/logs/eval_val.log" python reeval/progress_cps_v4/eval_val.py --runs-root "$OUT/runs" \
    --conditions "$(echo $CONDS | tr ' ' ',')" --seeds "$(echo $SEEDS | tr ' ' ',')" --horizon-sec "$H" \
    "${EVAL_COMMON[@]}" --lock-file "$OUT/LOCK.json" --out-dir "$OUT/eval_val" --research
  ;;
sens)
  ARGS=(--condition P --seed 42 --horizon-sec "$SENS_H" --lambda-prog "$LAMBDA" "${COMMON[@]}"
        --audit "$OUT/audit/audit_summary.json" --lock-file "$OUT/LOCK.json" --full-run)
  python reeval/progress_cps_v4/train_bps.py "${ARGS[@]}" --check-complete >> "$OUT/logs/sens.log" 2>&1 \
    || run_logged "$OUT/logs/sens.log" python -u reeval/progress_cps_v4/train_bps.py "${ARGS[@]}"
  run_logged "$OUT/logs/eval_sens.log" python reeval/progress_cps_v4/eval_val.py --runs-root "$OUT/runs" \
    --conditions B,P --seeds 42 --horizon-sec "$SENS_H" "${EVAL_COMMON[@]}" --lock-file "$OUT/LOCK.json" \
    --out-dir "$OUT/eval_val_sens_H${SENS_H}"
  echo ">>> sensitivity is reported outside the main decision (1 seed, no research gate)."
  ;;
test-freeze)
  run_logged "$OUT/logs/test.log" python reeval/progress_cps_v4/official_test_protocol.py freeze --eval-dir "$OUT/eval_val" \
    --lock-file "$OUT/LOCK.json" --out-dir "$OUT/test"
  ;;
test-infer)
  run_logged "$OUT/logs/test.log" python reeval/progress_cps_v4/official_test_protocol.py infer --out-dir "$OUT/test" --hf-raw "$DATA/nexar_hf_test"
  ;;
test-score)
  run_logged "$OUT/logs/test.log" python reeval/progress_cps_v4/official_test_protocol.py score --out-dir "$OUT/test" --hf-raw "$DATA/nexar_hf_test"
  ;;
pack)
  DST=$REPO/results/progress_cps_v4
  mkdir -p "$DST/log"
  for f in LOCK.json LOCK.json.sha256; do [ -f "$OUT/$f" ] && cp -v "$OUT/$f" "$DST/log/"; done
  for d in prepare preflight audit sidecar sidecar_sample; do
    [ -d "$OUT/$d" ] || continue
    mkdir -p "$DST/log/$d"
    find "$OUT/$d" -maxdepth 1 -type f \( -name '*.json' -o -name '*.txt' -o -name '*.csv' -o -name '*.md' \) \
      ! -name 'sidecar.json' -exec cp {} "$DST/log/$d/" \;
  done
  if [ -d "$OUT/test" ]; then
    mkdir -p "$DST/test"
    find "$OUT/test" -maxdepth 1 -type f \( -name '*.json' -o -name '*.csv' -o -name '*.md' \) -exec cp {} "$DST/test/" \;
    find "$OUT/test" -maxdepth 1 -type f \( -name '*.log' -o -name '*.txt' \) -exec mkdir -p "$DST/log/test" \; -exec cp {} "$DST/log/test/" \;
  fi
  for r in "$OUT"/runs/*/; do
    [ -d "$r" ] || continue; n=$(basename "$r"); mkdir -p "$DST/log/runs/$n"
    cp "$r"config.json "$r"train_log.csv "$DST/log/runs/$n/"; [ -f "$r"COMPLETE.json ] && cp "$r"COMPLETE.json "$DST/log/runs/$n/"
  done
  for e in "$OUT"/eval_val*; do
    [ -d "$e" ] || continue; n=$(basename "$e"); mkdir -p "$DST/$n"
    find "$e" -maxdepth 1 -type f \( -name '*.md' -o -name '*.csv' -o -name '*.json' \) -exec cp {} "$DST/$n/" \;
    find "$e" -maxdepth 1 -type f \( -name '*.log' -o -name '*.txt' \) -exec mkdir -p "$DST/log/$n" \; -exec cp {} "$DST/log/$n/" \;
    cp -r "$e/preds_val" "$e/dense_val" "$DST/$n/" 2>/dev/null || true
  done
  cp "$OUT"/logs/*.log "$DST/log/" 2>/dev/null || true
  cp "$OUT"/logs/*.txt "$DST/log/" 2>/dev/null || true
  if grep -rlE "hf_[A-Za-z0-9]{20,}" "$DST"; then die "token-like string found in $DST -- do not commit"; fi
  du -sh "$DST"
  echo "Ready: git add results/progress_cps_v4 (no .pth, no video, no cache)"
  ;;
*)
  sed -n '2,25p' "$0"
  ;;
esac
