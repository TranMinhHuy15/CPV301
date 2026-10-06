# RQ3: Progressive RiskProp, v4 run guide

**Progressive RiskProp** is RiskProp with an extra Continuous Progress Supervision (CPS) term. CPS is a *PRE-ACT-inspired loss-level adaptation* (PRE-ACT reference: `giddyyupp/PRE-ACT@5e0d38fb`). It is **not** a reproduction of PRE-ACT:

| | This project |
|---|---|
| Backbone and head | SlowOnly-R50 with one sigmoid head, unchanged |
| Not used | progress head, preference loss, VideoMAE, FLaRA |
| λ | 1, not PRE-ACT's 10 |
| Which snippets get CPS | positive videos only |
| Excluded from CPS | the two BCE anchors and τ ≤ 0 (PRE-ACT gives τ ≤ 0 a target of 1) |

RQ1b is out of scope.

## RQ numbering (v4 scope)

| RQ | Question | Status |
|---|---|---|
| RQ1 | TOP vs AdaLEA vs RiskProp (adaptations under the team protocol) | results unchanged (`results/RESULTS_RQ1.md`) |
| RQ2 | Random vs fixed-lag temporal pairing (formerly RQ3) | results unchanged (`results/RESULTS_RQ3.md`) |
| (component analysis) | FFR/AMC 2×2 (formerly RQ2) | results unchanged (`results/RESULTS_RQ2.md`) |
| **RQ3** | *Can event-distance-aware progress supervision improve RiskProp's early collision anticipation and temporal risk progression while maintaining predictive accuracy under the temporal-pairing configuration selected in RQ2?* | this pipeline |

The result files keep their historical names and numbers, and none of them is rewritten. The root `README.md` is untouched in v4.

## Locked design (team decision, 2026-10-05)

| Item | Value |
|---|---|
| Conditions | **B** = BCE + 0.5 FFR + 0.5 AMC (matched baseline, retrained) · **P** = B + λ·CPS(continuous) · **S** = B + λ·CPS(binary 1[0<τ<H]), with the same mask, reduction and λ as P |
| Target | r = 0 if τ ≥ H, otherwise (1 − e^(−α(1−τ/H))) / (1 − e^(−α)). τ = time_of_event − **actual PTS** of the last frame used. H = 2 s, α = 3. Examples: r(1.5) ≈ 0.5553, r(1.0) ≈ 0.8176 |
| Mask | Positive videos only. Excluded: k = 0 and k = 11 (BCE anchors), decode/zero snippets, non-finite τ, τ ≤ 0, duplicate end frames (latest kept), non-monotonic times. Nominal clamp at 0.4 s is logged separately. Negatives get no CPS. |
| Loss | SmoothL1(β = 1) on sigmoid(z) in FP32, target detached. Mean over valid snippets per video, then mean over positive videos with ≥ 1 valid snippet; 0 (graph-connected) if a batch has none |
| Pairing | fixed lag 1.0 s for B/P/S. RQ2 found no accuracy difference between fixed and random (mAP −0.0026 [−0.031, 0.025]), so fixed was not chosen for looking better |
| λ | 1.0, set before training and not tuned. SmoothL1 on probabilities = 0.5·(a − r)², the same algebraic scale as FFR (0.5·MSE). Head gradient norms of both terms are logged as a diagnostic only |
| Budget | seeds 42/43/44; 50 epochs; SGD lr 0.01, momentum 0.9, weight decay 1e-4, batch 2, LR decay at 20/40; split seed 42 (1,200 / 300) |
| Sensitivity | P with H = 1.5 s, seed 42, compared with the same B; outside the main decision |
| **Decision** | RQ3 is supported **iff** P − B on AP@1.5s has CI lower bound > 0 **and** P − B on mAP has CI lower bound > −0.02. CIs are computed on raw floats, and the research gate must pass. P − S on AP@1.5s is reported regardless. PVR/ADS/RCJ, rise, negative score level, seed stability, mAUC@0.1 and Coverage are supporting evidence only |
| Checkpoint | The trainer keeps `best` (lowest val BCE @1.0 s) and `latest` (epoch 50). The evaluator picks per run the one with the higher mean val AP over 0.5/1.0/1.5 s; an exact tie goes to `best` |

## Files

| File | Role | Gate |
|---|---|---|
| `progress_cps_v4/common.py` | pinned constants, hashing, content digests, atomic writes, code/env identity, RNG restore | all |
| `progress_cps_v4/prepare_data_pinned.py` | HF `train/**` at revision `7535d065…`, ids from `results/repro/id_to_hf_source.csv`, train.csv sha must equal the manifest, split recomputed | G1 |
| `progress_cps_v4/preflight.py` | versions vs the recorded env, CUDA, slow_r50 weights identity, fwd+bwd peak memory, disk, token | G3 |
| `progress_cps_v4/audit_data.py` | split / ids / mapping / 1,500 videos / PTS / event vs duration / 1,200+300×3 cache scan; writes `val_timing.csv` | G2 |
| `progress_cps_v4/build_sidecar.py` | per-snippet actual end frame, PTS, τ, mask and reason; suspect zeros re-decoded; ≥ 24 pixel replays plus abnormal videos; cache content digest | G2 |
| `progress_cps_v4/supervision_targets.py` | targets, mask rules, CPS loss | — |
| `progress_cps_v4/train_bps.py` | training; fingerprint; atomic checkpoints; `COMPLETE.json`; `--check-complete` | G3/G4/G6 |
| `progress_cps_v4/lock_config.py` | immutable `LOCK.json`, written only when G1–G4 evidence exists | G5 |
| `progress_cps_v4/eval_val.py` | validation: all 300 videos for accuracy, 150 positives for temporal, dense timing gate, research gate, raw-float decision | G7 |
| `progress_cps_v4/official_test_protocol.py` | official test: freeze → infer (exactly 1,344 ids) → score once (public 667 / private 677) | G8 |
| `progress_cps_v4/unit_tests_v3.py`, `progress_cps_v4/regression_tests_v4.py` | 33 v3 tests + 25 v4 regression tests (CPU) | G0 |
| `progress_cps_v4/run_all.sh` | fail-fast stage runner | — |
| `progress_cps_v4/requirements.txt` | pinned small dependencies (torch comes from the image) | — |

Baseline files (`pipeline/cell*.py`, `reeval/precache_val`–`eval_sensitivity`) are only imported and never modified.

## Fresh vast.ai run

**Prerequisites.**
- An RTX 4080-class GPU with ≥ 16 GB.
- An image with Python 3.12, `torch==2.11.0+cu128` and `torchvision==0.26.0+cu128` (the environment of RQ1–RQ2). The scripts never install or upgrade torch.
- At least 150 GB of disk: HF train videos (tens of GB, not measured here), train cache ≈ 11 GB (1,200 × 12 × 5 × 224² × 3 bytes), val cache ≈ 0.7 GB, dense cache ≈ 7 GB, 18 checkpoints × 242 MB.
- The repo must live at `/workspace/CPV301`, because `riskprop_precache_snippet_sequence` hard-codes that path.

`/workspace` is **not** guaranteed to survive on every vast.ai offer. Copy `outputs_progress/` (checkpoints, logs) off the instance after each stage that matters.

```bash
cd /workspace && git clone https://github.com/TranMinhHuy15/CPV301.git && cd CPV301
unzip -o /path/CPV301_rq3_progress_v4.zip          # overlay on the base commit, see RELEASE_MANIFEST_V4.json
export HF_TOKEN=...                                  # never written to a file
tmux new -s rq3
bash reeval/progress_cps_v4/run_all.sh setup                # pinned small deps + /kaggle/working/data link
bash reeval/progress_cps_v4/run_all.sh preflight            # G3 environment; FAIL -> fix the image, not the code
bash reeval/progress_cps_v4/run_all.sh prepare              # G1; FAIL -> stop (do not edit the manifest)
bash reeval/progress_cps_v4/run_all.sh cache                # riskprop_precache_snippet_sequence train cache + precache_val val cache
bash reeval/progress_cps_v4/run_all.sh check                # G0 tests + G2 audit + sidecar; must print usable_for_full_run: True
bash reeval/progress_cps_v4/run_all.sh smoke                # G3: real model, CUDA, 2 steps x B/P/S
bash reeval/progress_cps_v4/run_all.sh mini                 # G4: 40 videos x 2 epochs, resume test, dev eval (not a result)
LOCK_CONFIRMED=1 bash reeval/progress_cps_v4/run_all.sh lock    # G5: immutable LOCK.json
bash reeval/progress_cps_v4/run_all.sh full                 # G6: 9 runs; rerun the same command after any interruption
bash reeval/progress_cps_v4/run_all.sh eval                 # G7: research-mode validation report + decision
bash reeval/progress_cps_v4/run_all.sh sens                 # optional sensitivity (H = 1.5 s)
bash reeval/progress_cps_v4/run_all.sh test-freeze && bash reeval/progress_cps_v4/run_all.sh test-infer \
  && bash reeval/progress_cps_v4/run_all.sh test-score      # G8: official test, once
bash reeval/progress_cps_v4/run_all.sh pack                 # small files -> results/progress_cps_v4/ (no .pth/video/token)
```

**Outputs.**

| Stage | Files |
|---|---|
| prepare | `outputs_progress/prepare/prepare_report.json` |
| preflight | `outputs_progress/preflight/preflight.json` |
| check | `outputs_progress/audit/{audit_summary.json,audit_report.txt,audit_videos.csv,val_timing.csv}` and `outputs_progress/sidecar/{sidecar.json,sidecar_summary.json,sidecar_snippets.csv}` |
| lock | `outputs_progress/LOCK.json` (+ `.sha256`) |
| full | `outputs_progress/runs/<run>/{best,latest}.pth, config.json, train_log.csv, COMPLETE.json` |
| eval | `outputs_progress/eval_val/{summary_rq3_progress.md, rq3_decision.json, bootstrap_ci.csv, per_run_metrics.csv, mean_sd.csv, alert_gap_strata_*.csv, dense_gate.json, chosen_checkpoints.json}` |
| test | `outputs_progress/test/{test_freeze.json, submission_*.csv, infer_receipt.json, score_receipt.json, test_summary.md}` |

## Retry and resume policy

- **Every stage is fail-fast.** On a non-zero exit the stage stops, the original error stays in `outputs_progress/logs/<stage>.log`, and later stages are not run.
- **Resume after an interruption.** Rerun `full`. A run is skipped only when `train_bps --check-complete` confirms `COMPLETE.json` with the same fingerprint and checkpoint hashes:
  - The fingerprint covers config, code digest, init digest, split/index/val-cache/sidecar/audit/LOCK hashes and the train-cache digest.
  - An unfinished run resumes from `latest.pth` at an **epoch boundary**. Checkpoints are loaded on CPU and RNG states restored on the right device. A resumed run is not guaranteed to be bitwise identical in the middle of an epoch.
  - On CPU, an interrupted-then-resumed dummy run is bitwise equal to an uninterrupted one (test `test_resume_equals_uninterrupted_and_marker`). On CUDA the same is checked by `mini` and `test_cuda_resume`.
- **When resume is refused.** If anything in the fingerprint changed (λ, α, H, sidecar, LOCK, code, cache, init), resume is refused and nothing is overwritten. Use a new `OUT` or move the old run away. Never edit the old fingerprint.
- **Checkpoint safety.** Checkpoints are written atomically (tmp → fsync → reload → replace). A crash, OOM, non-finite loss or corrupt write leaves the previous valid checkpoint and never produces `COMPLETE.json`.
- **No bypasses for full runs.** The following cannot be overridden: missing CUDA, an incomplete audit, an unusable sidecar (pixel mismatch, cache-only zero snippets, missing PTS), a changed cache, or a LOCK mismatch. `--accept-cache-zeros` and `--allow-index-timing` exist only for dev diagnostics and never make a sidecar usable.

## Common failures

| Message | Meaning | What to do |
|---|---|---|
| `prepare: rebuilt train.csv sha256 … != manifest` | the HF metadata at this revision differs from the one the split was made on | stop; do not edit the manifest; check `--revision` |
| `F7 … resolve to a different HF file` | ids are mapped to other videos (unpinned data_download_from_hf download) | rerun `prepare` into a clean `data/nexar_kaggle_style` |
| `F8 PTS not strictly increasing` / `F6 event beyond the last frame` | corrupt or variable-timestamp media | report the video; no silent fallback |
| `sidecar … cache-only zero snippets` | the riskprop_precache_snippet_sequence cache holds zeros although the video decodes | rebuild the riskprop_precache_snippet_sequence cache, then rerun `check` |
| `REFUSE full run … differs from LOCK` | code/data changed after the lock | rerun the gates; a new LOCK needs a new `OUT` |
| `Dense timing gate: FAIL` | dense endpoints cannot be verified | PVR/ADS/RCJ are diagnostic only; the accuracy decision is unaffected |
| `RESEARCH GATE FAIL` | missing run/seed, dev run, changed checkpoint, different init/code | no research conclusion; fix and re-evaluate |

## Reporting rules

- **mTTA_detected** is a *discrete lead-time summary under FAR ≤ 0.1*: the largest of 0.5/1.0/1.5 s detected at the per-lead FAR-0.1 threshold. Always report it with Coverage and the actual FAR. It is not PRE-ACT's mTTA and not an onset metric.
- **Temporal metrics** (PVR/ADS/RCJ) are always read together with rise, range and negative score level. A flat curve is not better early warning. If the dense gate fails, make no temporal claim.
- **Validation CIs** are internal and computed after checkpoint choice. A video bootstrap does not capture training-seed variance, so also report mean ± SD over the 3 seeds.
- **Wording when only primary + NI pass:** say "accuracy support under the pre-registered rule". If P > B but P ≈ S, this supports extra supervision, not the continuous target shape.
- **Alert–event gap.** 347/750 train positives have a gap > 1.5 s and 200/750 > 2 s. Results per gap group (validation counts) are in `alert_gap_strata_mean.csv`. These labels are not "wrong".
- **Episode-level leakage** cannot be checked, because Nexar has no episode key.
