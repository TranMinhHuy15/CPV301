# RQ3 (new): Progressive RiskProp, run guide (vast.ai)

**Progressive RiskProp** is RiskProp plus a continuous progress supervision term (CPS) inspired by PRE-ACT. It is **not** a reproduction of PRE-ACT: there is no VideoMAE, no progress head and no preference loss, and λ is not PRE-ACT's 10.

All files are new; no existing file in `pipeline/` or `reeval/re01–re10` is changed. RQ1b is out of scope.

## What each file does

| File | Role | GPU |
|---|---|---|
| `re13_progress_supervision.py` | CPS target (PRE-ACT `exp_above`, H = 2 s, α = 3), dense-binary control, validity rules, loss | no |
| `re16_test_progress_protocol.py` | 30 unit tests: targets, boundaries, masks, gradients, parity with `cell32`, model-input contract, full-run gate, PTS, zero snippets, sidecar binding, resume fingerprint, stale predictions | no |
| `re11_audit_progress_data.py` | Read-only audit of the split, metadata, cache, PTS/VFR, decode, and ID → HF mapping. Exits with code 2 on any fatal issue | no |
| `re12_build_progress_sidecar.py` | Sidecar holding the real endpoint frame and PTS, `tau_actual`, CPS mask and reason for each snippet. Verifies that pixels match the cache | no |
| `re14_train_progress_riskprop.py` | Trains B / P / S (optionally F / FP). The base loss is `cell32.riskprop_loss`, called unchanged | yes |
| `re15_eval_progress_riskprop.py` | Validation only: locked `per_run` checkpoint rule, re03 metrics, re07 dense curves, re10 PVR/ADS/RCJ plus score amplitude, alert-gap strata, paired bootstrap, non-inferiority | yes |
| `run_progress_rq3.sh` | Runs the stages `setup → data → check → smoke → mini → lock → full → eval → (sens) → pack` | |

## Conditions

| | Loss | Purpose |
|---|---|---|
| **B** | BCE + 0.5 FFR + 0.5 AMC | Matched RiskProp baseline, retrained with the same script |
| **P** | B + λ · SmoothL1(σ(z), r(τ)) | Continuous progress target |
| **S** | B + λ · SmoothL1(σ(z), 1[0<τ<H]) | Dense binary target with the same mask |

**CPS mask:** positive videos only; τ is computed from the real PTS of the last frame. The mask excludes:
- the two BCE anchors (k = 0 and k = 11);
- decode failures and zero-valued snippets;
- snippets with τ ≤ 0 (PRE-ACT gives these a target of 1);
- duplicate endpoints caused by the clamp. With event 3.032 s at 30 fps, **7** snippets share one frame, so only k = 6 is kept.

**Nominal targets on the 12-snippet grid:**

| τ (s) | 5.6…2.1 | 1.6 | 1.1 | 0.6 |
|---|---|---|---|---|
| P target | 0 | 0.475 | 0.780 | 0.924 |
| S target | 0 | 1 | 1 | 1 |

The two targets differ on only about 3 snippets, so read P − S with caution.

## Commands (vast.ai, RTX 4080 16 GB, disk ≥ 120 GB)

```bash
cd /workspace && git clone https://github.com/TranMinhHuy15/CPV301.git && cd CPV301
# copy the 8 new files into reeval/ (or pull after they are pushed)
export HF_TOKEN=...            # never written to any file
tmux new -s rq3                # training survives a closed SSH session
bash reeval/run_progress_rq3.sh setup
bash reeval/run_progress_rq3.sh data     # HF download + cell30 (~1 h) + re01; checks split == results/repro
bash reeval/run_progress_rq3.sh check    # MUST end with AUDIT PASS + sidecar pass; otherwise stop and send the report
bash reeval/run_progress_rq3.sh smoke    # 2 steps per condition, real model
bash reeval/run_progress_rq3.sh mini     # 40 videos x 2 epochs + eval; pipeline check only
# team agrees on the design block at the top of run_progress_rq3.sh, then:
LOCK_CONFIRMED=1 bash reeval/run_progress_rq3.sh lock
bash reeval/run_progress_rq3.sh full     # 9 runs, about 12-16 GPU-h; rerun the same command to resume
bash reeval/run_progress_rq3.sh eval     # outputs_progress/eval_val/summary_rq3_progress.md
bash reeval/run_progress_rq3.sh sens     # optional: P with H = 1.5 s, seed 42
bash reeval/run_progress_rq3.sh pack     # small files -> results/rq3_progress/ (no .pth, no video, no token)
```

**Design values proposed in the script; the team must confirm them before `lock`:**
- pairing = `fixed`;
- H = 2.0, α = 3, λ = 1.0 (a single value, no sweep);
- seeds 42/43/44;
- primary metric = AP@1.5s (P − B);
- non-inferiority on mAP with margin 0.02.

These can be overridden with environment variables, for example `LAMBDA=0.5 bash ... lock`.

## Rules the scripts enforce

- `--full-run` is refused unless all of the following hold:
  - the audit passed (complete, videos probed, cache scanned);
  - the sidecar is complete and its pixels were verified;
  - the cache IDs match the manifest;
  - there are no limits and the model is the real one with 50 epochs;
  - the arguments equal `LOCK.json`.
- Dev runs (`smoke`, `mini`) go to `outputs_progress/dev/` and are marked `research_result: false`.
- The official test set is not used. After the validation result is frozen, the test is scored once with a fresh `solution.csv`.
- **B is retrained.** The old `riskprop_fixed` checkpoints have no recorded commit, cache hash or config, so they are not treated as a matched baseline.

## Data-integrity checks (v2)

These are cases where the code would still run without errors but produce wrong numbers. The scripts now stop instead.

| Risk | Check |
|---|---|
| PTS does not start at 0, or does not increase | `re11` F8 and `re12`: non-increasing, non-finite or negative PTS is fatal. A first-frame offset is subtracted and recorded as `pts_offset_s`. A video without PTS cannot be used for a full run unless `--allow-index-timing` is passed. |
| Sidecar does not match the data | The sidecar stores sha256 of the manifest, of `train_index.json` and of every `.pt` file (`cache_digest`). `re14` checks the manifest and index hashes on every run, and recomputes `cache_digest` on a full run. A full run also needs at least 24 videos re-decoded with identical pixels. |
| An all-zero snippet is not automatically a decode failure | `re12` re-decodes every zero snippet and labels it `zero_decode_fail` (decoding fails), `zero_black` (the frame really is black) or `zero_cache_only` (the video decodes, but the cache holds zeros). All three are excluded from CPS. If `cache_only` > 0, rebuild the `cell30` cache, or pass `ACCEPT_CACHE_ZEROS=1`, which is recorded. |
| Old checkpoints or predictions reused by mistake | Each checkpoint stores a `fingerprint` (config, plus hashes of manifest, train index, val index, sidecar and lock). Resume is refused if anything differs. `re15` reuses predictions or dense curves only if the checkpoint sha256, the val index and the dense meta all match; otherwise it prints `[STALE]` and recomputes. `--analysis-only` refuses when a checkpoint has changed, and evaluation refuses runs trained on different data. |

## Known limits (report them)

- **Episode-level leakage cannot be checked:** Nexar has no episode key.
- **mTTA is not PRE-ACT's mTTA.** It is the largest of the three discrete leads detected at the FAR 0.1 threshold, and is reported together with Coverage.
- **Temporal metrics must be read with amplitude.** PVR/ADS/RCJ appear in the same table as score amplitude (rise, range, negative level), because a flat curve can look "monotonic".
- **The alert–event gap can exceed H.** 347/750 positives have a gap > 1.5 s, so results are also reported per gap bin.
