# CHANGELOG v4: Progressive RiskProp (RQ3)

**Base.** GitHub `TranMinhHuy15/CPV301@cf7de850` (v2 pipeline), overlaid with the v3 changes (re14/re15/re16, run script, README). This is the base the v4 overlay applies to.

**No design change.** The following are exactly as locked on 2026-10-05:
- backbone and head;
- targets, mask and loss;
- split, pairing, H/α/λ, seeds, budget;
- checkpoint and decision rules.

**Status values used below.**
- **fixed:** a problem was observed (E) or needed verification (V) and the code now handles it. A test is named for each.
- **hardening:** H, a protection added; no observed failure.
- **NOT RUN:** needs real media or CUDA, which this environment does not have.

## Item table

| ID | Type | File / function | Change | Test(s) | Status |
|---|---|---|---|---|---|
| V4-01 | E | packaging, `progress_common.py` | full overlay of all RQ3 files; manifest of sha256 for every file; base commit and prerequisite baseline files listed; `verify_release.py` | `test_v4_01_all_modules_import_and_cli_help`; fresh-extract log | fixed |
| V4-02 | E | `run_progress_rq3.sh` (`run_logged`, `set -euo pipefail`) | every python/pip call checks its own exit status through `tee`; filters run on logs afterwards with `\|\| true`; a stage stops at the first failure and the full log keeps the original error | `TestShell.test_failing_step_stops_stage_and_keeps_log`, `test_filter_without_matches_is_not_a_failure_and_child_failure_is` | fixed |
| V4-03 | E | `re14.complete_ok`, `--check-complete`; run script `full` | done markers replaced by `COMPLETE.json` (fingerprint + best/latest sha256), written only after the final checkpoint re-loads; `full` skips a run only if `--check-complete` exits 0, and fails if training ends without a verifiable marker | `test_v4_03_complete_marker`, `TestShell.test_full_skips_only_verified_and_fails_unverified`, `test_resume_equals_uninterrupted_and_marker` | fixed |
| V4-04 | V | new `re17_prepare_data_pinned.py` | HF download pinned to revision `7535d065…` (train/** only); ids from `id_to_hf_source.csv`; train.csv rebuilt with cell09 columns (LF) and its sha256 required to equal the manifest; split recomputed; 750/750 labels, events and alerts checked; existing different files are never overwritten | `TestPrepare.*`; synthetic `prepare` stage | fixed (code); **NOT RUN on the real HF revision** |
| V4-05 | V | `re11` | audit bound to the train.csv / manifest / id map / train+val index hashes and the prepare report (HF revision); `complete` only if 1,500 probed + 1,200 train + 300×3 val scanned, mapping verifiable, not limited; re12/re14/re20 refuse incomplete audits | `test_full_run_refused_with_limits_or_wrong_lambda` (incomplete audit) | fixed (code); **NOT RUN on real media** |
| V4-06 | V | `re11.pts_policy` (shared by re12) | PTS finite, strictly increasing, first ≥ 0 (fatal F8); offset subtracted and logged; frame = 1/avg_fps of *that* video; VFR suspect flagged; event after the last frame fatal (F6); no silent index-timing fallback for full runs | `test_v4_06_pts_policy`, `TestPtsAndZero.*` | fixed (code); **NOT RUN on real media** |
| V4-07 | V | `re12`, `re13` | ≥ 24 deterministic pixel replays with both labels, plus every abnormal video (zero snippet, nominal clamp, PTS offset); mismatch fatal; `cache_only` zeros make the sidecar unusable (flags are dev-only); nominal clamp logged apart from duplicate end frames; CPS coverage per positive reported (positives without valid snippets stay in BCE/FFR/AMC) | `TestMasks.*`, `test_zero_kinds`; synthetic `check` (24/24 replays) | fixed (code); **NOT RUN on real media** |
| V4-08 | E | `progress_common.val_cache_digest / dense_cache_digest`, `re15.preds_fresh / dense_fresh`, `re14.build_inputs` | content digests of every val `.pt` and of `windows_u8.npy`+meta; predictions and dense curves are reused only with the same checkpoint sha256 and the same content digest; full runs bind val-cache and train-cache content | `test_v4_08_pixel_mutation_changes_val_and_dense_digest`, `test_stale_predictions_detected` | fixed |
| V4-09 | E/H | `re14.run_fingerprint`, `lock_errors` | fingerprint = config + code digest (SOURCE_FILES) + init digest + split/index/val/train-cache/sidecar/audit/LOCK hashes + budget; full runs never resume dev runs (separate dirs + `full_run` in the fingerprint); LOCK fingerprints compared at full start | `test_v4_09_fingerprint_code_init_lock`, `test_resume_refused_when_config_or_data_changes`, code/sidecar/val-cache cases in the gate test | fixed |
| V4-10 | V | `re14` resume, `progress_common.restore_rng` | checkpoints loaded with `map_location="cpu"`; torch/CUDA RNG forced to CPU ByteTensor; DataLoader generator state saved and restored; resume at epoch boundaries only | `test_resume_equals_uninterrupted_and_marker` (CPU, bitwise-equal weights); `test_cuda_resume` | fixed (CPU verified); **CUDA: NOT RUN** (skipped, no GPU) |
| V4-11 | H | `progress_common.torch_save_atomic`, `re14` | tmp → fsync → reload-verify → `os.replace`; previous checkpoint kept on any failure; full requires CUDA; non-finite train/val loss exits 4 without a marker | `test_v4_11_atomic_save_keeps_last_valid` | fixed (hardening) |
| V4-12 | V | `re14` | model built right after seeding (init digest recorded; B/P/S of one seed share it); loss untouched (`cell32` called as is); CPS consumes no RNG; AMP skipped-step count, CPS coverage and head grad norms of both terms logged; no clipping or freezing added | `test_parity_with_cell32`, `test_v4_12_same_init_for_conditions_and_lambda0_parity` | fixed |
| V4-13 | E/V | `re11.val_timing_rows` → `val_timing.csv`; `re15.dense_gate` | 300 videos × (3 leads + 30 dense windows): requested vs actual endpoint (PTS), τ, future-frame flag, deviation in s and in frames of that video; gate checks coverage 300/150, requested == dense meta, no future frame, no duplicates, no clamp, \|dev\| ≤ 1 frame, no VFR; lead windows with a future frame are fatal in re11 | `test_v4_13_val_timing_rows`, `TestDenseGateAndDecision.*`, `test_lead_and_dense_rules_match_re01_re07` | fixed (code); **NOT RUN on real media** |
| V4-14 | E | `re15.make_cohorts`, bootstrap | accuracy metrics and CIs on all 300 videos; temporal on positives with dense curves; a missing dense curve never shrinks the accuracy cohort (v3 bug: the bootstrap used the dense-filtered ids) | `test_v4_14_accuracy_cohort_not_shrunk_by_dense` | fixed |
| V4-15 | E | `re15.decision` | decision on raw float CIs (CSV written with full precision; rounding only in printed tables); non-finite CI → not available; P − S always reported when S exists | `test_v4_15_raw_float_precision` | fixed |
| V4-16 | E/V | `re15.research_gate`, `--research` | conclusion only if B/P/S × 42/43/44 present, all full research runs under the same LOCK, `COMPLETE.json` matching checkpoints, same code, same init per seed, 300/150 cohorts, finite predictions, manifest order; otherwise "no research conclusion" and exit 5 in research mode | `test_v4_16_research_gate` | fixed |
| V4-17 | E | new `re19_test_progress.py` (`infer`) | reuses `re04.extract_tail_window` / `to_tensor` / `predict` and `re05.run_official` / `grouped_metrics`; exactly 1,344 unique ids matched one-to-one to video files; any undecodable clip fatal (no 0/0.5 fill); finite scores; public 667 / private 677 reported separately | `test_v4_17_exact_ids` | fixed (code); **NOT RUN** (no test media, no trained B/P/S) |
| V4-18 | H | `re19` (`freeze`, `score`) | freeze requires a passed research gate and stores chosen checkpoint hashes; `infer` re-verifies them; `score` runs once (byte-identical submissions only re-print); never triggered by train/mini/eval | `test_v4_18_freeze_requires_research_gate` | fixed (hardening) |
| V4-19 | E | `re15`, README | mTTA_detected + Coverage + ActualFAR reported; documented as a discrete lead-time summary under FAR ≤ 0.1 (re03 definition unchanged) | report text | fixed (docs) |
| V4-20 | V | `re15`, README | dense grid 3.0 → 0.1 s (early → late), cohort n shown; amplitude (early/late/rise/range/negative) next to PVR/ADS/RCJ; target fit diagnostic; temporal rows labelled "diagnostic" when the gate fails | `TestDenseGateAndDecision.*` | fixed |
| V4-21 | V | `re15` strata | validation groups: 5 gap bins plus > 1.5 s and > 2.0 s, with n_pos/n_neg per group (validation counts, not the 750 train) | synthetic eval output | fixed |
| V4-22 | E/V | `README_PROGRESS_RQ3.md` | RQ numbering table (old/new), locked design and rationale, deviations from PRE-ACT, commands, artifact paths, retry policy, common failures; result files and root README unchanged | — | fixed (docs) |
| V4-23 | E | `re16` (33 kept, 4 adapted to the v4 API), new `re21` (25) | regression tests for V4-02/03/04/06/08/09/10/11/12/13/14/15/16/17/18 and G5 | `logs/cpu_re16_tests.txt`, `logs/cpu_re21_tests.txt` | fixed (CPU); CUDA test skipped = NOT RUN |
| V4-24 | H | `requirements_progress_rq3.txt`, `re18_preflight.py`, `release_v4/*` | pinned small deps (torch from the image, never upgraded); preflight checks versions vs the recorded env, CUDA, slow_r50 weights identity and peak memory, disk, token; manifest + verify script; acceptance report | fresh-extract log | fixed (code); preflight on GPU **NOT RUN** |

## Additions not in the handover list

- **G5:** new `re20_lock_progress.py` writes LOCK.json only from real G1–G4 evidence. LOCK stores the design, the decision rule and all fingerprints. It is immutable: an identical rerun is a no-op, anything different is refused. It is never created when `PROGRESS_SYNTHETIC_COUNTS` is set.
- **Synthetic plumbing mode:** `PROGRESS_SYNTHETIC_COUNTS` lets the whole chain run on tiny fake data. Everything produced in this mode is marked `synthetic`, so it can never pass the research gate, LOCK or test freeze. The tests always unset it.

## Deviations or decisions declared here (not research changes)

- **Checkpoint tie-break:** an exact float tie in mean val AP goes to `best`. This was the existing re03 rule, now written down.
- **Validation BCE for `best.pth`:** computed in fp32 from autocast logits. cell34 computed it in fp16. The same code is used for B, P and S.
- **DataLoader shuffle:** uses a dedicated generator seeded with the run seed, which can be saved for resume. The batch order therefore differs from v3 runs, but is identical across B/P/S for a seed.
- **Budget:** v3 quoted "12–16 GPU-h" without measuring it; v4 makes no fixed claim. `train_log.csv` records time per epoch and peak memory, so the real throughput comes from `mini` and the first `full` epoch.

## v4.0.1 patch (2026-10-05, found by the first real GPU run of `check`)

- **`re21_test_v4_regression.py::test_cuda_resume` (test-only bug).** The test removed the value `cpu` from the argument list and then overwrote the element after `--device`, which was `--lambda-prog`. As a result argparse exited with code 2. Fix: replace the value of `--device` in place.
- **Impact.** No training, evaluation or data code changed, and the code fingerprint is unaffected (re21 is not in `SOURCE_FILES`). The test was reported **NOT RUN** on CPU and is now exercised on the GPU by the `check` stage.

## v4.0.2 patch (2026-10-05, from the real G2 audit, before any training result)

- **Real data.** The audit of the 300 validation videos found that 34 of 4,500 dense positive windows (in 5 videos) had |actual − requested| endpoint = 1.000002–1.000031 frames, i.e. one frame plus at most 0.001 ms. No video was VFR-suspect, the maximum PTS deviation from index/avg_fps was 6e-5 frames, and no endpoint fell at or after the event.
- **Cause.** `int(t·fps)` (re07/cell10 rule) can land exactly one frame early when t·fps falls on a frame boundary; the excess is float rounding.
- **Change.** The `re15.dense_gate` tolerance is now |dev_s| ≤ frame_period of that video + 1 ms (`DENSE_TOL_S`); it was ≤ 1.0 frame + 1e-6.
- **Status of the change.** Fixed before LOCK and before any validation result, so this is a numeric-precision correction of the gate, not a tuning step. Test: `TestDenseTolerance`. re15 is not in `SOURCE_FILES`, so the code fingerprint of training runs is unaffected.
