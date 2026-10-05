# V4 acceptance report: Progressive RiskProp (RQ3)

**Verdict: v4 code complete. NOT "ready for full".** The gates G1–G5 have not been run on the real Nexar data and a GPU. This environment has no CUDA, no HF access and no raw media. Each status below says what was actually run.

## Environment of these checks

| Item | Value |
|---|---|
| Machine | sandbox CPU container, no GPU, no HF access |
| Software | Python 3.11.15 · torch 2.14.0 (CUDA build, `cuda_available=False`) · numpy 2.4.4 · pandas 3.0.2 · scikit-learn 1.8.0 · decord 0.6.0 · Pillow 12.2.0 · torchvision not installed · bash 5.2.21 · ffmpeg 6.1.1 |
| Difference from the recorded RQ1–RQ2 environment | That environment was Python 3.12.14, torch 2.11.0+cu128, torchvision 0.26.0+cu128, RTX 4080. The difference is why every GPU and media gate below is **NOT RUN**. |
| Base | GitHub `cf7de850` (fresh clone) + this overlay → `verify_release.py` PASS (`logs/fresh_extract_test.txt`) |

## Gate status

| Gate | What | Status | Evidence |
|---|---|---|---|
| G0 package | syntax, imports, CLI `--help`, unit/regression tests, fresh extraction | **PASS (CPU)** | `logs/static_checks.txt`, `logs/cpu_re16_tests.txt` (33/33), `logs/cpu_re21_tests.txt` (24 pass, 1 skipped = CUDA), `logs/fresh_extract_test.txt` |
| G1 metadata | pinned HF revision, id map, 1,500 records, train.csv sha, split | **NOT RUN on real data.** PASS on synthetic data (30 videos, synthetic manifest) | `logs/synthetic_e2e_cpu_dummy.txt` (stage prepare) |
| G2 media/cache | probe 1,500, scan 1,200 + 300×3, PTS/event timing, zero classification, pixel replay, cache digest, val timing | **NOT RUN on real data.** PASS on synthetic data (24/24 pixel replays, audit complete, val timing 6 videos) | `logs/synthetic_e2e_cpu_dummy.txt` (stage check) |
| G3 GPU smoke | real SlowOnly-R50, B/P/S × 2 steps, CUDA/AMP/memory, preflight | **NOT RUN** (no GPU). Dummy-model CPU plumbing ran (`smoke` with `MODEL=dummy`) | synthetic log (stage smoke) |
| G4 mini | 40 videos × 2 epochs B/P/S, resume test, dev eval | **NOT RUN with the real model/GPU.** CPU dummy: PASS, including interrupted-then-resumed training and the dev evaluation | synthetic log (stage mini); `test_resume_equals_uninterrupted_and_marker` |
| G5 lock | immutable LOCK with fingerprints | **NOT RUN.** Correctly refused on synthetic data. Logic tested (`test_g5_lock_requires_evidence_and_is_immutable`) | synthetic log (stage lock) |
| G6 full | B/P/S × 3 seeds × 50 epochs | **NOT RUN.** Correctly refused without LOCK / CUDA | synthetic log (stage full) |
| G7 validation | 300 videos, research gate, decision | **NOT RUN on real runs.** Logic tested; dev evaluation ran on synthetic data | `test_v4_14/15/16`, eval_mini in synthetic log |
| G8 official test | freeze / 1,344 ids / score once | **NOT RUN** (no trained runs, no test media). Id and freeze logic tested | `test_v4_17_exact_ids`, `test_v4_18_freeze_requires_research_gate` |
| CUDA resume | restore on CUDA, RNG/optimizer/scaler | **NOT RUN** (`test_cuda_resume` skipped). Covered on GPU by `mini` | — |

> "PASS" above means PASS in the stated environment only. Synthetic and dummy runs prove plumbing and gates, not data quality, timing of the real videos, GPU behaviour or any research result.

## What the user/Codex must run (in order) before calling it "ready for full"

1. `setup`, then `preflight`. preflight must PASS with `--require-cuda`. A WARN on versions means the image differs from torch 2.11+cu128; decide before continuing.
2. `prepare`. G1: `prepare_report.json` must have `"pass": true` and the train.csv sha must equal the manifest.
3. `cache`, then `check`. G2: `AUDIT PASS complete=True` and `SIDECAR usable_for_full_run: True`. Keep `audit_report.txt`, `sidecar_summary.json` and `val_timing.csv`.
4. `smoke`. G3: real model, CUDA, 2 steps × B/P/S. Note the peak memory and seconds per step.
5. `mini`. G4: three COMPLETE dev runs, the "Resumed P_…" line, and `eval_mini`. The scores are not a result.
6. `LOCK_CONFIRMED=1 … lock`. G5.

Only after steps 1–6 pass is the pipeline **ready for full**. Throughput for the budget estimate comes from `train_log.csv` (`time_sec`, `peak_mem_gb`) of `mini` and the first full epoch. No fixed GPU-hour figure is promised.

## Remaining blockers / known limits

- **Unverified against the real dataset:**
  - real HF metadata layout at revision `7535d065…`. re17 discovers CSVs under `train/`; if the layout differs it fails with the columns it found. It never guesses.
  - real video PTS behaviour.
  - cell30 pixel replay on the vast.ai decoder.
- **GPU behaviour unverified:** memory, AMP skipped steps, CUDA resume.
- **Episode-level leakage** cannot be checked (no episode key in Nexar).
- **Fixed path:** `cell30` hard-codes `/workspace/CPV301`, so the repo must live there (the `cache` stage checks this).
- **Statistics:** validation CIs are internal and computed after checkpoint choice; the video bootstrap does not include training-seed variance.
- **Not in v4:** F/FP optional conditions are supported by the code but not part of the main plan.
