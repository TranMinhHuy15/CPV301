# Progressive RiskProp / CPS v4 results

This folder contains the results for the paper's RQ3: **Progressive RiskProp / CPS v4**. This is a **PRE-ACT-inspired loss-level adaptation, not a reproduction of PRE-ACT**. The only data folders at this level are `eval_val/` for internal validation, `test/` for the official Nexar test, and `log/` for logs and supporting run/audit files. This README is the partner landing page.

**Headline:** the pre-registered accuracy criterion was **not met**. The validation result does not show that continuous CPS (P) improves the primary metric over the matched baseline (B). The official test is reported as a final descriptive check; the validation decision remains the RQ3 decision.

## Start here

| File | What it contains |
|---|---|
| [eval_val/summary_rq3_progress.md](eval_val/summary_rq3_progress.md) | Internal validation report, pre-registered decision, confidence intervals, and supporting metrics |
| [eval_val/per_run_metrics.csv](eval_val/per_run_metrics.csv) | Validation metrics for each of the nine runs, one row per run |
| [eval_val/mean_sd.csv](eval_val/mean_sd.csv) | Validation means and standard deviations over the three seeds |
| [test/test_summary.md](test/test_summary.md) | Aggregate Nexar official-test results and paired bootstrap intervals |
| [test/test_per_run.csv](test/test_per_run.csv) | Official-test metrics for each run, including public and private mAP |
| [eval_val/rq3_decision.json](eval_val/rq3_decision.json) | Machine-readable validation decision |
| [../../reeval/progress_cps_v4/README.md](../../reeval/progress_cps_v4/README.md) | Locked design, protocol, metric definitions, and reproduction instructions |
| [../RESULTS.md](../RESULTS.md) · [../../README.md](../../README.md) | Overall results and repository map |

## B, P, and S: the nine runs

All conditions use the same SlowOnly-R50 model and fixed-lag pairing. B is a retrained, matched RiskProp baseline.

| Code | Meaning |
|---|---|
| **B** | Baseline: BCE + 0.5 FFR + 0.5 AMC, with no progress supervision (CPS mode none) |
| **P** | Continuous CPS: B plus the continuous progress-supervision loss |
| **S** | Dense binary control: B plus binary supervision for 0 < τ < H, using the same mask and loss weight as P |

CPS is a PRE-ACT-inspired loss adaptation, **not a reproduction of PRE-ACT**. The locked run uses H = 2.0 s, α = 3, λ = 1, and training seeds **42, 43, and 44**.

That is **3 conditions × 3 seeds = 9 runs**:

- **B:** B_fixed_seed42, B_fixed_seed43, B_fixed_seed44
- **P:** P_fixed_H2_a3_lam1_seed42, P_fixed_H2_a3_lam1_seed43, P_fixed_H2_a3_lam1_seed44
- **S:** S_fixed_H2_a3_lam1_seed42, S_fixed_H2_a3_lam1_seed43, S_fixed_H2_a3_lam1_seed44

## Validation results

Validation accuracy uses 300 videos; temporal metrics use the 150 positive videos. Each seed cell is **mAP / AP@1.5 s**. The mean and SD are across the three training seeds.

| Condition | Seed 42 (mAP / AP@1.5 s) | Seed 43 (mAP / AP@1.5 s) | Seed 44 (mAP / AP@1.5 s) | Mean mAP ± SD | Mean AP@1.5 s ± SD |
|---|---:|---:|---:|---:|---:|
| B (baseline) | 0.7504 / 0.6984 | 0.7427 / 0.6613 | 0.7263 / 0.6426 | 0.7398 ± 0.0123 | 0.6674 ± 0.0284 |
| P (continuous CPS) | 0.7375 / 0.6540 | 0.7122 / 0.6216 | 0.7455 / 0.6411 | 0.7317 ± 0.0173 | 0.6389 ± 0.0163 |
| S (binary control) | 0.7010 / 0.6127 | 0.7526 / 0.6735 | 0.7346 / 0.6489 | 0.7294 ± 0.0262 | 0.6450 ± 0.0306 |

The pre-registered P − B primary comparison was **−0.0285 AP@1.5 s** (95% CI −0.0614 to +0.0084). The mAP difference was **−0.0081** (95% CI −0.0366 to +0.0207); the required non-inferiority margin was 0.02, and this check was not met. See the [full validation report](eval_val/summary_rq3_progress.md) for the decision rule and supporting analyses.

## Official Nexar test results

The official test has **1,344 clips**: 667 public and 677 private. It was scored once after validation. The table gives **all-clip mAP for each seed**; mean ± SD is across seeds.

| Condition | Seed 42 | Seed 43 | Seed 44 | All-clip mAP mean ± SD |
|---|---:|---:|---:|---:|
| B (baseline) | 0.7642 | 0.7360 | 0.7836 | 0.7612 ± 0.0239 |
| P (continuous CPS) | 0.7903 | 0.7132 | 0.7753 | 0.7596 ± 0.0409 |
| S (binary control) | 0.7723 | 0.7973 | 0.7522 | 0.7739 ± 0.0226 |

Across seeds, public / private / all-clip mAP was: B **0.7254 / 0.8088 / 0.7612**, P **0.7413 / 0.7839 / 0.7596**, and S **0.7465 / 0.8024 / 0.7739**. The official-test P − B all-clip difference was −0.0016 (95% CI −0.0184 to +0.0156), which is inconclusive. The [official summary](test/test_summary.md) and [per-run CSV](test/test_per_run.csv) contain the full results.

## Folder map

| Folder | What is inside |
|---|---|
| [eval_val/](eval_val/) | Validation summaries, per-run metrics, decision record, and supporting prediction files |
| [test/](test/) | Official-test summaries, per-run scores, receipts, freeze record, and nine submitted prediction files |
| [log/](log/) | Every `.log` and `.txt` file, the sealed LOCK and checksum, and supporting run/audit folders |
| [log/runs/](log/runs/) | Config, completion record, and training-curve CSV for each of the nine runs |
| [log/audit/](log/audit/), [log/prepare/](log/prepare/), [log/preflight/](log/preflight/) | Data audit, preparation, and environment checks |
| [log/sidecar/](log/sidecar/) and [log/sidecar_sample/](log/sidecar_sample/) | Snippet timing/target audit tables and summaries |

`eval_val/`, `test/`, and `log/` are the only data folders at the top level. The README stays beside them so GitHub can show the partner guide when the folder is opened.
## Logs and audit files you can skip on a first read

“Skip” means **you do not need to open these to understand the reported results**. Keep them in the repository: they preserve the run history and can help with an audit or reproduction.

- **Console logs in `log/`:** audit.log; cache_cell30.log; cache_re01.log; eval_mini.log; eval_val.log; full_master.log; full_seed42_B_H2.0.log; full_seed42_P_H2.0.log; full_seed42_S_H2.0.log; full_seed43_B_H2.0.log; full_seed43_P_H2.0.log; full_seed43_S_H2.0.log; full_seed44_B_H2.0.log; full_seed44_P_H2.0.log; full_seed44_S_H2.0.log; lock.log; mini_B.log; mini_P.log; mini_S.log; mini_resume.log; preflight.log; preflight_full.log; preflight_smoke.log; prepare.log; setup.log; sidecar.log; sidecar_sample.log; smoke_B.log; smoke_P.log; smoke_S.log; test.log; unit_tests.log.
- **Console/text snapshots in `log/`:** all_epochs.txt, eval_summary.txt, mini_summary.txt, report.txt. For the findings, use the Markdown and CSV summaries linked above.
- **Detailed data and environment audit records:** log/audit/audit_report.txt, log/audit/audit_summary.json, log/audit/audit_videos.csv, log/audit/val_timing.csv, log/preflight/preflight.json, log/prepare/prepare_report.json, log/sidecar/sidecar_snippets.csv, log/sidecar/sidecar_summary.json, log/sidecar_sample/sidecar_snippets.csv, log/sidecar_sample/sidecar_summary.json.
- **Training curves:** log/runs/<run>/train_log.csv (one per run). Open that run's config.json and COMPLETE.json only if you need its exact settings or completion evidence. Model checkpoint .pth files are not committed.
- **Detailed validation artifacts:** eval_val/dense_val/*.npz, eval_val/preds_val/*.{json,npz}, eval_val/dense_gate.json, eval_val/chosen_checkpoints.json, eval_val/bootstrap_ci.csv, and eval_val/alert_gap_strata_*.csv. These support the concise validation report.
- **Official-test audit artifacts:** test/submission_*.csv, test/infer_receipt.json, test/score_receipt.json, test/test_freeze.json, and test/test_bootstrap_ci.csv. Use these to inspect the submitted predictions and one-time scoring trail.

**Preserve the sealed protocol:** do not edit or replace log/LOCK.json or log/LOCK.json.sha256. The run configs, completion records, raw metrics, bootstrap files, and test freeze/receipts are evidence for the experiment even if you do not need to open them for a quick read.
