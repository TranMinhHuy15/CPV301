# Supplementary — Prediction Ensemble and Calibration Diagnostics

> Status: post-hoc, supplementary only. Not used for checkpoint or method selection, and does
> not replace the mean ± SD tables in `RESULTS_RQ1.md` / `RESULTS_RQ2.md` / `RESULTS_RQ3.md`.
> Both analyses below are run on the **internal validation split**, using the existing
> per-seed checkpoints — no retraining was done.

## 1. Three-seed prediction ensemble

For each condition, the three seeds' per-video scores were averaged (mean of the raw
prediction, not of the metric) before computing AP, using the validation split.

| Condition | Ensemble mAP | Mean of per-seed mAP (published) | Difference |
|---|---|---|---|
| RiskProp random | 0.763 | 0.745 ± 0.033 | +0.018 |
| RiskProp fixed (τ=1.0s) | 0.766 | 0.743 ± 0.005 | +0.023 |
| RQ2 A (BCE only) | 0.657 | 0.635 ± 0.061 | +0.022 |
| RQ2 B (FFR-only) | 0.769 | 0.743 ± 0.016 | +0.026 |
| RQ2 C (AMC-only) | 0.667 | 0.653 ± 0.023 | +0.014 |

**How to read this:** a supplementary three-seed prediction ensemble improves validation mAP
relative to averaging the individual seed-level metrics. This is a routine ensembling effect
(combining correlated but non-identical predictors), not evidence that any one method is
"better" than another — the ensemble was computed identically for every condition shown, and
no condition was selectively re-run or cherry-picked. It should not be read as "ensemble
proves RiskProp is better."

**Not done:** the equivalent ensemble for TOP and AdaLEA (no per-video raw validation
predictions were available in this working session; the checkpoints exist, but inference has
not been re-run — see §3).

**Retracted:** an earlier attempt to compute the official-test ensemble used a locally cached
copy of `solution.csv` that turned out to be corrupted (it failed a sanity check against
`sample_submission.csv`, which should score 0.841/0.862 but scored 0.234–0.500 on the cached
copies). Those official-test ensemble numbers were wrong and are not used anywhere in this
project. Any future official-test ensemble must re-download `solution.csv` and
`evaluate_submission.py` fresh from the source (`nexar-ai/nexar_collision_prediction` on
Hugging Face) before scoring.

## 2. Calibration diagnostics

Brier score and Expected Calibration Error (ECE, 10 bins), computed on the same
three-seed-ensembled validation scores, per horizon.

| Condition | Brier @0.5s / 1.0s / 1.5s | ECE @0.5s / 1.0s / 1.5s |
|---|---|---|
| RiskProp random | 0.172 / 0.231 / 0.280 | 0.138 / 0.196 / 0.249 |
| RiskProp fixed (τ=1.0s) | 0.193 / 0.242 / 0.296 | 0.163 / 0.214 / 0.275 |
| RQ2 A (BCE only) | 0.316 / 0.393 / 0.423 | 0.321 / 0.382 / 0.413 |
| RQ2 B (FFR-only) | 0.183 / 0.215 / 0.257 | 0.130 / 0.166 / 0.229 |
| RQ2 C (AMC-only) | 0.348 / 0.434 / 0.466 | 0.376 / 0.452 / 0.472 |

**Findings:**
- RiskProp random is better calibrated than the ablations that remove FFR (A, C).
- Among the RQ2 conditions, FFR-only (B) has the best calibration; AMC-only (C) has the worst.
- ECE increases at longer horizons (1.5s) for every condition — predictions get less
  reliable further from the event.
- ECE is non-trivial everywhere (0.13–0.47): predicted scores should **not** be interpreted
  literally as calibrated collision probabilities without further calibration (e.g. temperature
  scaling) if used in a real warning system.

**How to read this:** this is a calibration diagnostic, not evidence about which condition has
higher mAP. It answers "how trustworthy are the raw probability outputs", a different question
from accuracy (RQ1/RQ2/RQ3's own metrics).

**Not done:** calibration for TOP and AdaLEA (same reason as §1) — so this is **not** a
three-way model comparison. It only compares RiskProp against its own ablations.

## 3. What would be needed to extend this to TOP and AdaLEA

All 27 runs' checkpoints (TOP, AdaLEA, RiskProp random/fixed + all RQ2/RQ3 ablations, both
`best` and `latest`) are confirmed present and complete on the team's local machine
(`D:\FA26\CPV\text_checkpoint`, 54 files, verified against the expected 27 × 2 set with no
gaps). Having the checkpoints is not the same as having predictions: TOP and AdaLEA still need
one inference pass over the validation split (and, if wanted, a fresh official-test pass using
newly downloaded `solution.csv`) before their ensemble/calibration numbers can be added here.
This is not required to finish RQ1–RQ3 — it only extends the supplementary analysis to all
three methods instead of RiskProp alone.

## 4. Bottom line

None of the above changes the main conclusions:
- **RQ1:** TOP and RiskProp are statistically indistinguishable on the official test set;
  both outperform AdaLEA.
- **RQ2:** FFR accounts for essentially all of the accuracy gain; AMC's effect on accuracy is
  not established, and its temporal-consistency benefit is smaller once FFR is present.
- **RQ3:** fixed-lag pairing preserves accuracy and gives a small PVR improvement, but does not
  reliably reduce seed-to-seed variance.

This document is supplementary. The reproducibility/writing checklist for RQ1–RQ3 is complete
independently of it.
