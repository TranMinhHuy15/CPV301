# RQ3 -- Progressive RiskProp, internal validation

Generated 2026-10-05 23:02 | accuracy cohort 300 videos (150 positives) | temporal cohort 150 positives | pairing=fixed H=2.0 alpha=3.0 lambda=1.0 | checkpoint rule: per_run, tie -> best | research_result=True

Dense timing gate: PASS

## 0. Pre-registered decision (LOCK.json; raw floats)

- Research gate: PASS
- Primary: P - B on AP@1.5s = -0.0285 [-0.0614, +0.0084] -> NOT met
- Non-inferiority: P - B on mAP = -0.0081 [-0.0366, +0.0207], margin 0.02 -> NOT met
- **RQ3 supported (accuracy-based pre-registered rule): NO**
- P - S on AP@1.5s (pre-specified, reported regardless): -0.0061 [-0.0399, +0.0252] -> inconclusive (CI includes 0)
- A YES is accuracy support only; temporal improvement needs the dense gate to pass AND supportive PVR/ADS/RCJ with non-flat curves (rise/range). P > B with P ~ S supports extra supervision, not the continuous shape specifically.

## 1. Accuracy / low-FAR (mean +/- SD over seeds; all validation videos)

| Condition | n_seeds | mAP | AP@0.5s | AP@1.0s | AP@1.5s | mAUC01 | Recall@1.0s | ActualFAR@1.0s | mTTA_detected | Coverage |
|---|---|---|---|---|---|---|---|---|---|---|
| B | 3 | 0.7398 +/- 0.0123 | 0.8158 +/- 0.0103 | 0.7362 +/- 0.0119 | 0.6674 +/- 0.0284 | 0.2333 +/- 0.0178 | 0.3822 +/- 0.0269 | 0.1000 +/- 0.0000 | 1.0667 +/- 0.0359 | 0.6244 +/- 0.0518 |
| P | 3 | 0.7317 +/- 0.0173 | 0.8370 +/- 0.0192 | 0.7192 +/- 0.0216 | 0.6389 +/- 0.0163 | 0.2442 +/- 0.0480 | 0.3400 +/- 0.0742 | 0.1000 +/- 0.0000 | 0.9891 +/- 0.0182 | 0.6378 +/- 0.0849 |
| S | 3 | 0.7294 +/- 0.0262 | 0.8159 +/- 0.0222 | 0.7274 +/- 0.0295 | 0.6450 +/- 0.0306 | 0.2222 +/- 0.0512 | 0.4022 +/- 0.0749 | 0.1000 +/- 0.0000 | 1.1142 +/- 0.0649 | 0.6311 +/- 0.0844 |

mTTA_detected = largest of the 3 discrete leads detected at FAR <= 0.1 (discrete lead-time summary, not PRE-ACT's mTTA, not an onset metric); always read with Coverage.

## 2. Temporal metrics WITH score amplitude (positives, dense curves; PVR/ADS/RCJ lower = better)

| Condition | PVR | ADS | RCJ | score_early_pos | score_late_pos | rise_pos | range_pos | score_neg_mean | score_neg_p95 |
|---|---|---|---|---|---|---|---|---|---|
| B | 0.1831 +/- 0.0169 | 0.0274 +/- 0.0030 | 0.1038 +/- 0.0127 | 0.3836 +/- 0.1988 | 0.7421 +/- 0.1597 | 0.3585 +/- 0.0667 | 0.7109 +/- 0.1013 | 0.2101 +/- 0.1976 | 0.9994 +/- 0.0007 |
| P | 0.1809 +/- 0.0109 | 0.0253 +/- 0.0035 | 0.0997 +/- 0.0131 | 0.1178 +/- 0.0288 | 0.5536 +/- 0.0404 | 0.4358 +/- 0.0313 | 0.7514 +/- 0.0086 | 0.0593 +/- 0.0101 | 0.9378 +/- 0.0372 |
| S | 0.1887 +/- 0.0225 | 0.0281 +/- 0.0024 | 0.1063 +/- 0.0096 | 0.1338 +/- 0.0226 | 0.5385 +/- 0.0053 | 0.4047 +/- 0.0279 | 0.7573 +/- 0.0229 | 0.0704 +/- 0.0107 | 0.9892 +/- 0.0028 |

A low PVR/RCJ with a small rise/range means a flat curve, not better early warning.

## 3. Target fit (diagnostic only, unclamped dense windows)

| Condition | fit_MAE_continuous | fit_MAE_binary |
|---|---|---|
| B | 0.3691 +/- 0.0232 | 0.4043 +/- 0.0531 |
| P | 0.3332 +/- 0.0091 | 0.4652 +/- 0.0053 |
| S | 0.3387 +/- 0.0082 | 0.4600 +/- 0.0087 |

## 4. Alert-event gap groups (validation; mean over seeds; positives of the group + all negatives)

| gap_group | condition | n_pos | mAP_group | AP@1.5s | PVR | rise_pos |
|---|---|---|---|---|---|---|
| <=0.5 | B | 10.0 | 0.1396 | 0.1131 | 0.2847 | 0.1323 |
| 0.5-1.0 | B | 29.0 | 0.3982 | 0.3395 | 0.1458 | 0.3475 |
| 1.0-1.5 | B | 42.0 | 0.4757 | 0.371 | 0.1808 | 0.3716 |
| 1.5-2.0 | B | 29.0 | 0.3616 | 0.2727 | 0.2105 | 0.3492 |
| >2.0 | B | 40.0 | 0.5448 | 0.4268 | 0.1674 | 0.4162 |
| >1.5 s | B | 69.0 | 0.6134 | 0.511 | 0.1855 | 0.388 |
| >2.0 s | B | 40.0 | 0.5448 | 0.4268 | 0.1674 | 0.4162 |
| <=0.5 | P | 10.0 | 0.1452 | 0.1227 | 0.2165 | 0.1956 |
| 0.5-1.0 | P | 29.0 | 0.3829 | 0.2902 | 0.1558 | 0.4563 |
| 1.0-1.5 | P | 42.0 | 0.4473 | 0.3059 | 0.2093 | 0.3951 |
| 1.5-2.0 | P | 29.0 | 0.3887 | 0.2679 | 0.1778 | 0.4354 |
| >2.0 | P | 40.0 | 0.5694 | 0.4098 | 0.1627 | 0.5238 |
| >1.5 s | P | 69.0 | 0.6294 | 0.498 | 0.169 | 0.4867 |
| >2.0 s | P | 40.0 | 0.5694 | 0.4098 | 0.1627 | 0.5238 |
| <=0.5 | S | 10.0 | 0.1338 | 0.1181 | 0.2299 | 0.1146 |
| 0.5-1.0 | S | 29.0 | 0.3592 | 0.2651 | 0.1727 | 0.4 |
| 1.0-1.5 | S | 42.0 | 0.425 | 0.3127 | 0.1889 | 0.4197 |
| 1.5-2.0 | S | 29.0 | 0.4086 | 0.3063 | 0.187 | 0.3864 |
| >2.0 | S | 40.0 | 0.5472 | 0.4153 | 0.191 | 0.4782 |
| >1.5 s | S | 69.0 | 0.63 | 0.5185 | 0.1893 | 0.4396 |
| >2.0 s | S | 40.0 | 0.5472 | 0.4153 | 0.191 | 0.4782 |

## 5. Contrasts (paired label-stratified bootstrap 95% CI, B=2000, seed 12345)

| contrast | metric | role | cohort | estimate | ci_low | ci_high | verdict | direction | non_inferiority |
|---|---|---|---|---|---|---|---|---|---|
| P - B | mAP | accuracy | accuracy n=300 videos | -0.0081 | -0.0366 | +0.0207 | inconclusive (CI includes 0) |  | non-inferiority NOT shown (margin 0.02) |
| P - B | AP@1.5s | primary | accuracy n=300 videos | -0.0285 | -0.0614 | +0.0084 | inconclusive (CI includes 0) |  |  |
| P - B | mAUC01 | accuracy | accuracy n=300 videos | +0.0109 | -0.0421 | +0.0582 | inconclusive (CI includes 0) |  |  |
| P - B | PVR | supporting | temporal n=150 positives | -0.0022 | -0.0236 | +0.0192 | inconclusive (CI includes 0) |  |  |
| P - B | ADS | supporting | temporal n=150 positives | -0.0021 | -0.0052 | +0.0009 | inconclusive (CI includes 0) |  |  |
| P - B | RCJ | supporting | temporal n=150 positives | -0.0041 | -0.0142 | +0.0057 | inconclusive (CI includes 0) |  |  |
| S - B | mAP | accuracy | accuracy n=300 videos | -0.0104 | -0.0422 | +0.0210 | inconclusive (CI includes 0) |  | non-inferiority NOT shown (margin 0.02) |
| S - B | AP@1.5s | accuracy | accuracy n=300 videos | -0.0224 | -0.0574 | +0.0173 | inconclusive (CI includes 0) |  |  |
| S - B | mAUC01 | accuracy | accuracy n=300 videos | -0.0111 | -0.0752 | +0.0552 | inconclusive (CI includes 0) |  |  |
| S - B | PVR | supporting | temporal n=150 positives | +0.0056 | -0.0135 | +0.0233 | inconclusive (CI includes 0) |  |  |
| S - B | ADS | supporting | temporal n=150 positives | +0.0007 | -0.0024 | +0.0038 | inconclusive (CI includes 0) |  |  |
| S - B | RCJ | supporting | temporal n=150 positives | +0.0025 | -0.0070 | +0.0120 | inconclusive (CI includes 0) |  |  |
| P - S | mAP | accuracy | accuracy n=300 videos | +0.0023 | -0.0259 | +0.0277 | inconclusive (CI includes 0) |  |  |
| P - S | AP@1.5s | pre-specified | accuracy n=300 videos | -0.0061 | -0.0399 | +0.0252 | inconclusive (CI includes 0) |  |  |
| P - S | mAUC01 | accuracy | accuracy n=300 videos | +0.0220 | -0.0398 | +0.0734 | inconclusive (CI includes 0) |  |  |
| P - S | PVR | supporting | temporal n=150 positives | -0.0078 | -0.0224 | +0.0067 | inconclusive (CI includes 0) |  |  |
| P - S | ADS | supporting | temporal n=150 positives | -0.0028 | -0.0051 | -0.0007 | decrease (CI < 0) | better |  |
| P - S | RCJ | supporting | temporal n=150 positives | -0.0066 | -0.0136 | +0.0002 | inconclusive (CI includes 0) |  |  |

Notes: validation only (internal; CIs after checkpoint choice are not an independent confirmation; video bootstrap does not cover training-seed uncertainty -- see mean +/- SD). Official test: re19.