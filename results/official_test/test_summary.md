# Official Nexar test set -- generated 2026-09-25 01:52

Checkpoints and TOP head rule LOCKED on the internal validation split (results/reeval_corrected/locked_config.json; RQ2 runs: rq2_chosen_checkpoints.json). RQ2 rows are exploratory.

## 1. Official scorer (Nexar evaluate_submission.py, unmodified) -- PRIMARY

| Method | n_seeds | mAP public | mAP private |
|---|---|---|---|
| TOP | 3 | 0.7687 +/- 0.0164 | 0.8111 +/- 0.0273 |
| AdaLEA | 3 | 0.7003 +/- 0.0115 | 0.7951 +/- 0.0188 |
| RiskProp random | 3 | 0.7563 +/- 0.0230 | 0.8115 +/- 0.0042 |
| RiskProp fixed | 3 | 0.7450 +/- 0.0183 | 0.8048 +/- 0.0151 |
| RQ2 A neither | 3 | 0.6536 +/- 0.0293 | 0.6865 +/- 0.0816 |
| RQ2 B FFR-only | 3 | 0.7625 +/- 0.0235 | 0.8238 +/- 0.0268 |
| RQ2 C AMC-only | 3 | 0.6549 +/- 0.0083 | 0.7098 +/- 0.0238 |

## 2. Re-implemented official metric on the whole test set (public + private)

AP inside each solution.csv group, averaged over groups (official definition); mAUC0.1 = partial AUC at FPR<=0.1 on the same groups. Group labels = horizon of their positives.

| Method | mAP | AP g0 (0.5s) | AP g1 (1.0s) | AP g2 (1.5s) | mAUC01 |
|---|---|---|---|---|---|
| TOP | 0.7852 +/- 0.0210 | 0.8112 +/- 0.0314 | 0.7843 +/- 0.0162 | 0.7602 +/- 0.0263 | 0.2728 +/- 0.0348 |
| AdaLEA | 0.7419 +/- 0.0114 | 0.7764 +/- 0.0147 | 0.7327 +/- 0.0145 | 0.7167 +/- 0.0146 | 0.2107 +/- 0.0137 |
| RiskProp random | 0.7808 +/- 0.0152 | 0.8316 +/- 0.0170 | 0.7641 +/- 0.0235 | 0.7469 +/- 0.0199 | 0.2542 +/- 0.0375 |
| RiskProp fixed | 0.7721 +/- 0.0133 | 0.8313 +/- 0.0216 | 0.7525 +/- 0.0207 | 0.7324 +/- 0.0190 | 0.2586 +/- 0.0299 |
| RQ2 A neither | 0.6670 +/- 0.0556 | 0.7294 +/- 0.0800 | 0.6556 +/- 0.0579 | 0.6161 +/- 0.0337 | 0.1530 +/- 0.0528 |
| RQ2 B FFR-only | 0.7905 +/- 0.0213 | 0.8411 +/- 0.0201 | 0.7646 +/- 0.0268 | 0.7658 +/- 0.0175 | 0.2929 +/- 0.0385 |
| RQ2 C AMC-only | 0.6788 +/- 0.0164 | 0.7717 +/- 0.0218 | 0.6610 +/- 0.0228 | 0.6037 +/- 0.0411 | 0.1880 +/- 0.0200 |

Check: max |our mAP - official mAP| over runs and both subsets = 0.000000 (identical to the official scorer).

## 3. Paired bootstrap 95% CI on the whole test set (B=2000, resampling within every group x label cell)

Estimate = first minus second; value per method = mean over its 3 seeds. RQ2 rows exploratory.

| comparison | metric | estimate | ci_low | ci_high | verdict |
|---|---|---|---|---|---|
| TOP - AdaLEA | mAP | 0.0433 | 0.0245 | 0.0596 | positive (CI > 0) |
| TOP - AdaLEA | mAUC01 | 0.0621 | 0.026 | 0.0966 | positive (CI > 0) |
| TOP - RiskProp random | mAP | 0.0044 | -0.018 | 0.0281 | inconclusive (CI includes 0) |
| TOP - RiskProp random | mAUC01 | 0.0186 | -0.0311 | 0.0678 | inconclusive (CI includes 0) |
| AdaLEA - RiskProp random | mAP | -0.0389 | -0.0599 | -0.0146 | negative (CI < 0) |
| AdaLEA - RiskProp random | mAUC01 | -0.0435 | -0.0845 | -0.0015 | negative (CI < 0) |
| RiskProp fixed - RiskProp random | mAP | -0.0087 | -0.0221 | 0.006 | inconclusive (CI includes 0) |
| RiskProp fixed - RiskProp random | mAUC01 | 0.0044 | -0.0246 | 0.033 | inconclusive (CI includes 0) |
| FFR effect, AMC absent (B-A) | mAP | 0.1235 | 0.0995 | 0.1471 | positive (CI > 0) |
| FFR effect, AMC absent (B-A) | mAUC01 | 0.1399 | 0.1022 | 0.1827 | positive (CI > 0) |
| AMC effect, FFR absent (C-A) | mAP | 0.0118 | -0.0069 | 0.0312 | inconclusive (CI includes 0) |
| AMC effect, FFR absent (C-A) | mAUC01 | 0.0349 | 0.0079 | 0.0581 | positive (CI > 0) |
| FFR effect, AMC present (D-C) | mAP | 0.1021 | 0.0777 | 0.1234 | positive (CI > 0) |
| FFR effect, AMC present (D-C) | mAUC01 | 0.0663 | 0.0324 | 0.106 | positive (CI > 0) |
| AMC effect, FFR present (D-B) | mAP | -0.0097 | -0.0236 | 0.0048 | inconclusive (CI includes 0) |
| AMC effect, FFR present (D-B) | mAUC01 | -0.0387 | -0.0703 | -0.0075 | negative (CI < 0) |
| Interaction (D-C)-(B-A) | mAP | -0.0214 | -0.0446 | 0.0027 | inconclusive (CI includes 0) |
| Interaction (D-C)-(B-A) | mAUC01 | -0.0736 | -0.1122 | -0.0316 | negative (CI < 0) |

