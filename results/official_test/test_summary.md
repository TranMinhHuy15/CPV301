# Official Nexar test set -- generated 2026-09-25 01:47

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

