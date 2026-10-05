# RQ3 official Nexar test (scored once) -- 2026-10-05 23:22

Public 667 / private 677 clips; checkpoints frozen from validation (test_freeze.json). The RQ3 decision is the validation decision (re15); this is the final report.

| Condition | n_seeds | mAP public (official) | mAP private (official) | mAP all | mAUC@0.1 all |
|---|---|---|---|---|---|
| B | 3 | 0.7254 +/- 0.0288 | 0.8088 +/- 0.0203 | 0.7612 +/- 0.0239 | 0.2528 +/- 0.0339 |
| P | 3 | 0.7413 +/- 0.0243 | 0.7839 +/- 0.0527 | 0.7596 +/- 0.0409 | 0.2574 +/- 0.0308 |
| S | 3 | 0.7465 +/- 0.0306 | 0.8024 +/- 0.0280 | 0.7739 +/- 0.0226 | 0.2784 +/- 0.0446 |

## Paired bootstrap 95% CI (all 1,344 clips, within group x label cells, B=2000)

| contrast | metric | estimate | ci_low | ci_high | verdict |
|---|---|---|---|---|---|
| P - B | mAP | -0.0016 | -0.0184 | +0.0156 | inconclusive (CI includes 0) |
| P - B | mAUC01 | +0.0046 | -0.0260 | +0.0352 | inconclusive (CI includes 0) |
| S - B | mAP | +0.0127 | -0.0048 | +0.0275 | inconclusive (CI includes 0) |
| S - B | mAUC01 | +0.0257 | -0.0085 | +0.0559 | inconclusive (CI includes 0) |
| P - S | mAP | -0.0143 | -0.0282 | +0.0016 | inconclusive (CI includes 0) |
| P - S | mAUC01 | -0.0211 | -0.0496 | +0.0077 | inconclusive (CI includes 0) |
