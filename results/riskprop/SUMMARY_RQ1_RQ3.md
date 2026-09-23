# RiskProp Results Summary (RQ1 + RQ3)

## RQ1 anchor — RiskProp random-offset (3 seeds)
| Metric | seed42 | seed43 | seed44 | mean±std |
|---|---|---|---|---|
| Proposal mAP | 0.5449 | 0.7264 | 0.6474 | 0.6396 ± 0.0910 |
| Video AUC | 0.5141 | 0.7463 | 0.6514 | 0.6373 ± 0.1167 |
| mTTA@FAR<=0.1 (s) | 0.974 | 1.081 | 1.177 | 1.0773 ± 0.1015 |
| mAUC^0.1 | 0.0670 | 0.1804 | 0.1356 | 0.1277 ± 0.0571 |

## RQ3 — RiskProp fixed-lag (tau=1.0s) vs random-offset
| Metric | Fixed (3 seeds) | Random (3 seeds) |
|---|---|---|
| Proposal mAP | 0.6738 ± 0.0309 | 0.6396 ± 0.0910 |
| Video AUC | 0.6774 ± 0.0219 | 0.6373 ± 0.1167 |
| mTTA@FAR<=0.1 (s) | 1.0027 ± 0.0635 | 1.0773 ± 0.1015 |
| mAUC^0.1 | 0.1535 ± 0.0452 | 0.1277 ± 0.0571 |

## Note (limitation)
seed42-random: val_loss diverged after epoch 0 while val_mAP kept improving
(epoch0 mAP=0.52 vs epoch17 mAP=0.71). Checkpoint selection (by val_loss)
picked epoch 0 as "best", likely under-reporting this seed's true performance.
Only best/latest checkpoints are saved (no per-epoch), so not retroactively fixable.
