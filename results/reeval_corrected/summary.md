# Re-evaluation on the internal validation split

Generated 2026-09-24 09:26 | 300 videos (150 positive, 150 negative)

## 1. Legacy check

Original protocol (best checkpoint, TOP matched head) vs the eval_results_*.txt already on GitHub. Tolerance 0.0015.

| run | file | old_mAP | new_mAP | max_abs_diff | ok |
|---|---|---|---|---|---|
| adalea_seed42 | adalea/eval_results_rq1_adalea5f_seed42.txt | 0.761 | 0.761 | 0.00033 | True |
| adalea_seed43 | adalea/eval_results_rq1_adalea5f_seed43.txt | 0.6444 | 0.6444 | 4e-05 | True |
| adalea_seed44 | adalea/eval_results_rq1_adalea5f_seed44.txt | 0.6493 | 0.6493 | 4e-05 | True |
| riskprop_fixed_seed42 | riskprop/eval_results_rq3_riskprop_fixed_seed42.txt | 0.6908 | 0.6908 | 0.00033 | True |
| riskprop_fixed_seed43 | riskprop/eval_results_rq3_riskprop_fixed_seed43.txt | 0.6925 | 0.6925 | 0.00033 | True |
| riskprop_fixed_seed44 | riskprop/eval_results_rq3_riskprop_fixed_seed44.txt | 0.6382 | 0.6382 | 3e-05 | True |
| riskprop_random_seed42 | riskprop/eval_results_rq1_riskprop_random_seed42.txt | 0.5449 | 0.5449 | 5e-05 | True |
| riskprop_random_seed43 | riskprop/eval_results_rq1_riskprop_random_seed43.txt | 0.7264 | 0.7264 | 0.00033 | True |
| riskprop_random_seed44 | riskprop/eval_results_rq1_riskprop_random_seed44.txt | 0.6474 | 0.6474 | 5e-05 | True |
| top_seed42 | top/eval_results_rq1_top5f_seed42.txt | 0.6519 | 0.6519 | 0.00033 | True |
| top_seed43 | top/eval_results_rq1_top5f_seed43.txt | 0.7107 | 0.7107 | 0.00033 | True |
| top_seed44 | top/eval_results_rq1_top5f_seed44.txt | 0.6923 | 0.6923 | 0.00033 | True |

**Verdict: PASS -- validation split reproduced**

## 2. TOP fixed head rule

Chosen by mean validation mAP over TOP's 6 prediction sets (3 seeds x best/latest). The oracle matched head is not a candidate.

| rule | n_sets | mean_val_mAP |
|---|---|---|
| head_0.5 | 6 | 0.6613 |
| head_1.0 | 6 | 0.7023 |
| head_1.5 | 6 | 0.7192 |
| head_2.0 | 6 | 0.726 |
| mean3 | 6 | 0.7025 |
| max | 6 | 0.7257 |

**Locked: `head_2.0`**

## 3. Checkpoint rule: `per_run`

per_run = each run keeps the higher-val-mAP checkpoint of {best-val-loss, last epoch}; global = one checkpoint type for all runs. Same rule for all 12 runs.

| run | best_mAP | best_epoch | latest_mAP | latest_epoch | chosen |
|---|---|---|---|---|---|
| adalea_seed42 | 0.761 | 7 | 0.6937 | 50 | best |
| adalea_seed43 | 0.6444 | 2 | 0.7303 | 50 | latest |
| adalea_seed44 | 0.6493 | 6 | 0.7269 | 50 | latest |
| riskprop_fixed_seed42 | 0.6908 | 1 | 0.7393 | 50 | latest |
| riskprop_fixed_seed43 | 0.6925 | 3 | 0.7408 | 50 | latest |
| riskprop_fixed_seed44 | 0.6382 | 4 | 0.748 | 50 | latest |
| riskprop_random_seed42 | 0.5449 | 1 | 0.7108 | 50 | latest |
| riskprop_random_seed43 | 0.7264 | 1 | 0.7481 | 50 | latest |
| riskprop_random_seed44 | 0.6474 | 2 | 0.7769 | 50 | latest |
| top_seed42 | 0.6808 | 5 | 0.7626 | 50 | latest |
| top_seed43 | 0.73 | 11 | 0.7373 | 50 | latest |
| top_seed44 | 0.6913 | 3 | 0.7541 | 50 | latest |

## 4. RQ1 main table (mean +/- SD over 3 seeds; TOP head `head_2.0`)

| Model | n_seeds | mAP | AP@0.5s | AP@1.0s | AP@1.5s | mAUC01 | VideoAUC |
|---|---|---|---|---|---|---|---|
| TOP | 3 | 0.7513 +/- 0.0129 | 0.8065 +/- 0.0237 | 0.7544 +/- 0.0114 | 0.6931 +/- 0.0069 | 0.2228 +/- 0.0270 | 0.7720 +/- 0.0096 |
| AdaLEA | 3 | 0.7394 +/- 0.0187 | 0.7841 +/- 0.0174 | 0.7366 +/- 0.0215 | 0.6975 +/- 0.0216 | 0.2405 +/- 0.0241 | 0.7523 +/- 0.0097 |
| RiskProp random | 3 | 0.7453 +/- 0.0331 | 0.8171 +/- 0.0218 | 0.7390 +/- 0.0379 | 0.6797 +/- 0.0399 | 0.2294 +/- 0.0734 | 0.7679 +/- 0.0290 |

| Model | n_seeds | Recall@0.5s | Recall@1.0s | Recall@1.5s | ActualFAR@0.5s | ActualFAR@1.0s | ActualFAR@1.5s | mTTA_detected | mTTA_all | Coverage |
|---|---|---|---|---|---|---|---|---|---|---|
| TOP | 3 | 0.4756 +/- 0.0336 | 0.3911 +/- 0.0192 | 0.2778 +/- 0.0102 | 0.1000 +/- 0.0000 | 0.0978 +/- 0.0038 | 0.1000 +/- 0.0000 | 1.0865 +/- 0.0477 | 0.6578 +/- 0.0212 | 60.7 +/- 4.4% |
| AdaLEA | 3 | 0.4778 +/- 0.0857 | 0.4000 +/- 0.0240 | 0.3578 +/- 0.0743 | 0.0978 +/- 0.0038 | 0.0978 +/- 0.0038 | 0.1000 +/- 0.0000 | 1.2081 +/- 0.0124 | 0.6900 +/- 0.0960 | 57.1 +/- 7.9% |
| RiskProp random | 3 | 0.4911 +/- 0.1128 | 0.3667 +/- 0.1058 | 0.2733 +/- 0.0987 | 0.0978 +/- 0.0038 | 0.0978 +/- 0.0038 | 0.0978 +/- 0.0038 | 1.1038 +/- 0.0793 | 0.6289 +/- 0.1644 | 56.4 +/- 11.4% |

Note: each negative clip has one cached window, so false alarms per clip at the FAR<=0.1 threshold equal ActualFAR.

### Per-seed ranking by val mAP (RQ1)

| seed | #1 | #2 | #3 |
|---|---|---|---|
| 42 | TOP 0.7626 | AdaLEA 0.7610 | RiskProp random 0.7108 |
| 43 | RiskProp random 0.7481 | TOP 0.7373 | AdaLEA 0.7303 |
| 44 | RiskProp random 0.7769 | TOP 0.7541 | AdaLEA 0.7269 |

## 5. RQ3 table (mean +/- SD over 3 seeds)

| Model | n_seeds | mAP | AP@0.5s | AP@1.0s | AP@1.5s | mAUC01 | VideoAUC |
|---|---|---|---|---|---|---|---|
| RiskProp random | 3 | 0.7453 +/- 0.0331 | 0.8171 +/- 0.0218 | 0.7390 +/- 0.0379 | 0.6797 +/- 0.0399 | 0.2294 +/- 0.0734 | 0.7679 +/- 0.0290 |
| RiskProp fixed | 3 | 0.7427 +/- 0.0046 | 0.8169 +/- 0.0093 | 0.7392 +/- 0.0010 | 0.6720 +/- 0.0064 | 0.2242 +/- 0.0181 | 0.7598 +/- 0.0169 |

| Model | n_seeds | Recall@0.5s | Recall@1.0s | Recall@1.5s | ActualFAR@0.5s | ActualFAR@1.0s | ActualFAR@1.5s | mTTA_detected | mTTA_all | Coverage |
|---|---|---|---|---|---|---|---|---|---|---|
| RiskProp random | 3 | 0.4911 +/- 0.1128 | 0.3667 +/- 0.1058 | 0.2733 +/- 0.0987 | 0.0978 +/- 0.0038 | 0.0978 +/- 0.0038 | 0.0978 +/- 0.0038 | 1.1038 +/- 0.0793 | 0.6289 +/- 0.1644 | 56.4 +/- 11.4% |
| RiskProp fixed | 3 | 0.5222 +/- 0.0077 | 0.3756 +/- 0.0102 | 0.2822 +/- 0.0252 | 0.1000 +/- 0.0000 | 0.1000 +/- 0.0000 | 0.1000 +/- 0.0000 | 1.0785 +/- 0.0304 | 0.6544 +/- 0.0299 | 60.7 +/- 1.3% |

### Paired seeds, fixed - random (mAP)

| seed | fixed_mAP | random_mAP | diff |
|---|---|---|---|
| 42 | 0.7393 | 0.7108 | 0.0285 |
| 43 | 0.7408 | 0.7481 | -0.0073 |
| 44 | 0.748 | 0.7769 | -0.0289 |

mean diff = -0.0026, SD of diff = 0.0290

## Appendix A. Best run per method (descriptive only)

| Model | seed | ckpt | mAP | AP@0.5s | AP@1.0s | AP@1.5s | mAUC01 | VideoAUC | Coverage |
|---|---|---|---|---|---|---|---|---|---|
| TOP | 42 | latest | 0.7626 | 0.82 | 0.7669 | 0.701 | 0.254 | 0.7825 | 61.3% |
| AdaLEA | 42 | best | 0.761 | 0.804 | 0.757 | 0.7219 | 0.2655 | 0.7631 | 61.3% |
| RiskProp random | 44 | latest | 0.7769 | 0.8377 | 0.7732 | 0.7197 | 0.297 | 0.8011 | 63.3% |
| RiskProp fixed | 44 | latest | 0.748 | 0.827 | 0.7404 | 0.6765 | 0.2416 | 0.7419 | 62.0% |

## Appendix B. Sensitivity: all-best vs all-latest checkpoints (mAP mean +/- SD)

| Model | all-best | all-latest |
|---|---|---|
| TOP | 0.7007 +/- 0.0259 | 0.7513 +/- 0.0129 |
| AdaLEA | 0.6849 +/- 0.0659 | 0.7170 +/- 0.0203 |
| RiskProp random | 0.6396 +/- 0.0910 | 0.7453 +/- 0.0331 |
| RiskProp fixed | 0.6739 +/- 0.0309 | 0.7427 +/- 0.0046 |

## Appendix C. TOP with oracle matched head (uses horizon labels; NOT comparable)

| Model | mAP | AP@0.5s | AP@1.0s | AP@1.5s | mAUC01 | VideoAUC |
|---|---|---|---|---|---|---|
| TOP oracle head | 0.7180 +/- 0.0026 | 0.7899 +/- 0.0217 | 0.7093 +/- 0.0066 | 0.6549 +/- 0.0125 | 0.2001 +/- 0.0022 | 0.7718 +/- 0.0097 |

## 6. Paired bootstrap 95% CI (B=2000, stratified by label, same video resamples for every run; metric averaged over 3 seeds)

| comparison | metric | estimate | ci_low | ci_high | verdict |
|---|---|---|---|---|---|
| TOP | mAP | 0.7513 | 0.7016 | 0.8024 |  |
| AdaLEA | mAP | 0.7394 | 0.6907 | 0.792 |  |
| RiskProp random | mAP | 0.7453 | 0.6954 | 0.8009 |  |
| RiskProp fixed | mAP | 0.7427 | 0.6907 | 0.8004 |  |
| TOP - AdaLEA | mAP | 0.0119 | -0.0262 | 0.046 | inconclusive (CI includes 0) |
| TOP - RiskProp random | mAP | 0.0061 | -0.0397 | 0.0451 | inconclusive (CI includes 0) |
| AdaLEA - RiskProp random | mAP | -0.0059 | -0.0473 | 0.031 | inconclusive (CI includes 0) |
| RiskProp fixed - RiskProp random | mAP | -0.0026 | -0.0311 | 0.0249 | inconclusive (CI includes 0) |
| TOP | mAUC01 | 0.2228 | 0.1544 | 0.3132 |  |
| AdaLEA | mAUC01 | 0.2405 | 0.164 | 0.3337 |  |
| RiskProp random | mAUC01 | 0.2294 | 0.1531 | 0.3242 |  |
| RiskProp fixed | mAUC01 | 0.2242 | 0.146 | 0.3269 |  |
| TOP - AdaLEA | mAUC01 | -0.0177 | -0.0868 | 0.0545 | inconclusive (CI includes 0) |
| TOP - RiskProp random | mAUC01 | -0.0066 | -0.0964 | 0.0705 | inconclusive (CI includes 0) |
| AdaLEA - RiskProp random | mAUC01 | 0.0111 | -0.0612 | 0.0768 | inconclusive (CI includes 0) |
| RiskProp fixed - RiskProp random | mAUC01 | -0.0052 | -0.0643 | 0.0579 | inconclusive (CI includes 0) |

Bootstrap resamples validation videos only; seed-to-seed variation is reported separately as SD.

## 7. Locked configuration

```json
{
  "created": "2026-09-24 09:28:05",
  "legacy_check_passed": true,
  "top_head_rule": "head_2.0",
  "top_head_rule_criterion": "max mean val mAP over 3 seeds x {best, latest}; oracle matched head excluded",
  "checkpoint_rule": "per_run",
  "checkpoint_rule_criterion": "val mAP (mean AP over 0.5/1.0/1.5 s), candidates {best-val-loss, last-epoch}, tie -> best",
  "chosen_checkpoints": {
    "adalea_seed42": {
      "kind": "best",
      "file": "best_adalea_seed42.pth",
      "epoch": 7
    },
    "adalea_seed43": {
      "kind": "latest",
      "file": "latest_adalea_seed43.pth",
      "epoch": 50
    },
    "adalea_seed44": {
      "kind": "latest",
      "file": "latest_adalea_seed44.pth",
      "epoch": 50
    },
    "riskprop_fixed_seed42": {
      "kind": "latest",
      "file": "latest_riskprop_fixed_seed42.pth",
      "epoch": 50
    },
    "riskprop_fixed_seed43": {
      "kind": "latest",
      "file": "latest_riskprop_fixed_seed43.pth",
      "epoch": 50
    },
    "riskprop_fixed_seed44": {
      "kind": "latest",
      "file": "latest_riskprop_fixed_seed44.pth",
      "epoch": 50
    },
    "riskprop_random_seed42": {
      "kind": "latest",
      "file": "latest_riskprop_random_seed42.pth",
      "epoch": 50
    },
    "riskprop_random_seed43": {
      "kind": "latest",
      "file": "latest_riskprop_random_seed43.pth",
      "epoch": 50
    },
    "riskprop_random_seed44": {
      "kind": "latest",
      "file": "latest_riskprop_random_seed44.pth",
      "epoch": 50
    },
    "top_seed42": {
      "kind": "latest",
      "file": "latest_cached_seed42.pth",
      "epoch": 50
    },
    "top_seed43": {
      "kind": "latest",
      "file": "latest_cached_seed43.pth",
      "epoch": 50
    },
    "top_seed44": {
      "kind": "latest",
      "file": "latest_cached_seed44.pth",
      "epoch": 50
    }
  },
  "val_predictions_sha256": "fe4d578dcce70f78dc695556cc07e9aff72f5dc3b5ee2e9982d2b9f5ca067488",
  "note": "Created before opening solution.csv / time_to_accident_test_map.csv. The official test run must use exactly these rules."
}
```
