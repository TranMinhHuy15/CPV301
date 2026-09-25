# RQ2 -- FFR x AMC ablation (internal validation split)

Generated 2026-09-25 01:15 | 300 videos (150 pos / 150 neg); temporal metrics on 150 positives with a known event time. Checkpoint per run: locked per_run rule (see rq2_chosen_checkpoints.json). EPS=0.01 (locked before scoring).

## 0. Training sanity check (loss terms actually on/off)

Expected: A reg=0 mono=0; B reg>0 mono=0; C reg=0 mono>0.

| run | epochs | mean_train_reg | mean_train_mono | best_val_loss | best_epoch | last_val_mAP |
|---|---|---|---|---|---|---|
| riskprop_random_seed42_neither | 50 | 0.0 | 0.0 | 4.0241 | 3 | 0.634 |
| riskprop_random_seed43_neither | 50 | 0.0 | 0.0 | 4.2879 | 1 | 0.6842 |
| riskprop_random_seed44_neither | 50 | 0.0 | 0.0 | 3.3483 | 0 | 0.553 |
| riskprop_random_seed42_ffronly | 50 | 0.05482 | 0.0 | 3.2449 | 0 | 0.7126 |
| riskprop_random_seed43_ffronly | 50 | 0.05346 | 0.0 | 3.3734 | 0 | 0.7433 |
| riskprop_random_seed44_ffronly | 50 | 0.05325 | 0.0 | 4.7562 | 1 | 0.7592 |
| riskprop_random_seed42_amconly | 50 | 0.0 | 0.00996 | 3.9758 | 0 | 0.6251 |
| riskprop_random_seed43_amconly | 50 | 0.0 | 0.01094 | 3.5307 | 4 | 0.6758 |
| riskprop_random_seed44_amconly | 50 | 0.0 | 0.01103 | 4.6571 | 1 | 0.5864 |

## 1. Accuracy (mean +/- SD over 3 seeds; mAP is the primary metric)

| Condition | n_seeds | mAP | AP@0.5s | AP@1.0s | AP@1.5s |
|---|---|---|---|---|---|
| A neither | 3 | 0.6407 +/- 0.0633 | 0.7124 +/- 0.0920 | 0.6224 +/- 0.0688 | 0.5872 +/- 0.0325 |
| B FFR-only | 3 | 0.7432 +/- 0.0163 | 0.8119 +/- 0.0241 | 0.7344 +/- 0.0197 | 0.6833 +/- 0.0089 |
| C AMC-only (random) | 3 | 0.6538 +/- 0.0234 | 0.7669 +/- 0.0189 | 0.6321 +/- 0.0453 | 0.5624 +/- 0.0183 |
| D FFR+AMC (RiskProp random) | 3 | 0.7453 +/- 0.0331 | 0.8171 +/- 0.0218 | 0.7390 +/- 0.0379 | 0.6797 +/- 0.0399 |
| F FFR+AMC fixed-lag (RQ3) | 3 | 0.7427 +/- 0.0046 | 0.8169 +/- 0.0093 | 0.7392 +/- 0.0010 | 0.6720 +/- 0.0064 |

## 2. Low-FAR anticipation (FAR <= 0.1)

| Condition | mAUC01 | Recall@0.5s | Recall@1.0s | Recall@1.5s | ActualFAR@1.0s | mTTA_detected | mTTA_all | Coverage |
|---|---|---|---|---|---|---|---|---|
| A neither | 0.1502 +/- 0.0804 | 0.3800 +/- 0.1568 | 0.2422 +/- 0.1110 | 0.1667 +/- 0.0636 | 0.1000 +/- 0.0000 | 0.9911 +/- 0.0746 | 0.4444 +/- 0.1760 | 0.4511 +/- 0.1772 |
| B FFR-only | 0.2198 +/- 0.0394 | 0.5178 +/- 0.0582 | 0.3956 +/- 0.0454 | 0.2933 +/- 0.0240 | 0.1000 +/- 0.0000 | 1.1208 +/- 0.0025 | 0.6700 +/- 0.0603 | 0.5978 +/- 0.0535 |
| C AMC-only (random) | 0.1643 +/- 0.0300 | 0.4000 +/- 0.0751 | 0.2222 +/- 0.0758 | 0.1489 +/- 0.0278 | 0.0956 +/- 0.0077 | 0.9483 +/- 0.0136 | 0.4433 +/- 0.0984 | 0.4667 +/- 0.0987 |
| D FFR+AMC (RiskProp random) | 0.2294 +/- 0.0734 | 0.4911 +/- 0.1128 | 0.3667 +/- 0.1058 | 0.2733 +/- 0.0987 | 0.0978 +/- 0.0038 | 1.1038 +/- 0.0793 | 0.6289 +/- 0.1644 | 0.5644 +/- 0.1136 |
| F FFR+AMC fixed-lag (RQ3) | 0.2242 +/- 0.0181 | 0.5222 +/- 0.0077 | 0.3756 +/- 0.0102 | 0.2822 +/- 0.0252 | 0.1000 +/- 0.0000 | 1.0785 +/- 0.0304 | 0.6544 +/- 0.0299 | 0.6067 +/- 0.0133 |

## 3. Temporal monotonicity (dense curves; lower = smoother / more monotone)

viol/down/jitter_pos: positives; *_neg: negatives (false-alarm behaviour); viol_pos_eps0 = sensitivity with EPS=0. Row F is for RQ3 only.

| Condition | viol_pos | down_pos | jitter_pos | viol_pos_eps0 | jitter_neg | meanscore_neg |
|---|---|---|---|---|---|---|
| A neither | 0.1826 +/- 0.0689 | 0.0210 +/- 0.0057 | 0.0786 +/- 0.0295 | 0.3534 +/- 0.0403 | 0.0471 +/- 0.0098 | 0.1680 +/- 0.1393 |
| B FFR-only | 0.2157 +/- 0.0118 | 0.0342 +/- 0.0014 | 0.1245 +/- 0.0053 | 0.3191 +/- 0.0088 | 0.0878 +/- 0.0160 | 0.1493 +/- 0.0317 |
| C AMC-only (random) | 0.1292 +/- 0.0885 | 0.0127 +/- 0.0017 | 0.0523 +/- 0.0154 | 0.3220 +/- 0.0302 | 0.0198 +/- 0.0091 | 0.1190 +/- 0.1787 |
| D FFR+AMC (RiskProp random) | 0.1991 +/- 0.0090 | 0.0323 +/- 0.0021 | 0.1209 +/- 0.0083 | 0.3175 +/- 0.0057 | 0.0851 +/- 0.0081 | 0.1406 +/- 0.0229 |
| F FFR+AMC fixed-lag (RQ3) | 0.1818 +/- 0.0121 | 0.0345 +/- 0.0033 | 0.1345 +/- 0.0108 | 0.3064 +/- 0.0058 | 0.0724 +/- 0.0091 | 0.1097 +/- 0.0169 |

## 4. Run-to-run stability

The +/- values above are the SD over seeds 42/43/44 (Group 4).

## 5. Main effects and interaction (paired stratified bootstrap 95% CI, B=2000)

`direction` says whether a conclusive change is better or worse for that metric (higher mAP/mAUC is better; lower violation/downward step/jitter is better).

| contrast | metric | estimate | ci_low | ci_high | verdict | direction |
|---|---|---|---|---|---|---|
| FFR effect, AMC absent (B-A) | mAP | 0.1025 | 0.0621 | 0.1443 | increase (CI > 0) | better |
| FFR effect, AMC absent (B-A) | mAUC01 | 0.0696 | 0.0076 | 0.1427 | increase (CI > 0) | better |
| FFR effect, AMC absent (B-A) | viol_pos | 0.0331 | 0.0098 | 0.0557 | increase (CI > 0) | worse |
| FFR effect, AMC absent (B-A) | down_pos | 0.0132 | 0.0093 | 0.0172 | increase (CI > 0) | worse |
| FFR effect, AMC absent (B-A) | jitter_pos | 0.0459 | 0.0335 | 0.0595 | increase (CI > 0) | worse |
| AMC effect, FFR absent (C-A) | mAP | 0.0131 | -0.0211 | 0.0444 | inconclusive (CI includes 0) |  |
| AMC effect, FFR absent (C-A) | mAUC01 | 0.0142 | -0.0248 | 0.0508 | inconclusive (CI includes 0) |  |
| AMC effect, FFR absent (C-A) | viol_pos | -0.0534 | -0.0704 | -0.0355 | decrease (CI < 0) | better |
| AMC effect, FFR absent (C-A) | down_pos | -0.0084 | -0.0109 | -0.0057 | decrease (CI < 0) | better |
| AMC effect, FFR absent (C-A) | jitter_pos | -0.0263 | -0.0343 | -0.0176 | decrease (CI < 0) | better |
| FFR effect, AMC present (D-C) | mAP | 0.0915 | 0.0544 | 0.13 | increase (CI > 0) | better |
| FFR effect, AMC present (D-C) | mAUC01 | 0.0651 | 0.0046 | 0.1345 | increase (CI > 0) | better |
| FFR effect, AMC present (D-C) | viol_pos | 0.07 | 0.0467 | 0.0925 | increase (CI > 0) | worse |
| FFR effect, AMC present (D-C) | down_pos | 0.0197 | 0.016 | 0.0234 | increase (CI > 0) | worse |
| FFR effect, AMC present (D-C) | jitter_pos | 0.0686 | 0.0564 | 0.0808 | increase (CI > 0) | worse |
| AMC effect, FFR present (D-B) | mAP | 0.0021 | -0.0256 | 0.028 | inconclusive (CI includes 0) |  |
| AMC effect, FFR present (D-B) | mAUC01 | 0.0097 | -0.0446 | 0.0675 | inconclusive (CI includes 0) |  |
| AMC effect, FFR present (D-B) | viol_pos | -0.0165 | -0.0305 | -0.0027 | decrease (CI < 0) | better |
| AMC effect, FFR present (D-B) | down_pos | -0.0019 | -0.0048 | 0.001 | inconclusive (CI includes 0) |  |
| AMC effect, FFR present (D-B) | jitter_pos | -0.0035 | -0.0128 | 0.0061 | inconclusive (CI includes 0) |  |
| Interaction (D-C)-(B-A) | mAP | -0.0111 | -0.0501 | 0.0309 | inconclusive (CI includes 0) |  |
| Interaction (D-C)-(B-A) | mAUC01 | -0.0045 | -0.067 | 0.0591 | inconclusive (CI includes 0) |  |
| Interaction (D-C)-(B-A) | viol_pos | 0.0369 | 0.0155 | 0.0585 | increase (CI > 0) |  |
| Interaction (D-C)-(B-A) | down_pos | 0.0064 | 0.0026 | 0.0101 | increase (CI > 0) |  |
| Interaction (D-C)-(B-A) | jitter_pos | 0.0227 | 0.0104 | 0.0348 | increase (CI > 0) |  |

## 6. RQ3 add-on: temporal metrics, fixed-lag (F) vs random (D)

| contrast | metric | estimate | ci_low | ci_high | verdict | direction |
|---|---|---|---|---|---|---|
| RQ3 temporal: fixed - random (F-D) | viol_pos | -0.0173 | -0.0337 | -0.0013 | decrease (CI < 0) | better |
| RQ3 temporal: fixed - random (F-D) | down_pos | 0.0022 | -0.0012 | 0.0053 | inconclusive (CI includes 0) |  |
| RQ3 temporal: fixed - random (F-D) | jitter_pos | 0.0136 | 0.0021 | 0.0245 | increase (CI > 0) | worse |

## 7. Comparison with the RiskProp paper (Table 3, Nexar, Only Collision Label)

Paper: official Nexar test set, single run, lambda1=1.5 / lambda2=1.1, LR 0.002, batch 64, frame-level sequences. Ours: internal validation split, 3 seeds, lambda1=lambda2=0.5, LR 0.01, 12-snippet sequences (see RQ2_THEORY.md deviations). Absolute levels are therefore NOT comparable; only the direction of each effect is. mTTA is omitted because the paper's mTTA^0.1 is defined differently from ours.

| Condition | paper Exp. | paper mAP (test) | ours mAP (val) | paper mAUC0.1 (test) | ours mAUC0.1 (val) |
|---|---|---|---|---|---|
| A neither | I | 0.781 | 0.6407 +/- 0.0633 | 0.298 | 0.1502 +/- 0.0804 |
| B FFR-only | III | 0.854 | 0.7432 +/- 0.0163 | 0.453 | 0.2198 +/- 0.0394 |
| C AMC-only (random) | II | 0.785 | 0.6538 +/- 0.0234 | 0.302 | 0.1643 +/- 0.0300 |
| D FFR+AMC (RiskProp random) | IV | 0.87 | 0.7453 +/- 0.0331 | 0.472 | 0.2294 +/- 0.0734 |

| contrast | metric | paper_delta | ours_delta | ours_95CI | agreement |
|---|---|---|---|---|---|
| FFR effect, AMC absent (B-A) | mAP | 0.073 | 0.1025 | [0.0621, 0.1443] | agrees (same sign) |
| FFR effect, AMC absent (B-A) | mAUC01 | 0.155 | 0.0696 | [0.0076, 0.1427] | agrees (same sign) |
| AMC effect, FFR absent (C-A) | mAP | 0.004 | 0.0131 | [-0.0211, 0.0444] | not confirmed (our CI includes 0) |
| AMC effect, FFR absent (C-A) | mAUC01 | 0.004 | 0.0142 | [-0.0248, 0.0508] | not confirmed (our CI includes 0) |
| FFR effect, AMC present (D-C) | mAP | 0.085 | 0.0915 | [0.0544, 0.13] | agrees (same sign) |
| FFR effect, AMC present (D-C) | mAUC01 | 0.17 | 0.0651 | [0.0046, 0.1345] | agrees (same sign) |
| AMC effect, FFR present (D-B) | mAP | 0.016 | 0.0021 | [-0.0256, 0.028] | not confirmed (our CI includes 0) |
| AMC effect, FFR present (D-B) | mAUC01 | 0.019 | 0.0097 | [-0.0446, 0.0675] | not confirmed (our CI includes 0) |
| Interaction (D-C)-(B-A) | mAP | 0.012 | -0.0111 | [-0.0501, 0.0309] | not confirmed (our CI includes 0) |
| Interaction (D-C)-(B-A) | mAUC01 | 0.015 | -0.0045 | [-0.067, 0.0591] | not confirmed (our CI includes 0) |
