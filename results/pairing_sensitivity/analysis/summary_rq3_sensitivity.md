# RQ3 sensitivity -- FixedLag tau 0.5 / 1.0 / 1.5 s (internal validation)

Generated 2026-09-25 09:59 | 300 videos (150 pos); temporal metrics on 150 positives; EPS=0.01. Checkpoints: locked per_run rule (tau 1.0 and random reuse locked_config). Secondary analysis (proposal Sec. 5.3).

## 1. Accuracy and low-FAR (mean +/- SD over 3 seeds)

| Condition | n | mAP | AP@0.5s | AP@1.0s | AP@1.5s | mAUC01 | Recall@1.0s | mTTA_detected | Coverage |
|---|---|---|---|---|---|---|---|---|---|
| FixedLag tau=0.5s | 3 | 0.7280 +/- 0.0153 | 0.8040 +/- 0.0228 | 0.7184 +/- 0.0157 | 0.6616 +/- 0.0174 | 0.2106 +/- 0.0240 | 0.3600 +/- 0.0570 | 1.0935 +/- 0.0551 | 0.5644 +/- 0.0518 |
| FixedLag tau=1.0s (RQ3 main) | 3 | 0.7427 +/- 0.0046 | 0.8169 +/- 0.0093 | 0.7392 +/- 0.0010 | 0.6720 +/- 0.0064 | 0.2242 +/- 0.0181 | 0.3756 +/- 0.0102 | 1.0785 +/- 0.0304 | 0.6067 +/- 0.0133 |
| FixedLag tau=1.5s | 3 | 0.7504 +/- 0.0303 | 0.8246 +/- 0.0189 | 0.7474 +/- 0.0334 | 0.6790 +/- 0.0396 | 0.2598 +/- 0.0428 | 0.4200 +/- 0.0800 | 1.0899 +/- 0.0790 | 0.6467 +/- 0.0643 |
| RiskProp random-offset | 3 | 0.7453 +/- 0.0331 | 0.8171 +/- 0.0218 | 0.7390 +/- 0.0379 | 0.6797 +/- 0.0399 | 0.2294 +/- 0.0734 | 0.3667 +/- 0.1058 | 1.1038 +/- 0.0793 | 0.5644 +/- 0.1136 |

## 2. Temporal (dense curves, positives; lower = better)

| Condition | PVR | ADS | RCJ | PVR_eps0 |
|---|---|---|---|---|
| FixedLag tau=0.5s | 0.1735 +/- 0.0170 | 0.0313 +/- 0.0018 | 0.1207 +/- 0.0081 | 0.3198 +/- 0.0014 |
| FixedLag tau=1.0s (RQ3 main) | 0.1818 +/- 0.0121 | 0.0345 +/- 0.0033 | 0.1345 +/- 0.0108 | 0.3064 +/- 0.0058 |
| FixedLag tau=1.5s | 0.1911 +/- 0.0008 | 0.0324 +/- 0.0033 | 0.1216 +/- 0.0140 | 0.3113 +/- 0.0114 |
| RiskProp random-offset | 0.1991 +/- 0.0090 | 0.0323 +/- 0.0021 | 0.1209 +/- 0.0083 | 0.3175 +/- 0.0057 |

## 3. Contrasts (paired stratified bootstrap 95% CI, B=2000)

| contrast | metric | estimate | ci_low | ci_high | verdict | direction |
|---|---|---|---|---|---|---|
| tau0.5 - tau1.0 | mAP | -0.0147 | -0.0475 | 0.0163 | inconclusive (CI includes 0) |  |
| tau0.5 - tau1.0 | mAUC01 | -0.0137 | -0.0832 | 0.0516 | inconclusive (CI includes 0) |  |
| tau0.5 - tau1.0 | PVR | -0.0083 | -0.0237 | 0.0072 | inconclusive (CI includes 0) |  |
| tau0.5 - tau1.0 | ADS | -0.0032 | -0.0062 | -0.0 | decrease (CI < 0) | better |
| tau0.5 - tau1.0 | RCJ | -0.0139 | -0.0239 | -0.0036 | decrease (CI < 0) | better |
| tau1.5 - tau1.0 | mAP | 0.0076 | -0.0207 | 0.0353 | inconclusive (CI includes 0) |  |
| tau1.5 - tau1.0 | mAUC01 | 0.0356 | -0.0285 | 0.0923 | inconclusive (CI includes 0) |  |
| tau1.5 - tau1.0 | PVR | 0.0093 | -0.006 | 0.025 | inconclusive (CI includes 0) |  |
| tau1.5 - tau1.0 | ADS | -0.0021 | -0.0051 | 0.0011 | inconclusive (CI includes 0) |  |
| tau1.5 - tau1.0 | RCJ | -0.013 | -0.0236 | -0.0028 | decrease (CI < 0) | better |
| tau0.5 - random | mAP | -0.0173 | -0.0459 | 0.0114 | inconclusive (CI includes 0) |  |
| tau0.5 - random | mAUC01 | -0.0189 | -0.077 | 0.0449 | inconclusive (CI includes 0) |  |
| tau0.5 - random | PVR | -0.0256 | -0.0426 | -0.0091 | decrease (CI < 0) | better |
| tau0.5 - random | ADS | -0.001 | -0.0042 | 0.0022 | inconclusive (CI includes 0) |  |
| tau0.5 - random | RCJ | -0.0002 | -0.0113 | 0.0105 | inconclusive (CI includes 0) |  |
| tau1.0 - random | mAP | -0.0026 | -0.0311 | 0.0249 | inconclusive (CI includes 0) |  |
| tau1.0 - random | mAUC01 | -0.0052 | -0.0643 | 0.0579 | inconclusive (CI includes 0) |  |
| tau1.0 - random | PVR | -0.0173 | -0.0337 | -0.0013 | decrease (CI < 0) | better |
| tau1.0 - random | ADS | 0.0022 | -0.0012 | 0.0053 | inconclusive (CI includes 0) |  |
| tau1.0 - random | RCJ | 0.0136 | 0.0021 | 0.0245 | increase (CI > 0) | worse |
| tau1.5 - random | mAP | 0.0051 | -0.0205 | 0.032 | inconclusive (CI includes 0) |  |
| tau1.5 - random | mAUC01 | 0.0304 | -0.0211 | 0.0821 | inconclusive (CI includes 0) |  |
| tau1.5 - random | PVR | -0.008 | -0.0248 | 0.0081 | inconclusive (CI includes 0) |  |
| tau1.5 - random | ADS | 0.0001 | -0.003 | 0.0031 | inconclusive (CI includes 0) |  |
| tau1.5 - random | RCJ | 0.0007 | -0.0098 | 0.0111 | inconclusive (CI includes 0) |  |
