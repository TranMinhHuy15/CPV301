# Results

This chapter summarises the three research questions. The detailed reports, with every table, contrast and deviation, are:
- [`RESULTS_RQ1.md`](RESULTS_RQ1.md): TOP vs AdaLEA vs RiskProp
- [`RESULTS_RQ2.md`](RESULTS_RQ2.md): FFR × AMC ablation
- [`RESULTS_RQ3.md`](RESULTS_RQ3.md): fixed-lag vs random-offset AMC pairing, including the τ sensitivity runs

All numbers are mean ± SD over three training seeds (42/43/44) unless stated otherwise.

## 5.1 Evaluation protocol

**Correction before analysis.** An initial evaluation of TOP, AdaLEA and RiskProp revealed two methodological issues that would have biased the comparisons:
1. **TOP head selection.** The original TOP evaluation chose, for each clip, the output head matching the clip's known lead time (0.5/1.0/1.5 s). AdaLEA and RiskProp emit a single risk score per window and have no access to that information, so this "oracle" scoring is not comparable across methods.
2. **Checkpoint selection.** Every run saves a `best` checkpoint (lowest validation loss) and a `latest` checkpoint (final epoch). For 11 of the 12 original runs, validation mAP kept improving well after the epoch chosen by validation loss.

All 24 saved checkpoints (4 configurations × 3 seeds × {best, latest}) were re-evaluated on the original internal validation split: 300 videos, 150 positive and 150 negative, rebuilt from the original preprocessing code. The rebuilt split reproduced the previously reported metrics (max |Δ| = 0.0003 mAP). Two rules were then locked from validation data only, before any further analysis ([`reeval_corrected/locked_config.json`](reeval_corrected/locked_config.json)):
- **TOP head rule:** one fixed head, `head_2.0`, chosen by mean validation mAP; the oracle head was excluded from candidacy.
- **Checkpoint rule:** per run, keep whichever of {best, latest} has the higher mean validation AP over 0.5/1.0/1.5 s.

The same checkpoint rule was later applied unchanged to the RQ2 and RQ3 runs.

**Outcomes.**
- **Primary outcome for RQ1: mAP on the official Nexar test set.** The set has 1,344 clips and was scored once, after all rules were locked, with Nexar's unmodified `evaluate_submission.py`. A group-wise re-implementation reproduces the official public and private mAP of every run exactly and adds mAUC@0.1.
- **RQ2 and RQ3 are answered on the internal validation split.** Their temporal metrics need known event times: pairwise violation rate PVR (ε = 0.01), average downward step ADS and risk-curve jitter RCJ, computed on 30 causal windows per positive video. Official-test rows for these RQs are supplementary.

**Statistics.** Paired bootstrap 95% CIs (B = 2,000; seed 12345) apply the same resampled videos to every method:
- validation: label-stratified
- test: stratified within every solution group × label cell

A CI that contains 0 is reported as *inconclusive*.

Code: [`../reeval/`](../reeval/). Data: [`reeval_corrected/`](reeval_corrected/), [`official_test/`](official_test/), [`rq2/`](rq2/), [`rq3_sensitivity/`](rq3_sensitivity/), [`repro/`](repro/).

## 5.2 RQ1: TOP vs. AdaLEA vs. RiskProp

**Official test set (primary outcome)**

| Method | mAP public (official) | mAP private (official) | mAP all | mAUC@0.1 all |
|---|---|---|---|---|
| TOP (`head_2.0`) | **0.769 ± 0.016** | **0.811 ± 0.027** | **0.785 ± 0.021** | **0.273 ± 0.035** |
| AdaLEA | 0.700 ± 0.012 | 0.795 ± 0.019 | 0.742 ± 0.011 | 0.211 ± 0.014 |
| RiskProp (random-offset) | 0.756 ± 0.023 | **0.811 ± 0.004** | 0.781 ± 0.015 | 0.254 ± 0.037 |

| Comparison (whole test set) | Δ mAP [95% CI] | Δ mAUC@0.1 [95% CI] |
|---|---|---|
| TOP − AdaLEA | **+0.043 [0.025, 0.060]** | **+0.062 [0.026, 0.097]** |
| RiskProp − AdaLEA | **+0.039 [0.015, 0.060]** | **+0.044 [0.002, 0.085]** |
| TOP − RiskProp | +0.004 [−0.018, 0.028], inconclusive | +0.019 [−0.031, 0.068], inconclusive |

**Internal validation: early warning at FAR ≤ 0.1 (secondary outcome)**

| Method | mAP | mAUC@0.1 | Recall@1.0 s | mTTA, detected (s) | Coverage |
|---|---|---|---|---|---|
| TOP | 0.751 ± 0.013 | 0.223 ± 0.027 | 0.391 ± 0.019 | 1.086 ± 0.048 | 60.7 ± 4.4% |
| AdaLEA | 0.739 ± 0.019 | 0.240 ± 0.024 | 0.400 ± 0.024 | 1.208 ± 0.012 | 57.1 ± 7.9% |
| RiskProp (random-offset) | 0.745 ± 0.033 | 0.229 ± 0.073 | 0.367 ± 0.106 | 1.104 ± 0.079 | 56.4 ± 11.4% |

Every pairwise validation difference is inconclusive, for example TOP − AdaLEA mAP +0.012 [−0.026, 0.046].

**Answer.**
- **Official test:** TOP and RiskProp both outperform AdaLEA, by about 0.04 mAP. TOP has the slightly higher mean (0.785 vs 0.781), but the CI of TOP − RiskProp includes zero, so the two cannot be separated.
- **Single best runs:** these are reported descriptively only (`RESULTS_RQ1.md`, Appendix A). RiskProp seed 44 has the highest validation mAP of any single run, but on the official test it is RiskProp's weakest seed, and TOP seed 42 scores higher (0.809 vs 0.768).
- **Validation:** the ordering is the same, but no difference is significant. The official test has about 4.5× more clips and was never used for selection.
- **Early warning:** AdaLEA warns earliest among detected positives, but the early-warning metrics do not separate the methods on validation.
- **Agreement with the paper:** the RiskProp − AdaLEA gap matches the RiskProp paper's Table 1 almost exactly (+0.039 vs +0.038 mAP). Our absolute scores are lower (0.781 vs 0.870), as expected from a much smaller training setup.

**Conclusion for RQ1.** No single method is best. TOP has the highest mean official-test mAP, but it does not significantly outperform RiskProp. TOP and RiskProp perform comparably, and both significantly outperform AdaLEA.

**From RQ1 to RQ2 and RQ3.** RiskProp does not significantly outperform TOP, but it performs comparably and significantly outperforms AdaLEA. It is the published method that this project reproduces. Unlike TOP and AdaLEA, which are trained here on independent windows, RiskProp also shapes how the risk score evolves over time through two dedicated losses, FFR and AMC. This makes it a competitive basis for the component ablation in RQ2 and the fixed-temporal-lag study in RQ3. Both follow-up questions were planned in the proposal and do not depend on RiskProp ranking first in RQ1.

## 5.3 RQ2: Contribution of FFR and random-pair AMC

2×2 ablation inside RiskProp on internal validation. D is the RQ1 RiskProp model.

| Condition | mAP | mAUC@0.1 | PVR ↓ | ADS ↓ | RCJ ↓ |
|---|---|---|---|---|---|
| A: BCE only | 0.641 ± 0.063 | 0.150 ± 0.080 | 0.183 ± 0.069 | 0.021 ± 0.006 | 0.079 ± 0.029 |
| B: + FFR | 0.743 ± 0.016 | 0.220 ± 0.039 | 0.216 ± 0.012 | 0.034 ± 0.001 | 0.124 ± 0.005 |
| C: + AMC | 0.654 ± 0.023 | 0.164 ± 0.030 | 0.129 ± 0.089 | 0.013 ± 0.002 | 0.052 ± 0.015 |
| D: + FFR + AMC | 0.745 ± 0.033 | 0.229 ± 0.073 | 0.199 ± 0.009 | 0.032 ± 0.002 | 0.121 ± 0.008 |

| Contrast | Δ mAP [95% CI] | Δ mAUC@0.1 [95% CI] | Δ PVR [95% CI] |
|---|---|---|---|
| FFR, AMC off (B − A) | **+0.103 [0.062, 0.144]** | **+0.070 [0.008, 0.143]** | +0.033 [0.010, 0.056] (worse) |
| FFR, AMC on (D − C) | **+0.092 [0.054, 0.130]** | **+0.065 [0.005, 0.135]** | +0.070 [0.047, 0.093] (worse) |
| AMC, FFR off (C − A) | +0.013 [−0.021, 0.044], inconclusive | +0.014 [−0.025, 0.051], inconclusive | **−0.053 [−0.070, −0.036] (better)** |
| AMC, FFR on (D − B) | +0.002 [−0.026, 0.028], inconclusive | +0.010 [−0.045, 0.068], inconclusive | **−0.017 [−0.031, −0.003] (better)** |
| Interaction | −0.011 [−0.050, 0.031], inconclusive | −0.005 [−0.067, 0.059], inconclusive | +0.037 [0.016, 0.059] |

**Answer.**
- **FFR provides essentially all of RiskProp's accuracy gain:** about +0.10 mAP and +0.07 mAUC@0.1, with or without AMC.
- **AMC does not change accuracy.** It makes risk curves more monotone (lower PVR), mostly when FFR is absent. The positive interaction on PVR means AMC's temporal benefit shrinks once FFR is present.
- **FFR's higher absolute PVR/ADS/RCJ is partly an amplitude effect.** Models without FFR produce lower, often flat curves. With ε = 0, every condition with FFR or AMC has fewer violations than A (`RESULTS_RQ2.md` §6).
- **The official test shows the same pattern** (supplementary): FFR +0.12 / +0.10 mAP, and AMC has no accuracy effect.
- **Agreement with the paper:** the result reproduces the direction of the RiskProp paper's Table 3 for FFR (+0.073 / +0.085 mAP in the paper). The paper's small AMC gain (+0.016 mAP on top of FFR) lies within our uncertainty and could not be confirmed.

## 5.4 RQ3: Fixed-lag vs. random-offset AMC pairing

Only the way AMC pairs are drawn differs. Random-offset uses gaps of 0.5–5.0 s (mean 2.75 s); FixedLag uses a constant gap τ. τ = 1.0 s is the main condition and was trained together with the RQ1 models (logs in [`riskprop/`](riskprop/)). τ = 0.5 / 1.5 s are secondary sensitivity runs, trained afterwards (6 new runs).

| Condition | Val mAP | Val PVR ↓ | Val RCJ ↓ | Official test mAP (all) |
|---|---|---|---|---|
| Random-offset | 0.745 ± 0.033 | 0.199 ± 0.009 | 0.121 ± 0.008 | 0.781 ± 0.015 |
| FixedLag τ = 0.5 s | 0.728 ± 0.015 | 0.174 ± 0.017 | 0.121 ± 0.008 | not scored |
| **FixedLag τ = 1.0 s (main)** | 0.743 ± 0.005 | 0.182 ± 0.012 | 0.135 ± 0.011 | 0.772 ± 0.013 |
| FixedLag τ = 1.5 s | 0.750 ± 0.030 | 0.191 ± 0.001 | 0.122 ± 0.014 | not scored |

| Contrast vs random-offset | Δ mAP [95% CI] | Δ PVR [95% CI] | Δ RCJ [95% CI] |
|---|---|---|---|
| τ = 0.5 s | −0.0173 [−0.0459, 0.0114] | **−0.0256 [−0.0426, −0.0091]** | −0.0002 [−0.0113, 0.0105] |
| τ = 1.0 s (validation) | −0.0026 [−0.0311, 0.0249] | **−0.0173 [−0.0337, −0.0013]** | +0.0136 [0.0021, 0.0245] (worse) |
| τ = 1.5 s | +0.0051 [−0.0205, 0.0320] | −0.0080 [−0.0248, 0.0081] | +0.0007 [−0.0098, 0.0111] |
| τ = 1.0 s (official test) | −0.0087 [−0.0221, 0.0060] | not applicable | not applicable |

**Answer.**
- **Accuracy and early warning are unchanged**, on validation and on the official test.
- **Fixed-lag pairing slightly reduces large pairwise violations, more so with shorter lags.** PVR: τ 0.5 < 1.0 < 1.5 < random. The trend holds for ε = 0.01 only, and differences between adjacent lags are inconclusive.
- **τ = 1.0 s also increases jitter, but τ = 0.5 / 1.5 s do not.** This most likely reflects those three seeds rather than a mechanism.
- **Our earlier interim finding that fixed-lag training reduces seed-to-seed variance is withdrawn.** It rested on τ = 1.0 s alone (SD 0.005 vs 0.033). At τ = 1.5 s the SD is 0.030, and on the official test τ = 1.0 s and random have similar SDs (0.013 vs 0.015).
- **Overall:** the proposal's RQ3 success criterion (an improvement in temporal violations *or* seed variance) is met only weakly, through PVR.

## 5.5 Summary

1. **Method comparison (RQ1).** On the official Nexar test set, TOP and RiskProp both beat AdaLEA by about 0.04 mAP, and the two cannot be separated. The 300-video validation split with three seeds cannot resolve differences of this size, which is why the official test was used as the primary outcome.
2. **What makes RiskProp work (RQ2).** RiskProp's accuracy comes from future-frame regularization (FFR). The adaptive monotonic constraint (AMC) affects the shape of the risk curve (fewer violations), not ranking accuracy, and that effect is small once FFR is present.
3. **How AMC pairs are sampled (RQ3).** Replacing random offsets with a fixed lag does not affect accuracy. It gives a small, metric-dependent temporal benefit and no reliable reduction in seed variance.
4. **Accuracy and temporal consistency are separate outcomes.** Components that raise mAP (FFR) can raise absolute violation counts, and components that improve monotonicity (AMC, fixed lag) leave mAP unchanged. Both need to be reported when collision-anticipation models are compared.

This summary replaces the interim §5.4 of the earlier draft ("none of the RQ1 or RQ3 comparisons reach significance … fixed-lag's main benefit is variance reduction"). That draft predated the official test run and the RQ3 sensitivity runs.

## 5.6 Limitations

- **Three seeds per condition.** Bootstrap CIs resample videos, not seeds, and seed SDs from three runs are imprecise.
- **Smaller training setup than the papers.** Batch 2 videos on one GPU, λ1 = λ2 = 0.5, and 12-snippet sequences instead of frame-level ones. Absolute scores are therefore lower than published, and only effect directions are compared with the RiskProp paper.
- **Only two checkpoints saved per run.** The per-run rule therefore picks the better of {lowest validation loss, final epoch}, not the best of all 50 epochs. The final epoch was chosen for 11 of 12 RQ1/RQ3 runs, 7 of 9 new RQ2 runs and all 6 sensitivity runs.
- **Temporal metrics on validation only.** Test clips hide the event time, so dense risk curves cannot be built for them. The ε = 0.01 threshold was fixed before these metrics were computed; the curve-amplitude analyses are post hoc.
- **Multiple comparisons.** Many contrasts are reported without multiplicity correction. CIs whose bound lies within about 0.002 of zero should be read as weak evidence.
- **Different training sessions.** Runs were trained in different sessions (the RQ2 and RQ3 sensitivity runs on one RTX 4080), with identical code and hyperparameters.

**Compute** (training time from the logs):

| Experiment | Runs | GPU-hours |
|---|---|---|
| TOP | 3 | 0.8 |
| AdaLEA | 3 | 1.0 |
| RiskProp random-offset + fixed τ = 1.0 s | 6 | 10.7 |
| RQ2 ablation (A, B, C) | 9 | 11.4 |
| RQ3 sensitivity (τ = 0.5 / 1.5 s) | 6 | 7.7 |
| **Total** | **27** | **31.5** |

Official-test inference for all 21 runs took about 5 minutes on one RTX 4080.

## Ghi chú nội bộ để họp nhóm (xóa mục này trước khi nộp)

1. **File này thay hẳn bản `RESULTS.md` cũ.** Bản cũ có §5.4 ghi "không có khác biệt nào có ý nghĩa" và "fixed-lag chủ yếu giảm phương sai", nay đã sai sau khi có official test và các run sensitivity. Nhóm đồng ý bỏ bản cũ không?
2. **Kết luận tổng (mục 5.5).** Có 4 ý: TOP ≈ RiskProp > AdaLEA; FFR là thành phần chính; fixed-lag không đổi độ chính xác; accuracy và tính nhất quán theo thời gian là hai trục khác nhau. Ý 4 có nên đưa vào Discussion/Conclusion không?
3. **Các file chi tiết `RESULTS_RQ1/2/3.md`.** Giữ trong repo làm phụ lục, còn bài báo chỉ dùng bảng rút gọn như file này. Hay gộp hết vào một file?
4. **Đã sửa lỗi làm tròn** (lệch 0,001) trong RQ1 và RQ2, theo số tính trực tiếp từ dữ liệu từng run. CI giữ đúng như file CSV (4 chữ số, làm tròn một lần). Bảng RQ3 để CI 4 chữ số vì nhiều cận sát 0. Nhóm muốn thống nhất 3 hay 4 chữ số cho CI trong bài?
5. **Trước khi nộp:** xóa mục ghi chú nội bộ ở cả 4 file (RESULTS, RQ1 §7, RQ2 §10, RQ3 §9).
6. **Kết luận RQ1 và đoạn chuyển sang RQ2/RQ3 (§5.2, đã thống nhất với partner ngày 25/9).** Không có model tốt nhất tuyệt đối: TOP có mean cao nhất nhưng không hơn RiskProp có ý nghĩa thống kê. Lưu ý khi viết bài: RiskProp là phương pháp đã công bố mà nhóm tái lập, không phải "proposed method" của nhóm. Biến thể nhóm đề xuất là FixedLag-RiskProp (RQ3).
7. **Chi phí tiền thuê máy:** bổ sung số credit vast.ai đã dùng vào mục 5.6 nếu thầy yêu cầu báo cáo chi phí.
