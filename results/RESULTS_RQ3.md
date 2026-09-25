# Results — RQ3: Fixed-lag vs. random-offset AMC pairing in RiskProp

> RQ3: Does training RiskProp's adaptive monotonic constraint (AMC) on pairs separated by a fixed lag (FixedLag-RiskProp, τ = 1.0 s), instead of the published random offsets, change accuracy, temporal consistency of the risk curve, or run-to-run stability?

Which runs are new: the random-offset and FixedLag τ = 1.0 s runs (3 seeds each) were trained together with the RQ1 models, and their training logs are in `results/riskprop/`. Only the six τ = 0.5 / 1.5 s sensitivity runs were trained for this analysis.

Sources:
- Main comparison on validation: `results/reeval_corrected/`, plus the temporal add-on in `results/rq2/analysis/summary_rq2.md` §6.
- Official test: `results/official_test/`.
- τ sensitivity: `results/rq3_sensitivity/`, produced by `reeval/re09_train_rq3_sensitivity.py` and `reeval/re10_rq3_sensitivity_eval.py`.

The RQ3 conclusions come from the **internal validation split**, because the temporal metrics need the known event time. The official-test rows in Section 3 are supplementary.

## 1. Protocol

**Conditions.** Everything except the way AMC pairs are drawn is identical to full RiskProp (FFR and AMC both on).

| Condition | AMC pair gap (snippet steps of 0.5 s) | Gap in seconds | Role | Runs |
|---|---|---|---|---|
| Random-offset | round(d·(N−1)), d ~ U(0.1, 0.9) → 1–10 steps | 0.5–5.0 s (mean 2.75 s) | baseline; the published objective, the RQ1 anchor and RQ2 condition D | 3 (seeds 42/43/44), trained with RQ1 |
| FixedLag τ = 1.0 s | 2 | 1.0 s | **main RQ3 condition** ("RiskProp fixed") | 3, trained with RQ1 |
| FixedLag τ = 0.5 s | 1 | 0.5 s | secondary sensitivity | 3 new |
| FixedLag τ = 1.5 s | 3 | 1.5 s | secondary sensitivity | 3 new |

The proposal states that "secondary sensitivity runs use τ = 0.5 and 1.5 seconds and will not be promoted as separate contributions". Section 4 is reported in that spirit.

**Controls.** All conditions share:
- the same train/validation split (1,200 / 300 videos, 150 positive + 150 negative in validation; `results/repro/split_manifest_seed42.json`)
- SlowOnly-R50 on 12 causal snippets of 5 frames at 10 fps, dt = 0.5 s
- SGD, LR 0.01 with the same step schedule, 50 epochs, batch 2 videos
- λ1 = λ2 = 0.5, W_coll = 8.0, M = 8 AMC pairs per positive video, margin δ = 0.01 · Δt · c̄ (so the required rise grows with the lag)

The random-offset and τ = 1.0 s runs come from `cell34_train_riskprop_cached.py`, with `RISKPROP_PAIRING=random` and `fixed` respectively. The τ = 0.5 / 1.5 s runs come from `reeval/re09_train_rq3_sensitivity.py`. That script is a copy of `cell34` that only passes `fixed_gap = round(τ / dt)` to `cell32.riskprop_loss`; `cell32` and `cell34` are unchanged.

**Training sanity checks.**

| Condition | First-epoch L_AMC | L_AMC, mean of last 10 epochs | Epoch of lowest val loss | Final-epoch val loss | s / epoch |
|---|---|---|---|---|---|
| Random-offset | 0.032–0.037 | 0.004 | 1–2 | 8.9–9.3 | 129–148 |
| τ = 0.5 s | 0.026–0.030 | 0.005–0.006 | 1–2 | 10.3–18.4 | 92–93 |
| τ = 1.0 s | 0.034–0.040 | 0.005–0.006 | 1–4 | 9.0–16.9 | 105–123 |
| τ = 1.5 s | 0.038–0.039 | 0.005–0.006 | 1–3 | 9.2–13.5 | 92 |

Ranges are over the three seeds. The first-epoch AMC loss rises with τ, which is consistent with the lag switch taking effect (a longer lag allows larger drops and demands a larger margin). AMC stays active until the end of training in every run.

**Checkpoint rule.** This is the locked per-run rule (`results/reeval_corrected/locked_config.json`): among the two saved checkpoints (lowest validation loss; final epoch), keep the one with the higher mean validation AP over 0.5/1.0/1.5 s. The random-offset and τ = 1.0 s runs keep their locked choices. The same rule chose the final epoch for all six new runs (`rq3_sensitivity_chosen_checkpoints.json`), so all 12 RQ3 runs use their final epoch.

**Metrics.** These are the same as RQ2 (`RESULTS_RQ2.md` §1):
- mAP over 0.5/1.0/1.5 s is the accuracy metric.
- Low-FAR metrics at FAR ≤ 0.1: mAUC@0.1, recall, mTTA and coverage.
- Temporal metrics are computed on dense causal risk curves of the 150 validation positives: 30 windows ending at t_event − d, d = 3.0 … 0.1 s. PVR (ε = 0.01, fixed in advance), ADS and RCJ; lower is better for all three. PVR with ε = 0 is a sensitivity check.

**Statistics.** A condition's value is the mean of its three seeds. Contrasts use a paired, label-stratified bootstrap (B = 2,000, seed 12345) with the same resampled videos for every condition. A 95% CI containing 0 is *inconclusive*. Contrast estimates are given to four decimals because several CI bounds lie within 0.002 of zero.

## 2. Main comparison: τ = 1.0 s vs. random-offset (validation)

**Accuracy and low-FAR anticipation** (mean ± SD over 3 seeds)

| Condition | mAP | AP@0.5s | AP@1.0s | AP@1.5s | mAUC@0.1 | Recall@1.0s | mTTA detected (s) | Coverage |
|---|---|---|---|---|---|---|---|---|
| Random-offset | 0.745 ± 0.033 | 0.817 ± 0.022 | 0.739 ± 0.038 | 0.680 ± 0.040 | 0.229 ± 0.073 | 0.367 ± 0.106 | 1.104 ± 0.079 | 56.4 ± 11.4% |
| FixedLag τ = 1.0 s | 0.743 ± 0.005 | 0.817 ± 0.009 | 0.739 ± 0.001 | 0.672 ± 0.006 | 0.224 ± 0.018 | 0.376 ± 0.010 | 1.078 ± 0.030 | 60.7 ± 1.3% |

Per-seed mAP was 0.711 / 0.748 / 0.777 for random and 0.739 / 0.741 / 0.748 for fixed (seeds 42 / 43 / 44). The paired fixed − random differences are +0.029, −0.007 and −0.029.

**Temporal consistency** (validation positives, dense curves)

| Condition | PVR (ε = 0.01) | ADS | RCJ | PVR (ε = 0, sensitivity) |
|---|---|---|---|---|
| Random-offset | 0.199 ± 0.009 | 0.032 ± 0.002 | 0.121 ± 0.008 | 0.317 ± 0.006 |
| FixedLag τ = 1.0 s | 0.182 ± 0.012 | 0.034 ± 0.003 | 0.135 ± 0.011 | 0.306 ± 0.006 |

**Bootstrap contrast, τ = 1.0 s − random** (95% CI)

| Metric | Estimate [95% CI] | Verdict |
|---|---|---|
| mAP | −0.0026 [−0.0311, 0.0249] | inconclusive |
| mAUC@0.1 | −0.0052 [−0.0643, 0.0579] | inconclusive |
| PVR | −0.0173 [−0.0337, −0.0013] | lower (better) |
| ADS | +0.0022 [−0.0012, 0.0053] | inconclusive |
| RCJ | +0.0136 [0.0021, 0.0245] | higher (worse) |

## 3. Supplementary: official Nexar test set

The two RQ3 main conditions were scored in the same single pass as RQ1 (`RESULTS_RQ1.md` §6). No test label was used to choose anything. Values are mean ± SD over 3 seeds; "all" means public + private under Nexar's group-wise definition.

| Condition | mAP public (official) | mAP private (official) | mAP all | AP 0.5 s | AP 1.0 s | AP 1.5 s | mAUC@0.1 all |
|---|---|---|---|---|---|---|---|
| Random-offset | 0.756 ± 0.023 | 0.811 ± 0.004 | 0.781 ± 0.015 | 0.832 ± 0.017 | 0.764 ± 0.023 | 0.747 ± 0.020 | 0.254 ± 0.037 |
| FixedLag τ = 1.0 s | 0.745 ± 0.018 | 0.805 ± 0.015 | 0.772 ± 0.013 | 0.831 ± 0.022 | 0.753 ± 0.021 | 0.732 ± 0.019 | 0.259 ± 0.030 |

Paired bootstrap on the whole test set (B = 2,000, resampling within every group × label cell):
- **fixed − random mAP = −0.0087 [−0.0221, 0.0060], inconclusive**
- mAUC@0.1 = +0.0044 [−0.0246, 0.0330], inconclusive

Per seed, the fixed − random mAP-all difference is +0.008, −0.024 and −0.011.

The τ = 0.5 / 1.5 s runs were not scored on the test set, because they are secondary sensitivity runs.

## 4. Secondary sensitivity: τ = 0.5 / 1.0 / 1.5 s (validation)

**Accuracy and low-FAR anticipation**

| Condition | mAP | AP@0.5s | AP@1.0s | AP@1.5s | mAUC@0.1 | Recall@1.0s | mTTA detected (s) | Coverage |
|---|---|---|---|---|---|---|---|---|
| FixedLag τ = 0.5 s | 0.728 ± 0.015 | 0.804 ± 0.023 | 0.718 ± 0.016 | 0.662 ± 0.017 | 0.211 ± 0.024 | 0.360 ± 0.057 | 1.093 ± 0.055 | 56.4 ± 5.2% |
| FixedLag τ = 1.0 s | 0.743 ± 0.005 | 0.817 ± 0.009 | 0.739 ± 0.001 | 0.672 ± 0.006 | 0.224 ± 0.018 | 0.376 ± 0.010 | 1.078 ± 0.030 | 60.7 ± 1.3% |
| FixedLag τ = 1.5 s | 0.750 ± 0.030 | 0.825 ± 0.019 | 0.747 ± 0.033 | 0.679 ± 0.040 | 0.260 ± 0.043 | 0.420 ± 0.080 | 1.090 ± 0.079 | 64.7 ± 6.4% |
| Random-offset | 0.745 ± 0.033 | 0.817 ± 0.022 | 0.739 ± 0.038 | 0.680 ± 0.040 | 0.229 ± 0.073 | 0.367 ± 0.106 | 1.104 ± 0.079 | 56.4 ± 11.4% |

The actual FAR at the chosen thresholds was at most 0.10 for every run and horizon (range 0.087–0.100).

**Temporal consistency**

| Condition | PVR (ε = 0.01) | ADS | RCJ | PVR (ε = 0, sensitivity) |
|---|---|---|---|---|
| FixedLag τ = 0.5 s | 0.174 ± 0.017 | 0.031 ± 0.002 | 0.121 ± 0.008 | 0.320 ± 0.001 |
| FixedLag τ = 1.0 s | 0.182 ± 0.012 | 0.034 ± 0.003 | 0.135 ± 0.011 | 0.306 ± 0.006 |
| FixedLag τ = 1.5 s | 0.191 ± 0.001 | 0.032 ± 0.003 | 0.122 ± 0.014 | 0.311 ± 0.011 |
| Random-offset | 0.199 ± 0.009 | 0.032 ± 0.002 | 0.121 ± 0.008 | 0.317 ± 0.006 |

**Bootstrap contrasts** (95% CI). Bold marks a CI that excludes 0; "better" and "worse" refer to the direction of that change.

| Contrast | mAP | mAUC@0.1 | PVR | ADS | RCJ |
|---|---|---|---|---|---|
| τ0.5 − τ1.0 | −0.0147 [−0.0475, 0.0163] | −0.0137 [−0.0832, 0.0516] | −0.0083 [−0.0237, 0.0072] | **−0.0032 [−0.0062, −0.0000] better** | **−0.0139 [−0.0239, −0.0036] better** |
| τ1.5 − τ1.0 | +0.0076 [−0.0207, 0.0353] | +0.0356 [−0.0285, 0.0923] | +0.0093 [−0.0060, 0.0250] | −0.0021 [−0.0051, 0.0011] | **−0.0130 [−0.0236, −0.0028] better** |
| τ0.5 − random | −0.0173 [−0.0459, 0.0114] | −0.0189 [−0.0770, 0.0449] | **−0.0256 [−0.0426, −0.0091] better** | −0.0010 [−0.0042, 0.0022] | −0.0002 [−0.0113, 0.0105] |
| τ1.0 − random | −0.0026 [−0.0311, 0.0249] | −0.0052 [−0.0643, 0.0579] | **−0.0173 [−0.0337, −0.0013] better** | +0.0022 [−0.0012, 0.0053] | **+0.0136 [0.0021, 0.0245] worse** |
| τ1.5 − random | +0.0051 [−0.0205, 0.0320] | +0.0304 [−0.0211, 0.0821] | −0.0080 [−0.0248, 0.0081] | +0.0001 [−0.0030, 0.0031] | +0.0007 [−0.0098, 0.0111] |

## 5. Seed-to-seed variability

The interim draft of `RESULTS.md` (its old §5.3, now replaced) reported lower seed variance as the main effect of fixed-lag training. The sensitivity runs let us test whether that holds beyond τ = 1.0 s.

| Condition | Seed SD of validation mAP | 95% CI of the SD (χ², n = 3; post hoc) | Seed SD of official test mAP (all / public / private) |
|---|---|---|---|
| Random-offset | 0.033 | [0.017, 0.208] | 0.015 / 0.023 / 0.004 |
| FixedLag τ = 0.5 s | 0.015 | [0.008, 0.096] | not scored |
| FixedLag τ = 1.0 s | 0.005 | [0.002, 0.029] | 0.013 / 0.018 / 0.015 |
| FixedLag τ = 1.5 s | 0.030 | [0.016, 0.190] | not scored |

The CI column assumes normally distributed seed effects and is only a rough guide: an SD from three seeds has two degrees of freedom.

## 6. Exploratory (post hoc): curve level and shape

This was not planned in advance. It is computed from the saved dense curves (`results/rq2/raw/dense_rq2/`, `results/rq3_sensitivity/raw/dense_rq3s/`) to check whether the PVR differences come from flatter curves, as they partly did in RQ2 (`RESULTS_RQ2.md` §6). Each value is the mean over the three seeds.

| Condition | Mean positive score | Rise (last 0.3 s − first 0.3 s) | Positive curves with range < 0.05 | RCJ / curve range (median) | Mean negative score | RCJ on negatives |
|---|---|---|---|---|---|---|
| FixedLag τ = 0.5 s | 0.41 | 0.43 | 7.6% | 0.144 | 0.11 | 0.071 |
| FixedLag τ = 1.0 s | 0.41 | 0.43 | 6.7% | 0.150 | 0.11 | 0.072 |
| FixedLag τ = 1.5 s | 0.42 | 0.44 | 6.9% | 0.140 | 0.11 | 0.066 |
| Random-offset | 0.50 | 0.41 | 8.7% | 0.145 | 0.14 | 0.085 |

Fixed-lag curves are **not** flatter. At every τ they rise slightly more before the event and are less often near-flat than random-offset curves, so their lower PVR is not an artefact of flat curves. What does differ is the overall score level: fixed-lag training lowers scores on both positives (0.41 vs 0.50) and negatives (0.11 vs 0.14) and gives less jitter on negatives. Ranking accuracy (AP) is unchanged. Relative jitter (RCJ divided by curve range) is essentially the same for all four conditions (0.140–0.150).

## 7. Findings

1. **Fixed-lag pairing does not change accuracy or early warning.**
   - On validation, mAP is 0.728–0.750 across the three lags vs 0.745 for random, and every mAP and mAUC@0.1 contrast is inconclusive.
   - On the official test, τ = 1.0 s is −0.009 mAP [−0.022, 0.006] relative to random.
   - Recall, mTTA and coverage differences are within seed variation.
2. **Fixed-lag pairing slightly reduces large pairwise violations, and the reduction grows as the lag gets shorter.**
   - PVR is ordered τ = 0.5 s (0.174) < 1.0 s (0.182) < 1.5 s (0.191) < random (0.199; mean random lag 2.75 s).
   - τ0.5 − random (−0.026 [−0.043, −0.009]) and τ1.0 − random (−0.017 [−0.034, −0.001]) exclude 0. The τ1.0 CI does so only narrowly.
   - τ1.5 − random and the differences between adjacent lags are inconclusive, so the ordering is a trend rather than an established dose–response.
   - The trend is specific to violations larger than ε = 0.01. With ε = 0, τ = 1.0 s has the fewest violations (0.306) and τ = 0.5 s the most (0.320), with random in between (0.317).
3. **The extra jitter at τ = 1.0 s is not a property of fixed-lag pairing in general.**
   - τ = 1.0 s has the highest RCJ of the four conditions: +0.014 vs random, and higher than both τ = 0.5 s and τ = 1.5 s (both CIs exclude 0).
   - τ = 0.5 s and τ = 1.5 s have the same RCJ as random.
   - Because the pattern is not monotone in τ and relative jitter is similar across conditions (§6), it most likely reflects these three τ = 1.0 s seeds rather than a mechanism.
4. **The earlier seed-variance claim does not hold up.**
   - τ = 1.0 s had a validation mAP SD of 0.005 vs 0.033 for random. At τ = 1.5 s, however, the SD is 0.030, about the same as random.
   - The rough χ² intervals for τ = 1.0 s and random overlap.
   - On the official test, the τ = 1.0 s and random SDs are similar (0.013 vs 0.015 on all clips; the private subset is lower for random).
   - The point SD is lower than random at all three lags, which leaves room for a modest reduction, but three seeds per condition cannot establish one. **We therefore withdraw "fixed-lag reduces seed variance" as a finding.**
5. **Answer to RQ3.** Replacing random-offset AMC pairs with a fixed 1.0 s lag leaves accuracy and early warning unchanged. It gives a small reduction in large pairwise violations, together with a small increase in jitter at that lag. It does not reliably reduce seed variance. The proposal's success criterion (an improvement in temporal violations *or* seed variance) is therefore met only weakly, through PVR, and not through seed variance. Shorter lags (0.5 s) give the clearest PVR benefit without the jitter cost, but that is a secondary, exploratory observation.

## 8. Deviations and limitations

- **Change from the interim draft.** The old §5.3 of `RESULTS.md` (now replaced by the final version) said that the main measurable effect of fixed-lag training is lower seed variance. The τ-sensitivity runs and the official test do not support that (Finding 4), and this document replaces that statement. `RESULTS_RQ2.md` Finding 5 was worded to match.
- **Training sessions.** The random-offset and τ = 1.0 s runs were trained earlier, together with the RQ1 models (105–148 s per epoch; logs in `results/riskprop/`). The τ = 0.5 / 1.5 s runs were trained later on one RTX 4080 (about 92 s per epoch). Code, data, split and hyperparameters are identical, but GPU non-determinism across sessions may add some run-to-run variation.
- **Multiple comparisons.** Section 4 has 25 validation contrasts with no multiplicity correction, so about one CI could exclude 0 by chance. Three of the conclusive CIs have a bound within 0.0022 of zero (τ1.0 − random PVR, τ0.5 − τ1.0 ADS, τ1.0 − random RCJ) and should be read as weak evidence.
- **Three seeds.** Bootstrap CIs resample videos, not training seeds, so they are conditional on the three trained models per condition. The seed-SD intervals in §5 are post hoc and assume normality.
- **Temporal metrics.** The formulas and ε = 0.01 were fixed before any temporal metric was computed (see `RESULTS_RQ2.md` §1). The curve-shape analysis (§6) is post hoc.
- **Sensitivity runs on test.** τ = 0.5 / 1.5 s were evaluated on validation only, in line with the proposal's secondary status for these runs.
- **Cost.** The six sensitivity runs took about 7.7 GPU-hours on one RTX 4080. Their validation and dense-curve inference took a few minutes.

## 9. Ghi chú nội bộ để họp nhóm (xóa mục này trước khi nộp)

Những điểm cần partner và cả nhóm xem xét, thống nhất trước khi viết bản cuối:

1. **Câu hỏi RQ3 ở đầu file** đang là câu mình diễn đạt lại. Nhờ chép đúng nguyên văn câu RQ3 trong proposal vào.
2. **Rút lại kết luận "fixed-lag giảm phương sai giữa các seed".** Bản nháp `RESULTS.md` cũ (§5.3, nay đã thay bằng bản cuối) lấy đây làm kết quả chính. Với τ = 1.5 s (SD 0.030 ≈ random 0.033) và trên official test (0.013 so với 0.015), kết luận này không còn đứng được. Đề xuất: bỏ hẳn, chỉ ghi "không kết luận được với 3 seed". Nhóm đồng ý không? `RESULTS_RQ2.md` (Finding 5 và ghi chú số 7) đã được sửa cho khớp.
3. **Success criterion của proposal** ("temporal violations *hoặc* seed variance cải thiện"). Hiện chỉ PVR cải thiện, CI của τ1.0 sát 0, và RCJ ở τ1.0 còn tệ hơn. Viết là "đạt một phần, yếu" (như mục 7.5) hay "không đạt"?
4. **Câu kết luận chính đề xuất:** *"Fixed-lag không làm thay đổi độ chính xác (val và test), giảm nhẹ vi phạm đơn điệu lớn (PVR), nhưng không giảm được độ dao động giữa các seed."* Đồng ý không?
5. **Xu hướng PVR theo τ** (lag càng ngắn càng ít vi phạm) chỉ đúng với ε = 0.01; với ε = 0 thì thứ tự khác. Proposal ghi các run sensitivity "không được coi là đóng góp riêng". Đưa mục 4 vào bài chính (1 đoạn) hay để phụ lục?
6. **RCJ cao nhất ở τ = 1.0 s** nhưng không tăng/giảm đều theo τ. Mình diễn giải là do 3 seed đó, không phải cơ chế. Có cần bàn trong Discussion không?
7. **RQ3 trên official test** đang để là supplementary (khớp RQ1 §6). Có muốn đưa lên bảng chính của RQ3 không?
8. **τ = 0.5 / 1.5 s không chạy trên official test.** Máy vast đã/đang destroy, nếu muốn chạy phải thuê lại máy và tải lại dữ liệu test. Đề xuất bỏ, vì đây là phân tích phụ. Đồng ý?
9. **Mục 6 (mức điểm và hình dạng đường risk)** là phân tích làm thêm sau khi có kết quả (post hoc). Điểm đáng chú ý: fixed-lag hạ mức điểm tổng thể (0.41 so với 0.50) mà không làm đường risk phẳng hơn. Đưa vào bài chính hay phụ lục?
10. **Nhiều phép so sánh.** Có 25 CI không hiệu chỉnh đa kiểm định (multiple comparisons). Giữ câu cảnh báo như mục 8 là đủ, hay cần hiệu chỉnh (ví dụ Bonferroni)?
11. **So với paper RiskProp.** Theo ghi chú trong `cell32`, paper chỉ dùng random pairs (Fig. 3), không có thí nghiệm fixed-lag, nên RQ3 không có bảng so với paper. Nhờ partner xác nhận lại.
12. **Hai đợt train khác nhau.** τ1.0 và random train ở đợt trước, τ0.5/1.5 train đợt sau trên RTX 4080. Ghi là hạn chế như mục 8 được chưa?
13. **Chi phí:** điền số credit đã dùng sau khi destroy máy.
