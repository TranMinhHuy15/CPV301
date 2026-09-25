# Results — RQ1: TOP vs. AdaLEA vs. RiskProp

> RQ1: How do AdaLEA, TOP, and RiskProp compare in official test mAP and, on labeled internal validation data, early-warning performance at FAR ≤ 0.1?

Sources: `results/official_test/` (`reeval/re04_infer_test.py`, `reeval/re05_eval_test.py`) for the primary outcome, and `results/reeval_corrected/` (`reeval/re01`–`re03`) for the validation analysis.

## 1. Protocol

**Methods.** TOP, AdaLEA and RiskProp (random-offset AMC, i.e. the published objective), three predetermined seeds each (42/43/44). All three share:
- the SlowOnly-R50 backbone on causal 5-frame snippets (224×224, 10 fps)
- the same train/validation split (1,200 / 300 videos; `results/repro/split_manifest_seed42.json`)
- a matched training budget (SGD, LR 0.01, 50 epochs)

**Configuration locked on validation before any test inference.** Two evaluation choices were fixed from validation data only (`results/reeval_corrected/locked_config.json`; see the protocol-correction note in `RESULTS.md` §5.1):
- **TOP head rule.** TOP outputs 20 horizon heads, and the official test gives no horizon information. We use one fixed head, `head_2.0` ("collision within 2.0 s"), chosen by mean validation mAP. The oracle head matched to each clip's true horizon is excluded, because AdaLEA and RiskProp cannot use that information.
- **Checkpoint rule.** Per run, keep whichever of {lowest-validation-loss checkpoint, final epoch} has the higher mean validation AP over 0.5/1.0/1.5 s. This selected the final epoch for 11 of the 12 runs.

A legacy check passed first: re-scoring every original checkpoint reproduced the previously reported validation metrics (max |Δ| ≤ 0.0003).

**Primary outcome: official Nexar test mAP.** The official test set has 1,344 clips, pre-cut to end 0.5, 1.0 or 1.5 s before the event, and is split into public and private subsets. Each clip is scored once, on the causal 5-frame window ending at its last frame, with the same preprocessing as training. All 1,344 clips were scored (none skipped). Scores are computed by Nexar's unmodified `evaluate_submission.py`: AP inside each solution group, averaged over groups, separately for public and private.

For uncertainty, we re-implemented the same group-wise definition. It reproduces the official public and private mAP of every run exactly (max |Δ| = 0.000000). We applied it to the whole test set (public + private) and added mAUC@0.1 on the same groups. Paired bootstrap 95% CIs use B = 2,000, resampling within every group × label cell, with the same resamples applied to every run. A method's value is the mean of its three seeds.

**Secondary outcome: early warning at FAR ≤ 0.1, on internal validation.** On the 300 validation videos (150 positive / 150 negative), with three causal windows per video at 0.5, 1.0 and 1.5 s before the event:
- mAUC@0.1 (partial AUC for FPR ≤ 0.1)
- Recall at FAR ≤ 0.1, and the actual FAR achieved
- mTTA (the earliest horizon at which a detected positive clears the FAR ≤ 0.1 threshold)
- coverage

## 2. Official test set (primary outcome)

| Method | mAP public (official) | mAP private (official) | mAP all | AP 0.5 s | AP 1.0 s | AP 1.5 s | mAUC@0.1 all |
|---|---|---|---|---|---|---|---|
| TOP | **0.769 ± 0.016** | **0.811 ± 0.027** | **0.785 ± 0.021** | 0.811 ± 0.031 | **0.784 ± 0.016** | **0.760 ± 0.026** | **0.273 ± 0.035** |
| AdaLEA | 0.700 ± 0.012 | 0.795 ± 0.019 | 0.742 ± 0.011 | 0.776 ± 0.015 | 0.733 ± 0.015 | 0.717 ± 0.015 | 0.211 ± 0.014 |
| RiskProp (random) | 0.756 ± 0.023 | **0.811 ± 0.004** | 0.781 ± 0.015 | **0.832 ± 0.017** | 0.764 ± 0.023 | 0.747 ± 0.020 | 0.254 ± 0.037 |

Mean ± SD over three seeds. "All" means public + private, using the official group-wise definition. The AP columns are the three solution groups, labelled by the horizon of their positives.

**Paired bootstrap 95% CIs on the whole test set:**

| Comparison | Δ mAP [95% CI] | Δ mAUC@0.1 [95% CI] |
|---|---|---|
| TOP − AdaLEA | **+0.043 [0.025, 0.060]** | **+0.062 [0.026, 0.097]** |
| RiskProp − AdaLEA | **+0.039 [0.015, 0.060]** | **+0.044 [0.002, 0.085]** |
| TOP − RiskProp | +0.004 [−0.018, 0.028], inconclusive | +0.019 [−0.031, 0.068], inconclusive |

On the public subset, AdaLEA ranks last for every seed (seed 42: 0.687 vs 0.749 for RiskProp and 0.787 for TOP; seed 43: 0.706 vs 0.782 / 0.762; seed 44: 0.708 vs 0.738 / 0.757). TOP and RiskProp trade first place (TOP leads at seeds 42 and 44, RiskProp at seed 43).

## 3. Internal validation: early warning at FAR ≤ 0.1 (secondary outcome)

| Method | mAP | mAUC@0.1 | Recall@0.5 s | Recall@1.0 s | Recall@1.5 s | mTTA detected (s) | mTTA all (s) | Coverage |
|---|---|---|---|---|---|---|---|---|
| TOP | 0.751 ± 0.013 | 0.223 ± 0.027 | 0.476 ± 0.034 | 0.391 ± 0.019 | 0.278 ± 0.010 | 1.086 ± 0.048 | 0.658 ± 0.021 | 60.7 ± 4.4% |
| AdaLEA | 0.739 ± 0.019 | 0.240 ± 0.024 | 0.478 ± 0.086 | 0.400 ± 0.024 | 0.358 ± 0.074 | 1.208 ± 0.012 | 0.690 ± 0.096 | 57.1 ± 7.9% |
| RiskProp (random) | 0.745 ± 0.033 | 0.229 ± 0.073 | 0.491 ± 0.113 | 0.367 ± 0.106 | 0.273 ± 0.099 | 1.104 ± 0.079 | 0.629 ± 0.164 | 56.4 ± 11.4% |

The actual FAR at the chosen thresholds was at most 0.10 for every run and horizon (range 0.093–0.100).

**Validation bootstrap CIs:** every pairwise difference is inconclusive.

| Comparison | mAP | mAUC@0.1 |
|---|---|---|
| TOP − AdaLEA | +0.012 [−0.026, 0.046] | −0.018 [−0.087, 0.055] |
| TOP − RiskProp | +0.006 [−0.040, 0.045] | −0.007 [−0.096, 0.071] |
| AdaLEA − RiskProp | −0.006 [−0.047, 0.031] | +0.011 [−0.061, 0.077] |

## 4. Findings

1. **On the official test set, TOP and RiskProp both outperform AdaLEA, and TOP and RiskProp are statistically indistinguishable.** Relative to AdaLEA, TOP is +0.043 mAP and RiskProp is +0.039 mAP (both CIs exclude 0), and the same holds for mAUC@0.1. The TOP − RiskProp difference is +0.004 mAP with a CI spanning 0: TOP has the slightly higher mean (0.785 vs 0.781), but it does not significantly outperform RiskProp. RiskProp is strongest at the shortest horizon (AP 0.832 vs 0.811 at 0.5 s), and TOP at the longer horizons.
2. **Validation shows the same ordering but not a significant difference.** Validation mAP is ordered TOP ≥ RiskProp ≥ AdaLEA, but every CI includes 0. The official test contains about 4.5× more clips than the validation split and its scores were not used for any selection, so it is both better powered and unbiased. Validation was used to lock the TOP head and the checkpoints and is therefore slightly optimistic.
3. **Early-warning performance at FAR ≤ 0.1 is not separable on validation.** AdaLEA has the longest warning time among detected positives (1.21 s) and the highest recall at 1.5 s, but it has lower coverage than TOP, and none of the mAUC@0.1 differences is significant. With 150 positive validation videos, early-warning differences of this size cannot be resolved.
4. **Seed variability.** RiskProp has the largest seed SD on validation mAP (0.033, vs 0.013 for TOP and 0.019 for AdaLEA), but the smallest on the private test subset (0.004). With three seeds, SD estimates are too imprecise to rank the methods by stability. RQ2/RQ3 examine the RiskProp-specific sources of variance.
5. **Agreement with the RiskProp paper (Table 1, Nexar test).** The paper reports RiskProp over AdaLEA by +0.038 mAP (0.870 vs 0.832) and +0.094 mAUC@0.1 (0.472 vs 0.378). We observe +0.039 mAP [0.015, 0.060] and +0.044 mAUC@0.1 [0.002, 0.085]. The direction agrees, and the mAP gap is almost identical. Our absolute scores are lower (RiskProp 0.781 vs 0.870 mAP), as expected from a much smaller training setup: batch 2 videos on one GPU vs batch 64 on eight A800s, λ1 = λ2 = 0.5 vs 1.5 / 1.1, and snippet-level rather than frame-level sequences. TOP is not included in the paper's table.
6. **Single best runs do not indicate which method is better** (Appendix A). RiskProp seed 44 has the highest validation mAP of any single run (0.777). On the official test, however, it is RiskProp's weakest seed (0.768), while TOP seed 42 scores highest of all nine RQ1 runs (0.809). This is why the main comparison uses three-seed means.

## 5. Deviations and limitations

- **Evaluation protocol correction.** The original TOP evaluation used the oracle horizon-matched head, and checkpoints were selected by validation loss. Both were replaced by rules locked on validation before any test inference (§1; details in `RESULTS.md` §5.1).
- **Test inference code.** A missing ImageNet normalisation in the test inference script was found and fixed **before** its first run, so no test score was computed with the faulty version. The secondary re-implementation of the test metric was aligned with Nexar's group-wise definition after reading `evaluate_submission.py`. The official numbers were unaffected, and the re-implementation matches them exactly.
- **TOP uses one fixed head for every clip.** TOP's best possible per-horizon head is not used, because it requires horizon knowledge that the other methods do not have.
- **Three seeds.** Bootstrap CIs resample videos, not training seeds, and the seed SD is reported separately.
- **Example submission.** The `sample_submission.csv` distributed with the dataset scores 0.841 / 0.862 (public / private). Its provenance is undocumented, so it is not treated as a baseline.
- **Cost.** RQ1 models were trained on a single GPU. Official-test inference for all runs (frame extraction plus scoring) took about 5 minutes on one RTX 4080.

## 6. Role of the official test set

The official Nexar test set is the proposal's primary outcome for RQ1. It was run once, after every evaluation choice had been locked on validation, for four reasons:

1. **Unbiased estimate.** The TOP head and every checkpoint were chosen on the internal validation split, so validation scores are slightly optimistic. The official test labels were never used for any choice (checkpoint, head, threshold, or re-run), so its scores estimate generalisation without that bias.
2. **Statistical power.** Validation has 300 videos (150 positive). The official test has 1,344 clips. The TOP/RiskProp-vs-AdaLEA gap was inconclusive on validation and becomes resolvable on the test set.
3. **Comparability with the literature.** Published Nexar results, including RiskProp's Table 1, are reported on this test set with Nexar's own scorer. We use the same unmodified scorer, so our mAP is on the official scale.
4. **Horizon-blind evaluation.** Test clips end 0.5, 1.0 or 1.5 s before the event without revealing which. This is why TOP must use one fixed head and cannot use the oracle head, which makes the comparison fair across methods.

What the test set was **not** used for: selecting models, checkpoints, heads or thresholds, or deciding what to report. RQ2 and RQ3 conditions were scored in the same single pass, but only as supplementary or exploratory results. RQ2 conclusions come from validation.

## Appendix A. Best single validation run per method (post hoc, descriptive)

For each method, the seed with the highest validation mAP was taken, using the checkpoint chosen by the locked rule. Every metric in a row comes from that same run; no metrics are combined across runs.

This comparison is **descriptive only** and is not used for any conclusion, for two reasons:
- It uses validation twice: first to choose the checkpoint, then to choose the seed.
- Taking the best of three seeds favours the method with the largest seed-to-seed variance. The validation mAP SD is 0.033 for RiskProp, 0.019 for AdaLEA and 0.013 for TOP.

| Method | Seed / checkpoint | Val mAP | Val AP@0.5s | Val AP@1.0s | Val AP@1.5s | Val mAUC@0.1 | Val Video AUC | Val coverage | Official test mAP all (public / private) | Test rank among the method's 3 seeds |
|---|---|---|---|---|---|---|---|---|---|---|
| TOP (`head_2.0`) | seed 42, final epoch (50) | 0.7626 | 0.8200 | 0.7669 | 0.7010 | 0.2540 | 0.7825 | 61.3% | **0.8090** (0.7874 / 0.8385) | 1 of 3 |
| AdaLEA | seed 42, lowest-val-loss epoch (7) | 0.7610 | 0.8040 | 0.7570 | **0.7219** | 0.2655 | 0.7631 | 61.3% | 0.7329 (0.6870 / 0.7951) | 3 of 3 |
| RiskProp (random) | seed 44, final epoch (50) | **0.7769** | **0.8377** | **0.7732** | 0.7197 | **0.2970** | **0.8011** | **63.3%** | 0.7684 (0.7378 / 0.8068) | 3 of 3 |

Sources: `results/reeval_corrected/per_run_metrics.csv` (validation), `results/official_test/test_per_run.csv` (test) and `results/reeval_corrected/locked_config.json` (chosen checkpoints). The RiskProp row is the random-offset model, not the fixed-lag variant of RQ3.

Among these selected runs, RiskProp seed 44 has the highest validation mAP. The comparison is post hoc, however, and on the official test TOP seed 42 scores higher than RiskProp seed 44 (0.809 vs 0.768). For both RiskProp and AdaLEA, the best validation run is the weakest of the three seeds on the test set. The main RQ1 result therefore remains the three-seed comparison in §2.

## 7. Ghi chú nội bộ để họp nhóm (xóa mục này trước khi nộp)

Những điểm cần partner và cả nhóm xem xét, thống nhất trước khi viết bản cuối:

1. **Câu kết luận chính của RQ1.** Đề xuất: *"Trên official test, TOP và RiskProp đều hơn AdaLEA (mAP +0.043 và +0.039, CI > 0), còn TOP và RiskProp không phân định được (+0.004, CI chứa 0)."* Nhóm đồng ý cách diễn đạt này không?
2. **Val khác test.** Trên val mọi khác biệt đều inconclusive, trên test thì AdaLEA thua rõ. Đề xuất: lấy test làm kết quả chính, val để minh họa, giải thích bằng cỡ mẫu và thiên lệch do chọn cấu hình (mục 4.2). Có cần nói thêm gì không?
3. **Số nào làm tiêu đề.** Hiện đang báo cáo cả mAP public/private chính thức và mAP toàn bộ test (có CI). Đề xuất: bảng chính ghi cả public và private, CI dùng toàn bộ test. Hay chỉ dùng public như leaderboard Kaggle?
4. **TOP chỉ dùng 1 head (`head_2.0`).** Cách này công bằng với AdaLEA và RiskProp, nhưng làm TOP thấp hơn bản dùng head "oracle" trong paper gốc của TOP. Có đưa kết quả oracle (chỉ trên val) vào phụ lục không?
5. **AdaLEA trên val có mTTA dài nhất (1.21s) và recall@1.5s cao nhất**, nhưng trên test mAP lại thấp nhất. Có cần bàn trong Discussion không (ví dụ AdaLEA báo động sớm nhưng xếp hạng kém)?
6. **Khoảng cách với paper** (RiskProp 0.781 so với 0.870 mAP). Giải thích bằng ngân sách tính toán (batch, số GPU, λ, snippet-level) có đủ thuyết phục chưa? Hướng so sánh với AdaLEA thì khớp paper gần như tuyệt đối (+0.039 so với +0.038).
7. **`sample_submission.csv` của Nexar đạt 0.841/0.862**, cao hơn cả 3 model. Không rõ nó là dự đoán của model nào. Có nhắc trong bài không, hay bỏ?
8. **Kết quả RQ2 và RQ3 trên test.** Theo protocol của partner, RQ2 chỉ kết luận từ val. Đưa các dòng RQ2/RQ3 trên test vào phụ lục (ghi rõ exploratory) hay bỏ hẳn?
9. **Minh bạch về lỗi chuẩn hóa ảnh** trong code test, đã sửa trước lần chạy đầu tiên. Ghi trong mục Deviations như hiện tại có đủ không?
10. **Chỉ có 3 seed.** CI đang resample video, không resample seed. Nhóm có muốn thêm seed không? Mỗi seed thêm tốn khoảng 1.3 giờ GPU cho mỗi phương pháp. Hay ghi là hạn chế?
11. **Chỉ số cảnh báo sớm chỉ đo trên val.** Đề cương ghi đúng như vậy ("on labeled internal validation data"). Xác nhận không cần đo trên test.
12. **Bảng best single run (đã thống nhất với partner, 25/9).** Bảng này đặt ở Phụ lục A, chỉ để mô tả (post hoc), không dùng cho kết luận. Không viết "RiskProp is the best single validation run". Kết luận chính: TOP và RiskProp không phân định được trên official test và cả hai hơn AdaLEA; TOP có mean cao hơn một chút nhưng CI của TOP − RiskProp chứa 0.
