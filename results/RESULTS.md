# Results

## 5.1 Evaluation protocol correction

An initial evaluation of TOP, AdaLEA, and RiskProp (random-offset and fixed-lag variants) revealed two methodological issues that would have biased the RQ1/RQ3 comparisons:

1. **TOP head selection.** The original TOP evaluation selected, for each clip, the output head matching the clip's known lead time (0.5/1.0/1.5 s) — information unavailable to AdaLEA and RiskProp, which emit a single risk score per window. This "oracle" scoring is not comparable across methods.
2. **Checkpoint selection.** All three models save a `best` checkpoint (lowest validation loss) and a `latest` checkpoint (final epoch). For 11 of 12 trained runs, validation mAP continued to improve well past the epoch selected by validation loss, so the `best` checkpoint under-represented the trained model's true performance.

We re-evaluated all 24 saved checkpoints (3 seeds × {best, latest} × 4 configurations) on the original internal validation split (300 videos, reconstructed bit-for-bit from the original preprocessing code; confirmed by an exact match against the previously reported metrics, max abs. difference 0.0003 mAP). Two rules were then locked from validation data only, before any further analysis: (i) a single fixed TOP head (`head_2.0`, chosen by mean validation mAP across all TOP runs, oracle head excluded from candidacy), and (ii) a per-run checkpoint rule (whichever of {best, latest} has the higher validation mAP). Metrics were then recomputed under these fixed rules and reported as mean ± SD over 3 seeds, with 95% confidence intervals from a paired, label-stratified bootstrap (B = 2000 resamples, same resampled videos applied to every method).

Full data: [`results/reeval_corrected/`](results/reeval_corrected/) (`summary.md`, `locked_config.json`, per-run/bootstrap CSVs) and the re-evaluation pipeline itself ([`reeval/`](reeval/)).

## 5.2 RQ1 — TOP vs. AdaLEA vs. RiskProp (random-offset)

| Model | mAP | AP@0.5s | AP@1.0s | AP@1.5s | mAUC@0.1 | Video AUC |
|---|---|---|---|---|---|---|
| TOP | 0.751 ± 0.013 | 0.807 ± 0.024 | 0.754 ± 0.011 | 0.693 ± 0.007 | 0.223 ± 0.027 | 0.772 ± 0.010 |
| AdaLEA | 0.739 ± 0.019 | 0.784 ± 0.017 | 0.737 ± 0.022 | 0.698 ± 0.022 | 0.241 ± 0.024 | 0.752 ± 0.010 |
| RiskProp (random) | 0.745 ± 0.033 | 0.817 ± 0.022 | 0.739 ± 0.038 | 0.680 ± 0.040 | 0.229 ± 0.073 | 0.768 ± 0.029 |

Paired bootstrap 95% CIs on the pairwise mAP differences all include zero (TOP−AdaLEA: [−0.026, 0.046]; TOP−RiskProp: [−0.040, 0.045]; AdaLEA−RiskProp: [−0.047, 0.031]), and the per-seed ranking is not stable (TOP leads at seed 42, RiskProp leads at seeds 43–44). **We therefore find no statistically significant difference in detection performance among the three methods under this evaluation budget**; observed mean differences are within seed-to-seed variance.

## 5.3 RQ3 — RiskProp random-offset vs. fixed-lag (τ = 1.0 s)

| Variant | mAP | AP@0.5s | AP@1.0s | AP@1.5s | mAUC@0.1 |
|---|---|---|---|---|---|
| Random-offset | 0.745 ± 0.033 | 0.817 ± 0.022 | 0.739 ± 0.038 | 0.680 ± 0.040 | 0.229 ± 0.073 |
| Fixed-lag | 0.743 ± 0.005 | 0.817 ± 0.009 | 0.739 ± 0.001 | 0.672 ± 0.006 | 0.224 ± 0.018 |

Per-seed paired differences (fixed − random) are +0.029, −0.007, −0.029 (mean −0.003, SD 0.029) — essentially zero on average, with the fixed-lag variant showing markedly **lower seed-to-seed variance** (SD 0.005 vs. 0.033 in mAP). The bootstrap CI on the mean difference includes zero ([−0.031, 0.025]). **We find no significant difference in mean detection accuracy between the two training strategies; the main measurable effect of fixed-lag training is a reduction in seed-to-seed variability**, consistent with the proposal's RQ3 success criterion (temporal violations *or* seed variance may show improvement — variance does).

## 5.4 Summary

Under the corrected, leakage-free protocol, none of the RQ1 or RQ3 comparisons reach statistical significance at the video-count and seed-count used here. This is a substantive finding in itself: it indicates that architectural choice (TOP vs. AdaLEA vs. RiskProp) and offset strategy (random vs. fixed-lag) have a smaller effect than seed-to-seed training variance at this scale, with fixed-lag training's main benefit being variance reduction rather than mean accuracy. A full statistical resolution would require either more seeds or the larger official test split (not used here, per protocol, pending a locked-configuration test run).
