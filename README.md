# CPV301: Collision anticipation on Nexar, a controlled reproduction of RiskProp

Research-based learning project, CPV301 (Computer Vision), FPT University.
Team: Dương Duy Lợi, Trần Minh Huy, Hoàng Duy Lưu.

We compare three collision-anticipation methods (**TOP**, **AdaLEA**, **RiskProp**) on the [Nexar Collision Prediction](https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction) dataset under one controlled protocol. We then take RiskProp apart: which of its two losses matter, and whether its pairing of time steps can be improved.

**Status:** RQ1, RQ2 and RQ3 are complete. RQ1b (RiskProp re-trained with its authors' own code) is in progress.

---

## Key findings

| | Finding |
|---|---|
| **RQ1** | On the official Nexar test set, **TOP (0.785 mAP) and RiskProp (0.781) are statistically indistinguishable, and both beat AdaLEA (0.742)** by about 0.04 mAP. |
| **RQ2** | RiskProp's accuracy comes from its **future-frame regularization (FFR)** (+0.09 to +0.10 mAP). Its **monotonic constraint (AMC)** does not change accuracy; it only makes risk curves more monotone. |
| **RQ3** | Pairing time steps at a **fixed lag** instead of random offsets does not change accuracy, slightly reduces temporal violations, and does **not** reliably reduce seed-to-seed variance. |

All numbers are means over 3 training seeds, with 95% bootstrap confidence intervals in the result files.

---

## Where to find what

| I want to… | Open |
|---|---|
| Read all results in one place | [`results/RESULTS.md`](results/RESULTS.md) |
| Read one research question in detail | [`RESULTS_RQ1.md`](results/RESULTS_RQ1.md), [`RESULTS_RQ2.md`](results/RESULTS_RQ2.md), [`RESULTS_RQ3.md`](results/RESULTS_RQ3.md) |
| See the raw numbers (CSV, predictions, logs) | the sub-folders of [`results/`](results/) (table below) |
| Read the original training and evaluation code | [`pipeline/`](pipeline/) |
| Re-run the corrected evaluation, RQ2 or RQ3 | [`reeval/`](reeval/) |
| Run RQ1b (RiskProp with the authors' code) | [`riskprop_full/README_RQ1B.md`](riskprop_full/README_RQ1B.md) |
| Look up the RQ2 losses and metric formulas | [`RQ2_THEORY.md`](RQ2_THEORY.md) |

---

## Research questions

| RQ | Question | Evaluated on |
|---|---|---|
| RQ1 | How do AdaLEA, TOP and RiskProp compare in official test mAP and in early warning at a false-alarm rate ≤ 0.1? | Official Nexar test (mAP); internal validation (early warning) |
| RQ2 | How much do RiskProp's two losses, FFR and AMC, contribute, alone and together? (2×2 ablation) | Internal validation |
| RQ3 | Does pairing time steps at a fixed lag (τ = 1.0 s; sensitivity at 0.5 and 1.5 s) help compared with random offsets? | Internal validation |
| RQ1b | Is the project's RiskProp weaker because of how it was re-implemented? (the authors' own code, same protocol) | In progress |

## The three methods

All three use the same SlowOnly-R50 backbone, the same 5-frame causal windows (10 fps, 224×224), the same train/validation split and the same optimiser budget (SGD, lr 0.01, 50 epochs, seeds 42/43/44). At evaluation each method scores the same single window.

| Method | Idea | Output per window |
|---|---|---|
| **TOP** | Predicts "collision within 0.1 s, 0.2 s, …, 2.0 s" (20 heads), trained on labels derived from the time to collision | 20 scores; evaluated with one fixed head (`head_2.0`), chosen on validation |
| **AdaLEA** (Suzuki et al., CVPR 2018) | A loss that penalises late anticipation more, pushing the model to warn earlier | 1 risk score |
| **RiskProp** (Zou et al., CVPR 2026) | Trained on ordered snippet sequences: only the first and the collision snippet are labelled, and FFR + AMC propagate risk to the rest | 1 risk score |

## Main results (official Nexar test set, 1,344 clips)

| Method | mAP public | mAP private | mAP all |
|---|---|---|---|
| TOP (single fixed head) | 0.769 ± 0.016 | 0.811 ± 0.027 | 0.785 ± 0.021 |
| AdaLEA | 0.700 ± 0.012 | 0.795 ± 0.019 | 0.742 ± 0.011 |
| RiskProp | 0.756 ± 0.023 | 0.811 ± 0.004 | 0.781 ± 0.015 |

Scored with Nexar's unmodified `evaluate_submission.py`. The TOP − RiskProp difference is +0.004 mAP, with a 95% CI of [−0.018, 0.028].

---

## Repository structure

```
README.md             this file
RQ2_THEORY.md         RQ2 background: losses, metric formulas, predictions made before running RQ2
pipeline/             ORIGINAL code, kept unchanged (one file per notebook cell)
  cell09              download Nexar from Hugging Face
  cell10, cell30      pre-compute 5-frame windows (TOP, AdaLEA, evaluation) / 12-snippet sequences (RiskProp)
  cell11b-cell16      TOP: dataset, model, training, original evaluation
  cell20-cell24       AdaLEA: model, loss, dataset, training, original evaluation
  cell31-cell35       RiskProp: model, losses (FFR + AMC), dataset, training, original evaluation
reeval/               corrected evaluation and the RQ2 / RQ3 experiments (new files; pipeline/ is never edited)
  re01-re03           rebuild validation cache, score the 24 original checkpoints, lock the evaluation rules
  re04-re05           official test inference + official scorer, bootstrap CIs
  re06-re08           RQ2: ablation training (copy of cell34 with FFR/AMC switches), inference, analysis
  re09-re10           RQ3 sensitivity: fixed lag 0.5 / 1.5 s training, evaluation
riskprop_full/        RQ1b: run the authors' public RiskProp code on Nexar with the same protocol
results/
  RESULTS.md, RESULTS_RQ1.md, RESULTS_RQ2.md, RESULTS_RQ3.md   write-up
  reeval_corrected/   locked evaluation rules, per-run validation metrics, CIs (RQ1 / RQ3)
  official_test/      submissions of all 21 runs, official scorer output, test CIs
  rq2/                RQ2 logs, analysis, raw predictions and risk curves
  rq3_sensitivity/    RQ3 sensitivity logs, analysis, raw predictions and risk curves
  repro/              split manifest, video-id map, environment
  top/, adalea/, riskprop/   original training logs (evaluation there is superseded by reeval_corrected/)
```

---

## Reproducing the results

<details>
<summary><b>Setup: environment and data</b></summary>

- Python 3.12, PyTorch 2.11 (CUDA 12.8), torchvision 0.26, decord, scikit-learn, pandas, datasets. The full list is in [`results/repro/environment_vast_rtx4080.txt`](results/repro/environment_vast_rtx4080.txt). One RTX 4080 (16 GB) was used on vast.ai.
- The data comes from Hugging Face `nexar-ai/nexar_collision_prediction`. Accept the dataset terms, then `export HF_TOKEN=<your token>`.
- Split: 1,200 training / 300 validation videos (stratified, seed 42). The validation split is recorded in [`results/repro/split_manifest_seed42.json`](results/repro/split_manifest_seed42.json).
- The official test set has 1,344 clips. Nexar's `solution.csv` and `evaluate_submission.py` are downloaded by `re04` and are **not** redistributed here.
- Paths are set at the top of each script (Kaggle `/kaggle/...` or vast.ai `/workspace/CPV301/...`).
</details>

<details>
<summary><b>Steps: run from the repository root; every training script takes its seed from an environment variable (42, 43, 44)</b></summary>

```bash
# 1. data and caches
python pipeline/cell09_prepare_hf_data.py
python pipeline/cell10_precache_5f.py
python pipeline/cell30_precache_riskprop.py

# 2. train (50 epochs; best_* = lowest validation loss, latest_* = final epoch)
TOP_SEED=42      python pipeline/cell15_train_cached_5f.py
ADALEA_SEED=42   python pipeline/cell23_train_adalea_cached.py
RISKPROP_SEED=42 RISKPROP_PAIRING=random python pipeline/cell34_train_riskprop_cached.py   # RiskProp (RQ1, RQ2-D)
RISKPROP_SEED=42 RISKPROP_PAIRING=fixed  python pipeline/cell34_train_riskprop_cached.py   # fixed lag 1.0 s (RQ3)

# 3. corrected evaluation, rules locked on validation
python reeval/re01_precache_val.py
python reeval/re02_infer_val.py --ckpt-dir <folder with the 24 checkpoints>
python reeval/re03_analyze_rq.py --preds-dir reeval_out/preds_val --results-dir results --out-dir reeval_out/analysis

# 4. RQ2 ablation
bash reeval/run_rq2_all.sh && python reeval/re07_infer_rq2.py && python reeval/re08_analyze_rq2.py

# 5. official test (once, after the rules are locked)
python reeval/re04_infer_test.py --ckpt-dir <checkpoints> --locked-config reeval_out/analysis/locked_config.json --with-rq2
python reeval/re05_eval_test.py

# 6. RQ3 sensitivity
bash reeval/run_rq3_sens.sh && python reeval/re10_rq3_sensitivity_eval.py
```
`pipeline/cell16`, `cell24` and `cell35` are the original evaluation scripts. Their oracle TOP head and their choice of checkpoint by validation loss are replaced by step 3.
</details>

<details>
<summary><b>Checkpoints (not in the repository, about 242 MB each; available from the team)</b></summary>

| Set | Files | Naming |
|---|---|---|
| RQ1 / RQ3 | 24 | `{best,latest}_cached_seedS.pth` (TOP), `{best,latest}_adalea_seedS.pth`, `{best,latest}_riskprop_{random,fixed}_seedS.pth` |
| RQ2 | 18 | `{best,latest}_riskprop_random_seedS_{neither,ffronly,amconly}.pth` |
| RQ3 sensitivity | 12 | `{best,latest}_riskprop_fixed_seedS_tau{0.5,1.5}.pth` |

The checkpoint actually used for each run is listed in `results/reeval_corrected/locked_config.json`, `results/rq2/analysis/rq2_chosen_checkpoints.json` and `results/rq3_sensitivity/analysis/rq3_sensitivity_chosen_checkpoints.json`.
</details>

<details>
<summary><b>Compute</b></summary>

Training took about 31.5 GPU-hours in total:

| Runs | GPU-hours |
|---|---|
| TOP | 0.8 |
| AdaLEA | 1.0 |
| RiskProp random + fixed | 10.7 |
| RQ2 | 11.4 |
| RQ3 sensitivity | 7.7 |

Official-test inference for all 21 runs takes about 5 minutes.
</details>

---

## Differences from the original papers

The training setup is much smaller than in the published papers:
- RiskProp: batch of 2 videos on one GPU, λ1 = λ2 = 0.5, and 12 snippets 0.5 s apart. The authors' public code uses 30 clips 0.1 s apart and λ = 1.5 / 1.1.
- All three methods share one training budget.

Absolute scores are therefore lower than published, and comparisons with the papers use effect directions only. Every deviation is listed in the "Deviations and limitations" section of each results file. RQ1b re-trains RiskProp with the authors' own code to measure how much of the gap comes from these differences.
