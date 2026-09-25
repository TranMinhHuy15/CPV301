# CPV301: Collision anticipation on Nexar, reproducing and extending RiskProp

Research-based learning project for CPV301 (Computer Vision), FPT University.
Team: Dương Duy Lợi, Trần Minh Huy, Hoàng Duy Lưu.

We compare three collision-anticipation methods (TOP, AdaLEA, RiskProp) on the [Nexar Collision Prediction](https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction) dataset. We then take RiskProp apart: which of its two losses matter, and whether its AMC loss works better with fixed-lag pairs.

## Research questions

| RQ | Question | Primary evidence |
|---|---|---|
| RQ1 | How do AdaLEA, TOP and RiskProp compare in official test mAP and, on internal validation, in early warning at FAR ≤ 0.1? | Official Nexar test set |
| RQ2 | How much do future-frame regularization (FFR) and random-pair adaptive monotonic constraint (AMC) contribute, individually and jointly? (2×2 ablation) | Internal validation |
| RQ3 | Does fixed-lag AMC pairing (τ = 1.0 s; sensitivity runs at 0.5 and 1.5 s) change accuracy, temporal consistency or seed stability compared with random offsets? | Internal validation |

## Main results

Full write-up: [`results/RESULTS.md`](results/RESULTS.md). Details per question: [`RESULTS_RQ1.md`](results/RESULTS_RQ1.md), [`RESULTS_RQ2.md`](results/RESULTS_RQ2.md), [`RESULTS_RQ3.md`](results/RESULTS_RQ3.md).

| Method (official Nexar test, 3 seeds) | mAP public | mAP private | mAP all |
|---|---|---|---|
| TOP (single fixed head) | 0.769 ± 0.016 | 0.811 ± 0.027 | 0.785 ± 0.021 |
| AdaLEA | 0.700 ± 0.012 | 0.795 ± 0.019 | 0.742 ± 0.011 |
| RiskProp | 0.756 ± 0.023 | 0.811 ± 0.004 | 0.781 ± 0.015 |

- **RQ1:** TOP and RiskProp both beat AdaLEA by about 0.04 mAP (95% CIs exclude 0), and TOP and RiskProp cannot be separated.
- **RQ2:** FFR gives essentially all of RiskProp's accuracy gain (+0.09 to +0.10 mAP). AMC does not change accuracy; it only makes risk curves more monotone, mainly when FFR is absent.
- **RQ3:** Fixed-lag pairing leaves accuracy unchanged and slightly reduces large pairwise violations. It does not reliably reduce seed-to-seed variance.

## Repository layout

```
cell09_prepare_hf_data.py        download Nexar from Hugging Face into a Kaggle-style layout
cell10_precache_5f.py            5-frame causal windows: train + val at 0.5/1.0/1.5 s (TOP, AdaLEA, all evaluation)
cell30_precache_riskprop.py      12-snippet ordered sequences for RiskProp training
cell11b / cell13 / cell15 / cell16    TOP: dataset, model, training, original evaluation
cell20 – cell24                  AdaLEA: model, loss, dataset, training, original evaluation
cell31 – cell35                  RiskProp: model, losses (FFR + AMC), dataset, training, original evaluation
RQ2_THEORY.md                    RQ2 background: losses, metric definitions, predictions
reeval/                          corrected evaluation, official test, RQ2 and RQ3-sensitivity pipeline
  re01_precache_val.py           rebuild the validation cache
  re02_infer_val.py              score the 24 original checkpoints on validation
  re03_analyze_rq.py             legacy check, lock TOP head + checkpoint rule, RQ1/RQ3 validation tables
  re04_infer_test.py             official test inference with the locked configuration
  re05_eval_test.py              official scorer + group-wise mAP/mAUC and bootstrap on the test set
  re06_train_rq2_ablation.py     copy of cell34 with FFR/AMC switches (RQ2)
  run_rq2_all.sh                 runs the 9 RQ2 trainings (A/B/C × 3 seeds)
  re07_infer_rq2.py              RQ2 validation predictions, checkpoint choice, dense risk curves
  re08_analyze_rq2.py            RQ2 tables, contrasts and paper comparison
  re09_train_rq3_sensitivity.py  copy of cell34 with a configurable fixed lag τ (RQ3 sensitivity)
  run_rq3_sens.sh                runs the 6 sensitivity trainings (τ 0.5/1.5 × 3 seeds)
  re10_rq3_sensitivity_eval.py   RQ3 sensitivity predictions, dense curves and analysis
results/
  RESULTS.md, RESULTS_RQ1/2/3.md final write-up
  reeval_corrected/              locked configuration, per-run validation metrics, bootstrap CIs (RQ1/RQ3)
  official_test/                 submissions for all 21 runs, official scorer output, test summary and CIs
  rq2/                           RQ2 training logs, analysis, raw predictions and dense curves
  rq3_sensitivity/               RQ3 sensitivity training logs, analysis, raw predictions and dense curves
  repro/                         split manifest, video-id map, dense-sweep metadata, environment
  top/, adalea/, riskprop/       original training logs and evaluation files (superseded by reeval_corrected/)
```

The original pipeline files (`cell*.py`) are kept unchanged. New experiments use copies (`re06`, `re09`) that only expose the switches they need.

## Setup

**Environment.** Python 3.12, PyTorch 2.11 (CUDA 12.8), torchvision 0.26, decord 0.6, scikit-learn 1.9, pandas 3.0, numpy 2.5 and datasets 5.0. The backbone is SlowOnly-R50 from `torch.hub` (`facebookresearch/pytorchvideo`, pretrained). The full package list is in [`results/repro/environment_vast_rtx4080.txt`](results/repro/environment_vast_rtx4080.txt). RQ2, the RQ3 sensitivity runs and all re-evaluation ran on one RTX 4080 (16 GB) rented on vast.ai.

**Data.** The dataset is downloaded from Hugging Face (`nexar-ai/nexar_collision_prediction`), which needs a Hugging Face account that has accepted the dataset terms:

```bash
export HF_TOKEN=<your Hugging Face token>
```

- The 1,500 training videos are split 1,200 / 300 (stratified, seed 42). The 300-video validation split (150 positive / 150 negative) is recorded in [`results/repro/split_manifest_seed42.json`](results/repro/split_manifest_seed42.json).
- The official test set has 1,344 clips.
- Nexar's `solution.csv`, `time_to_accident_test_map.csv` and `evaluate_submission.py` are downloaded by `re04` and are **not** redistributed in this repository.

**Paths.** Each script sets its paths in a CONFIG block or in argparse defaults: Kaggle `/kaggle/...` or vast.ai `/workspace/CPV301/...`. Edit these for another machine.

## Reproducing the results

Run from the repository root. Every training script takes its seed from an environment variable; run it for seeds 42, 43 and 44.

**1. Data and caches**
```bash
python cell09_prepare_hf_data.py
python cell10_precache_5f.py
python cell30_precache_riskprop.py
```

**2. Train the RQ1 / RQ3 models** (50 epochs each; `best_*` = lowest validation loss, `latest_*` = final epoch)
```bash
TOP_SEED=42      python cell15_train_cached_5f.py                                  # TOP
ADALEA_SEED=42   python cell23_train_adalea_cached.py                              # AdaLEA
RISKPROP_SEED=42 RISKPROP_PAIRING=random python cell34_train_riskprop_cached.py    # RiskProp (RQ1; RQ2 condition D)
RISKPROP_SEED=42 RISKPROP_PAIRING=fixed  python cell34_train_riskprop_cached.py    # FixedLag τ = 1.0 s (RQ3)
```
The original evaluation scripts (`cell16`, `cell24`, `cell35`) produced `results/top|adalea|riskprop/`. Their TOP oracle head and validation-loss checkpoint choice are superseded by step 3.

**3. Corrected re-evaluation and locked configuration**
```bash
python reeval/re01_precache_val.py
python reeval/re02_infer_val.py --ckpt-dir <folder with the 24 checkpoints>
python reeval/re03_analyze_rq.py --preds-dir reeval_out/preds_val --results-dir results --out-dir reeval_out/analysis
```
This writes `locked_config.json` (TOP head `head_2.0`, per-run checkpoint choice). Published copy: `results/reeval_corrected/`.

**4. RQ2 ablation**
```bash
bash reeval/run_rq2_all.sh              # 9 runs; resumable, DRY_RUN=1 to list the commands
python reeval/re07_infer_rq2.py
python reeval/re08_analyze_rq2.py
```

**5. Official test set** (run once, after the configuration is locked)
```bash
python reeval/re04_infer_test.py --ckpt-dir <folder with the 24 checkpoints> \
    --locked-config reeval_out/analysis/locked_config.json --with-rq2
python reeval/re05_eval_test.py
```

**6. RQ3 sensitivity (τ = 0.5 / 1.5 s)**
```bash
bash reeval/run_rq3_sens.sh             # 6 runs
python reeval/re10_rq3_sensitivity_eval.py
```

By default, `re07` reads the original checkpoints from `/workspace/CPV301/ckpts` and the new ones from `/workspace/CPV301/outputs_riskprop`; both folders can be changed with command-line options. `re10` reads the new τ checkpoints from `outputs_riskprop` and reuses `re07`'s outputs for τ = 1.0 s and random-offset.

## Checkpoints

Checkpoints are not stored in the repository: `*.pth` is git-ignored, and each file is about 242 MB. There are 54 in total, and the team can share them on request.

| Set | Files | Naming |
|---|---|---|
| Original RQ1 / RQ3 | 24 | `{best,latest}_cached_seedS.pth` (TOP), `{best,latest}_adalea_seedS.pth`, `{best,latest}_riskprop_{random,fixed}_seedS.pth` |
| RQ2 ablation | 18 | `{best,latest}_riskprop_random_seedS_{neither,ffronly,amconly}.pth` |
| RQ3 sensitivity | 12 | `{best,latest}_riskprop_fixed_seedS_tau{0.5,1.5}.pth` |

The checkpoint each run actually uses is recorded in `results/reeval_corrected/locked_config.json`, `results/rq2/analysis/rq2_chosen_checkpoints.json` and `results/rq3_sensitivity/analysis/rq3_sensitivity_chosen_checkpoints.json`.

## Compute

Training took about 31.5 GPU-hours in total:
- TOP: 0.8 h
- AdaLEA: 1.0 h
- RiskProp random + fixed: 10.7 h
- RQ2: 11.4 h
- RQ3 sensitivity: 7.7 h

A RiskProp epoch takes about 90–93 s on an RTX 4080. Official-test inference for all 21 runs takes about 5 minutes.

## Main differences from the original papers

The training setup is much smaller than in the published papers:
- RiskProp: batch of 2 videos on one GPU (the paper uses 64 on eight GPUs)
- λ1 = λ2 = 0.5 instead of 1.5 / 1.1
- 12 causal 5-frame snippets per video instead of frame-level sequences
- a matched budget for all three methods (SGD, LR 0.01, 50 epochs)

Absolute scores are therefore lower than published, and comparisons with the papers are made on effect directions only. All deviations are listed in the "Deviations and limitations" section of each results file.
