# CPV301: Collision anticipation on Nexar, a controlled reproduction of RiskProp

Research-based learning project, CPV301 (Computer Vision), FPT University.
Team: Dương Duy Lợi, Trần Minh Huy, Hoàng Duy Lưu.

We compare three collision-anticipation methods (**TOP**, **AdaLEA**, **RiskProp**) on the [Nexar Collision Prediction](https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction) dataset under one controlled protocol. We then take RiskProp apart: which of its two losses matter, and whether its pairing of time steps can be improved.

**Status:** RQ1, RQ2 and RQ3 (original scope) are complete. RQ3 v4 — Progressive RiskProp / CPS — is also complete (validation + official test), result: **not supported**. RQ1b (RiskProp re-trained with its authors' own code) is in progress.

---

## ⚠️ Naming: the RQ numbers below are the *original* numbering, kept on purpose

The file and folder names in this repo (`RESULTS_RQ2.md`, `RESULTS_RQ3.md`, `rq2/`, `rq3_sensitivity/`, `re01`–`re21`, …) all use the **original 2026-09 numbering** (RQ1 / RQ2 = FFR+AMC ablation / RQ3 = pairing). They are **never renamed**, even where the write-up below renumbers things, for two concrete reasons:
1. `pipeline/cellNN_*.py` files `import` each other by exact filename (e.g. `cell34_train_riskprop_cached.py` does `from cell33_riskprop_dataset import ...`), and the README itself documents them as "kept unchanged" from the original notebook.
2. `reeval/re11`–`re21` are invoked by exact filename from three shell scripts (`run_progress_rq3.sh`, `run_rq2_all.sh`, `run_rq3_sens.sh`), **and** those filenames are baked into `results/rq3_progress/LOCK.json` — an intentionally **immutable**, hash-sealed record (`LOCK.json.sha256`) that `re20_lock_progress.py` refuses to overwrite once written. Renaming any of `re11`–`re21` would make that sealed provenance record self-inconsistent, which defeats the point of having a pre-registered, tamper-evident lock behind the RQ3 result.

So: **if you want a different RQ numbering for the paper, change the prose/labels, never the file names.** The current team renumbering (used going forward) is:

| In the paper, call it… | …which is the file/folder named… | Status |
|---|---|---|
| RQ1 | `RESULTS_RQ1.md` (unchanged) | complete |
| RQ2 | `RESULTS_RQ3.md`, `rq3_sensitivity/` (**formerly "RQ3"** — pairing, fixed vs random lag) | complete |
| Supporting analysis (not its own RQ) | `RESULTS_RQ2.md`, `rq2/` (**formerly "RQ2"** — FFR/AMC 2×2 ablation) | complete, demoted to supporting evidence |
| RQ3 (new) | `results/rq3_progress/` + `reeval/re11`–`re21` (Progressive RiskProp / CPS, v4) | complete — **not supported** |
| RQ1b | `riskprop_full/` | in progress |

This table already existed, half-written, inside [`reeval/README_PROGRESS_RQ3.md`](reeval/README_PROGRESS_RQ3.md#rq-numbering-v4-scope) — it just never made it up to this root README, which is why it's easy to lose track of which name means what. Read that file for the full v4 file-by-file index.

---

## Key findings

| | Finding (original numbering) |
|---|---|
| **RQ1** | On the official Nexar test set, **TOP (0.785 mAP) and RiskProp (0.781) are statistically indistinguishable, and both beat AdaLEA (0.742)** by about 0.04 mAP. |
| **RQ2** | RiskProp's accuracy comes from its **future-frame regularization (FFR)** (+0.09 to +0.10 mAP). Its **monotonic constraint (AMC)** does not change accuracy; it only makes risk curves more monotone. |
| **RQ3** | Pairing time steps at a **fixed lag** instead of random offsets does not change accuracy, slightly reduces temporal violations, and does **not** reliably reduce seed-to-seed variance. |
| **RQ3 v4 (Progressive RiskProp / CPS)** | Adding Continuous Progress Supervision on top of RiskProp changes the shape of the risk curve (steeper rise, better negative suppression) but **does not** improve accuracy (AP/mAP), on validation or on the official test set. See [`reeval/README_PROGRESS_RQ3.md`](reeval/README_PROGRESS_RQ3.md) and [`results/rq3_progress/eval_val/summary_rq3_progress.md`](results/rq3_progress/eval_val/summary_rq3_progress.md). |

All numbers are means over 3 training seeds, with 95% bootstrap confidence intervals in the result files.

---

## Where to find what

| I want to… | Open |
|---|---|
| Read all results in one place | [`results/RESULTS.md`](results/RESULTS.md) |
| Read one research question in detail | [`RESULTS_RQ1.md`](results/RESULTS_RQ1.md), [`RESULTS_RQ2.md`](results/RESULTS_RQ2.md) (FFR/AMC, supporting evidence), [`RESULTS_RQ3.md`](results/RESULTS_RQ3.md) (pairing, "RQ2" in the paper) |
| Read the new RQ3 (Progressive RiskProp / CPS, v4) write-up | [`reeval/README_PROGRESS_RQ3.md`](reeval/README_PROGRESS_RQ3.md) (run guide + file index), [`results/rq3_progress/eval_val/summary_rq3_progress.md`](results/rq3_progress/eval_val/summary_rq3_progress.md) (validation), and [`results/rq3_progress/test/test_summary.md`](results/rq3_progress/test/test_summary.md) (official test) |
| See the raw numbers (CSV, predictions, logs) | the sub-folders of [`results/`](results/) (table below) |
| Read the original training and evaluation code | [`pipeline/`](pipeline/) — never modified, see the naming note above |
| Re-run the corrected evaluation, old RQ2/RQ3 | [`reeval/`](reeval/) scripts `re01`–`re10` |
| Re-run the new RQ3 (Progressive RiskProp / CPS) pipeline | [`reeval/run_progress_rq3.sh`](reeval/run_progress_rq3.sh), scripts `re11`–`re21` |
| Run RQ1b (RiskProp with the authors' code) | [`riskprop_full/README_RQ1B.md`](riskprop_full/README_RQ1B.md) |
| Look up the RQ2 losses and metric formulas | [`RQ2_THEORY.md`](RQ2_THEORY.md) |

---

## Research questions

| Paper RQ | Question | Evaluated on |
|---|---|---|
| RQ1 | How do AdaLEA, TOP and RiskProp compare in official test mAP and early warning at a false-alarm rate ≤ 0.1? | Official Nexar test (mAP); internal validation (early warning) |
| RQ2 | Does pairing time steps at a fixed lag (τ = 1.0 s; sensitivity at 0.5 and 1.5 s) help compared with random offsets? | Internal validation |
| Supporting analysis | How much do RiskProp's two losses, FFR and AMC, contribute, alone and together? (2×2 ablation) | Internal validation |
| RQ3 | Does Progressive RiskProp with continuous progress supervision improve early anticipation while maintaining predictive accuracy? | Internal validation; official Nexar test |
| RQ1b | Is the project's RiskProp weaker because of how it was re-implemented? (the authors' own code, same protocol) | In progress |
## The three methods

All three use the same SlowOnly-R50 backbone, the same 5-frame causal windows (10 fps, 224×224), the same train/validation split and the same optimiser budget (SGD, lr 0.01, 50 epochs, seeds 42/43/44). At evaluation each method scores the same single window.

| Method | Idea | Output per window |
|---|---|---|
| **TOP** | Predicts "collision within 0.1 s, 0.2 s, …, 2.0 s" (20 heads), trained on labels derived from the time to collision | 20 scores; evaluated with one fixed head (`head_2.0`), chosen on validation |
| **AdaLEA** (Suzuki et al., CVPR 2018) | A loss that penalises late anticipation more, pushing the model to warn earlier | 1 risk score |
| **RiskProp** (Zou et al., CVPR 2026) | Trained on ordered snippet sequences: only the first and the collision snippet are labelled, and FFR + AMC propagate risk to the rest | 1 risk score |

## Main results (original RQ1; official Nexar test set, 1,344 clips)

| Method | mAP public | mAP private | mAP all |
|---|---|---|---|
| TOP (single fixed head) | 0.769 ± 0.016 | 0.811 ± 0.027 | 0.785 ± 0.021 |
| AdaLEA | 0.700 ± 0.012 | 0.795 ± 0.019 | 0.742 ± 0.011 |
| RiskProp | 0.756 ± 0.023 | 0.811 ± 0.004 | 0.781 ± 0.015 |

Scored with Nexar's unmodified `evaluate_submission.py`. The TOP − RiskProp difference is +0.004 mAP, with a 95% CI of [−0.018, 0.028].

---

## Repository structure

All names below are historical and **locked** (see the naming note above) — nothing here gets renamed; this tree is the map, not a to-do list.

```
README.md             this file
RQ2_THEORY.md         RQ2 (old numbering) background: losses, metric formulas, predictions made before running it
pipeline/             ORIGINAL code, kept unchanged (one file per notebook cell) -- never touched by reeval/
  cell09              download Nexar from Hugging Face
  cell10, cell30      pre-compute 5-frame windows (TOP, AdaLEA, evaluation) / 12-snippet sequences (RiskProp)
  cell11b-cell16      TOP: dataset, model, training, original evaluation
  cell20-cell24       AdaLEA: model, loss, dataset, training, original evaluation
  cell31-cell35       RiskProp: model, losses (FFR + AMC), dataset, training, original evaluation

reeval/               corrected evaluation + all experiments after the original pipeline/ (pipeline/ is never edited)
  -- old scope (RQ1 corrected eval, old RQ2 ablation, old RQ3 pairing) --
  re01-re03           rebuild validation cache, score the 24 original checkpoints, lock the evaluation rules
  re04-re05           official NEXAR TEST SET inference + official scorer, bootstrap CIs
  re06-re08           old "RQ2": FFR/AMC ablation training (copy of cell34 with switches), inference, analysis
  re09-re10           old "RQ3": pairing sensitivity, fixed lag 0.5 / 1.5 s training, evaluation
  -- v4 scope: new RQ3, Progressive RiskProp / CPS -- see reeval/README_PROGRESS_RQ3.md for the full index --
  re11-re13           audit progress data, build per-snippet sidecar (tau/mask), CPS target + loss definitions
  re14-re15           train B/P/S conditions, evaluate on validation (bootstrap CI, pre-registered decision)
  re16, re21          SOFTWARE regression tests (CPU, pytest-style) -- NOT the Nexar test set, despite the name
  re17-re18           pin HF data revision, preflight checks (env/CUDA/weights identity)
  re19                official NEXAR TEST SET: freeze checkpoints -> infer -> score, once
  re20                write the immutable LOCK.json (gate G5); refuses to overwrite with different content
  run_progress_rq3.sh fail-fast runner for the whole v4 stage sequence (prepare -> ... -> test)
  release_v4/         changelog, release manifest, acceptance report + CPU test logs for the v4 patch itself

riskprop_full/        RQ1b: run the authors' public RiskProp code on Nexar with the same protocol (in progress)

results/
  RESULTS.md, RESULTS_RQ1.md, RESULTS_RQ2.md, RESULTS_RQ3.md   old-numbering write-ups (see naming note above
                                                                 for what RQ2.md/RQ3.md are called in the paper)
  RESULTS_SUPPLEMENTARY_ensemble_calibration.md   ensemble/calibration side-analysis, not a numbered RQ
  reeval_corrected/   locked evaluation rules, per-run validation metrics, CIs (RQ1 / old RQ3 pairing)
  official_test/      submissions of all 21 original-scope runs, official scorer output, test CIs
  rq2/                old "RQ2" (FFR/AMC) logs, analysis, raw predictions and risk curves
  rq3_sensitivity/    old "RQ3" pairing-sensitivity logs, analysis, raw predictions and risk curves
  rq3_progress/       v4 / new-RQ3 (CPS): LOCK.json, all 9 runs, validation eval, official test -- see
                       rq3_progress/eval_val/summary_rq3_progress.md and rq3_progress/test/test_summary.md
  repro/              split manifest, video-id map, environment
  top/, adalea/, riskprop/   original training logs (evaluation there is superseded by reeval_corrected/)
```

**"test" means two different things in this repo** — the official held-out Nexar test set (1,344 clips, scored once: `re04`/`re05`, `re19`, `results/official_test/`, `results/rq3_progress/test/`) vs. a software/regression test (`re16`, `re21`, `release_v4/logs/*_tests.txt`, `unit_tests.log` — plain pytest-style checks that run on CPU with no real data). Same English word, unrelated meaning; when in doubt, "official test" = the Nexar leaderboard set, "unit/regression test" = code correctness checks.

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
