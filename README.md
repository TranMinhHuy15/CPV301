# CPV301: Collision anticipation on Nexar, a controlled reproduction of RiskProp

Research-based learning project, CPV301 (Computer Vision), FPT University.
Team: Dương Duy Lợi, Trần Minh Huy, Hoàng Duy Lưu.

We compare three collision-anticipation methods (**TOP**, **AdaLEA**, **RiskProp**) on the [Nexar Collision Prediction](https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction) dataset under one controlled protocol. We then take RiskProp apart: which of its two losses matter, whether its pairing of time steps can be improved, and whether adding progress supervision (CPS) helps.

**Status:** RQ1, the FFR/AMC ablation and the pairing experiment are complete. The CPS experiment (Progressive RiskProp, v4) is also complete — result: **not supported**. RQ1b (RiskProp re-trained with its authors' own code) is in progress.

---

## Folders are named by topic, not by RQ number

The paper's RQ numbering changed once already (the pairing experiment was promoted to RQ2 and the FFR/AMC ablation was demoted to supporting evidence — team decision, 2026-10-05) and may change again. Code and result **folders** are named by what they test (`rq2_ablation_riskprop_ffr_amc/`, `pairing_sensitivity/`, `progress_cps_v4/`, …), so a future renumbering is a one-line edit to the table below, never a file rename. The `RESULTS_RQ*.md` **write-up files** still carry their original numbers, because that's what the first version of the paper cites them as — use this table to go from "the paper calls it RQ…" to "the file/folder is called…":

| Paper calls it… | Write-up | Code | Raw results |
|---|---|---|---|
| RQ1 (TOP vs AdaLEA vs RiskProp) | `results/RESULTS_RQ1.md` | `pipeline/` (`top_*`, `adalea_*`, `riskprop_*`) + `reeval/*.py` (flat files) | `results/official_test/`, `results/reeval_corrected/`, `results/top/`, `results/adalea/`, `results/riskprop/` |
| RQ2 (pairing, fixed vs random lag) | `results/RESULTS_RQ3.md` *(old filename, kept)* | `reeval/pairing_sensitivity/` | `results/pairing_sensitivity/` |
| Supporting evidence, not its own RQ (FFR/AMC 2×2 ablation) | `results/RESULTS_RQ2.md` *(old filename, kept)* | `reeval/rq2_ablation_riskprop_ffr_amc/` | `results/rq2_ablation_riskprop_ffr_amc/` |
| RQ3 (PRE-ACT-inspired CPS adaptation, v4) — **not supported** | [partner results page](results/progress_cps_v4/README.md); [validation](results/progress_cps_v4/eval_val/summary_rq3_progress.md); [official test](results/progress_cps_v4/test/test_summary.md) | `reeval/progress_cps_v4/` | `results/progress_cps_v4/` |
| RQ1b (authors' own code) | `rq1b_authors_riskprop/README_RQ1B.md` | `rq1b_authors_riskprop/` | `results/reeval_corrected/` (shares the RQ1 checkpoint rule) |

One more overloaded word: **"test"** means two different things here — the official held-out Nexar test set (1,344 clips, scored once: `reeval/infer_official_test.py`, `reeval/score_official_test.py`, `reeval/progress_cps_v4/official_test_protocol.py`, `results/official_test/`, `results/progress_cps_v4/test/`) vs. a software/regression test (`reeval/progress_cps_v4/unit_tests_v3.py`, `regression_tests_v4.py` — plain checks that run on CPU with no real data). When in doubt: "official test" = the Nexar leaderboard set, "unit/regression test" = code correctness checks.

`pipeline/` is the one folder that is genuinely never touched after the fact: it is imported by filename from several places (`reeval/`, `rq1b_authors_riskprop/`), so a file disappearing or changing behavior there would silently change every downstream result. `results/progress_cps_v4/log/LOCK.json` and `results/progress_cps_v4/log/LOCK.json.sha256` are immutable, hash-sealed snapshots of one specific run — they may mention script names as they existed on 2026-10, which can differ slightly from today's names below; that is expected archival behavior, not a bug.

---

## Key findings

| | Finding |
|---|---|
| **RQ1** | On the official Nexar test set, **TOP (0.785 mAP) and RiskProp (0.781) are statistically indistinguishable, and both beat AdaLEA (0.742)** by about 0.04 mAP. |
| **Ablation (FFR/AMC)** | RiskProp's accuracy comes from its **future-frame regularization (FFR)** (+0.09 to +0.10 mAP). Its **monotonic constraint (AMC)** does not change accuracy; it only makes risk curves more monotone. |
| **Pairing (RQ2)** | Pairing time steps at a **fixed lag** instead of random offsets does not change accuracy, slightly reduces temporal violations, and does **not** reliably reduce seed-to-seed variance. |
| **Progress supervision / CPS (RQ3, v4)** | Adding Continuous Progress Supervision on top of RiskProp changes the *shape* of the risk curve (steeper rise, better negative suppression) but **does not** improve accuracy (AP/mAP), on validation or on the official test. |

All numbers are means over 3 training seeds, with 95% bootstrap confidence intervals in the result files.

---

## Where to find what

| I want to… | Open |
|---|---|
| Read all results in one place | [`results/RESULTS.md`](results/RESULTS.md) |
| Read RQ1 / pairing / the ablation in detail | [`RESULTS_RQ1.md`](results/RESULTS_RQ1.md), [`RESULTS_RQ2.md`](results/RESULTS_RQ2.md) (ablation, supporting evidence), [`RESULTS_RQ3.md`](results/RESULTS_RQ3.md) (pairing, "RQ2" in the paper — see the table above) |
| Read the CPS experiment (Progressive RiskProp, v4, "RQ3" in the paper) | [partner results page](results/progress_cps_v4/README.md) (PRE-ACT-inspired, not a reproduction), [run guide](reeval/progress_cps_v4/README.md), [validation](results/progress_cps_v4/eval_val/summary_rq3_progress.md), [official test](results/progress_cps_v4/test/test_summary.md) |
| See the raw numbers (CSV, predictions, logs) | the sub-folders of [`results/`](results/) (table below) |
| Read the original training and evaluation code | [`pipeline/`](pipeline/) — never modified after the fact, see the note above |
| Re-run the corrected evaluation | [`reeval/`](reeval/) flat scripts (`precache_val.py` → `score_official_test.py`) |
| Re-run the ablation / pairing / CPS experiments | [`reeval/rq2_ablation_riskprop_ffr_amc/`](reeval/rq2_ablation_riskprop_ffr_amc/), [`reeval/pairing_sensitivity/`](reeval/pairing_sensitivity/), [`reeval/progress_cps_v4/`](reeval/progress_cps_v4/) (each has its own `run_all.sh`) |
| Run RQ1b (RiskProp with the authors' code) | [`rq1b_authors_riskprop/README_RQ1B.md`](rq1b_authors_riskprop/README_RQ1B.md) |
| Look up the ablation's losses and metric formulas | [`RQ2_THEORY.md`](RQ2_THEORY.md) |

---

## Research questions

| RQ (paper numbering) | Question | Evaluated on |
|---|---|---|
| RQ1 | How do AdaLEA, TOP and RiskProp compare in official test mAP and in early warning at a false-alarm rate ≤ 0.1? | Official Nexar test (mAP); internal validation (early warning) |
| RQ2 | Does pairing time steps at a fixed lag (τ = 1.0 s; sensitivity at 0.5 and 1.5 s) help compared with random offsets? | Internal validation |
| RQ3 | Does adding Continuous Progress Supervision (CPS) to RiskProp improve early collision anticipation while keeping accuracy? | Internal validation + official Nexar test |
| RQ1b | Is the project's RiskProp weaker because of how it was re-implemented? (the authors' own code, same protocol) | In progress |
| *Supporting evidence (not a numbered RQ)* | How much do RiskProp's two losses, FFR and AMC, contribute, alone and together? (2×2 ablation) | Internal validation |

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
README.md                      this file
RQ2_THEORY.md                  ablation background: losses, metric formulas, predictions made before running it

pipeline/                      ORIGINAL code, never modified after the fact (one file per notebook cell, renamed by topic)
  data_download_from_hf.py             download Nexar from Hugging Face
  data_precache_5frame_windows.py      pre-compute 5-frame windows (TOP, AdaLEA, evaluation)
  top_dataset.py, top_model.py, top_train.py, top_eval_original.py
  adalea_model.py, adalea_loss.py, adalea_dataset.py, adalea_train.py, adalea_eval_original.py
  riskprop_precache_snippet_sequence.py   pre-compute 12-snippet sequences (RiskProp)
  riskprop_model.py, riskprop_loss_ffr_amc.py, riskprop_dataset.py, riskprop_train.py, riskprop_eval_original.py

reeval/                        corrected evaluation + every experiment after the original pipeline (pipeline/ never edited)
  precache_val.py, infer_val_24ckpt.py, analyze_rq1_and_pairing.py,
  infer_official_test.py, score_official_test.py        shared RQ1 infra: rebuild val cache, score the
                                                          24 original checkpoints, lock the eval rules,
                                                          official-test inference + scorer
  rq2_ablation_riskprop_ffr_amc/              FFR/AMC 2x2 ablation (supporting evidence, not its own RQ)
    train_ablation.py, infer_val.py, analyze_ablation.py, run_all.sh
  pairing_sensitivity/           pairing experiment ("RQ2" in the paper): fixed vs random lag, tau sensitivity
    train_tau_sweep.py, eval_sensitivity.py, run_all.sh
  progress_cps_v4/               CPS experiment ("RQ3" in the paper) -- see progress_cps_v4/README.md for the full index
    common.py, audit_data.py, build_sidecar.py, supervision_targets.py,
    train_bps.py, eval_val.py, prepare_data_pinned.py, preflight.py,
    official_test_protocol.py, lock_config.py,
    unit_tests_v3.py, regression_tests_v4.py     SOFTWARE tests (CPU) -- not the Nexar test set
    run_all.sh, requirements.txt, README.md
    release/                      changelog, release manifest, acceptance report + CPU test logs (historical,
                                   point-in-time -- may reference pre-reorg script names)

rq1b_authors_riskprop/         RQ1b: run the authors' public RiskProp code on Nexar with the same protocol (in progress)
                                "riskprop_full" still appears INSIDE this folder as a variant label (the authors'
                                full 30-clip/0.1s protocol, as opposed to this project's reduced "random"/"fixed"
                                RiskProp) -- that is a result name, not a path, and is unrelated to this folder's name.

results/
  RESULTS.md, RESULTS_RQ1.md, RESULTS_RQ2.md, RESULTS_RQ3.md   write-ups (old numbering -- see the table above)
  RESULTS_SUPPLEMENTARY_ensemble_calibration.md                 ensemble/calibration side-analysis, not a numbered RQ
  reeval_corrected/             locked evaluation rules, per-run validation metrics, CIs (RQ1 + pairing)
  official_test/                submissions of all 21 original-scope runs, official scorer output, test CIs
  rq2_ablation_riskprop_ffr_amc/             FFR/AMC ablation logs, analysis, raw predictions and risk curves
  pairing_sensitivity/          pairing-experiment logs, analysis, raw predictions and risk curves
  progress_cps_v4/              Partner README, validation and official test at root; log/ contains the sealed lock, 9 run records and audit trail
  repro/                        split manifest, video-id map, environment
  top/, adalea/, riskprop/      original training logs (evaluation there is superseded by reeval_corrected/)
```

---

## Reproducing the results

<details>
<summary><b>Setup: environment and data</b></summary>

- Python 3.12, PyTorch 2.11 (CUDA 12.8), torchvision 0.26, decord, scikit-learn, pandas, datasets. The full list is in [`results/repro/environment_vast_rtx4080.txt`](results/repro/environment_vast_rtx4080.txt). One RTX 4080 (16 GB) was used on vast.ai.
- The data comes from Hugging Face `nexar-ai/nexar_collision_prediction`. Accept the dataset terms, then `export HF_TOKEN=<your token>`.
- Split: 1,200 training / 300 validation videos (stratified, seed 42). The validation split is recorded in [`results/repro/split_manifest_seed42.json`](results/repro/split_manifest_seed42.json).
- The official test set has 1,344 clips. Nexar's `solution.csv` and `evaluate_submission.py` are downloaded by `infer_official_test.py` and are **not** redistributed here.
- Paths are set at the top of each script (Kaggle `/kaggle/...` or vast.ai `/workspace/CPV301/...`).
</details>

<details>
<summary><b>Steps: run from the repository root; every training script takes its seed from an environment variable (42, 43, 44)</b></summary>

```bash
# 1. data and caches
python pipeline/data_download_from_hf.py
python pipeline/data_precache_5frame_windows.py
python pipeline/riskprop_precache_snippet_sequence.py

# 2. train (50 epochs; best_* = lowest validation loss, latest_* = final epoch)
TOP_SEED=42      python pipeline/top_train.py
ADALEA_SEED=42   python pipeline/adalea_train.py
RISKPROP_SEED=42 RISKPROP_PAIRING=random python pipeline/riskprop_train.py   # RiskProp (RQ1, ablation-D)
RISKPROP_SEED=42 RISKPROP_PAIRING=fixed  python pipeline/riskprop_train.py   # fixed lag 1.0 s (pairing / "RQ2")

# 3. corrected evaluation, rules locked on validation
python reeval/precache_val.py
python reeval/infer_val_24ckpt.py --ckpt-dir <folder with the 24 checkpoints>
python reeval/analyze_rq1_and_pairing.py --preds-dir reeval_out/preds_val --results-dir results --out-dir reeval_out/analysis

# 4. FFR/AMC ablation (supporting evidence)
bash reeval/rq2_ablation_riskprop_ffr_amc/run_all.sh && python reeval/rq2_ablation_riskprop_ffr_amc/infer_val.py && python reeval/rq2_ablation_riskprop_ffr_amc/analyze_ablation.py

# 5. official test (once, after the rules are locked)
python reeval/infer_official_test.py --ckpt-dir <checkpoints> --locked-config reeval_out/analysis/locked_config.json --with-rq2
python reeval/score_official_test.py

# 6. pairing sensitivity ("RQ2")
bash reeval/pairing_sensitivity/run_all.sh && python reeval/pairing_sensitivity/eval_sensitivity.py

# 7. CPS experiment ("RQ3", v4) -- see reeval/progress_cps_v4/README.md for the full, gated stage-by-stage guide
bash reeval/progress_cps_v4/run_all.sh prepare && bash reeval/progress_cps_v4/run_all.sh cache && ...
```
`pipeline/top_eval_original.py`, `adalea_eval_original.py` and `riskprop_eval_original.py` are the original evaluation scripts. Their oracle TOP head and their choice of checkpoint by validation loss are replaced by step 3.
</details>

<details>
<summary><b>Checkpoints (not in the repository, about 242 MB each; available from the team)</b></summary>

| Set | Files | Naming |
|---|---|---|
| RQ1 / pairing | 24 | `{best,latest}_cached_seedS.pth` (TOP), `{best,latest}_adalea_seedS.pth`, `{best,latest}_riskprop_{random,fixed}_seedS.pth` |
| Ablation (FFR/AMC) | 18 | `{best,latest}_riskprop_random_seedS_{neither,ffronly,amconly}.pth` |
| Pairing sensitivity | 12 | `{best,latest}_riskprop_fixed_seedS_tau{0.5,1.5}.pth` |

Checkpoint *filenames* on disk were not touched by the repo reorganization — only the code/result **folders** that reference them were renamed. The checkpoint actually used for each run is listed in `results/reeval_corrected/locked_config.json`, `results/rq2_ablation_riskprop_ffr_amc/analysis/rq2_chosen_checkpoints.json` and `results/pairing_sensitivity/analysis/rq3_sensitivity_chosen_checkpoints.json`.
</details>

<details>
<summary><b>Compute</b></summary>

Training took about 31.5 GPU-hours in total for the original scope (RQ1, ablation, pairing sensitivity), plus about 12.5 GPU-hours for the CPS experiment (v4, 9 runs):

| Runs | GPU-hours |
|---|---|
| TOP | 0.8 |
| AdaLEA | 1.0 |
| RiskProp random + fixed | 10.7 |
| FFR/AMC ablation | 11.4 |
| Pairing sensitivity | 7.7 |
| CPS experiment (v4, 9 runs) | ~12.5 |

Official-test inference for the original 21 runs takes about 5 minutes.
</details>

---

## Differences from the original papers

The training setup is much smaller than in the published papers:
- RiskProp: batch of 2 videos on one GPU, λ1 = λ2 = 0.5, and 12 snippets 0.5 s apart. The authors' public code uses 30 clips 0.1 s apart and λ = 1.5 / 1.1.
- All three methods share one training budget.

Absolute scores are therefore lower than published, and comparisons with the papers use effect directions only. Every deviation is listed in the "Deviations and limitations" section of each results file. RQ1b re-trains RiskProp with the authors' own code (in `rq1b_authors_riskprop/`) to measure how much of the gap comes from these differences.
