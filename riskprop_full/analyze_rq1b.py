"""
RQ1b -- compare RiskProp-full with TOP, AdaLEA and the project's RiskProp,
using exactly the RQ1 protocol and code.

Stage "val": validation metrics (re03.full_metrics) and the paired,
label-stratified bootstrap of re03 (B = 2000, seed 12345; the resampled videos
do not depend on which methods are compared, so the old RQ1 contrasts must come
out identical -- this is checked against results/reeval_corrected/).
Inputs:
  * RiskProp-full predictions from infer_rq1b.py --stage val
  * TOP / AdaLEA predictions from reeval/re02_infer_val.py (their 6 chosen checkpoints)
  * RiskProp (random) predictions already on GitHub: results/rq2/raw/preds_val_rq2
  Checkpoints of the old runs follow results/reeval_corrected/locked_config.json.

Stage "test": runs reeval/re05_eval_test.py unchanged on the 21 old submissions
(results/official_test/submissions) plus the 3 RiskProp-full submissions, with
RiskProp-full added to the method list and to the comparisons. The old numbers
must be reproduced exactly (checked against results/official_test/).

Usage:
    python riskprop_full/analyze_rq1b.py --stage val
    python riskprop_full/analyze_rq1b.py --stage test
"""
import argparse
import glob
import json
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd

SEEDS = [42, 43, 44]
OLD = {"TOP": "top", "AdaLEA": "adalea", "RiskProp random": "riskprop_random"}
NEW = {"RiskProp-full": "riskprop_full"}
COMPARISONS = [("RiskProp-full", "TOP"), ("RiskProp-full", "AdaLEA"),
               ("RiskProp-full", "RiskProp random"),
               ("TOP", "AdaLEA"), ("TOP", "RiskProp random"), ("AdaLEA", "RiskProp random")]
MAIN_COLS = ["mAP", "AP@0.5s", "AP@1.0s", "AP@1.5s", "mAUC01", "VideoAUC",
             "Recall@0.5s", "Recall@1.0s", "Recall@1.5s", "mTTA_detected", "mTTA_all", "Coverage"]


def fmt(v):
    v = np.asarray(v, float)
    return f"{v.mean():.4f} +/- {v.std(ddof=1):.4f}" if len(v) > 1 else f"{v.mean():.4f}"


def md(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for _, r in df.iterrows()]
    return "\n".join(out)


def stage_val(args):
    sys.path.insert(0, os.path.join(args.repo_dir, "reeval"))
    import re03_analyze_rq as r3

    locked = json.load(open(args.locked_config))
    top_rule = locked["top_head_rule"]
    kinds = {r: v["kind"] for r, v in locked["chosen_checkpoints"].items()}
    new_ch = json.load(open(args.rq1b_chosen))["chosen"]
    kinds.update({r: v["kind"] for r, v in new_ch.items()})

    preds = {}
    for d in args.old_preds + [args.new_preds]:
        preds.update(r3.load_preds(d))

    methods = {**OLD, **NEW}
    method_runs, missing = {}, []
    for m, p in methods.items():
        rs = []
        for s in SEEDS:
            r = f"{p}_seed{s}"
            if r in kinds and (r, kinds[r]) in preds:
                rs.append(r)
            else:
                missing.append(f"{r} ({kinds.get(r, '?')})")
        if rs:
            method_runs[m] = rs
    if missing:
        print("WARNING: missing predictions:", ", ".join(missing))

    ref_ids = targets = None
    for rs in method_runs.values():
        for r in rs:
            p = preds[(r, kinds[r])]
            if ref_ids is None:
                ref_ids, targets = p["vid_ids"], p["targets"].astype(int)
            if not np.array_equal(p["vid_ids"], ref_ids):
                sys.exit(f"video order of {r} differs from the others -- stop")

    rows, run_scores = [], {}
    for m, rs in method_runs.items():
        for r in rs:
            ls, vauc = r3.lead_scores_for(preds[(r, kinds[r])], top_rule=top_rule)
            run_scores[r] = ls
            rows.append({"method": m, "run": r, "checkpoint": kinds[r], **r3.full_metrics(ls, targets, vauc)})
    per_run = pd.DataFrame(rows)
    os.makedirs(args.out_dir, exist_ok=True)
    per_run.to_csv(os.path.join(args.out_dir, "rq1b_val_per_run.csv"), index=False)

    # consistency with the published RQ1 numbers
    checks = []
    if os.path.exists(args.ref_per_run):
        ref = pd.read_csv(args.ref_per_run)
        d, n = 0.0, 0
        for _, row in per_run[per_run.method.isin(OLD)].iterrows():
            rr = ref[(ref.run == row.run) & (ref.checkpoint == row.checkpoint)]
            if len(rr):
                n += 1
                d = max(d, abs(rr.mAP.iloc[0] - row.mAP), abs(rr.mAUC01.iloc[0] - row.mAUC01))
        checks.append(f"old runs vs results/reeval_corrected/per_run_metrics.csv: " + (
            f"max |diff| = {d:.2e} over {n} runs -> {'PASS' if d < 1e-6 else 'FAIL'}" if n
            else "no old run available -> NOT CHECKED"))

    # bootstrap (identical to re03)
    rng = np.random.default_rng(args.boot_seed)
    pos_idx, neg_idx = np.where(targets == 1)[0], np.where(targets == 0)[0]
    boot = {m: {"mAP": [], "mAUC01": []} for m in method_runs}
    t0 = time.time()
    for b in range(args.n_boot):
        idx = np.concatenate([rng.choice(pos_idx, len(pos_idx), replace=True),
                              rng.choice(neg_idx, len(neg_idx), replace=True)])
        y = targets[idx]
        for m, rs in method_runs.items():
            vals = [r3.fast_map_mauc([s[idx] for s in run_scores[r]], y) for r in rs]
            boot[m]["mAP"].append(np.mean([v[0] for v in vals]))
            boot[m]["mAUC01"].append(np.mean([v[1] for v in vals]))
        if (b + 1) % 500 == 0:
            print(f"  bootstrap {b + 1}/{args.n_boot} ({time.time() - t0:.0f}s)")
    boot = {m: {k: np.array(v) for k, v in d.items()} for m, d in boot.items()}

    def point(m, k):
        return float(per_run[per_run.method == m][k].mean())

    ci = []
    for k in ("mAP", "mAUC01"):
        for a, b_ in COMPARISONS:
            if a not in boot or b_ not in boot:
                continue
            dd = boot[a][k] - boot[b_][k]
            lo, hi = np.percentile(dd, [2.5, 97.5])
            ci.append(dict(comparison=f"{a} - {b_}", metric=k, estimate=round(point(a, k) - point(b_, k), 4),
                           ci_low=round(lo, 4), ci_high=round(hi, 4),
                           verdict="inconclusive (CI includes 0)" if lo <= 0 <= hi
                           else (f"{a} higher" if lo > 0 else f"{b_} higher")))
    ci_df = pd.DataFrame(ci)
    ci_df.to_csv(os.path.join(args.out_dir, "rq1b_val_bootstrap_ci.csv"), index=False)
    if os.path.exists(args.ref_ci):
        ref = pd.read_csv(args.ref_ci)
        d, n = 0.0, 0
        for _, row in ci_df[~ci_df.comparison.str.contains("RiskProp-full")].iterrows():
            rr = ref[(ref.comparison == row.comparison) & (ref.metric == row.metric)]
            if len(rr):
                n += 1
                d = max(d, *(abs(rr[c].iloc[0] - row[c]) for c in ("estimate", "ci_low", "ci_high")))
        checks.append(f"old contrasts vs results/reeval_corrected/bootstrap_ci.csv: " + (
            f"max |diff| = {d:.4f} over {n} rows -> {'PASS' if d < 1e-4 else 'FAIL'}" if n
            else "no old contrast available (TOP/AdaLEA predictions missing?) -> NOT CHECKED"))

    # appendix: authors' checkpoint rule
    app = []
    for r, info in new_ch.items():
        key = (r, "authorbest")
        if key in preds:
            ls, vauc = r3.lead_scores_for(preds[key])
            app.append({"run": r, "epoch": info.get("authorbest_epoch"), **r3.full_metrics(ls, targets, vauc)})
    app_df = pd.DataFrame(app)

    tab = pd.DataFrame([{"Method": m, "n": len(rs), **{c: fmt(per_run[per_run.method == m][c]) for c in MAIN_COLS}}
                        for m, rs in method_runs.items()])
    ch_tab = pd.DataFrame([{"run": r, "chosen": v["kind"], "epoch": v["epoch"], "best-loss epoch": v["best_loss_epoch"],
                            "val mAP best-loss": round(v["val_mAP_best"], 4),
                            "val mAP final": round(v["val_mAP_latest"], 4),
                            "authors' best-mAUC@ epoch": v.get("authorbest_epoch")} for r, v in new_ch.items()])
    rep = [f"# RQ1b -- RiskProp-full on the internal validation split ({time.strftime('%Y-%m-%d %H:%M')})\n",
           f"300 videos ({int(targets.sum())} positive). TOP head rule {top_rule}; old checkpoints from "
           "locked_config.json; RiskProp-full checkpoints from rq1b_chosen_checkpoints.json (same rule).\n",
           "## Consistency checks\n", *[f"- {c}" for c in checks], "",
           "## RiskProp-full checkpoint choice\n", md(ch_tab), "",
           "## Mean +/- SD over seeds\n", md(tab), "",
           f"## Paired label-stratified bootstrap (B={args.n_boot}, seed {args.boot_seed})\n", md(ci_df), ""]
    if len(app_df):
        rep += ["## Appendix: authors' checkpoint rule (best mAUC@ on validation), validation only\n",
                md(pd.DataFrame([{"n": len(app_df), **{c: fmt(app_df[c]) for c in ("mAP", "mAUC01", "AP@0.5s",
                                                                                   "AP@1.0s", "AP@1.5s")}}])), ""]
    open(os.path.join(args.out_dir, "summary_rq1b_val.md"), "w").write("\n".join(rep) + "\n")
    print("\n".join(rep))


def stage_test(args):
    sys.path.insert(0, os.path.join(args.repo_dir, "reeval"))
    import re05_eval_test as r5

    comb = os.path.join(args.work_root, "test_infer_all")
    os.makedirs(comb, exist_ok=True)
    n_old = 0
    for p in glob.glob(os.path.join(args.old_submissions, "submission_*.csv")):
        shutil.copy(p, comb)
        n_old += 1
    new = glob.glob(os.path.join(args.new_submissions, "submission_riskprop_full_seed*.csv"))
    for p in new:
        shutil.copy(p, comb)
    print(f"{n_old} old + {len(new)} RiskProp-full submissions in {comb}")
    if len(new) != 3:
        print("WARNING: expected 3 RiskProp-full submissions")

    r5.METHODS = {"TOP": ("top", ""), "AdaLEA": ("adalea", ""),
                  "RiskProp random": ("riskprop_random", ""), "RiskProp-full": ("riskprop_full", "")}
    r5.RQ1_RQ3 = [(a, b) for a, b in COMPARISONS]
    r5.RQ2 = []
    sys.argv = ["re05_eval_test.py", "--infer-dir", comb, "--hf-local-raw", args.hf_local_raw,
                "--out-dir", args.out_dir, "--n-boot", str(args.n_boot), "--boot-seed", str(args.boot_seed)]
    r5.main()

    checks = []
    new_pr = pd.read_csv(os.path.join(args.out_dir, "test_per_run.csv"))
    if os.path.exists(args.ref_test_per_run):
        ref = pd.read_csv(args.ref_test_per_run)
        m = new_pr.merge(ref, on="run", suffixes=("", "_ref"))
        d = max((m.mAP_public_official - m.mAP_public_official_ref).abs().max(),
                (m.mAP_private_official - m.mAP_private_official_ref).abs().max())
        checks.append(f"old runs vs results/official_test/test_per_run.csv: max |diff| = {d:.2e} over "
                      f"{len(m)} runs -> {'PASS' if d < 1e-6 else 'FAIL'}")
    if os.path.exists(args.ref_test_ci):
        ref = pd.read_csv(args.ref_test_ci)
        new_ci = pd.read_csv(os.path.join(args.out_dir, "test_bootstrap_ci.csv"))
        m = new_ci.merge(ref, on=["comparison", "metric"], suffixes=("", "_ref"))
        d = max((m[c] - m[c + "_ref"]).abs().max() for c in ("estimate", "ci_low", "ci_high")) if len(m) else float("nan")
        checks.append(f"old contrasts vs results/official_test/test_bootstrap_ci.csv: max |diff| = {d:.4f} over "
                      f"{len(m)} rows -> {'PASS' if d < 1e-4 else 'FAIL'}")
    with open(os.path.join(args.out_dir, "test_summary.md"), "a") as f:
        f.write("\n## RQ1b consistency checks\n\n" + "\n".join(f"- {c}" for c in checks) + "\n")
    print("\n".join(checks))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["val", "test"], required=True)
    ap.add_argument("--repo-dir", default="/workspace/CPV301")
    ap.add_argument("--work-root", default="/workspace/CPV301/reeval_out/rq1b")
    ap.add_argument("--new-preds", default="/workspace/CPV301/reeval_out/rq1b/preds_val_rq1b")
    ap.add_argument("--rq1b-chosen", default="/workspace/CPV301/reeval_out/rq1b/rq1b_chosen_checkpoints.json")
    ap.add_argument("--old-preds", nargs="+", default=["/workspace/CPV301/reeval_out/preds_val",
                                                       "/workspace/CPV301/results/rq2/raw/preds_val_rq2"])
    ap.add_argument("--locked-config", default="/workspace/CPV301/results/reeval_corrected/locked_config.json")
    ap.add_argument("--ref-per-run", default="/workspace/CPV301/results/reeval_corrected/per_run_metrics.csv")
    ap.add_argument("--ref-ci", default="/workspace/CPV301/results/reeval_corrected/bootstrap_ci.csv")
    ap.add_argument("--old-submissions", default="/workspace/CPV301/results/official_test/submissions")
    ap.add_argument("--new-submissions", default="/workspace/CPV301/reeval_out/rq1b/test_infer_rq1b")
    ap.add_argument("--hf-local-raw", default="/workspace/CPV301/data/nexar_hf_raw")
    ap.add_argument("--ref-test-per-run", default="/workspace/CPV301/results/official_test/test_per_run.csv")
    ap.add_argument("--ref-test-ci", default="/workspace/CPV301/results/official_test/test_bootstrap_ci.csv")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--boot-seed", type=int, default=12345)
    args = ap.parse_args()
    if not args.out_dir:
        args.out_dir = os.path.join(args.work_root, "analysis_val" if args.stage == "val" else "analysis_test")
    (stage_val if args.stage == "val" else stage_test)(args)


if __name__ == "__main__":
    main()
