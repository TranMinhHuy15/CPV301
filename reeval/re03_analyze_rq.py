"""
re03 -- Offline analysis of the validation predictions written by re02.
Runs on any machine (numpy / pandas / scikit-learn only, no GPU, no video).

What it does, in this fixed order:
  1. LEGACY CHECK  -- recompute every run's metrics with the ORIGINAL
     protocol (best checkpoint; TOP = matched head, Video AUC = max head)
     and compare with the eval_results_*.txt files already on GitHub.
     If they do not match, the validation split / cache is not the same as
     before and the script stops (old tables would not be comparable).
  2. TOP HEAD RULE -- choose ONE fixed head rule for TOP (used for every
     clip, no horizon information), by mean validation mAP over TOP's six
     prediction sets (3 seeds x best/latest). The oracle "matched head" is
     never a candidate; it is only reported in the appendix.
  3. CHECKPOINT RULE -- same rule for all 12 runs:
        per_run (default): each run keeps whichever of {best, latest} has
                           the higher validation mAP (tie -> best)
        global           : one checkpoint type for every run, the one with
                           the higher mean validation mAP over all runs
  4. TABLES -- per-run metrics, RQ1 (TOP / AdaLEA / RiskProp random) and
     RQ3 (RiskProp random vs fixed) as mean +/- SD over 3 seeds, per-seed
     ranking, best-run appendix, sensitivity (all-best vs all-latest),
     oracle-head appendix for TOP.
  5. PAIRED BOOTSTRAP 95% CI -- stratified by label, same resampled video
     indices for every run; metric averaged over the 3 seeds per method.
  6. LOCK FILE -- locked_config.json with the chosen head and checkpoint
     rules. Create it BEFORE opening solution.csv /
     time_to_accident_test_map.csv; the official test run must reuse it.

All metric code (AP, low-FAR mAUC / Recall / threshold / actual FAR, mTTA,
coverage) is a line-by-line copy of cell16 / cell24 / cell35.

Usage:
    python reeval/re03_analyze_rq.py \
        --preds-dir reeval_out/preds_val --results-dir results \
        --out-dir reeval_out/analysis
"""
import argparse
import glob
import hashlib
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, auc, roc_auc_score, roc_curve

LEADS = [0.5, 1.0, 1.5]
SEEDS = [42, 43, 44]
KINDS = ["best", "latest"]
TARGET_FAR = 0.1
# TOP head h answers "collision within (h+1)*0.1 s?" (cell13 / cell11b)
LEAD_TO_HIDX = {0.5: 4, 1.0: 9, 1.5: 14}
TOP_HEAD_RULES = ["head_0.5", "head_1.0", "head_1.5", "head_2.0", "mean3", "max"]
METHODS = {  # method -> run-name prefix
    "TOP": "top",
    "AdaLEA": "adalea",
    "RiskProp random": "riskprop_random",
    "RiskProp fixed": "riskprop_fixed",
}
RQ1_METHODS = ["TOP", "AdaLEA", "RiskProp random"]
RQ3_METHODS = ["RiskProp random", "RiskProp fixed"]
LEGACY_FILES = {
    "top": ["top/eval_results_rq1_top5f_seed{s}.txt"],
    "adalea": ["adalea/eval_results_rq1_adalea5f_seed{s}.txt"],
    "riskprop_random": ["riskprop/eval_results_rq1_riskprop_random_seed{s}.txt"],
    "riskprop_fixed": ["riskprop/eval_results_rq3_riskprop_fixed_seed{s}.txt",
                       "riskprop/eval_results_rq1_riskprop_fixed_seed{s}.txt"],
}


# ----------------------------------------------------------------------------
# Metrics -- copied from cell16/cell24/cell35
# ----------------------------------------------------------------------------
def low_far_metrics(targets, scores, target_far=TARGET_FAR):
    fpr, tpr, thr = roc_curve(targets, scores)
    mask = fpr <= target_far
    if np.any(mask):
        fpr_low = np.append(fpr[mask], target_far)
        tpr_low = np.append(tpr[mask], np.interp(target_far, fpr, tpr))
        mauc = auc(fpr_low, tpr_low) / target_far
        idx = np.where(fpr <= target_far)[0][-1]
        return mauc, tpr[idx], thr[idx], fpr[idx]
    return 0.0, 0.0, 1.0, 0.0


def full_metrics(lead_scores, targets, video_auc_scores):
    """lead_scores: list of 3 arrays (N,) aligned with LEADS."""
    res = {}
    per = {}
    for l, s in zip(LEADS, lead_scores):
        ap = average_precision_score(targets, s)
        mauc, rec, thr, far = low_far_metrics(targets, s)
        per[l] = dict(AP=ap, mAUC01=mauc, Recall01=rec, Thresh01=thr, FAR01=far)
    res["mAP"] = float(np.mean([per[l]["AP"] for l in LEADS]))
    for l in LEADS:
        res[f"AP@{l}s"] = float(per[l]["AP"])
    res["VideoAUC"] = float(roc_auc_score(targets, video_auc_scores))
    for l in LEADS:
        res[f"mAUC01@{l}s"] = float(per[l]["mAUC01"])
    res["mAUC01"] = float(np.mean([per[l]["mAUC01"] for l in LEADS]))
    for l in LEADS:
        res[f"Recall@{l}s"] = float(per[l]["Recall01"])
    for l in LEADS:
        res[f"ActualFAR@{l}s"] = float(per[l]["FAR01"])
    for l in LEADS:
        res[f"Threshold@{l}s"] = float(per[l]["Thresh01"])
    pos_idx = np.where(targets == 1)[0]
    tta = []
    for i in pos_idx:
        det = [l for l, s in zip(LEADS, lead_scores) if s[i] >= per[l]["Thresh01"]]
        if det:
            tta.append(max(det))
    mtta = float(np.mean(tta)) if tta else 0.0
    cov = len(tta) / max(1, len(pos_idx))
    res["mTTA_detected"] = mtta
    res["Coverage"] = float(cov)
    res["mTTA_all"] = mtta * cov
    return res


def fast_map_mauc(lead_scores, targets):
    """Only mAP and mAUC@0.1 (used inside the bootstrap loop)."""
    aps, maucs = [], []
    for s in lead_scores:
        aps.append(average_precision_score(targets, s))
        maucs.append(low_far_metrics(targets, s)[0])
    return float(np.mean(aps)), float(np.mean(maucs))


# ----------------------------------------------------------------------------
# Scores
# ----------------------------------------------------------------------------
def top_rule_scores(probs, rule):
    """probs: (N, 20) for one lead -> (N,) under a FIXED rule (no horizon info)."""
    if rule == "head_0.5":
        return probs[:, 4]
    if rule == "head_1.0":
        return probs[:, 9]
    if rule == "head_1.5":
        return probs[:, 14]
    if rule == "head_2.0":
        return probs[:, 19]
    if rule == "mean3":
        return probs[:, [4, 9, 14]].mean(axis=1)
    if rule == "max":
        return probs.max(axis=1)
    raise ValueError(rule)


def lead_scores_for(pred, top_rule=None, oracle=False):
    """Return (list of 3 per-lead score arrays, video-AUC score array)."""
    sc = pred["scores"]
    if sc.ndim == 3:  # TOP (3, N, 20)
        if oracle:
            ls = [sc[i][:, LEAD_TO_HIDX[l]] for i, l in enumerate(LEADS)]
            return ls, sc[1].max(axis=1)          # legacy cell16 definition
        ls = [top_rule_scores(sc[i], top_rule) for i in range(len(LEADS))]
        return ls, ls[1]
    ls = [sc[i] for i in range(len(LEADS))]
    return ls, ls[1]


# ----------------------------------------------------------------------------
# IO
# ----------------------------------------------------------------------------
def load_preds(preds_dir):
    preds = {}
    for p in sorted(glob.glob(os.path.join(preds_dir, "*.npz"))):
        name = os.path.basename(p)[:-4]
        run, kind = name.rsplit("_", 1)
        z = np.load(p, allow_pickle=False)
        d = {k: z[k] for k in z.files}
        meta_p = p[:-4] + ".json"
        d["meta"] = json.load(open(meta_p)) if os.path.exists(meta_p) else {}
        preds[(run, kind)] = d
    return preds


def parse_eval_txt(path):
    txt = open(path).read()
    pats = {
        "mAP": r"Proposal mAP:\s*([0-9.]+)",
        "AP@0.5s": r"AP@0\.5s:\s*([0-9.]+)",
        "AP@1.0s": r"AP@1\.0s:\s*([0-9.]+)",
        "AP@1.5s": r"AP@1\.5s:\s*([0-9.]+)",
        "VideoAUC": r"Video AUC:\s*([0-9.]+)",
        "Recall@0.5s": r"Recall@FAR<=0\.1@0\.5s:\s*([0-9.]+)",
        "Recall@1.0s": r"Recall@FAR<=0\.1@1\.0s:\s*([0-9.]+)",
        "Recall@1.5s": r"Recall@FAR<=0\.1@1\.5s:\s*([0-9.]+)",
        "mAUC01": r"mAUC01_mean=([0-9.]+)",
        "Coverage": r"coverage=([0-9.]+)\s*$",
    }
    out = {}
    for k, p in pats.items():
        m = re.search(p, txt, flags=re.M)
        if m:
            out[k] = float(m.group(1))
    return out


def fmt(mean, sd, pct=False):
    if pct:
        return f"{mean*100:.1f} +/- {sd*100:.1f}%"
    return f"{mean:.4f} +/- {sd:.4f}"


def md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for r in df.astype(object).itertuples(index=False):  # keep ints as ints
        lines.append("| " + " | ".join(str(v) for v in r) + " |")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds-dir", default="reeval_out/preds_val")
    ap.add_argument("--results-dir", default="results",
                    help="repo results/ folder with the old eval_results_*.txt")
    ap.add_argument("--out-dir", default="reeval_out/analysis")
    ap.add_argument("--ckpt-rule", choices=["per_run", "global"], default="per_run")
    ap.add_argument("--legacy-tol", type=float, default=0.0015)
    ap.add_argument("--allow-legacy-mismatch", action="store_true")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--boot-seed", type=int, default=12345)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    preds = load_preds(args.preds_dir)
    if not preds:
        sys.exit(f"No .npz found in {args.preds_dir}")
    runs = sorted({r for r, _ in preds})
    ref = next(iter(preds.values()))
    vid_ids, targets = ref["vid_ids"], ref["targets"].astype(int)
    for (r, k), d in preds.items():
        assert np.array_equal(d["vid_ids"], vid_ids), f"{r}/{k}: video order differs"
        assert np.array_equal(d["targets"].astype(int), targets)
    n_pos, n = int(targets.sum()), len(targets)
    print(f"Loaded {len(preds)} prediction sets | {n} val videos, {n_pos} positive")
    expected = {f"{p}_seed{s}" for p in METHODS.values() for s in SEEDS}
    miss = sorted((r, k) for r in expected for k in KINDS if (r, k) not in preds)
    if miss:
        print(f"WARNING: missing prediction sets: {miss}")

    def method_of(run):
        return run.rsplit("_seed", 1)[0]

    report = [f"# Re-evaluation on the internal validation split\n",
              f"Generated {time.strftime('%Y-%m-%d %H:%M')} | {n} videos "
              f"({n_pos} positive, {n-n_pos} negative)\n"]

    # ------------------------------------------------------------------ 1
    print("\n[1] Legacy check (original protocol, best checkpoints)")
    rows, worst = [], 0.0
    for run in runs:
        if (run, "best") not in preds:
            continue
        pref, seed = method_of(run), int(run.rsplit("_seed", 1)[1])
        txt = None
        for tmpl in LEGACY_FILES.get(pref, []):
            p = os.path.join(args.results_dir, tmpl.format(s=seed))
            if os.path.exists(p):
                txt = p
                break
        ls, va = lead_scores_for(preds[(run, "best")], oracle=True)
        now = full_metrics(ls, targets, va)
        if txt is None:
            rows.append(dict(run=run, file="(not found)", max_abs_diff=np.nan, ok=False))
            continue
        old = parse_eval_txt(txt)
        diffs = {k: abs(now[k] - v) for k, v in old.items() if k in now}
        md = max(diffs.values()) if diffs else np.nan
        worst = max(worst, md if not np.isnan(md) else 0)
        ok = bool(diffs) and md <= args.legacy_tol
        rows.append(dict(run=run, file=os.path.relpath(txt, args.results_dir),
                         old_mAP=old.get("mAP"), new_mAP=round(now["mAP"], 4),
                         max_abs_diff=round(md, 5), ok=ok))
        print(f"  {run:24s} old mAP={old.get('mAP')}  new={now['mAP']:.4f}  "
              f"max|diff|={md:.5f}  {'OK' if ok else 'MISMATCH'}")
    legacy = pd.DataFrame(rows)
    legacy.to_csv(os.path.join(args.out_dir, "legacy_check.csv"), index=False)
    legacy_pass = bool(len(legacy)) and bool(legacy["ok"].all())
    report.append("## 1. Legacy check\n")
    report.append("Original protocol (best checkpoint, TOP matched head) vs the "
                  "eval_results_*.txt already on GitHub. "
                  f"Tolerance {args.legacy_tol}.\n")
    report.append(md_table(legacy) + "\n")
    report.append(f"**Verdict: {'PASS -- validation split reproduced' if legacy_pass else 'FAIL'}**\n")
    if not legacy_pass:
        print("\n!! LEGACY CHECK FAILED: the rebuilt validation set does not "
              "reproduce the old results.")
        if not args.allow_legacy_mismatch:
            open(os.path.join(args.out_dir, "summary.md"), "w").write("\n".join(report))
            sys.exit(2)
        print("   continuing because --allow-legacy-mismatch was given")

    # ------------------------------------------------------------------ 2
    print("\n[2] TOP fixed head rule (candidates exclude the oracle matched head)")
    head_rows = []
    for rule in TOP_HEAD_RULES:
        vals = []
        for s in SEEDS:
            for k in KINDS:
                key = (f"top_seed{s}", k)
                if key in preds:
                    ls, va = lead_scores_for(preds[key], top_rule=rule)
                    vals.append(full_metrics(ls, targets, va)["mAP"])
        head_rows.append(dict(rule=rule, n_sets=len(vals),
                              mean_val_mAP=round(float(np.mean(vals)), 4) if vals else np.nan))
        print(f"  {rule:9s} mean val mAP = {np.mean(vals):.4f} over {len(vals)} sets")
    head_df = pd.DataFrame(head_rows)
    top_rule = head_df.sort_values("mean_val_mAP", ascending=False, kind="stable").iloc[0]["rule"]
    print(f"  -> TOP head rule locked: {top_rule}")
    report.append("## 2. TOP fixed head rule\n")
    report.append("Chosen by mean validation mAP over TOP's 6 prediction sets "
                  "(3 seeds x best/latest). The oracle matched head is not a candidate.\n")
    report.append(md_table(head_df) + f"\n\n**Locked: `{top_rule}`**\n")

    # ------------------------------------------------------------------ metrics for every set
    all_metrics = {}
    for key, d in preds.items():
        ls, va = lead_scores_for(d, top_rule=top_rule)
        m = full_metrics(ls, targets, va)
        m["epoch"] = d["meta"].get("ckpt_epoch")
        all_metrics[key] = m

    # ------------------------------------------------------------------ 3
    print(f"\n[3] Checkpoint rule: {args.ckpt_rule}")
    chosen = {}
    if args.ckpt_rule == "global":
        means = {k: np.mean([all_metrics[(r, k)]["mAP"] for r in runs if (r, k) in all_metrics])
                 for k in KINDS}
        g = "best" if means["best"] >= means["latest"] else "latest"
        chosen = {r: g for r in runs}
        print(f"  mean val mAP best={means['best']:.4f} latest={means['latest']:.4f} -> {g}")
    else:
        for r in runs:
            b = all_metrics.get((r, "best"), {}).get("mAP", -1)
            l = all_metrics.get((r, "latest"), {}).get("mAP", -1)
            chosen[r] = "best" if b >= l else "latest"
    ck_rows = []
    for r in runs:
        ck_rows.append(dict(run=r,
                            best_mAP=round(all_metrics.get((r, "best"), {}).get("mAP", np.nan), 4),
                            best_epoch=all_metrics.get((r, "best"), {}).get("epoch"),
                            latest_mAP=round(all_metrics.get((r, "latest"), {}).get("mAP", np.nan), 4),
                            latest_epoch=all_metrics.get((r, "latest"), {}).get("epoch"),
                            chosen=chosen[r]))
        print(f"  {r:24s} best={ck_rows[-1]['best_mAP']}  latest={ck_rows[-1]['latest_mAP']}"
              f"  -> {chosen[r]}")
    ck_df = pd.DataFrame(ck_rows)
    ck_df.to_csv(os.path.join(args.out_dir, "checkpoint_choice.csv"), index=False)
    report.append(f"## 3. Checkpoint rule: `{args.ckpt_rule}`\n")
    report.append("per_run = each run keeps the higher-val-mAP checkpoint of "
                  "{best-val-loss, last epoch}; global = one checkpoint type for all runs. "
                  "Same rule for all 12 runs.\n")
    report.append(md_table(ck_df) + "\n")

    # ------------------------------------------------------------------ 4
    metric_cols = ["mAP", "AP@0.5s", "AP@1.0s", "AP@1.5s", "VideoAUC",
                   "mAUC01@0.5s", "mAUC01@1.0s", "mAUC01@1.5s", "mAUC01",
                   "Recall@0.5s", "Recall@1.0s", "Recall@1.5s",
                   "ActualFAR@0.5s", "ActualFAR@1.0s", "ActualFAR@1.5s",
                   "Threshold@0.5s", "Threshold@1.0s", "Threshold@1.5s",
                   "mTTA_detected", "mTTA_all", "Coverage"]
    per_run = []
    for r in runs:
        k = chosen[r]
        if (r, k) not in all_metrics:
            continue
        row = dict(run=r, method=[m for m, p in METHODS.items() if p == method_of(r)][0],
                   seed=int(r.rsplit("_seed", 1)[1]), checkpoint=k,
                   epoch=all_metrics[(r, k)]["epoch"])
        row.update({c: all_metrics[(r, k)][c] for c in metric_cols})
        per_run.append(row)
    per_run_df = pd.DataFrame(per_run)
    per_run_df.to_csv(os.path.join(args.out_dir, "per_run_metrics.csv"), index=False)

    def mean_sd_table(df, methods, cols):
        out = []
        for m in methods:
            sub = df[df["method"] == m]
            row = {"Model": m, "n_seeds": len(sub)}
            for c in cols:
                v = sub[c].astype(float)
                row[c] = fmt(v.mean(), v.std(ddof=1) if len(v) > 1 else 0.0,
                             pct=(c == "Coverage"))
            out.append(row)
        return pd.DataFrame(out)

    main_cols = ["mAP", "AP@0.5s", "AP@1.0s", "AP@1.5s", "mAUC01", "VideoAUC"]
    warn_cols = ["Recall@0.5s", "Recall@1.0s", "Recall@1.5s", "ActualFAR@0.5s",
                 "ActualFAR@1.0s", "ActualFAR@1.5s", "mTTA_detected", "mTTA_all", "Coverage"]
    rq1_a = mean_sd_table(per_run_df, RQ1_METHODS, main_cols)
    rq1_b = mean_sd_table(per_run_df, RQ1_METHODS, warn_cols)
    rq3_a = mean_sd_table(per_run_df, RQ3_METHODS, main_cols)
    rq3_b = mean_sd_table(per_run_df, RQ3_METHODS, warn_cols)
    pd.concat([rq1_a.set_index("Model"), rq1_b.set_index("Model").drop(columns="n_seeds")],
              axis=1).to_csv(os.path.join(args.out_dir, "rq1_mean_sd.csv"))
    pd.concat([rq3_a.set_index("Model"), rq3_b.set_index("Model").drop(columns="n_seeds")],
              axis=1).to_csv(os.path.join(args.out_dir, "rq3_mean_sd.csv"))

    report.append(f"## 4. RQ1 main table (mean +/- SD over 3 seeds; TOP head `{top_rule}`)\n")
    report.append(md_table(rq1_a) + "\n\n" + md_table(rq1_b) + "\n")
    report.append("Note: each negative clip has one cached window, so false alarms "
                  "per clip at the FAR<=0.1 threshold equal ActualFAR.\n")

    # per-seed ranking
    rank_rows = []
    for s in SEEDS:
        sub = per_run_df[(per_run_df["seed"] == s) & per_run_df["method"].isin(RQ1_METHODS)]
        sub = sub.sort_values("mAP", ascending=False)
        rank_rows.append({"seed": s, **{f"#{i+1}": f"{r.method} {r.mAP:.4f}"
                                        for i, r in enumerate(sub.itertuples())}})
    rank_df = pd.DataFrame(rank_rows)
    report.append("### Per-seed ranking by val mAP (RQ1)\n")
    report.append(md_table(rank_df) + "\n")

    # RQ3
    report.append("## 5. RQ3 table (mean +/- SD over 3 seeds)\n")
    report.append(md_table(rq3_a) + "\n\n" + md_table(rq3_b) + "\n")
    pair_rows = []
    for s in SEEDS:
        a = per_run_df[(per_run_df["method"] == "RiskProp fixed") & (per_run_df["seed"] == s)]
        b = per_run_df[(per_run_df["method"] == "RiskProp random") & (per_run_df["seed"] == s)]
        if len(a) and len(b):
            pair_rows.append(dict(seed=s, fixed_mAP=round(a.mAP.iloc[0], 4),
                                  random_mAP=round(b.mAP.iloc[0], 4),
                                  diff=round(a.mAP.iloc[0] - b.mAP.iloc[0], 4)))
    if pair_rows:
        pdf = pd.DataFrame(pair_rows)
        report.append("### Paired seeds, fixed - random (mAP)\n")
        report.append(md_table(pdf) + f"\n\nmean diff = {pdf['diff'].mean():+.4f}, "
                      f"SD of diff = {pdf['diff'].std(ddof=1):.4f}\n")

    # appendices
    report.append("## Appendix A. Best run per method (descriptive only)\n")
    best_rows = []
    for m in METHODS:
        sub = per_run_df[per_run_df["method"] == m]
        if len(sub):
            r = sub.sort_values("mAP", ascending=False).iloc[0]
            best_rows.append({"Model": m, "seed": int(r.seed), "ckpt": r.checkpoint,
                              **{c: round(float(r[c]), 4) for c in main_cols},
                              "Coverage": f"{r.Coverage*100:.1f}%"})
    report.append(md_table(pd.DataFrame(best_rows)) + "\n")

    report.append("## Appendix B. Sensitivity: all-best vs all-latest checkpoints (mAP mean +/- SD)\n")
    sens = []
    for m, p in METHODS.items():
        row = {"Model": m}
        for k in KINDS:
            v = [all_metrics[(f"{p}_seed{s}", k)]["mAP"] for s in SEEDS
                 if (f"{p}_seed{s}", k) in all_metrics]
            row[f"all-{k}"] = fmt(np.mean(v), np.std(v, ddof=1)) if len(v) > 1 else "n/a"
        sens.append(row)
    report.append(md_table(pd.DataFrame(sens)) + "\n")

    report.append("## Appendix C. TOP with oracle matched head (uses horizon labels; NOT comparable)\n")
    orc = []
    for s in SEEDS:
        key = (f"top_seed{s}", chosen.get(f"top_seed{s}", "best"))
        if key in preds:
            ls, va = lead_scores_for(preds[key], oracle=True)
            orc.append(full_metrics(ls, targets, va))
    if orc:
        report.append(md_table(pd.DataFrame([{
            "Model": "TOP oracle head",
            **{c: fmt(np.mean([o[c] for o in orc]), np.std([o[c] for o in orc], ddof=1))
               for c in main_cols}}])) + "\n")

    # ------------------------------------------------------------------ 5
    print(f"\n[5] Paired stratified bootstrap, B={args.n_boot}")
    rng = np.random.default_rng(args.boot_seed)
    pos_idx, neg_idx = np.where(targets == 1)[0], np.where(targets == 0)[0]
    method_runs = {m: [f"{p}_seed{s}" for s in SEEDS
                       if (f"{p}_seed{s}", chosen.get(f"{p}_seed{s}", "best")) in preds]
                   for m, p in METHODS.items()}
    run_scores = {r: lead_scores_for(preds[(r, chosen[r])], top_rule=top_rule)[0]
                  for rs in method_runs.values() for r in rs}
    boot = {m: {"mAP": [], "mAUC01": []} for m in METHODS}
    t0 = time.time()
    for b in range(args.n_boot):
        idx = np.concatenate([rng.choice(pos_idx, len(pos_idx), replace=True),
                              rng.choice(neg_idx, len(neg_idx), replace=True)])
        y = targets[idx]
        for m, rs in method_runs.items():
            vals = [fast_map_mauc([s[idx] for s in run_scores[r]], y) for r in rs]
            boot[m]["mAP"].append(np.mean([v[0] for v in vals]))
            boot[m]["mAUC01"].append(np.mean([v[1] for v in vals]))
        if (b + 1) % 500 == 0:
            print(f"  {b+1}/{args.n_boot} ({time.time()-t0:.0f}s)")
    boot = {m: {k: np.array(v) for k, v in d.items()} for m, d in boot.items()}

    def point(m, metric):
        rs = method_runs[m]
        return float(np.mean([all_metrics[(r, chosen[r])][metric] for r in rs]))

    ci_rows = []
    comparisons = [("TOP", "AdaLEA"), ("TOP", "RiskProp random"),
                   ("AdaLEA", "RiskProp random"), ("RiskProp fixed", "RiskProp random")]
    for metric in ["mAP", "mAUC01"]:
        for m in METHODS:
            lo, hi = np.percentile(boot[m][metric], [2.5, 97.5])
            ci_rows.append(dict(comparison=m, metric=metric, estimate=round(point(m, metric), 4),
                                ci_low=round(lo, 4), ci_high=round(hi, 4), verdict=""))
        for a, b_ in comparisons:
            d = boot[a][metric] - boot[b_][metric]
            lo, hi = np.percentile(d, [2.5, 97.5])
            est = point(a, metric) - point(b_, metric)
            verdict = "inconclusive (CI includes 0)" if lo <= 0 <= hi else (
                f"{a} higher" if lo > 0 else f"{b_} higher")
            ci_rows.append(dict(comparison=f"{a} - {b_}", metric=metric,
                                estimate=round(est, 4), ci_low=round(lo, 4),
                                ci_high=round(hi, 4), verdict=verdict))
    ci_df = pd.DataFrame(ci_rows)
    ci_df.to_csv(os.path.join(args.out_dir, "bootstrap_ci.csv"), index=False)
    report.append(f"## 6. Paired bootstrap 95% CI (B={args.n_boot}, stratified by label, "
                  f"same video resamples for every run; metric averaged over 3 seeds)\n")
    report.append(md_table(ci_df) + "\n")
    report.append("Bootstrap resamples validation videos only; seed-to-seed variation "
                  "is reported separately as SD.\n")

    # ------------------------------------------------------------------ 6
    h = hashlib.sha256()
    for p in sorted(glob.glob(os.path.join(args.preds_dir, "*.npz"))):
        h.update(open(p, "rb").read())
    lock = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "legacy_check_passed": legacy_pass,
        "top_head_rule": top_rule,
        "top_head_rule_criterion": "max mean val mAP over 3 seeds x {best, latest}; "
                                   "oracle matched head excluded",
        "checkpoint_rule": args.ckpt_rule,
        "checkpoint_rule_criterion": "val mAP (mean AP over 0.5/1.0/1.5 s), "
                                     "candidates {best-val-loss, last-epoch}, tie -> best",
        "chosen_checkpoints": {r: {"kind": chosen[r],
                                   "file": preds[(r, chosen[r])]["meta"].get("ckpt_file"),
                                   "epoch": preds[(r, chosen[r])]["meta"].get("ckpt_epoch")}
                               for r in runs if (r, chosen[r]) in preds},
        "val_predictions_sha256": h.hexdigest(),
        "note": "Created before opening solution.csv / time_to_accident_test_map.csv. "
                "The official test run must use exactly these rules.",
    }
    with open(os.path.join(args.out_dir, "locked_config.json"), "w") as f:
        json.dump(lock, f, indent=2)
    report.append("## 7. Locked configuration\n")
    report.append("```json\n" + json.dumps(lock, indent=2) + "\n```\n")
    with open(os.path.join(args.out_dir, "summary.md"), "w") as f:
        f.write("\n".join(report))
    print(f"\nWrote {args.out_dir}/summary.md, locked_config.json and CSV tables.")


if __name__ == "__main__":
    main()
