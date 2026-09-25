"""
re08 -- RQ2 analysis (CPU only). Reads what re07 wrote and reports the four
RQ2 outcome groups of RQ2_THEORY.md for the 2x2 ablation A/B/C/D, plus the
RQ3 temporal comparison (fixed F vs random D).

  Group 1 accuracy          mAP (PRIMARY), AP@0.5/1.0/1.5 s
  Group 2 low-FAR           mAUC@0.1, Recall@FAR<=0.1, ActualFAR, mTTA, Coverage
                            (same code as re03 / cell35)
  Group 3 temporal          on dense risk curves (30 windows, 0.1 s apart):
                            violation rate  = frac. of pairs i<j with a_i > a_j + EPS
                            downward step   = mean max(0, a_t - a_{t+1})
                            jitter          = mean |a_{t+1} - 2 a_t + a_{t-1}|
                            PRIMARY set = positive videos with a known event time;
                            negatives reported separately (false-alarm jitter).
                            EPS = 0.01 LOCKED here before any temporal result was
                            computed (RQ2_THEORY.md names the tolerance but gives no
                            value); EPS = 0 is reported as a sensitivity check only.
  Group 4 stability         SD over seeds 42/43/44 of every per-run metric

Contrasts (RQ2_THEORY.md Sec. 4), each with a paired, label-stratified
bootstrap 95% CI (B=2000, seed 12345 -- same as re03; the SAME resampled
videos are applied to every run in a replicate). CI containing 0 is reported
as "inconclusive".
    FFR | AMC absent  = B - A        AMC | FFR absent  = C - A
    FFR | AMC present = D - C        AMC | FFR present = D - B
    Interaction       = (D - C) - (B - A)
Each condition value M(.) is the mean over its 3 seeds.

Usage:
    python reeval/re08_analyze_rq2.py
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd

EPS = 0.01
SEEDS = [42, 43, 44]
RQ2_CONDS = ["A", "B", "C", "D"]
COND_NAME = {"A": "A neither", "B": "B FFR-only", "C": "C AMC-only (random)",
             "D": "D FFR+AMC (RiskProp random)", "F": "F FFR+AMC fixed-lag (RQ3)"}
RUN_TMPL = {"A": "riskprop_random_seed{s}_neither", "B": "riskprop_random_seed{s}_ffronly",
            "C": "riskprop_random_seed{s}_amconly", "D": "riskprop_random_seed{s}",
            "F": "riskprop_fixed_seed{s}"}
CONTRASTS = [  # name, {cond: coefficient}
    ("FFR effect, AMC absent (B-A)", {"B": 1, "A": -1}),
    ("AMC effect, FFR absent (C-A)", {"C": 1, "A": -1}),
    ("FFR effect, AMC present (D-C)", {"D": 1, "C": -1}),
    ("AMC effect, FFR present (D-B)", {"D": 1, "B": -1}),
    ("Interaction (D-C)-(B-A)", {"D": 1, "C": -1, "B": -1, "A": 1}),
]
MAIN_COLS = ["mAP", "AP@0.5s", "AP@1.0s", "AP@1.5s", "mAUC01", "VideoAUC",
             "Recall@0.5s", "Recall@1.0s", "Recall@1.5s",
             "ActualFAR@0.5s", "ActualFAR@1.0s", "ActualFAR@1.5s",
             "mTTA_detected", "mTTA_all", "Coverage",
             "viol_pos", "down_pos", "jitter_pos", "viol_pos_eps0",
             "jitter_neg", "down_neg", "meanscore_neg"]
BOOT_COLS = ["mAP", "mAUC01", "viol_pos", "down_pos", "jitter_pos"]
# RiskProp paper (Zou et al., CVPR 2026), Table 3, Nexar, "Only Collision Label"
# column = the labelling scheme RiskProp (and our cell32) actually uses.
# Exp. I = neither, II = AMC only, III = FFR only, IV = both. Evaluated by the
# authors on the official Nexar test set (1,344 clips), single run, lambda1=1.5,
# lambda2=1.1, LR 0.002, batch 64 -- so only DIRECTIONS are comparable with ours.
PAPER_T3 = {
    "A": {"mAP": 0.781, "mAUC01": 0.298},   # Exp. I
    "C": {"mAP": 0.785, "mAUC01": 0.302},   # Exp. II  (AMC only)
    "B": {"mAP": 0.854, "mAUC01": 0.453},   # Exp. III (FFR only)
    "D": {"mAP": 0.870, "mAUC01": 0.472},   # Exp. IV  (full)
}
HIGHER_IS_BETTER = {"mAP": True, "mAUC01": True, "viol_pos": False,
                    "down_pos": False, "jitter_pos": False}


def temporal_per_video(a, eps=EPS):
    """a: (N, T) scores ordered earliest -> latest. Returns 3 arrays (N,)."""
    T = a.shape[1]
    iu, ju = np.triu_indices(T, k=1)
    viol = (a[:, iu] > a[:, ju] + eps).mean(axis=1)
    down = np.maximum(0.0, a[:, :-1] - a[:, 1:]).mean(axis=1)
    jit = np.abs(a[:, 2:] - 2 * a[:, 1:-1] + a[:, :-2]).mean(axis=1)
    return viol, down, jit


def fmt(m, s):
    return f"{m:.4f} +/- {s:.4f}"


def md_table(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-dir", default="/workspace/CPV301")
    ap.add_argument("--out-root", default="/workspace/CPV301/reeval_out")
    ap.add_argument("--data-dir", default="/workspace/CPV301/data/nexar_kaggle_style")
    ap.add_argument("--train-log-dir", default="/workspace/CPV301/outputs_riskprop")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--boot-seed", type=int, default=12345)
    args = ap.parse_args()
    sys.path.insert(0, os.path.join(args.repo_dir, "reeval"))
    import re03_analyze_rq as r3

    preds_dir = os.path.join(args.out_root, "preds_val_rq2")
    dense_dir = os.path.join(args.out_root, "dense_rq2")
    out_dir = os.path.join(args.out_root, "analysis_rq2")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(args.out_root, "rq2_chosen_checkpoints.json")) as f:
        chosen = json.load(f)["chosen"]

    # event times -> exclude positives without t_event from the temporal set
    df = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    df["vid"] = df["id"].apply(lambda x: str(int(x)).zfill(5))
    toe = dict(zip(df["vid"], df["time_of_event"]))

    runs, lead_scores, dense = {}, {}, {}
    ref_ids = ref_t = None
    for cond in RQ2_CONDS + ["F"]:
        for s in SEEDS:
            run = RUN_TMPL[cond].format(s=s)
            if run not in chosen:
                print(f"  [MISSING] {run} (not in rq2_chosen_checkpoints.json)")
                continue
            kind = chosen[run]["kind"]
            z = np.load(os.path.join(preds_dir, f"{run}_{kind}.npz"))
            d = np.load(os.path.join(dense_dir, f"{run}.npz"))
            if ref_ids is None:
                ref_ids, ref_t = z["vid_ids"], z["targets"].astype(int)
            assert np.array_equal(z["vid_ids"], ref_ids) and np.array_equal(d["vid_ids"], ref_ids)
            assert np.array_equal(z["targets"].astype(int), ref_t)
            runs[run] = (cond, s, kind)
            lead_scores[run] = [z["scores"][i] for i in range(3)]
            dense[run] = d["scores"].astype(np.float64)
    targets = ref_t
    N = len(targets)
    pos_idx, neg_idx = np.where(targets == 1)[0], np.where(targets == 0)[0]
    tpos_mask = np.array([t == 1 and not pd.isna(toe.get(v)) for v, t in zip(ref_ids, targets)])
    print(f"{len(runs)} runs | {N} val videos ({len(pos_idx)} pos) | "
          f"temporal positives with t_event: {int(tpos_mask.sum())}")

    # ---------------- per-run metrics ----------------
    per_video = {}
    rows = []
    for run, (cond, s, kind) in runs.items():
        ls = lead_scores[run]
        m = r3.full_metrics(ls, targets, ls[1])
        v, dn, j = temporal_per_video(dense[run], EPS)
        v0, _, _ = temporal_per_video(dense[run], 0.0)
        per_video[run] = {"viol_pos": v, "down_pos": dn, "jitter_pos": j}
        m.update(viol_pos=float(v[tpos_mask].mean()), down_pos=float(dn[tpos_mask].mean()),
                 jitter_pos=float(j[tpos_mask].mean()), viol_pos_eps0=float(v0[tpos_mask].mean()),
                 jitter_neg=float(j[neg_idx].mean()), down_neg=float(dn[neg_idx].mean()),
                 meanscore_neg=float(dense[run][neg_idx].mean()))
        rows.append(dict(condition=cond, seed=s, run=run, checkpoint=kind, **m))
    per_run = pd.DataFrame(rows).sort_values(["condition", "seed"])
    per_run.to_csv(os.path.join(out_dir, "per_run_metrics_rq2.csv"), index=False)

    # ---------------- condition mean +/- SD ----------------
    agg = []
    for cond in RQ2_CONDS + ["F"]:
        sub = per_run[per_run.condition == cond]
        if sub.empty:
            continue
        r = {"Condition": COND_NAME[cond], "n_seeds": len(sub)}
        for c in MAIN_COLS:
            r[c] = fmt(sub[c].mean(), sub[c].std(ddof=1) if len(sub) > 1 else 0.0)
        agg.append(r)
    agg_df = pd.DataFrame(agg)
    agg_df.to_csv(os.path.join(out_dir, "rq2_mean_sd.csv"), index=False)

    # ---------------- bootstrap ----------------
    print(f"\nPaired stratified bootstrap, B={args.n_boot}")
    rng = np.random.default_rng(args.boot_seed)

    def cond_values(idx):
        """M(cond) for BOOT_COLS on resampled video indices idx."""
        tp = idx[tpos_mask[idx]]
        out = {}
        for cond in set(c for c, _, _ in runs.values()):
            vals = {c: [] for c in BOOT_COLS}
            for run, (c, s, k) in runs.items():
                if c != cond:
                    continue
                mp, mu = r3.fast_map_mauc([x[idx] for x in lead_scores[run]], targets[idx])
                vals["mAP"].append(mp)
                vals["mAUC01"].append(mu)
                for key in ("viol_pos", "down_pos", "jitter_pos"):
                    vals[key].append(per_video[run][key][tp].mean())
            out[cond] = {c: float(np.mean(v)) for c, v in vals.items()}
        return out

    full = cond_values(np.arange(N))
    comps = CONTRASTS + [("RQ3 temporal: fixed - random (F-D)", {"F": 1, "D": -1})]
    boot = {name: {c: [] for c in BOOT_COLS} for name, _ in comps}
    t0 = time.time()
    for b in range(args.n_boot):
        idx = np.concatenate([rng.choice(pos_idx, len(pos_idx), replace=True),
                              rng.choice(neg_idx, len(neg_idx), replace=True)])
        cv = cond_values(idx)
        for name, coef in comps:
            if all(c in cv for c in coef):
                for col in BOOT_COLS:
                    boot[name][col].append(sum(w * cv[c][col] for c, w in coef.items()))
        if (b + 1) % 200 == 0:
            print(f"  {b+1}/{args.n_boot} ({time.time()-t0:.0f}s)")

    ci_rows = []
    for name, coef in comps:
        if not all(c in full for c in coef):
            continue
        cols = BOOT_COLS if not name.startswith("RQ3") else ["viol_pos", "down_pos", "jitter_pos"]
        for col in cols:
            est = sum(w * full[c][col] for c, w in coef.items())
            lo, hi = np.percentile(boot[name][col], [2.5, 97.5])
            if lo > 0:
                verdict = "increase (CI > 0)"
            elif hi < 0:
                verdict = "decrease (CI < 0)"
            else:
                verdict = "inconclusive (CI includes 0)"
            better = ""
            if verdict != "inconclusive (CI includes 0)" and not name.startswith("Interaction"):
                up = verdict.startswith("increase")
                better = "better" if up == HIGHER_IS_BETTER[col] else "worse"
            ci_rows.append(dict(contrast=name, metric=col, estimate=round(est, 4),
                                ci_low=round(lo, 4), ci_high=round(hi, 4),
                                verdict=verdict, direction=better))
    ci_df = pd.DataFrame(ci_rows)
    ci_df.to_csv(os.path.join(out_dir, "rq2_effects_ci.csv"), index=False)

    # ---------------- training-log sanity check ----------------
    san = []
    for cond in ["A", "B", "C"]:
        for s in SEEDS:
            p = os.path.join(args.train_log_dir, f"training_log_{RUN_TMPL[cond].format(s=s)}.csv")
            if not os.path.exists(p):
                continue
            lg = pd.read_csv(p)
            san.append(dict(run=RUN_TMPL[cond].format(s=s), epochs=len(lg),
                            mean_train_reg=round(lg["train_reg"].mean(), 5),
                            mean_train_mono=round(lg["train_mono"].mean(), 5),
                            best_val_loss=round(lg["val_loss"].min(), 4),
                            best_epoch=int(lg.loc[lg["val_loss"].idxmin(), "epoch"]),
                            last_val_mAP=round(lg["val_mAP"].iloc[-1], 4)))
    san_df = pd.DataFrame(san)

    # ---------------- comparison with the RiskProp paper ----------------
    pc_rows = []
    for name, coef in CONTRASTS:
        for col in ("mAP", "mAUC01"):
            pdelta = sum(w * PAPER_T3[c][col] for c, w in coef.items())
            ours = ci_df[(ci_df.contrast == name) & (ci_df.metric == col)]
            if ours.empty:
                continue
            o = ours.iloc[0]
            if o.ci_low > 0 or o.ci_high < 0:
                agree = "agrees (same sign)" if np.sign(o.estimate) == np.sign(pdelta) \
                    else "DISAGREES (opposite sign)"
            else:
                agree = "not confirmed (our CI includes 0)"
            pc_rows.append({"contrast": name, "metric": col,
                            "paper_delta": round(pdelta, 3),
                            "ours_delta": o.estimate,
                            "ours_95CI": f"[{o.ci_low}, {o.ci_high}]",
                            "agreement": agree})
    pc_df = pd.DataFrame(pc_rows)
    pc_df.to_csv(os.path.join(out_dir, "paper_comparison.csv"), index=False)
    lvl = []
    for cond, exp in (("A", "I"), ("B", "III"), ("C", "II"), ("D", "IV")):
        sub = per_run[per_run.condition == cond]
        if sub.empty:
            continue
        lvl.append({"Condition": COND_NAME[cond], "paper Exp.": exp,
                    "paper mAP (test)": PAPER_T3[cond]["mAP"],
                    "ours mAP (val)": fmt(sub["mAP"].mean(), sub["mAP"].std(ddof=1)),
                    "paper mAUC0.1 (test)": PAPER_T3[cond]["mAUC01"],
                    "ours mAUC0.1 (val)": fmt(sub["mAUC01"].mean(), sub["mAUC01"].std(ddof=1))})
    lvl_df = pd.DataFrame(lvl)

    # ---------------- report ----------------
    rep = [f"# RQ2 -- FFR x AMC ablation (internal validation split)\n",
           f"Generated {time.strftime('%Y-%m-%d %H:%M')} | {N} videos "
           f"({len(pos_idx)} pos / {len(neg_idx)} neg); temporal metrics on "
           f"{int(tpos_mask.sum())} positives with a known event time. "
           f"Checkpoint per run: locked per_run rule (see rq2_chosen_checkpoints.json). "
           f"EPS={EPS} (locked before scoring).\n"]
    if not san_df.empty:
        rep += ["## 0. Training sanity check (loss terms actually on/off)\n",
                "Expected: A reg=0 mono=0; B reg>0 mono=0; C reg=0 mono>0.\n",
                md_table(san_df), ""]
    g1 = ["Condition", "n_seeds", "mAP", "AP@0.5s", "AP@1.0s", "AP@1.5s"]
    g2 = ["Condition", "mAUC01", "Recall@0.5s", "Recall@1.0s", "Recall@1.5s",
          "ActualFAR@1.0s", "mTTA_detected", "mTTA_all", "Coverage"]
    g3 = ["Condition", "viol_pos", "down_pos", "jitter_pos", "viol_pos_eps0",
          "jitter_neg", "meanscore_neg"]
    rep += ["## 1. Accuracy (mean +/- SD over 3 seeds; mAP is the primary metric)\n",
            md_table(agg_df[g1]), "",
            "## 2. Low-FAR anticipation (FAR <= 0.1)\n", md_table(agg_df[g2]), "",
            "## 3. Temporal monotonicity (dense curves; lower = smoother / more monotone)\n",
            "viol/down/jitter_pos: positives; *_neg: negatives (false-alarm behaviour); "
            "viol_pos_eps0 = sensitivity with EPS=0. Row F is for RQ3 only.\n",
            md_table(agg_df[g3]), "",
            "## 4. Run-to-run stability\n",
            "The +/- values above are the SD over seeds 42/43/44 (Group 4).\n",
            "## 5. Main effects and interaction (paired stratified bootstrap 95% CI, "
            f"B={args.n_boot})\n",
            "`direction` says whether a conclusive change is better or worse for that metric "
            "(higher mAP/mAUC is better; lower violation/downward step/jitter is better).\n",
            md_table(ci_df[~ci_df.contrast.str.startswith("RQ3")]), "",
            "## 6. RQ3 add-on: temporal metrics, fixed-lag (F) vs random (D)\n",
            md_table(ci_df[ci_df.contrast.str.startswith("RQ3")]), ""]
    rep += ["## 7. Comparison with the RiskProp paper (Table 3, Nexar, Only Collision Label)\n",
            "Paper: official Nexar test set, single run, lambda1=1.5 / lambda2=1.1, LR 0.002, "
            "batch 64, frame-level sequences. Ours: internal validation split, 3 seeds, "
            "lambda1=lambda2=0.5, LR 0.01, 12-snippet sequences (see RQ2_THEORY.md deviations). "
            "Absolute levels are therefore NOT comparable; only the direction of each effect is. "
            "mTTA is omitted because the paper's mTTA^0.1 is defined differently from ours.\n",
            md_table(lvl_df) if not lvl_df.empty else "(no data)", "",
            md_table(pc_df) if not pc_df.empty else "(no data)", ""]
    with open(os.path.join(out_dir, "summary_rq2.md"), "w") as f:
        f.write("\n".join(rep))
    print("\n".join(rep))
    print(f"\nWritten to {out_dir}")


if __name__ == "__main__":
    main()
