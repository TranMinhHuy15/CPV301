"""
re10 -- RQ3 secondary sensitivity (FixedLag tau = 0.5 / 1.0 / 1.5 s) on the
internal validation split. Same protocol as RQ2 / RQ3 (re07 + re08):

  1. 3-lead validation predictions (re02 loader) for the new runs
     riskprop_fixed_seed{S}_tau0.5 / _tau1.5, both {best, latest} checkpoints.
  2. Checkpoint choice with the LOCKED per_run rule (higher mean val AP over
     0.5/1.0/1.5 s; tie -> best). tau 1.0 ("RiskProp fixed") and RiskProp
     random keep the choice already locked in locked_config.json.
  3. Dense causal risk curves (same 30-window sweep and cache as re07) for the
     chosen tau 0.5 / 1.5 checkpoints.
  4. Analysis: mean +/- SD over seeds for T05 / T10 / T15 / R, and paired
     label-stratified bootstrap CIs (B=2000, seed 12345) for
        T05 - T10, T15 - T10   (does the choice of tau matter?)
        T05 - R,   T15 - R,  T10 - R   (each lag vs random-offset AMC)
     on mAP, mAUC@0.1, PVR (eps 0.01), ADS, RCJ.

Reuses (read-only) what re07 already produced on this machine for tau 1.0 and
random: reeval_out/preds_val_rq2/ and reeval_out/dense_rq2/. Secondary
analysis per proposal Sec. 5.3 -- not promoted as a separate contribution.

Usage:  python reeval/re10_rq3_sensitivity_eval.py
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

LEADS = [0.5, 1.0, 1.5]
SEEDS = [42, 43, 44]
KINDS = ["best", "latest"]
EPS = 0.01
COND = {   # label -> (run template, location: new = outputs_riskprop, old = ckpts)
    "T05": ("riskprop_fixed_seed{s}_tau0.5", "new"),
    "T10": ("riskprop_fixed_seed{s}", "old"),
    "T15": ("riskprop_fixed_seed{s}_tau1.5", "new"),
    "R": ("riskprop_random_seed{s}", "old"),
}
NAME = {"T05": "FixedLag tau=0.5s", "T10": "FixedLag tau=1.0s (RQ3 main)",
        "T15": "FixedLag tau=1.5s", "R": "RiskProp random-offset"}
CONTRASTS = [("tau0.5 - tau1.0", {"T05": 1, "T10": -1}),
             ("tau1.5 - tau1.0", {"T15": 1, "T10": -1}),
             ("tau0.5 - random", {"T05": 1, "R": -1}),
             ("tau1.0 - random", {"T10": 1, "R": -1}),
             ("tau1.5 - random", {"T15": 1, "R": -1})]
BOOT_COLS = ["mAP", "mAUC01", "PVR", "ADS", "RCJ"]
HIGHER_BETTER = {"mAP": True, "mAUC01": True, "PVR": False, "ADS": False, "RCJ": False}


def temporal_per_video(a, eps=EPS):
    T = a.shape[1]
    iu, ju = np.triu_indices(T, k=1)
    pvr = (a[:, iu] > a[:, ju] + eps).mean(axis=1)
    ads = np.maximum(0.0, a[:, :-1] - a[:, 1:]).mean(axis=1)
    rcj = np.abs(a[:, 2:] - 2 * a[:, 1:-1] + a[:, :-2]).mean(axis=1)
    return pvr, ads, rcj


def md_table(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


# ----------------------------------------------------------------------------
# GPU part
# ----------------------------------------------------------------------------
def gpu_steps(args, new_runs):
    import torch
    sys.path.insert(0, args.repo_dir)
    sys.path.insert(0, os.path.join(args.repo_dir, "pipeline"))  # cell*.py live in pipeline/
    sys.path.insert(0, os.path.join(args.repo_dir, "reeval"))
    import re02_infer_val as r2
    from cell31_model_riskprop import RiskPropModel
    from cell22_adalea_dataset import _to_tensor
    from sklearn.metrics import average_precision_score

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    os.makedirs(args.preds_dir, exist_ok=True)
    os.makedirs(args.dense_dir, exist_ok=True)

    # 1. three-lead predictions
    print("\n[1] 3-lead validation predictions for tau 0.5 / 1.5 (best + latest)")
    vid_ids, targets, data = r2.load_val_tensors(args.cache_5f)
    taus = np.stack([data[l][1] for l in LEADS])
    model = RiskPropModel().to(device).eval()
    for run in new_runs:
        for kind in KINDS:
            out = os.path.join(args.preds_dir, f"{run}_{kind}.npz")
            if os.path.exists(out):
                continue
            p = os.path.join(args.new_ckpt_dir, f"{kind}_{run}.pth")
            if not os.path.exists(p):
                sys.exit(f"  [MISSING] {p} -- training not finished?")
            ck = torch.load(p, map_location=device, weights_only=False)
            model.load_state_dict(ck["model"])
            model.eval()
            scores = np.stack([r2.predict(model, data[l][0], device, 8) for l in LEADS])
            np.savez_compressed(out, vid_ids=np.array(vid_ids), targets=targets, taus=taus,
                                leads=np.array(LEADS), scores=scores)
            with open(out.replace(".npz", ".json"), "w") as f:
                json.dump({"run": run, "kind": kind, "ckpt_file": os.path.basename(p),
                           "ckpt_sha256_8MB": r2.sha256_head(p), "ckpt_epoch": ck.get("epoch")},
                          f, indent=2, default=float)
            print(f"  [OK] {run:32s} {kind:6s} epoch={ck.get('epoch')}")

    # 2. checkpoint choice (locked rule)
    print("\n[2] Checkpoint choice (locked per_run rule)")
    chosen = {}
    for run in new_runs:
        maps = {}
        for kind in KINDS:
            z = np.load(os.path.join(args.preds_dir, f"{run}_{kind}.npz"))
            maps[kind] = float(np.mean([average_precision_score(z["targets"], z["scores"][i])
                                        for i in range(3)]))
        kind = "best" if maps["best"] >= maps["latest"] else "latest"
        chosen[run] = {"kind": kind, "val_mAP_best": maps["best"], "val_mAP_latest": maps["latest"]}
        print(f"  {run:32s} best={maps['best']:.4f} latest={maps['latest']:.4f} -> {kind}")

    # 3. dense curves (cache built by re07)
    print("\n[3] Dense risk curves for the chosen tau 0.5 / 1.5 checkpoints")
    arr = np.load(os.path.join(args.dense_cache, "windows_u8.npy"), mmap_mode="r")
    with open(os.path.join(args.dense_cache, "meta.json")) as f:
        meta = json.load(f)
    todo = [r for r in new_runs if not os.path.exists(os.path.join(args.dense_dir, f"{r}.npz"))]
    models = []
    for run in todo:
        m = RiskPropModel().to(device).eval()
        ck = torch.load(os.path.join(args.new_ckpt_dir, f"{chosen[run]['kind']}_{run}.pth"),
                        map_location=device, weights_only=False)
        m.load_state_dict(ck["model"])
        m.eval()
        models.append(m)
    if todo:
        sc = np.zeros((len(todo), arr.shape[0], arr.shape[1]), dtype=np.float32)
        t0 = time.time()
        with torch.no_grad():
            for i in range(arr.shape[0]):
                x = torch.stack([_to_tensor({"frames": torch.from_numpy(np.array(arr[i, k]))})
                                 for k in range(arr.shape[1])]).to(device)
                for r, m in enumerate(models):
                    sc[r, i] = torch.sigmoid(m(x)).float().cpu().numpy()
                if (i + 1) % 50 == 0:
                    print(f"  [dense] {i+1}/{arr.shape[0]} videos ({time.time()-t0:.0f}s)")
        for r, run in enumerate(todo):
            np.savez_compressed(os.path.join(args.dense_dir, f"{run}.npz"), scores=sc[r],
                                vid_ids=np.array(meta["vid_ids"]), targets=np.array(meta["targets"]),
                                kind=np.array(chosen[run]["kind"]))
    return chosen


# ----------------------------------------------------------------------------
# Analysis (CPU)
# ----------------------------------------------------------------------------
def analyze(args, chosen_new):
    sys.path.insert(0, os.path.join(args.repo_dir, "reeval"))
    import re03_analyze_rq as r3
    with open(args.locked_config) as f:
        locked = json.load(f)["chosen_checkpoints"]
    df = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    df["vid"] = df["id"].apply(lambda x: str(int(x)).zfill(5))
    toe = dict(zip(df["vid"], df["time_of_event"]))

    runs, ls, dense = {}, {}, {}
    ref_ids = targets = None
    for c, (tmpl, loc) in COND.items():
        for s in SEEDS:
            run = tmpl.format(s=s)
            if loc == "new":
                kind, pdir, ddir = chosen_new[run]["kind"], args.preds_dir, args.dense_dir
            else:
                kind, pdir, ddir = locked[run]["kind"], args.rq2_preds_dir, args.rq2_dense_dir
            z = np.load(os.path.join(pdir, f"{run}_{kind}.npz"))
            d = np.load(os.path.join(ddir, f"{run}.npz"))
            if ref_ids is None:
                ref_ids, targets = z["vid_ids"], z["targets"].astype(int)
            assert np.array_equal(z["vid_ids"], ref_ids) and np.array_equal(d["vid_ids"], ref_ids)
            runs[run] = (c, s, kind)
            ls[run] = [z["scores"][i] for i in range(3)]
            dense[run] = d["scores"].astype(np.float64)
    N = len(targets)
    pos_idx, neg_idx = np.where(targets == 1)[0], np.where(targets == 0)[0]
    tpos = np.array([t == 1 and not pd.isna(toe.get(v)) for v, t in zip(ref_ids, targets)])

    rows, pv = [], {}
    for run, (c, s, kind) in runs.items():
        m = r3.full_metrics(ls[run], targets, ls[run][1])
        p, a, j = temporal_per_video(dense[run])
        p0, _, _ = temporal_per_video(dense[run], 0.0)
        pv[run] = {"PVR": p, "ADS": a, "RCJ": j}
        rows.append(dict(condition=c, seed=s, run=run, checkpoint=kind,
                         mAP=m["mAP"], **{f"AP@{l}s": m[f"AP@{l}s"] for l in LEADS},
                         mAUC01=m["mAUC01"], **{f"Recall@{l}s": m[f"Recall@{l}s"] for l in LEADS},
                         mTTA_detected=m["mTTA_detected"], Coverage=m["Coverage"],
                         PVR=p[tpos].mean(), ADS=a[tpos].mean(), RCJ=j[tpos].mean(),
                         PVR_eps0=p0[tpos].mean()))
    per_run = pd.DataFrame(rows)
    os.makedirs(args.out_dir, exist_ok=True)
    per_run.to_csv(os.path.join(args.out_dir, "per_run_metrics_rq3_sensitivity.csv"), index=False)

    cols = ["mAP", "AP@0.5s", "AP@1.0s", "AP@1.5s", "mAUC01", "Recall@1.0s",
            "mTTA_detected", "Coverage", "PVR", "ADS", "RCJ", "PVR_eps0"]
    agg = []
    for c in COND:
        sub = per_run[per_run.condition == c]
        agg.append({"Condition": NAME[c], "n": len(sub),
                    **{k: f"{sub[k].mean():.4f} +/- {sub[k].std(ddof=1):.4f}" for k in cols}})
    agg = pd.DataFrame(agg)
    agg.to_csv(os.path.join(args.out_dir, "rq3_sensitivity_mean_sd.csv"), index=False)

    def cvals(idx):
        tp = idx[tpos[idx]]
        out = {}
        for c in COND:
            v = {k: [] for k in BOOT_COLS}
            for run, (cc, s, k) in runs.items():
                if cc != c:
                    continue
                mp, mu = r3.fast_map_mauc([x[idx] for x in ls[run]], targets[idx])
                v["mAP"].append(mp)
                v["mAUC01"].append(mu)
                for key in ("PVR", "ADS", "RCJ"):
                    v[key].append(pv[run][key][tp].mean())
            out[c] = {k: float(np.mean(x)) for k, x in v.items()}
        return out

    print(f"\n[4] Paired stratified bootstrap, B={args.n_boot}")
    rng = np.random.default_rng(args.boot_seed)
    full = cvals(np.arange(N))
    boot = {n: {k: [] for k in BOOT_COLS} for n, _ in CONTRASTS}
    t0 = time.time()
    for b in range(args.n_boot):
        idx = np.concatenate([rng.choice(pos_idx, len(pos_idx), replace=True),
                              rng.choice(neg_idx, len(neg_idx), replace=True)])
        cv = cvals(idx)
        for n, coef in CONTRASTS:
            for k in BOOT_COLS:
                boot[n][k].append(sum(w * cv[c][k] for c, w in coef.items()))
        if (b + 1) % 500 == 0:
            print(f"  {b+1}/{args.n_boot} ({time.time()-t0:.0f}s)")
    ci = []
    for n, coef in CONTRASTS:
        for k in BOOT_COLS:
            est = sum(w * full[c][k] for c, w in coef.items())
            lo, hi = np.percentile(boot[n][k], [2.5, 97.5])
            if lo <= 0 <= hi:
                verdict, direction = "inconclusive (CI includes 0)", ""
            else:
                up = lo > 0
                verdict = "increase (CI > 0)" if up else "decrease (CI < 0)"
                direction = "better" if up == HIGHER_BETTER[k] else "worse"
            ci.append(dict(contrast=n, metric=k, estimate=round(est, 4), ci_low=round(lo, 4),
                           ci_high=round(hi, 4), verdict=verdict, direction=direction))
    ci = pd.DataFrame(ci)
    ci.to_csv(os.path.join(args.out_dir, "rq3_sensitivity_ci.csv"), index=False)
    with open(os.path.join(args.out_dir, "rq3_sensitivity_chosen_checkpoints.json"), "w") as f:
        json.dump({"new_runs": chosen_new,
                   "reused_locked": {r: k for r, (c, s, k) in runs.items() if COND[c][1] == "old"}},
                  f, indent=2)

    rep = ["# RQ3 sensitivity -- FixedLag tau 0.5 / 1.0 / 1.5 s (internal validation)\n",
           f"Generated {time.strftime('%Y-%m-%d %H:%M')} | {N} videos ({len(pos_idx)} pos); "
           f"temporal metrics on {int(tpos.sum())} positives; EPS={EPS}. Checkpoints: locked per_run "
           f"rule (tau 1.0 and random reuse locked_config). Secondary analysis (proposal Sec. 5.3).\n",
           "## 1. Accuracy and low-FAR (mean +/- SD over 3 seeds)\n",
           md_table(agg[["Condition", "n", "mAP", "AP@0.5s", "AP@1.0s", "AP@1.5s", "mAUC01",
                         "Recall@1.0s", "mTTA_detected", "Coverage"]]), "",
           "## 2. Temporal (dense curves, positives; lower = better)\n",
           md_table(agg[["Condition", "PVR", "ADS", "RCJ", "PVR_eps0"]]), "",
           f"## 3. Contrasts (paired stratified bootstrap 95% CI, B={args.n_boot})\n",
           md_table(ci), ""]
    with open(os.path.join(args.out_dir, "summary_rq3_sensitivity.md"), "w") as f:
        f.write("\n".join(rep))
    print("\n".join(rep))
    print(f"\nWritten to {args.out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-dir", default="/workspace/CPV301")
    ap.add_argument("--data-dir", default="/workspace/CPV301/data/nexar_kaggle_style")
    ap.add_argument("--cache-5f", default="/workspace/CPV301/data/nexar_cache_5f")
    ap.add_argument("--dense-cache", default="/workspace/CPV301/data/nexar_cache_dense")
    ap.add_argument("--new-ckpt-dir", default="/workspace/CPV301/outputs_riskprop")
    ap.add_argument("--locked-config",
                    default="/workspace/CPV301/results/reeval_corrected/locked_config.json")
    ap.add_argument("--out-root", default="/workspace/CPV301/reeval_out")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--boot-seed", type=int, default=12345)
    ap.add_argument("--analysis-only", action="store_true",
                    help="skip GPU steps; read chosen checkpoints from the saved json")
    args = ap.parse_args()
    args.preds_dir = os.path.join(args.out_root, "preds_val_rq3s")
    args.dense_dir = os.path.join(args.out_root, "dense_rq3s")
    args.rq2_preds_dir = os.path.join(args.out_root, "preds_val_rq2")
    args.rq2_dense_dir = os.path.join(args.out_root, "dense_rq2")
    args.out_dir = os.path.join(args.out_root, "analysis_rq3s")
    new_runs = [COND[c][0].format(s=s) for c in ("T05", "T15") for s in SEEDS]
    if args.analysis_only:
        with open(os.path.join(args.out_dir, "rq3_sensitivity_chosen_checkpoints.json")) as f:
            chosen = json.load(f)["new_runs"]
    else:
        chosen = gpu_steps(args, new_runs)
    analyze(args, chosen)


if __name__ == "__main__":
    main()
