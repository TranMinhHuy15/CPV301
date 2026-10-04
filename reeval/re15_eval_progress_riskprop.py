"""
re15 -- Validation evaluation for Progressive RiskProp (RQ3 new). Reuses the
LOCKED project protocol instead of a second evaluator:
  * 3-lead windows (0.5/1.0/1.5 s) from nexar_cache_5f/val via
    re02.load_val_tensors / re02.predict (same tensors as RQ1-RQ3);
  * checkpoint choice = locked "per_run" rule of re03 (higher mean val AP over
    the 3 leads among {best, latest}; tie -> best), identical for B/P/S;
  * metrics = re03.full_metrics (AP, mAUC@0.1, Recall/threshold at FAR 0.1,
    mTTA_detected + Coverage);
  * dense causal curves = re07 sweep (30 windows, d = 3.0..0.1 s) from the
    shared nexar_cache_dense (built by re07.build_dense_cache if missing);
    temporal metrics = re10.temporal_per_video (PVR eps 0.01, ADS, RCJ);
  * paired label-stratified bootstrap (B=2000, seed 12345) like re08/re10.
Added for RQ3 (all on validation, diagnostics unless stated in the lock):
  * score amplitude next to PVR/ADS/RCJ (early/late mean score, rise, per-video
    range, negative score level) -- flat curves can look "monotonic";
  * target fit of the dense curve to the CPS target (diagnostic only);
  * alert-event-gap strata (<=0.5, 0.5-1, 1-1.5, 1.5-2, >2 s) per condition;
  * primary-metric and non-inferiority verdicts read from the lock file.
mTTA caveat: mTTA here is the largest of the 3 discrete leads detected at the
FAR-0.1 threshold (re03 definition), NOT PRE-ACT's mTTA; report with Coverage.
The official test set is NOT touched by this script.

Usage:
  python reeval/re15_eval_progress_riskprop.py --runs-root outputs_progress/runs \
      --pairing-mode fixed --horizon-sec 2.0 --alpha 3 --lambda-prog 1.0 \
      --conditions B,P,S --seeds 42,43,44 --lock-file outputs_progress/LOCK.json \
      --data-dir data/nexar_kaggle_style --cache-5f data/nexar_cache_5f \
      --dense-cache data/nexar_cache_dense --out-dir outputs_progress/eval_val
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "pipeline"))
from re10_rq3_sensitivity_eval import temporal_per_video, md_table  # noqa: E402
from re13_progress_supervision import continuous_target_scalar  # noqa: E402
from re14_train_progress_riskprop import run_name, build_model  # noqa: E402

LEADS = [0.5, 1.0, 1.5]
KINDS = ["best", "latest"]
GAP_BINS = [-np.inf, 0.5, 1.0, 1.5, 2.0, np.inf]
GAP_LABELS = ["<=0.5", "0.5-1.0", "1.0-1.5", "1.5-2.0", ">2.0"]
BOOT_COLS = ["mAP", "AP@1.5s", "mAUC01", "PVR", "ADS", "RCJ"]
HIGHER_BETTER = {"mAP": True, "AP@1.5s": True, "mAUC01": True, "PVR": False, "ADS": False, "RCJ": False}
ALL_CONTRASTS = [("P - B", {"P": 1, "B": -1}), ("S - B", {"S": 1, "B": -1}),
                 ("P - S", {"P": 1, "S": -1}), ("FP - F", {"FP": 1, "F": -1}),
                 ("(P-B) - (FP-F)", {"P": 1, "B": -1, "FP": -1, "F": 1})]


def lead_maps(targets, lead_scores):
    from sklearn.metrics import average_precision_score
    return [average_precision_score(targets, s) for s in lead_scores]


def amplitude(dense, tpos, neg, d_grid):
    d = np.asarray(d_grid)
    early, late = d >= 2.5, d <= 0.5
    pos = dense[tpos]
    return {"score_early_pos": float(pos[:, early].mean()), "score_late_pos": float(pos[:, late].mean()),
            "rise_pos": float(pos[:, late].mean() - pos[:, early].mean()),
            "range_pos": float((pos.max(1) - pos.min(1)).mean()),
            "score_neg_mean": float(dense[neg].mean()),
            "score_neg_p95": float(np.percentile(dense[neg].max(1), 95))}


def target_fit(dense, tpos, d_grid, t_obs, toe_arr, horizon, alpha):
    """MAE of dense positive curves to the CPS targets, only on unclamped
    windows (t_obs == toe - d). Diagnostic only."""
    errs_c, errs_b = [], []
    for i in np.where(tpos)[0]:
        for k, d in enumerate(d_grid):
            if not np.isfinite(t_obs[i][k]) or abs(t_obs[i][k] - (toe_arr[i] - d)) > 1e-6:
                continue
            c = continuous_target_scalar(d, horizon, alpha)
            b = 1.0 if 0 < d < horizon else 0.0
            errs_c.append(abs(dense[i, k] - c))
            errs_b.append(abs(dense[i, k] - b))
    return {"fit_MAE_continuous": float(np.mean(errs_c)) if errs_c else float("nan"),
            "fit_MAE_binary": float(np.mean(errs_b)) if errs_b else float("nan"),
            "fit_windows": len(errs_c)}


_SHA = {}


def file_sha(path):
    """Full sha256 of a file (cached per process)."""
    if path not in _SHA:
        import hashlib
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        _SHA[path] = h.hexdigest()
    return _SHA[path]


def preds_fresh(out, ckpt, val_sha):
    """Reuse a prediction file only if it was made from THIS checkpoint file
    and THIS validation cache index."""
    js = out.replace(".npz", ".json")
    if not (os.path.exists(out) and os.path.exists(js)):
        return False
    m = json.load(open(js))
    ok = m.get("ckpt_sha256") == file_sha(ckpt) and m.get("val_index_sha256") == val_sha
    if not ok:
        print(f"  [STALE] {os.path.basename(out)} does not match the current checkpoint/val cache -> recompute")
    return ok


def dense_fresh(out, ckpt, dense_sha):
    if not os.path.exists(out):
        return False
    z = np.load(out)
    ok = ("ckpt_sha256" in z and str(z["ckpt_sha256"]) == file_sha(ckpt)
          and "dense_meta_sha256" in z and str(z["dense_meta_sha256"]) == dense_sha)
    if not ok:
        print(f"  [STALE] {os.path.basename(out)} -> recompute")
    return ok


def check_same_data(runs):
    """All runs must come from the same split / caches / sidecar."""
    keys = ("split_manifest_sha256", "train_index_sha256", "val_index_sha256", "sidecar_sha256")
    seen = {}
    for run, rdir in runs.items():
        inp = json.load(open(os.path.join(rdir, "config.json")))["inputs"]
        sig = tuple(inp.get(k) for k in keys)
        seen.setdefault(sig, []).append(run)
    if len(seen) > 1:
        sys.exit("[REFUSE] runs were trained on different data/sidecars:\n  " +
                 "\n  ".join(f"{dict(zip(keys, [x[:12] if x else x for x in sig]))}: {r}"
                              for sig, r in seen.items()))


def gpu_steps(a, runs):
    import torch
    import re02_infer_val as r2
    import re07_infer_rq2 as r7
    from cell22_adalea_dataset import _to_tensor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(a.preds_dir, exist_ok=True)
    os.makedirs(a.dense_dir, exist_ok=True)
    print(f"device={device}\n[1] 3-lead validation predictions (best + latest)")
    vid_ids, targets, data = r2.load_val_tensors(a.cache_5f)
    taus = np.stack([data[l][1] for l in LEADS])
    val_sha = file_sha(os.path.join(a.cache_5f, "val_index.json"))
    model = build_model(a.model).to(device).eval()
    for run, rdir in runs.items():
        for kind in KINDS:
            out = os.path.join(a.preds_dir, f"{run}_{kind}.npz")
            p = os.path.join(rdir, f"{kind}.pth")
            if not os.path.exists(p):
                sys.exit(f"  [MISSING] {p} -- training not finished?")
            if preds_fresh(out, p, val_sha):
                continue
            ck = torch.load(p, map_location=device, weights_only=False)
            model.load_state_dict(ck["model"])
            model.eval()
            scores = np.stack([r2.predict(model, data[l][0], device, 8) for l in LEADS])
            np.savez_compressed(out, vid_ids=np.array(vid_ids), targets=targets, taus=taus,
                                leads=np.array(LEADS), scores=scores)
            with open(out.replace(".npz", ".json"), "w") as f:
                json.dump({"run": run, "kind": kind, "ckpt": p, "ckpt_sha256": file_sha(p),
                           "val_index_sha256": val_sha,
                           "ckpt_epoch": ck.get("epoch"), "config": ck.get("config")}, f, indent=2, default=str)
            print(f"  [OK] {run} {kind} epoch={ck.get('epoch')}")

    print("\n[2] Checkpoint choice (locked per_run rule)")
    chosen = {}
    for run in runs:
        maps = {}
        for kind in KINDS:
            z = np.load(os.path.join(a.preds_dir, f"{run}_{kind}.npz"))
            maps[kind] = float(np.mean(lead_maps(z["targets"], z["scores"])))
        kind = "best" if maps["best"] >= maps["latest"] else "latest"
        chosen[run] = {"kind": kind, "val_mAP_best": maps["best"], "val_mAP_latest": maps["latest"],
                       "ckpt_sha256": file_sha(os.path.join(runs[run], f"{kind}.pth"))}
        print(f"  {run}: best={maps['best']:.4f} latest={maps['latest']:.4f} -> {kind}")

    print("\n[3] Dense risk curves (re07 sweep)")
    if not os.path.exists(os.path.join(a.dense_cache, "windows_u8.npy")):
        r7.build_dense_cache(a.data_dir, a.cache_5f, a.dense_cache)
    arr = np.load(os.path.join(a.dense_cache, "windows_u8.npy"), mmap_mode="r")
    dense_sha = file_sha(os.path.join(a.dense_cache, "meta.json"))
    todo = [r for r in runs if not dense_fresh(os.path.join(a.dense_dir, f"{r}.npz"),
                                               os.path.join(runs[r], f"{chosen[r]['kind']}.pth"), dense_sha)]
    with open(os.path.join(a.dense_cache, "meta.json")) as f:
        meta = json.load(f)
    n_vid = arr.shape[0] if not a.limit_dense else min(a.limit_dense, arr.shape[0])
    for run in todo:
        ck = torch.load(os.path.join(runs[run], f"{chosen[run]['kind']}.pth"), map_location=device,
                        weights_only=False)
        model.load_state_dict(ck["model"])
        model.eval()
        sc = np.full((arr.shape[0], arr.shape[1]), np.nan, dtype=np.float32)
        t0 = time.time()
        with torch.no_grad():
            for i in range(n_vid):
                x = torch.stack([_to_tensor({"frames": torch.from_numpy(np.array(arr[i, k]))})
                                 for k in range(arr.shape[1])]).to(device)
                sc[i] = torch.sigmoid(model(x)).float().cpu().numpy()
                if (i + 1) % 50 == 0:
                    print(f"  [dense {run}] {i+1}/{n_vid} ({time.time()-t0:.0f}s)")
        np.savez_compressed(os.path.join(a.dense_dir, f"{run}.npz"), scores=sc,
                            vid_ids=np.array(meta["vid_ids"]), targets=np.array(meta["targets"]),
                            kind=np.array(chosen[run]["kind"]),
                            ckpt_sha256=np.array(chosen[run]["ckpt_sha256"]),
                            dense_meta_sha256=np.array(dense_sha))
    with open(os.path.join(a.out_dir, "chosen_checkpoints.json"), "w") as f:
        json.dump(chosen, f, indent=2)
    return chosen


def analyze(a, runs, chosen):
    import re03_analyze_rq as r3
    with open(os.path.join(a.dense_cache, "meta.json")) as f:
        meta = json.load(f)
    df = pd.read_csv(os.path.join(a.data_dir, "train.csv"))
    df["vid"] = df["id"].apply(lambda x: str(int(x)).zfill(5))
    toe = dict(zip(df["vid"], df["time_of_event"]))
    toa = dict(zip(df["vid"], df["time_of_alert"]))
    lock = json.load(open(a.lock_file)) if a.lock_file and os.path.exists(a.lock_file) else {}

    ls, dense, info = {}, {}, {}
    ref_ids = targets = None
    for run in runs:
        c, s = a.run_cond[run]
        kind = chosen[run]["kind"]
        z = np.load(os.path.join(a.preds_dir, f"{run}_{kind}.npz"))
        d = np.load(os.path.join(a.dense_dir, f"{run}.npz"))
        if ref_ids is None:
            ref_ids, targets = z["vid_ids"], z["targets"].astype(int)
        assert np.array_equal(z["vid_ids"], ref_ids) and np.array_equal(d["vid_ids"], ref_ids)
        ls[run] = [z["scores"][i] for i in range(3)]
        dense[run] = d["scores"].astype(np.float64)
        info[run] = (c, s, kind)
    keep = np.all(np.isfinite(np.stack(list(dense.values()))), axis=(0, 2))   # --limit-dense (dev)
    N = len(targets)
    pos_idx, neg_idx = np.where(targets == 1)[0], np.where(targets == 0)[0]
    tpos = np.array([t == 1 and not pd.isna(toe.get(v)) for v, t in zip(ref_ids, targets)]) & keep
    negm = (targets == 0) & keep
    toe_arr = np.array([float(toe.get(v)) if not pd.isna(toe.get(v)) else np.nan for v in ref_ids])
    gap = np.array([float(toe.get(v)) - float(toa.get(v)) if not (pd.isna(toe.get(v)) or pd.isna(toa.get(v)))
                    else np.nan for v in ref_ids])
    gap_lab = pd.cut(gap, bins=GAP_BINS, labels=GAP_LABELS).astype(object)
    t_obs = np.array(meta["t_obs"], dtype=np.float64)
    d_grid = meta["dense_d"]

    rows, pv, strata = [], {}, []
    for run, (c, s, kind) in info.items():
        m = r3.full_metrics(ls[run], targets, ls[run][1])
        p, ad, j = temporal_per_video(np.nan_to_num(dense[run]))
        pv[run] = {"PVR": p, "ADS": ad, "RCJ": j}
        rows.append(dict(condition=c, seed=s, run=run, checkpoint=kind, mAP=m["mAP"],
                         **{f"AP@{l}s": m[f"AP@{l}s"] for l in LEADS}, mAUC01=m["mAUC01"],
                         **{f"Recall@{l}s": m[f"Recall@{l}s"] for l in LEADS},
                         mTTA_detected=m["mTTA_detected"], Coverage=m["Coverage"],
                         PVR=p[tpos].mean(), ADS=ad[tpos].mean(), RCJ=j[tpos].mean(),
                         **amplitude(dense[run], tpos, negm, d_grid),
                         **target_fit(dense[run], tpos, d_grid, t_obs, toe_arr, a.horizon_sec, a.alpha)))
        for lab in GAP_LABELS:
            bp = np.where((gap_lab == lab) & (targets == 1))[0]
            if len(bp) == 0:
                continue
            idx = np.concatenate([bp, neg_idx])
            aps = lead_maps(targets[idx], [x[idx] for x in ls[run]])
            bpt = bp[tpos[bp]]
            strata.append(dict(condition=c, seed=s, gap_bin=lab, n_pos=len(bp), mAP_bin=float(np.mean(aps)),
                               **{f"AP@{l}s": float(v) for l, v in zip(LEADS, aps)},
                               PVR=float(p[bpt].mean()) if len(bpt) else np.nan,
                               rise_pos=float((dense[run][bpt][:, np.array(d_grid) <= 0.5].mean()
                                               - dense[run][bpt][:, np.array(d_grid) >= 2.5].mean()))
                               if len(bpt) else np.nan))
    per_run = pd.DataFrame(rows)
    per_run.to_csv(os.path.join(a.out_dir, "per_run_metrics.csv"), index=False)
    st = pd.DataFrame(strata)
    st.to_csv(os.path.join(a.out_dir, "alert_gap_strata_per_run.csv"), index=False)

    conds = [c for c in a.conditions if c in set(per_run.condition)]
    cols = ["mAP", "AP@0.5s", "AP@1.0s", "AP@1.5s", "mAUC01", "Recall@1.0s", "mTTA_detected", "Coverage",
            "PVR", "ADS", "RCJ", "score_early_pos", "score_late_pos", "rise_pos", "range_pos",
            "score_neg_mean", "score_neg_p95", "fit_MAE_continuous", "fit_MAE_binary"]

    def msd(x):
        return f"{x.mean():.4f} +/- {x.std(ddof=1):.4f}" if len(x) > 1 else f"{x.mean():.4f}"
    agg = pd.DataFrame([{"Condition": c, "n_seeds": int((per_run.condition == c).sum()),
                         **{k: msd(per_run[per_run.condition == c][k]) for k in cols}} for c in conds])
    agg.to_csv(os.path.join(a.out_dir, "mean_sd.csv"), index=False)
    st_agg = (st.groupby(["gap_bin", "condition"], sort=False)
              [["n_pos", "mAP_bin", "AP@1.5s", "PVR", "rise_pos"]].mean().round(4).reset_index()) if len(st) else st
    st_agg.to_csv(os.path.join(a.out_dir, "alert_gap_strata_mean.csv"), index=False)

    # paired stratified bootstrap
    contrasts = [(n, co) for n, co in ALL_CONTRASTS if all(c in conds for c in co)]

    def cvals(idx):
        tp = idx[tpos[idx]]
        out = {}
        for c in conds:
            v = {k: [] for k in BOOT_COLS}
            for run, (cc, s, k) in info.items():
                if cc != c:
                    continue
                aps = lead_maps(targets[idx], [x[idx] for x in ls[run]])
                v["mAP"].append(np.mean(aps))
                v["AP@1.5s"].append(aps[2])
                v["mAUC01"].append(np.mean([r3.low_far_metrics(targets[idx], x[idx])[0] for x in ls[run]]))
                for key in ("PVR", "ADS", "RCJ"):
                    v[key].append(pv[run][key][tp].mean())
            out[c] = {k: float(np.mean(x)) for k, x in v.items()}
        return out

    ci = pd.DataFrame()
    if contrasts:
        print(f"\n[4] Paired stratified bootstrap B={a.n_boot}: {[n for n, _ in contrasts]}")
        rng = np.random.default_rng(a.boot_seed)
        pos_b, neg_b = pos_idx[keep[pos_idx]], neg_idx[keep[neg_idx]]
        full = cvals(np.concatenate([pos_b, neg_b]))
        boot = {n: {k: [] for k in BOOT_COLS} for n, _ in contrasts}
        t0 = time.time()
        for b in range(a.n_boot):
            idx = np.concatenate([rng.choice(pos_b, len(pos_b)), rng.choice(neg_b, len(neg_b))])
            cv = cvals(idx)
            for n, co in contrasts:
                for k in BOOT_COLS:
                    boot[n][k].append(sum(w * cv[c][k] for c, w in co.items()))
            if (b + 1) % 500 == 0:
                print(f"  {b+1}/{a.n_boot} ({time.time()-t0:.0f}s)")
        rows = []
        ni_metric, ni_margin = lock.get("ni_metric"), lock.get("ni_margin")
        for n, co in contrasts:
            for k in BOOT_COLS:
                est = sum(w * full[c][k] for c, w in co.items())
                lo, hi = np.percentile(boot[n][k], [2.5, 97.5])
                if lo <= 0 <= hi:
                    verdict, direction = "inconclusive (CI includes 0)", ""
                else:
                    up = lo > 0
                    verdict = "increase (CI > 0)" if up else "decrease (CI < 0)"
                    direction = "better" if up == HIGHER_BETTER[k] else "worse"
                ni = ""
                if ni_metric == k and ni_margin is not None and n in ("P - B", "S - B", "FP - F"):
                    bound = lo if HIGHER_BETTER[k] else -hi
                    ni = (f"non-inferior (margin {ni_margin})" if bound > -ni_margin
                          else f"non-inferiority NOT shown (margin {ni_margin})")
                rows.append(dict(contrast=n, metric=k, estimate=round(est, 5), ci_low=round(lo, 5),
                                 ci_high=round(hi, 5), verdict=verdict, direction=direction,
                                 primary="yes" if k == lock.get("primary_metric") else "", non_inferiority=ni))
        ci = pd.DataFrame(rows)
        ci.to_csv(os.path.join(a.out_dir, "bootstrap_ci.csv"), index=False)

    rep = ["# RQ3 (new) -- Progressive RiskProp, internal validation\n",
           f"Generated {time.strftime('%Y-%m-%d %H:%M')} | {int(keep.sum())}/{N} videos "
           f"({int(tpos.sum())} positives with dense curves) | pairing={a.pairing_mode} H={a.horizon_sec} "
           f"alpha={a.alpha} lambda={a.lambda_prog} | checkpoint rule: per_run (re03, locked) | "
           f"research_result={a.research}\n",
           f"Locked design: primary={lock.get('primary_metric')}, non-inferiority {lock.get('ni_metric')} "
           f"margin={lock.get('ni_margin')} (from {a.lock_file or 'no lock file'})\n",
           "## 1. Accuracy / low-FAR (mean +/- SD over seeds)\n",
           md_table(agg[["Condition", "n_seeds", "mAP", "AP@0.5s", "AP@1.0s", "AP@1.5s", "mAUC01",
                         "Recall@1.0s", "mTTA_detected", "Coverage"]]), "",
           "## 2. Temporal metrics WITH score amplitude (positives, dense curves; PVR/ADS/RCJ lower = better)\n",
           md_table(agg[["Condition", "PVR", "ADS", "RCJ", "score_early_pos", "score_late_pos", "rise_pos",
                         "range_pos", "score_neg_mean", "score_neg_p95"]]), "",
           "A low PVR/RCJ with a small rise/range means a flat curve, not better anticipation.\n",
           "## 3. Target fit (diagnostic only, unclamped dense windows)\n",
           md_table(agg[["Condition", "fit_MAE_continuous", "fit_MAE_binary"]]), "",
           "## 4. Alert-event gap strata (mean over seeds; positives of the bin + all negatives)\n",
           md_table(st_agg) if len(st_agg) else "(none)", "",
           f"## 5. Contrasts (paired label-stratified bootstrap 95% CI, B={a.n_boot}, seed {a.boot_seed})\n",
           md_table(ci) if len(ci) else "(need >= 2 conditions)", "",
           "Notes: validation only, official test not used. mTTA = largest of the 3 discrete leads detected "
           "at the FAR-0.1 threshold (re03), not PRE-ACT's mTTA. On the 12-snippet grid P and S targets "
           "coincide on most supervised snippets; P - S rests on few points."]
    with open(os.path.join(a.out_dir, "summary_rq3_progress.md"), "w") as f:
        f.write("\n".join(rep))
    print("\n".join(rep))
    print(f"\nWritten to {a.out_dir}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", required=True, help="outputs_progress/runs (or /dev for smoke)")
    ap.add_argument("--conditions", default="B,P,S")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--pairing-mode", required=True, choices=["fixed", "random"])
    ap.add_argument("--horizon-sec", type=float, default=2.0)
    ap.add_argument("--alpha", type=float, default=3.0)
    ap.add_argument("--lambda-prog", type=float, required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--cache-5f", required=True)
    ap.add_argument("--dense-cache", required=True)
    ap.add_argument("--lock-file", default="")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--boot-seed", type=int, default=12345)
    ap.add_argument("--model", default="riskprop", choices=["riskprop", "dummy"])
    ap.add_argument("--limit-dense", type=int, default=0, help="dev only: score first N dense videos")
    ap.add_argument("--analysis-only", action="store_true")
    a = ap.parse_args(argv)
    a.conditions = [c.strip() for c in a.conditions.split(",") if c.strip()]
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    a.preds_dir = os.path.join(a.out_dir, "preds_val")
    a.dense_dir = os.path.join(a.out_dir, "dense_val")
    os.makedirs(a.out_dir, exist_ok=True)
    runs, a.run_cond = {}, {}
    for c in a.conditions:
        for s in seeds:
            r = run_name(c, a.pairing_mode, a.horizon_sec, a.alpha, a.lambda_prog, s, a.tag)
            rdir = os.path.join(a.runs_root, r)
            if not os.path.isdir(rdir):
                sys.exit(f"[MISSING run dir] {rdir}")
            runs[r] = rdir
            a.run_cond[r] = (c, s)
    a.research = all(json.load(open(os.path.join(d, "config.json")))["config"].get("research_result")
                     for d in runs.values())
    check_same_data(runs)
    if a.analysis_only:
        chosen = json.load(open(os.path.join(a.out_dir, "chosen_checkpoints.json")))
        for run in runs:                      # saved choice must still match the files on disk
            p = os.path.join(runs[run], f"{chosen[run]['kind']}.pth")
            if chosen[run].get("ckpt_sha256") != file_sha(p):
                sys.exit(f"[REFUSE --analysis-only] {p} changed since chosen_checkpoints.json; rerun without it")
            pj = json.load(open(os.path.join(a.preds_dir, f"{run}_{chosen[run]['kind']}.json")))
            dz = np.load(os.path.join(a.dense_dir, f"{run}.npz"))
            if pj.get("ckpt_sha256") != file_sha(p) or str(dz.get("ckpt_sha256", "")) != file_sha(p):
                sys.exit(f"[REFUSE --analysis-only] predictions of {run} were not made from {p}")
    else:
        chosen = gpu_steps(a, runs)
    analyze(a, runs, chosen)


if __name__ == "__main__":
    main()
