"""
re15 (v4) -- Validation evaluation for Progressive RiskProp (RQ3). Reuses the
LOCKED project protocol instead of a second evaluator:
  * 3-lead windows (0.5/1.0/1.5 s) from nexar_cache_5f/val via
    re02.load_val_tensors / re02.predict (same tensors as RQ1-RQ3);
  * checkpoint choice, identical for every run: the trainer keeps `best`
    (lowest val BCE @1.0 s) and `latest` (last epoch); here the one with the
    higher mean val AP over the 3 leads is chosen; exact float tie -> best;
  * metrics = re03.full_metrics (AP, low-FAR mAUC@0.1, Recall / threshold / actual
    FAR at FAR <= 0.1, mTTA_detected, Coverage) -- unchanged definitions;
  * dense causal curves = re07 sweep (30 windows, d = 3.0 -> 0.1 s, early -> late);
    temporal metrics = re10.temporal_per_video (PVR eps 0.01, ADS, RCJ);
  * paired label-stratified video bootstrap (B=2000, seed 12345); a condition's
    value = mean over its seeds (not a seed ensemble).

v4 changes
  * COHORTS: accuracy metrics and their CIs always use all 300 validation videos;
    temporal metrics use the 150 positives. Missing dense curves never remove a
    video from the accuracy cohort. Research mode fails on any missing/non-finite
    prediction, wrong id order, or incomplete dense coverage.
  * FRESHNESS: predictions are reused only if made from the same checkpoint file
    (sha256) and the same validation-cache CONTENT digest (every val .pt); dense
    curves only if same checkpoint and same dense-cache content digest
    (windows_u8.npy + meta.json). A one-pixel change forces recomputation.
  * DENSE TIMING GATE uses re11's val_timing.csv (actual PTS of every dense
    endpoint): 300 videos / 150 positives, requested endpoints == this dense
    cache's meta, no endpoint at/after the event, no duplicate end frames, no
    clamping, |actual - requested| <= 1 frame of THAT video's fps + 1 ms, no VFR
    suspect. Fail -> temporal metrics are diagnostic only (--strict-dense stops).
  * DECISION from LOCK.json on RAW floats (rounding only in printed tables),
    finite CIs required, and only when the RESEARCH GATE passes: B/P/S x 3 seeds
    present, each run full_run=True + COMPLETE.json matching its checkpoints,
    same data/code fingerprints, same init digest per seed across conditions.
    Otherwise the report says "no research conclusion".
  * strata: alert-event gap bins plus >1.5 s and >2.0 s on the validation set.
mTTA_detected = the largest of the 3 discrete leads detected at the FAR-0.1
threshold (re03), reported with Coverage and actual FAR -- not PRE-ACT's mTTA and
not an onset metric. The official test set is NOT touched (see re19).
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "pipeline"))
import progress_common as pc  # noqa: E402
from re10_rq3_sensitivity_eval import temporal_per_video, md_table  # noqa: E402
from re13_progress_supervision import continuous_target_scalar  # noqa: E402
from re14_train_progress_riskprop import run_name, build_model  # noqa: E402

LEADS = [0.5, 1.0, 1.5]
KINDS = ["best", "latest"]
GAP_BINS = [-np.inf, 0.5, 1.0, 1.5, 2.0, np.inf]
GAP_LABELS = ["<=0.5", "0.5-1.0", "1.0-1.5", "1.5-2.0", ">2.0"]
BOOT_COLS = ["mAP", "AP@1.5s", "mAUC01", "PVR", "ADS", "RCJ"]
TEMPORAL = ("PVR", "ADS", "RCJ")
HIGHER_BETTER = {"mAP": True, "AP@1.5s": True, "mAUC01": True, "PVR": False, "ADS": False, "RCJ": False}
ALL_CONTRASTS = [("P - B", {"P": 1, "B": -1}), ("S - B", {"S": 1, "B": -1}),
                 ("P - S", {"P": 1, "S": -1}), ("FP - F", {"FP": 1, "F": -1}),
                 ("(P-B) - (FP-F)", {"P": 1, "B": -1, "FP": -1, "F": 1})]
MAIN_CONDITIONS, MAIN_SEEDS = ("B", "P", "S"), (42, 43, 44)
DENSE_TOL_S = 1e-3          # dense timing gate: |actual - requested| <= 1 frame + 1 ms


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
            "score_neg_mean": float(dense[neg].mean()) if neg.any() else float("nan"),
            "score_neg_p95": float(np.percentile(dense[neg].max(1), 95)) if neg.any() else float("nan")}


def target_fit(dense, tpos, d_grid, t_obs, toe_arr, horizon, alpha):
    """MAE of dense positive curves to the CPS targets on unclamped windows (diagnostic only)."""
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


# ----------------------------------------------------------------------------
# Provenance / freshness
# ----------------------------------------------------------------------------
_SHA = {}


def file_sha(path):
    """Full sha256 of a file (cached per process, keyed on path + size + mtime so a
    file rewritten during the run is re-hashed)."""
    st = os.stat(path)
    key = (path, st.st_size, st.st_mtime_ns)
    if key not in _SHA:
        _SHA[key] = pc.sha256_file(path)
    return _SHA[key]


def preds_fresh(out, ckpt, val_digest):
    """Reuse a prediction file only if made from THIS checkpoint file and THIS
    validation cache content (digest of every val .pt)."""
    js = out.replace(".npz", ".json")
    if not (os.path.exists(out) and os.path.exists(js)):
        return False
    m = json.load(open(js))
    ok = m.get("ckpt_sha256") == file_sha(ckpt) and m.get("val_cache_digest") == val_digest
    if not ok:
        print(f"  [STALE] {os.path.basename(out)} does not match the current checkpoint/val cache -> recompute")
    return ok


def dense_fresh(out, ckpt, dense_digest):
    if not os.path.exists(out):
        return False
    z = np.load(out)
    ok = ("ckpt_sha256" in z and str(z["ckpt_sha256"]) == file_sha(ckpt)
          and "dense_cache_digest" in z and str(z["dense_cache_digest"]) == dense_digest)
    if not ok:
        print(f"  [STALE] {os.path.basename(out)} -> recompute")
    return ok


DATA_KEYS = ("split_manifest_sha256", "train_index_sha256", "val_index_sha256", "sidecar_sha256",
             "val_cache_digest", "train_cache_digest")


def check_same_data(runs):
    """All runs must come from the same split / caches / sidecar."""
    seen = {}
    for run, rdir in runs.items():
        inp = json.load(open(os.path.join(rdir, "config.json")))["inputs"]
        sig = tuple(inp.get(k) for k in DATA_KEYS)
        seen.setdefault(sig, []).append(run)
    if len(seen) > 1:
        sys.exit("[REFUSE] runs were trained on different data/sidecars:\n  " +
                 "\n  ".join(f"{dict(zip(DATA_KEYS, [x[:12] if x else x for x in sig]))}: {r}"
                             for sig, r in seen.items()))


def research_gate(runs, run_cond, lock_file, conditions, seeds):
    """Conditions for a research conclusion. Returns list of issues (empty = OK)."""
    issues = []
    present = {(c, s) for c, s in run_cond.values()}
    for c in MAIN_CONDITIONS:
        for s in MAIN_SEEDS:
            if (c, s) not in present:
                issues.append(f"missing run {c} seed {s}")
    if not lock_file or not os.path.exists(lock_file):
        issues.append("no LOCK.json")
    lock_sha = pc.sha256_file(lock_file) if lock_file and os.path.exists(lock_file) else None
    inits, codes = {}, set()
    for run, rdir in runs.items():
        cfgp = os.path.join(rdir, "config.json")
        man = json.load(open(cfgp))
        if man["config"].get("synthetic") or pc.SYNTHETIC:
            issues.append(f"{run}: synthetic data")
        if not man["config"].get("research_result") or not man["config"].get("full_run"):
            issues.append(f"{run}: not a full research run")
        if man["inputs"].get("lock_sha256") != lock_sha:
            issues.append(f"{run}: trained under a different LOCK")
        comp = os.path.join(rdir, "COMPLETE.json")
        if not os.path.exists(comp):
            issues.append(f"{run}: no COMPLETE.json")
        else:
            cj = json.load(open(comp))
            for kind in KINDS:
                f = os.path.join(rdir, f"{kind}.pth")
                if not os.path.exists(f) or file_sha(f) != cj.get("checkpoints", {}).get(kind):
                    issues.append(f"{run}: {kind}.pth missing or changed after COMPLETE.json")
        c, s = run_cond[run]
        inits.setdefault(s, set()).add(man.get("init_state_sha256"))
        codes.add(man.get("code", {}).get("digest"))
    for s, v in inits.items():
        if len(v) != 1:
            issues.append(f"seed {s}: conditions start from different initial weights")
    if len(codes) > 1:
        issues.append("runs were trained with different code (code digest differs)")
    return issues


# ----------------------------------------------------------------------------
# Dense timing gate and decision
# ----------------------------------------------------------------------------
def make_cohorts(targets, has_toe, dense_list):
    """Accuracy cohort = ALL videos; temporal cohort = positives with an event and finite
    dense curves in every run. Missing dense curves never shrink the accuracy cohort."""
    acc = np.arange(len(targets))
    dense_ok = np.all(np.isfinite(np.stack(dense_list)), axis=(0, 2))
    return acc, has_toe & dense_ok, dense_ok


def dense_gate(meta, val_ids, toe, val_timing="", expected=(pc.N_VAL, pc.N_VAL_POS)):
    """Is the dense validation grid timed correctly? (see module docstring)."""
    issues, c = [], {"videos": len(meta.get("vid_ids", [])), "positives_checked": 0, "duplicate": 0,
                     "clamped": 0, "post_event": 0, "nonfinite": 0, "dev_gt_1_frame": 0, "vfr": 0,
                     "audit_mismatch": 0, "audit_missing": 0}
    if meta.get("failed"):
        issues.append(f"{len(meta['failed'])} dense videos failed to decode (zeros): {meta['failed'][:5]}")
    if list(meta["vid_ids"]) != list(val_ids):
        issues.append("dense cache video order/ids != validation index")
    d = np.asarray(meta["dense_d"], dtype=float)
    pos_vids = []
    for vid, tgt, t in zip(meta["vid_ids"], meta["targets"], meta["t_obs"]):
        e = toe.get(vid)
        if int(tgt) != 1 or e is None or pd.isna(e):
            continue
        pos_vids.append(vid)
        c["positives_checked"] += 1
        t = np.asarray(t, dtype=float)
        if not np.all(np.isfinite(t)):
            c["nonfinite"] += 1
            continue
        if len(np.unique(np.round(t, 6))) < len(t):
            c["duplicate"] += 1
        if np.any(np.abs(t - (float(e) - d)) > 1e-6):
            c["clamped"] += 1
        if np.any(t > float(e) - 0.1 + 1e-6):
            c["post_event"] += 1
    for k, msg in (("nonfinite", "positives with non-finite endpoints"),
                   ("duplicate", "positives with duplicate requested dense endpoints"),
                   ("clamped", "positives with clamped dense endpoints (not at t_event - d)"),
                   ("post_event", "positives with a requested endpoint later than t_event - 0.1 s")):
        if c[k]:
            issues.append(f"{c[k]} {msg}")
    if not val_timing or not os.path.exists(val_timing):
        issues.append("actual dense endpoint times not verified (pass --val-timing <re11 val_timing.csv>)")
        return {"pass": False, "issues": issues, "counts": c}
    vt = pd.read_csv(val_timing, dtype={"vid": str})
    vt["vid"] = vt["vid"].str.zfill(5)
    vd = vt[vt.kind == "dense"]
    tmeta = dict(zip(meta["vid_ids"], meta["t_obs"]))
    for vid in pos_vids:
        rows = vd[vd.vid == vid].sort_values("k")
        if len(rows) != len(d):
            c["audit_missing"] += 1
            continue
        if np.any(np.abs(rows["requested_end_s"].values - np.asarray(tmeta[vid], dtype=float)) > 1e-6):
            c["audit_mismatch"] += 1
        if rows["future_frame"].astype(bool).any():
            c["post_event"] += 1
        if rows["end_frame"].duplicated().any():
            c["duplicate"] += 1
        # tolerance: one frame of THAT video + 1 ms for timestamp / float rounding
        # (int(t*fps) can land exactly one frame early; observed excess <= 0.001 ms)
        if (rows["dev_s"].abs() > rows["frame_period_s"] + DENSE_TOL_S).any():
            c["dev_gt_1_frame"] += 1
        if rows.get("vfr_suspect") is not None and rows["vfr_suspect"].fillna(False).astype(bool).any():
            c["vfr"] += 1
    for k, msg in (("audit_missing", "positives without 30 audited dense endpoints"),
                   ("audit_mismatch", "positives whose audited requested endpoints != this dense cache"),
                   ("dev_gt_1_frame", "positives with |actual - requested| endpoint > 1 frame + 1 ms"),
                   ("vfr", "positives from VFR-suspect videos")):
        if c[k]:
            issues.append(f"{c[k]} {msg}")
    if c["post_event"] and not any("later than" in i for i in issues):
        issues.append(f"{c['post_event']} positives with an actual endpoint at/after the event")
    if expected and (c["videos"] != expected[0] or c["positives_checked"] != expected[1]):
        issues.append(f"dense cache covers {c['videos']} videos / {c['positives_checked']} positives "
                      f"(expected {expected[0]} / {expected[1]})")
    return {"pass": not issues, "issues": issues, "counts": c}


def _finite(*xs):
    return all(x is not None and isinstance(x, (int, float)) and math.isfinite(x) for x in xs)


def decision(ci, lock, research_issues=()):
    """Pre-registered RQ3 decision from LOCK.json on RAW floats (P-S always reported)."""
    if not lock or ci is None or not len(ci):
        return {"available": False, "reason": "no lock file or no contrasts"}
    pm, pcn = lock.get("primary_metric"), lock.get("primary_contrast", "P - B")
    nim, margin, psm = lock.get("ni_metric"), lock.get("ni_margin"), lock.get("ps_metric")

    def row(contrast, metric):
        r = ci[(ci.contrast == contrast) & (ci.metric == metric)]
        return r.iloc[0].to_dict() if len(r) else None
    prim, ni = row(pcn, pm), row(pcn, nim)
    ps = row("P - S", psm) if psm else None
    out = {"available": True, "primary_contrast": pcn, "primary_metric": pm, "ni_metric": nim,
           "ni_margin": margin, "ps_metric": psm, "p_minus_s": ps, "research_issues": list(research_issues)}
    if prim is None or ni is None:
        out.update(available=False, reason=f"contrast {pcn} / metrics not evaluated")
        return out
    if not _finite(prim["ci_low"], prim["ci_high"], ni["ci_low"], ni["ci_high"], margin):
        out.update(available=False, reason="non-finite CI or margin")
        return out
    better = prim["ci_low"] > 0 if HIGHER_BETTER[pm] else prim["ci_high"] < 0
    ni_ok = (ni["ci_low"] > -margin) if HIGHER_BETTER[nim] else (ni["ci_high"] < margin)
    out.update(primary=prim, primary_met=bool(better), non_inferiority=ni, non_inferiority_met=bool(ni_ok),
               research_gate_passed=not research_issues,
               rq3_supported=bool(better and ni_ok and not research_issues))
    if research_issues:
        out["reason"] = "research gate failed -> no research conclusion"
    return out


# ----------------------------------------------------------------------------
# GPU part
# ----------------------------------------------------------------------------
def gpu_steps(a, runs):
    import torch
    import re02_infer_val as r2
    import re07_infer_rq2 as r7
    from cell22_adalea_dataset import _to_tensor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(a.preds_dir, exist_ok=True)
    os.makedirs(a.dense_dir, exist_ok=True)
    print(f"device={device}\n[1] 3-lead validation predictions (best + latest)")
    val_digest = pc.val_cache_digest(a.cache_5f)
    vid_ids, targets, data = r2.load_val_tensors(a.cache_5f)
    taus = np.stack([data[l][1] for l in LEADS])
    model = build_model(a.model).to(device).eval()
    for run, rdir in runs.items():
        for kind in KINDS:
            out = os.path.join(a.preds_dir, f"{run}_{kind}.npz")
            p = os.path.join(rdir, f"{kind}.pth")
            if not os.path.exists(p):
                sys.exit(f"  [MISSING] {p} -- training not finished?")
            if preds_fresh(out, p, val_digest):
                continue
            ck = torch.load(p, map_location="cpu", weights_only=False)
            model.load_state_dict(ck["model"])
            model.eval()
            scores = np.stack([r2.predict(model, data[l][0], device, 8) for l in LEADS])
            np.savez_compressed(out, vid_ids=np.array(vid_ids), targets=targets, taus=taus,
                                leads=np.array(LEADS), scores=scores)
            pc.write_json_atomic(out.replace(".npz", ".json"),
                                 {"run": run, "kind": kind, "ckpt": p, "ckpt_sha256": file_sha(p),
                                  "val_cache_digest": val_digest, "ckpt_epoch": ck.get("epoch"),
                                  "config": ck.get("config")})
            print(f"  [OK] {run} {kind} epoch={ck.get('epoch')}")

    print("\n[2] Checkpoint choice (per_run rule; exact tie -> best)")
    chosen = {}
    for run in runs:
        maps = {}
        for kind in KINDS:
            z = np.load(os.path.join(a.preds_dir, f"{run}_{kind}.npz"))
            maps[kind] = float(np.mean(lead_maps(z["targets"], z["scores"])))
        kind = "best" if maps["best"] >= maps["latest"] else "latest"
        chosen[run] = {"kind": kind, "val_mAP_best": maps["best"], "val_mAP_latest": maps["latest"],
                       "ckpt": os.path.join(runs[run], f"{kind}.pth"),
                       "ckpt_sha256": file_sha(os.path.join(runs[run], f"{kind}.pth")),
                       "rule": "higher mean val AP over 0.5/1.0/1.5 s among best/latest; tie -> best"}
        print(f"  {run}: best={maps['best']:.4f} latest={maps['latest']:.4f} -> {kind}")

    print("\n[3] Dense risk curves (re07 sweep)")
    if not os.path.exists(os.path.join(a.dense_cache, "windows_u8.npy")):
        r7.build_dense_cache(a.data_dir, a.cache_5f, a.dense_cache)
    arr = np.load(os.path.join(a.dense_cache, "windows_u8.npy"), mmap_mode="r")
    dense_digest = pc.dense_cache_digest(a.dense_cache)
    todo = [r for r in runs if not dense_fresh(os.path.join(a.dense_dir, f"{r}.npz"), chosen[r]["ckpt"],
                                               dense_digest)]
    with open(os.path.join(a.dense_cache, "meta.json")) as f:
        meta = json.load(f)
    n_vid = arr.shape[0] if not a.limit_dense else min(a.limit_dense, arr.shape[0])
    for run in todo:
        ck = torch.load(chosen[run]["ckpt"], map_location="cpu", weights_only=False)
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
                            dense_cache_digest=np.array(dense_digest))
    out = {"chosen": chosen, "val_cache_digest": val_digest, "dense_cache_digest": dense_digest,
           "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    pc.write_json_atomic(os.path.join(a.out_dir, "chosen_checkpoints.json"), out)
    return chosen


# ----------------------------------------------------------------------------
# Analysis (CPU)
# ----------------------------------------------------------------------------
def analyze(a, runs, chosen):
    import re03_analyze_rq as r3
    with open(os.path.join(a.dense_cache, "meta.json")) as f:
        meta = json.load(f)
    df = pd.read_csv(os.path.join(a.data_dir, "train.csv"))
    df["vid"] = df["id"].apply(lambda x: str(int(x)).zfill(5))
    toe = dict(zip(df["vid"], df["time_of_event"]))
    toa = dict(zip(df["vid"], df["time_of_alert"]))
    lock = json.load(open(a.lock_file)) if a.lock_file and os.path.exists(a.lock_file) else {}
    man_val = json.load(open(a.split_manifest))["val_ids"] if a.split_manifest else None

    ls, dense, info = {}, {}, {}
    ref_ids = targets = None
    research_issues = list(a.research_issues)
    for run in runs:
        c, s = a.run_cond[run]
        kind = chosen[run]["kind"]
        z = np.load(os.path.join(a.preds_dir, f"{run}_{kind}.npz"))
        d = np.load(os.path.join(a.dense_dir, f"{run}.npz"))
        if ref_ids is None:
            ref_ids, targets = z["vid_ids"], z["targets"].astype(int)
        if not (np.array_equal(z["vid_ids"], ref_ids) and np.array_equal(d["vid_ids"], ref_ids)):
            sys.exit(f"[REFUSE] {run}: prediction/dense video order differs from the other runs")
        if not np.all(np.isfinite(z["scores"])):
            research_issues.append(f"{run}: non-finite 3-lead predictions")
        ls[run] = [z["scores"][i] for i in range(3)]
        dense[run] = d["scores"].astype(np.float64)
        info[run] = (c, s, kind)
    N = len(targets)
    if man_val is not None and list(map(str, ref_ids)) != list(man_val):
        research_issues.append("validation prediction ids/order != split manifest val_ids")
    if N != pc.N_VAL or int(targets.sum()) != pc.N_VAL_POS:
        research_issues.append(f"accuracy cohort has {N} videos / {int(targets.sum())} positives (need 300/150)")
    pos_idx, neg_idx = np.where(targets == 1)[0], np.where(targets == 0)[0]
    has_toe = np.array([t == 1 and not pd.isna(toe.get(v)) for v, t in zip(ref_ids, targets)])
    _, tpos, dense_ok = make_cohorts(targets, has_toe, list(dense.values()))
    negm = (targets == 0) & dense_ok
    if int(tpos.sum()) != int(has_toe.sum()):
        research_issues.append(f"dense curves cover {int(tpos.sum())}/{int(has_toe.sum())} positives")
    toe_arr = np.array([float(toe.get(v)) if not pd.isna(toe.get(v)) else np.nan for v in ref_ids])
    gap = np.array([float(toe.get(v)) - float(toa.get(v)) if not (pd.isna(toe.get(v)) or pd.isna(toa.get(v)))
                    else np.nan for v in ref_ids])
    t_obs = np.array(meta["t_obs"], dtype=np.float64)
    d_grid = meta["dense_d"]

    rows, pv, strata = [], {}, []
    groups = [(lab, pd.cut(gap, bins=GAP_BINS, labels=GAP_LABELS).astype(object) == lab) for lab in GAP_LABELS]
    groups += [(">1.5 s", gap > 1.5), (">2.0 s", gap > 2.0)]
    for run, (c, s, kind) in info.items():
        m = r3.full_metrics(ls[run], targets, ls[run][1])      # accuracy cohort = all N videos
        p, ad, j = temporal_per_video(np.nan_to_num(dense[run]))
        pv[run] = {"PVR": p, "ADS": ad, "RCJ": j}
        rows.append(dict(condition=c, seed=s, run=run, checkpoint=kind, n_accuracy=N, n_temporal=int(tpos.sum()),
                         mAP=m["mAP"], **{f"AP@{l}s": m[f"AP@{l}s"] for l in LEADS}, mAUC01=m["mAUC01"],
                         **{f"Recall@{l}s": m[f"Recall@{l}s"] for l in LEADS},
                         **{f"ActualFAR@{l}s": m[f"ActualFAR@{l}s"] for l in LEADS},
                         mTTA_detected=m["mTTA_detected"], Coverage=m["Coverage"],
                         PVR=p[tpos].mean(), ADS=ad[tpos].mean(), RCJ=j[tpos].mean(),
                         **amplitude(dense[run], tpos, negm, d_grid),
                         **target_fit(dense[run], tpos, d_grid, t_obs, toe_arr, a.horizon_sec, a.alpha)))
        for lab, gmask in groups:
            bp = np.where(np.asarray(gmask, dtype=bool) & (targets == 1))[0]
            if len(bp) == 0:
                continue
            idx = np.concatenate([bp, neg_idx])
            aps = lead_maps(targets[idx], [x[idx] for x in ls[run]])
            bpt = bp[tpos[bp]]
            dd = np.array(d_grid)
            strata.append(dict(condition=c, seed=s, gap_group=lab, n_pos=len(bp), n_neg=len(neg_idx),
                               mAP_group=float(np.mean(aps)), **{f"AP@{l}s": float(v) for l, v in zip(LEADS, aps)},
                               PVR=float(p[bpt].mean()) if len(bpt) else np.nan,
                               rise_pos=float(dense[run][bpt][:, dd <= 0.5].mean() - dense[run][bpt][:, dd >= 2.5].mean())
                               if len(bpt) else np.nan))
    per_run = pd.DataFrame(rows)
    per_run.to_csv(os.path.join(a.out_dir, "per_run_metrics.csv"), index=False)
    st = pd.DataFrame(strata)
    st.to_csv(os.path.join(a.out_dir, "alert_gap_strata_per_run.csv"), index=False)

    conds = [c for c in a.conditions if c in set(per_run.condition)]
    for c in conds:
        n = int((per_run.condition == c).sum())
        if c in MAIN_CONDITIONS and n != len(MAIN_SEEDS):
            research_issues.append(f"condition {c} has {n} seeds (need 3)")
    cols = ["mAP", "AP@0.5s", "AP@1.0s", "AP@1.5s", "mAUC01", "Recall@1.0s", "ActualFAR@1.0s", "mTTA_detected",
            "Coverage", "PVR", "ADS", "RCJ", "score_early_pos", "score_late_pos", "rise_pos", "range_pos",
            "score_neg_mean", "score_neg_p95", "fit_MAE_continuous", "fit_MAE_binary"]

    def msd(x):
        return f"{x.mean():.4f} +/- {x.std(ddof=1):.4f}" if len(x) > 1 else f"{x.mean():.4f}"
    agg = pd.DataFrame([{"Condition": c, "n_seeds": int((per_run.condition == c).sum()),
                         **{k: msd(per_run[per_run.condition == c][k]) for k in cols}} for c in conds])
    agg.to_csv(os.path.join(a.out_dir, "mean_sd.csv"), index=False)
    st_agg = (st.groupby(["gap_group", "condition"], sort=False)
              [["n_pos", "mAP_group", "AP@1.5s", "PVR", "rise_pos"]].mean().reset_index()) if len(st) else st
    st_agg.to_csv(os.path.join(a.out_dir, "alert_gap_strata_mean.csv"), index=False)

    gate = dense_gate(meta, list(ref_ids), toe, a.val_timing)
    pc.write_json_atomic(os.path.join(a.out_dir, "dense_gate.json"), gate)
    if not gate["pass"]:
        print("[DENSE GATE] FAIL -> PVR/ADS/RCJ/amplitude/target-fit are diagnostic only:\n  - "
              + "\n  - ".join(gate["issues"]))
        if a.strict_dense:
            sys.exit("[REFUSE] --strict-dense and the dense timing gate failed")

    # paired stratified bootstrap: accuracy on ALL videos, temporal on positives with dense curves
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
                for key in TEMPORAL:
                    v[key].append(pv[run][key][tp].mean() if len(tp) else np.nan)
            out[c] = {k: float(np.mean(x)) for k, x in v.items()}
        return out

    ci = pd.DataFrame()
    if contrasts:
        print(f"\n[4] Paired stratified bootstrap B={a.n_boot}: {[n for n, _ in contrasts]}")
        rng = np.random.default_rng(a.boot_seed)
        full = cvals(np.arange(N))
        boot = {n: {k: [] for k in BOOT_COLS} for n, _ in contrasts}
        t0 = time.time()
        for b in range(a.n_boot):
            idx = np.concatenate([rng.choice(pos_idx, len(pos_idx)), rng.choice(neg_idx, len(neg_idx))])
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
                lo, hi = (float(x) for x in np.percentile(boot[n][k], [2.5, 97.5]))
                if not (math.isfinite(lo) and math.isfinite(hi)):
                    verdict, direction = "non-finite CI", ""
                elif lo <= 0 <= hi:
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
                role = ("primary" if (k == lock.get("primary_metric") and n == lock.get("primary_contrast", "P - B"))
                        else "pre-specified" if (k == lock.get("ps_metric") and n == "P - S")
                        else "accuracy" if k in ("mAP", "AP@1.5s", "mAUC01")
                        else "supporting" if gate["pass"] else "diagnostic (dense gate failed)")
                cohort = (f"temporal n={int(tpos.sum())} positives" if k in TEMPORAL
                          else f"accuracy n={N} videos")
                rows.append(dict(contrast=n, metric=k, role=role, cohort=cohort, estimate=est, ci_low=lo,
                                 ci_high=hi, verdict=verdict, direction=direction, non_inferiority=ni))
        ci = pd.DataFrame(rows)
        ci.to_csv(os.path.join(a.out_dir, "bootstrap_ci.csv"), index=False, float_format="%.10g")

    dec = decision(ci, lock, research_issues)
    pc.write_json_atomic(os.path.join(a.out_dir, "rq3_decision.json"), dec)
    if dec.get("available"):
        p, n_ = dec["primary"], dec["non_inferiority"]
        dec_lines = [
            "## 0. Pre-registered decision (LOCK.json; raw floats)\n",
            f"- Research gate: {'PASS' if dec['research_gate_passed'] else 'FAIL -> no research conclusion'}"
            + ("" if dec["research_gate_passed"] else ": " + "; ".join(research_issues[:8])),
            f"- Primary: {dec['primary_contrast']} on {dec['primary_metric']} = {p['estimate']:+.4f} "
            f"[{p['ci_low']:+.4f}, {p['ci_high']:+.4f}] -> {'MET (CI > 0)' if dec['primary_met'] else 'NOT met'}",
            f"- Non-inferiority: {dec['primary_contrast']} on {dec['ni_metric']} = {n_['estimate']:+.4f} "
            f"[{n_['ci_low']:+.4f}, {n_['ci_high']:+.4f}], margin {dec['ni_margin']} -> "
            f"{'MET' if dec['non_inferiority_met'] else 'NOT met'}",
            f"- **RQ3 supported (accuracy-based pre-registered rule): {'YES' if dec['rq3_supported'] else 'NO'}**",
            (f"- P - S on {dec['ps_metric']} (pre-specified, reported regardless): "
             f"{dec['p_minus_s']['estimate']:+.4f} [{dec['p_minus_s']['ci_low']:+.4f}, "
             f"{dec['p_minus_s']['ci_high']:+.4f}] -> {dec['p_minus_s']['verdict']}")
            if dec.get("p_minus_s") else "- P - S: not evaluated (S missing)",
            "- A YES is accuracy support only; temporal improvement needs the dense gate to pass AND supportive "
            "PVR/ADS/RCJ with non-flat curves (rise/range). P > B with P ~ S supports extra supervision, not the "
            "continuous shape specifically.", ""]
    else:
        dec_lines = [f"## 0. Pre-registered decision: not available ({dec.get('reason')})\n"]
    gate_line = ("Dense timing gate: PASS" if gate["pass"] else
                 "Dense timing gate: **FAIL** -> temporal metrics are diagnostic only: " + "; ".join(gate["issues"]))
    ci_show = ci.copy()
    for k in ("estimate", "ci_low", "ci_high"):
        if k in ci_show:
            ci_show[k] = ci_show[k].map(lambda x: f"{x:+.4f}")
    st_show = st_agg.round(4) if len(st_agg) else st_agg
    rep = ["# RQ3 -- Progressive RiskProp, internal validation\n",
           f"Generated {time.strftime('%Y-%m-%d %H:%M')} | accuracy cohort {N} videos ({len(pos_idx)} positives) | "
           f"temporal cohort {int(tpos.sum())} positives | pairing={a.pairing_mode} H={a.horizon_sec} "
           f"alpha={a.alpha} lambda={a.lambda_prog} | checkpoint rule: per_run, tie -> best | "
           f"research_result={not research_issues}\n",
           gate_line + "\n", *dec_lines,
           "## 1. Accuracy / low-FAR (mean +/- SD over seeds; all validation videos)\n",
           md_table(agg[["Condition", "n_seeds", "mAP", "AP@0.5s", "AP@1.0s", "AP@1.5s", "mAUC01",
                         "Recall@1.0s", "ActualFAR@1.0s", "mTTA_detected", "Coverage"]]), "",
           "mTTA_detected = largest of the 3 discrete leads detected at FAR <= 0.1 (discrete lead-time summary, "
           "not PRE-ACT's mTTA, not an onset metric); always read with Coverage.\n",
           "## 2. Temporal metrics WITH score amplitude (positives, dense curves; PVR/ADS/RCJ lower = better)\n",
           md_table(agg[["Condition", "PVR", "ADS", "RCJ", "score_early_pos", "score_late_pos", "rise_pos",
                         "range_pos", "score_neg_mean", "score_neg_p95"]]), "",
           "A low PVR/RCJ with a small rise/range means a flat curve, not better early warning.\n",
           "## 3. Target fit (diagnostic only, unclamped dense windows)\n",
           md_table(agg[["Condition", "fit_MAE_continuous", "fit_MAE_binary"]]), "",
           "## 4. Alert-event gap groups (validation; mean over seeds; positives of the group + all negatives)\n",
           md_table(st_show) if len(st_show) else "(none)", "",
           f"## 5. Contrasts (paired label-stratified bootstrap 95% CI, B={a.n_boot}, seed {a.boot_seed})\n",
           md_table(ci_show) if len(ci_show) else "(need >= 2 conditions)", "",
           "Notes: validation only (internal; CIs after checkpoint choice are not an independent confirmation; "
           "video bootstrap does not cover training-seed uncertainty -- see mean +/- SD). Official test: re19."]
    with open(os.path.join(a.out_dir, "summary_rq3_progress.md"), "w") as f:
        f.write("\n".join(rep))
    print("\n".join(rep))
    print(f"\nWritten to {a.out_dir}")
    return dec


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
    ap.add_argument("--split-manifest", default=os.path.join(REPO, "results/repro/split_manifest_seed42.json"))
    ap.add_argument("--lock-file", default="")
    ap.add_argument("--val-timing", default="", help="re11 val_timing.csv (dense timing gate)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--boot-seed", type=int, default=12345)
    ap.add_argument("--model", default="riskprop", choices=["riskprop", "dummy"])
    ap.add_argument("--limit-dense", type=int, default=0, help="dev only: score first N dense videos")
    ap.add_argument("--analysis-only", action="store_true")
    ap.add_argument("--strict-dense", action="store_true",
                    help="stop if the dense timing gate fails (default: label temporal metrics diagnostic)")
    ap.add_argument("--research", action="store_true",
                    help="research mode: exit non-zero unless the research gate passes")
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
    check_same_data(runs)
    a.research_issues = research_gate(runs, a.run_cond, a.lock_file, a.conditions, seeds)
    if a.analysis_only:
        saved = json.load(open(os.path.join(a.out_dir, "chosen_checkpoints.json")))
        chosen = saved["chosen"]
        dense_digest = pc.dense_cache_digest(a.dense_cache)
        val_digest = pc.val_cache_digest(a.cache_5f)
        for run in runs:
            p = chosen[run]["ckpt"]
            pj = json.load(open(os.path.join(a.preds_dir, f"{run}_{chosen[run]['kind']}.json")))
            dz = np.load(os.path.join(a.dense_dir, f"{run}.npz"))
            if (chosen[run].get("ckpt_sha256") != file_sha(p) or pj.get("ckpt_sha256") != file_sha(p)
                    or pj.get("val_cache_digest") != val_digest
                    or str(dz.get("ckpt_sha256", "")) != file_sha(p)
                    or str(dz.get("dense_cache_digest", "")) != dense_digest):
                sys.exit(f"[REFUSE --analysis-only] saved predictions of {run} are stale; rerun without it")
    else:
        chosen = gpu_steps(a, runs)
    dec = analyze(a, runs, chosen)
    if a.research and (dec.get("research_issues") or not dec.get("available")):
        print("[RESEARCH GATE] FAIL: " + "; ".join(dec.get("research_issues") or [dec.get("reason", "")]))
        sys.exit(5)


if __name__ == "__main__":
    main()
