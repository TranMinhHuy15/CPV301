"""
re11 (v4) -- READ-ONLY data / split / media / cache audit for Progressive
RiskProp (gate G2, plus the G1 binding). Writes only into --output-dir.

Fatal checks (exit 2, audit_summary.json "pass": false):
  F1  train.csv sha256 == split manifest train_csv_sha256
  F2  seed-42 stratified split recomputed from train.csv == manifest (ids AND order)
  F3  train / val disjoint
  F4  RiskProp train cache ids == manifest train ids (1200); nexar_cache_5f
      val_index ids == manifest val ids in order (300), each with leads 0.5/1.0/1.5
  F5  positives: finite time_of_event >= 0; negatives: no event
  F6  every video opens; avg fps finite > 0; positives: time_of_event <= last
      frame time + one frame
  F7  data/train/<id>.mp4 resolves to the HF file recorded in
      results/repro/id_to_hf_source.csv (all 1500 must be verifiable)
  F8  PTS of every frame finite, strictly increasing, first PTS >= 0
  F9  counts: 1500 probed, 1200 train .pt scanned, 300 x 3 val .pt scanned
  F10 binding: --prepare-report (re17) passed, same train.csv sha256, revision recorded
  F11 validation lead windows (re01 rule) never end at/after time_of_event
      (actual PTS of the last frame)
`complete` is true only when nothing was limited and F7/F9/F10 could run;
re12/re20/re14 refuse audits that are not complete.

Timing policy (documented, applied by re12/re15):
  * clip-time origin = first frame; a non-zero first PTS is an offset that is
    subtracted (pts_offset_s) -- time_of_event is measured from the first frame;
  * one frame = 1 / avg_fps of THAT video (not a hard-coded 1/30);
  * VFR indicator = max |PTS - index/avg_fps| > one frame -> WARN here; such a
    video makes the dense timing gate fail (temporal metrics diagnostic only);
  * training tau always uses the actual PTS (re12), never index/fps silently.

Also written: audit_videos.csv (per video), val_timing.csv (per validation
window: lead 0.5/1.0/1.5 and the 30 dense windows of re07, requested vs actual
endpoint, actual tau, future-frame flag, deviation in seconds and frames).

Usage (vast.ai):
  python reeval/re11_audit_progress_data.py --data-dir data/nexar_kaggle_style \
     --split-manifest results/repro/split_manifest_seed42.json \
     --cache-root data/nexar_cache_riskprop --cache-5f data/nexar_cache_5f \
     --id-map results/repro/id_to_hf_source.csv \
     --prepare-report outputs_progress/prepare/prepare_report.json --output-dir outputs_progress/audit
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import progress_common as pc  # noqa: E402

SEED = 42
NUM_FRAMES, SAMPLE_FPS, SEQ_LEN, STRIDE_SEC = 5, 10, 12, 0.5   # = cell30 / re01
VAL_LEADS = (0.5, 1.0, 1.5)
DENSE_D = [round(3.0 - 0.1 * k, 1) for k in range(30)]         # = re07
GAP_BINS = [-np.inf, 0.5, 1.0, 1.5, 2.0, np.inf]
GAP_LABELS = ["<=0.5", "0.5-1.0", "1.0-1.5", "1.5-2.0", ">2.0"]
sha256_file = pc.sha256_file


def vid_str(x):
    return str(int(x)).zfill(5)


def make_stratify_key(df):
    """Verbatim rule of cell10 / cell30 / re01."""
    key = pd.Series("neg", index=df.index, dtype=object)
    pos_mask = df["target"] == 1
    a2e = df.loc[pos_mask, "time_of_event"] - df.loc[pos_mask, "time_of_alert"]
    bins = [-np.inf, 0.5, 1.0, 1.5, 2.0, np.inf]
    labels = ["b1", "b2", "b3", "b4", "b5"]
    key.loc[pos_mask] = "pos_" + pd.cut(a2e, bins=bins, labels=labels).astype(str)
    return key


def recompute_split(df):
    from sklearn.model_selection import train_test_split
    tr, va = train_test_split(df, test_size=0.2, stratify=make_stratify_key(df), random_state=SEED)
    return [vid_str(x) for x in tr["id"]], [vid_str(x) for x in va["id"]]


def nominal_t_obs(target, toe, duration):
    """cell30.get_sequence_t_obs (positives / negatives)."""
    min_t = (NUM_FRAMES - 1) / SAMPLE_FPS
    if target == 1:
        if toe is None or not np.isfinite(toe):
            toe = duration - 0.5
        t_last = max(min_t, min(toe - 0.1, duration - 0.1))
        return [max(min_t, t_last - (SEQ_LEN - 1 - k) * STRIDE_SEC) for k in range(SEQ_LEN)]
    center = duration * 0.5
    return [float(np.clip(center - (SEQ_LEN / 2 - 0.5 - k) * STRIDE_SEC,
                          min_t, max(min_t, duration - 0.1))) for k in range(SEQ_LEN)]


def val_lead_t_obs(target, toe, duration, lead):
    """re01.get_val_indices end time (verbatim rule)."""
    min_t = (NUM_FRAMES - 1) / SAMPLE_FPS
    if target == 1 and toe is not None and np.isfinite(toe):
        t = max(min_t, toe - lead)
        t = min(t, toe - 0.1)
        return max(min_t, t)
    return duration * 0.5


def dense_t_obs(target, toe, duration):
    """re07.dense_t_obs (verbatim rule)."""
    min_t = (NUM_FRAMES - 1) / SAMPLE_FPS
    if target == 1 and toe is not None and np.isfinite(toe):
        out = []
        for d in DENSE_D:
            t = max(min_t, toe - d)
            t = min(t, toe - 0.1)
            out.append(max(min_t, t))
        return out
    mid = duration * 0.5
    return [max(min_t, mid - (d - 0.1)) for d in DENSE_D]


def end_frame_of(t_obs, fps, total):
    """Last frame index chosen by cell10/cell30/re01/re07: min(int(t*fps), total-1)."""
    return min(int(t_obs * fps), total - 1)


def pts_policy(ts, fps):
    """Return (relative PTS array or None, info, error or None) -- same policy as re12."""
    ts = np.asarray(ts, dtype=np.float64)
    info = {"pts_first_s": float(ts[0]) if len(ts) else float("nan"),
            "pts_nonincreasing": int((np.diff(ts) <= 0).sum()) if len(ts) > 1 else 0,
            "pts_offset_s": 0.0, "frame_period_s": 1.0 / fps}
    if len(ts) == 0 or not np.all(np.isfinite(ts)):
        return None, info, "non-finite or missing PTS"
    if info["pts_nonincreasing"]:
        return None, info, f"PTS not strictly increasing ({info['pts_nonincreasing']} steps)"
    if ts[0] < -1e-3:
        return None, info, f"negative first PTS {ts[0]:.4f}"
    if abs(ts[0]) > 1e-3:
        info["pts_offset_s"] = float(ts[0])
    rel = ts - info["pts_offset_s"]
    dev = np.abs(rel - np.arange(len(rel)) / fps)
    info["pts_max_dev_s"] = float(dev.max())
    info["pts_max_dev_frames"] = float(dev.max() * fps)
    info["vfr_suspect"] = bool(dev.max() > 1.0 / fps)
    info["last_frame_s"] = float(rel[-1])
    return rel, info, None


def check_metadata(df):
    """F5 + metadata warnings."""
    fatal, warn, info = [], [], {}
    pos, neg = df[df["target"] == 1], df[df["target"] == 0]
    toe = pos["time_of_event"].astype(float)
    bad_pos = pos[~np.isfinite(toe) | (toe < 0)]
    if len(bad_pos):
        fatal.append(f"F5: {len(bad_pos)} positives with missing/negative time_of_event: "
                     f"{[vid_str(x) for x in bad_pos['id'][:10]]}")
    if neg["time_of_event"].notna().any():
        fatal.append(f"F5: {int(neg['time_of_event'].notna().sum())} negatives with a time_of_event")
    if pos["time_of_alert"].isna().any():
        warn.append(f"{int(pos['time_of_alert'].isna().sum())} positives without time_of_alert")
    if (pos["time_of_alert"] > pos["time_of_event"]).any():
        warn.append(f"{int((pos['time_of_alert'] > pos['time_of_event']).sum())} positives with alert > event")
    early = pos[pos["time_of_event"] < (SEQ_LEN - 1) * STRIDE_SEC + 0.5]
    if len(early):
        warn.append(f"{len(early)} positives with time_of_event < 6.0 s -> nominal clamp at 0.4 s "
                    f"(duplicate endpoints are masked by re12): {[vid_str(x) for x in early['id']]}")
    warn.append("Episode-level leakage NOT verifiable: Nexar metadata has no episode/source-drive key.")
    for _, r in df.iterrows():
        info[vid_str(r["id"])] = {"target": int(r["target"]),
                                  "time_of_event": None if pd.isna(r["time_of_event"]) else float(r["time_of_event"]),
                                  "time_of_alert": None if pd.isna(r["time_of_alert"]) else float(r["time_of_alert"])}
    return fatal, warn, info


def gap_distribution(df, ids):
    sub = df[df["id"].apply(vid_str).isin(ids) & (df["target"] == 1)]
    gap = (sub["time_of_event"] - sub["time_of_alert"]).astype(float)
    cut = pd.cut(gap, bins=GAP_BINS, labels=GAP_LABELS)
    out = {str(k): int(v) for k, v in cut.value_counts().sort_index().items()}
    out.update(n_pos=int(len(sub)), gap_gt_1_5=int((gap > 1.5).sum()), gap_gt_2_0=int((gap > 2.0).sum()),
               gap_median=float(np.nanmedian(gap)) if len(gap) else None)
    return out


def val_timing_rows(vid, target, toe, rel_pts, fps, total):
    """Requested vs actual endpoint for the 3 lead windows and 30 dense windows."""
    rows = []
    dur = total / fps
    period = 1.0 / fps
    specs = [("lead", l, val_lead_t_obs(target, toe, dur, l)) for l in VAL_LEADS]
    specs += [("dense", k, t) for k, t in enumerate(dense_t_obs(target, toe, dur))]
    for kind, k, t_req in specs:
        ef = end_frame_of(t_req, fps, total)
        actual = float(rel_pts[ef]) if rel_pts is not None else float("nan")
        tau = (toe - actual) if (target == 1 and toe is not None) else float("inf")
        rows.append({"vid": vid, "target": target, "kind": kind, "k": k, "requested_end_s": t_req,
                     "end_frame": ef, "actual_end_s": actual, "time_of_event": toe, "actual_tau_s": tau,
                     "future_frame": bool(target == 1 and toe is not None and actual >= toe),
                     "dev_s": actual - t_req, "dev_frames": (actual - t_req) / period,
                     "frame_period_s": period})
    return rows


def probe_video(path):
    from decord import VideoReader, cpu
    vr = VideoReader(path, ctx=cpu(0))
    n, fps = len(vr), float(vr.get_avg_fps())
    rec = {"frames": n, "avg_fps": fps, "duration_avg": n / fps if fps > 0 else float("nan")}
    try:
        ts = np.asarray(vr.get_frame_timestamp(np.arange(n)), dtype=np.float64)[:, 0]
        rec["pts_available"] = True
    except Exception as e:
        ts = None
        rec["pts_available"] = False
        rec["pts_error"] = str(e)[:120]
    return rec, ts


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--split-manifest", required=True)
    ap.add_argument("--cache-root", required=True, help="nexar_cache_riskprop (cell30)")
    ap.add_argument("--cache-5f", required=True, help="nexar_cache_5f (val windows, re01)")
    ap.add_argument("--id-map", required=True, help="results/repro/id_to_hf_source.csv")
    ap.add_argument("--prepare-report", default="", help="re17 prepare_report.json (required for complete)")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--limit-videos", type=int, default=0, help="dev only: probe the first N (not complete)")
    a = ap.parse_args(argv)
    os.makedirs(a.output_dir, exist_ok=True)
    t0 = time.time()
    fatal, warn = [], []
    csv_path = os.path.join(a.data_dir, "train.csv")
    tr_idx_path = os.path.join(a.cache_root, "train_index.json")
    va_idx_path = os.path.join(a.cache_5f, "val_index.json")
    for p in (csv_path, a.split_manifest, tr_idx_path, va_idx_path, a.id_map):
        if not os.path.exists(p):
            pc.fail(f"missing input {p}")
    df = pd.read_csv(csv_path)
    man = json.load(open(a.split_manifest))
    csv_sha = sha256_file(csv_path)
    if csv_sha != man["train_csv_sha256"]:
        fatal.append(f"F1: train.csv sha256 {csv_sha[:16]} != manifest {man['train_csv_sha256'][:16]}")
    tr_ids, va_ids = recompute_split(df)
    if tr_ids != man["train_ids"] or va_ids != man["val_ids"]:
        fatal.append(f"F2: recomputed split != manifest (train {tr_ids == man['train_ids']}, "
                     f"val {va_ids == man['val_ids']})")
    if set(man["train_ids"]) & set(man["val_ids"]):
        fatal.append("F3: train/val overlap")
    tr_index = json.load(open(tr_idx_path))
    va_index = json.load(open(va_idx_path))
    if set(tr_index) != set(man["train_ids"]) or len(tr_index) != pc.N_TRAIN:
        fatal.append(f"F4: train cache has {len(tr_index)} ids, {len(set(man['train_ids']) - set(tr_index))} "
                     f"missing, {len(set(tr_index) - set(man['train_ids']))} extra")
    if list(va_index) != man["val_ids"] or len(va_index) != pc.N_VAL:
        fatal.append("F4: val_index ids/order != manifest val ids")
    bad_leads = [v for v, d in va_index.items() if sorted(d) != sorted(str(l) for l in VAL_LEADS)]
    if bad_leads:
        fatal.append(f"F4: {len(bad_leads)} val videos without leads 0.5/1.0/1.5")
    f5, w5, meta = check_metadata(df)
    fatal += f5
    warn += w5

    # F7 mapping
    idm = pd.read_csv(a.id_map, dtype=str)
    bad, unverifiable = [], []
    for _, r in idm.iterrows():
        p = os.path.join(a.data_dir, "train", r["id_file"])
        if not os.path.lexists(p):
            unverifiable.append(r["id_file"])
            continue
        if not os.path.realpath(p).replace("\\", "/").endswith(r["hf_source"]):
            bad.append(r["id_file"])
    if bad:
        fatal.append(f"F7: {len(bad)} ids resolve to a different HF file than recorded: {bad[:10]}")
    if unverifiable:
        fatal.append(f"F7: {len(unverifiable)} ids missing or not resolvable to an HF path: {unverifiable[:10]}")

    # F10 binding to re17
    prep = None
    if a.prepare_report and os.path.exists(a.prepare_report):
        prep = json.load(open(a.prepare_report))
        if not prep.get("pass"):
            fatal.append("F10: prepare_report did not pass")
        if prep.get("train_csv_sha256") != csv_sha:
            fatal.append("F10: prepare_report was made for a different train.csv")
    else:
        warn.append("F10: no prepare_report (re17) -> HF revision not bound; audit is NOT complete")

    # probe + scan
    import torch
    ids = man["train_ids"] + man["val_ids"]
    limited = bool(a.limit_videos)
    if limited:
        ids = ids[:a.limit_videos]
    rows, vt_rows = [], []
    n_scan_tr = n_scan_va = 0
    for i, vid in enumerate(ids):
        split = "train" if vid in tr_index else "val"
        rec = {"vid": vid, "split": split, **meta.get(vid, {})}
        path = os.path.join(a.data_dir, "train", vid + ".mp4")
        try:
            info, ts = probe_video(path)
            rec.update(info)
        except Exception as e:
            rec["open_error"] = str(e)[:120]
            fatal.append(f"F6: {vid} cannot be opened: {str(e)[:80]}")
            rows.append(rec)
            continue
        fps, total = rec["avg_fps"], rec["frames"]
        if not (np.isfinite(fps) and fps > 0 and total > 0):
            fatal.append(f"F6: {vid} invalid fps/frames {fps}/{total}")
            rows.append(rec)
            continue
        rel = None
        if ts is None:
            fatal.append(f"F8: {vid} has no PTS")
        else:
            rel, pinfo, err = pts_policy(ts, fps)
            rec.update(pinfo)
            if err:
                fatal.append(f"F8: {vid}: {err}")
        toe = rec.get("time_of_event")
        last = rec.get("last_frame_s", total / fps)
        if rec.get("target") == 1 and toe is not None and toe > last + 1.0 / fps:
            fatal.append(f"F6: {vid} time_of_event {toe:.3f} s beyond the last frame {last:.3f} s")
            rec["event_beyond_duration"] = True
        if split == "train":
            d = torch.load(os.path.join(a.cache_root, "train", tr_index[vid]), weights_only=False)
            fr = d["frames"]
            rec["cache_shape"] = "x".join(map(str, fr.shape))
            rec["cache_zero_snippets"] = int((fr.reshape(fr.shape[0], -1).amax(dim=1) == 0).sum())
            if tuple(fr.shape) != (SEQ_LEN, NUM_FRAMES, 224, 224, 3):
                fatal.append(f"F9: {vid} train cache shape {tuple(fr.shape)}")
            if rec.get("target") == 1 and toe is not None:
                nom = nominal_t_obs(1, toe, total / fps)
                tau_nom = np.maximum(0.0, toe - np.array(nom))
                rec["cache_tau_max_absdiff"] = float(np.max(np.abs(tau_nom - d["tau"].numpy())))
                rec["nominal_clamped_snippets"] = int(sum(abs(t - 0.4) < 1e-9 for t in nom))
            n_scan_tr += 1
        else:
            zeros = 0
            for lead in VAL_LEADS:
                fn = os.path.join(a.cache_5f, "val", va_index[vid][str(lead)])
                dv = torch.load(fn, weights_only=False)
                if tuple(dv["frames"].shape) != (NUM_FRAMES, 224, 224, 3):
                    fatal.append(f"F9: {vid} val cache lead {lead} shape {tuple(dv['frames'].shape)}")
                zeros += int(dv["frames"].max() == 0)
            rec["val_zero_windows"] = zeros
            n_scan_va += 1
            vrows = val_timing_rows(vid, int(rec.get("target", 0)), toe, rel, fps, total)
            for r in vrows:
                r["vfr_suspect"] = rec.get("vfr_suspect")
            vt_rows += vrows
            fut = [r for r in vrows if r["kind"] == "lead" and r["future_frame"]]
            if fut:
                fatal.append(f"F11: {vid} lead windows end at/after the event: {[r['k'] for r in fut]}")
        rows.append(rec)
        if (i + 1) % 100 == 0:
            print(f"  [audit] {i+1}/{len(ids)} videos ({time.time()-t0:.0f}s)")

    vdf = pd.DataFrame(rows)
    vdf.to_csv(os.path.join(a.output_dir, "audit_videos.csv"), index=False)
    vt = pd.DataFrame(vt_rows)
    vt.to_csv(os.path.join(a.output_dir, "val_timing.csv"), index=False)
    if not limited and (len(vdf) != pc.N_TRAIN_CSV or n_scan_tr != pc.N_TRAIN or n_scan_va != pc.N_VAL):
        fatal.append(f"F9: probed {len(vdf)}/1500, scanned train {n_scan_tr}/1200, val {n_scan_va}/300")
    if "cache_zero_snippets" in vdf and vdf["cache_zero_snippets"].fillna(0).sum() > 0:
        warn.append(f"{int((vdf['cache_zero_snippets'].fillna(0) > 0).sum())} train videos have SUSPECT all-zero "
                    f"cached snippets ({int(vdf['cache_zero_snippets'].fillna(0).sum())}); re12 re-decodes them")
    if "val_zero_windows" in vdf and vdf["val_zero_windows"].fillna(0).sum() > 0:
        warn.append(f"{int(vdf['val_zero_windows'].fillna(0).sum())} validation windows are all-zero (suspect)")
    if "vfr_suspect" in vdf and vdf["vfr_suspect"].fillna(False).astype(bool).any():
        warn.append(f"{int(vdf['vfr_suspect'].fillna(False).astype(bool).sum())} videos: PTS > 1 frame away "
                    f"from index/avg_fps (VFR suspect)")
    if "pts_offset_s" in vdf and (vdf["pts_offset_s"].fillna(0) != 0).any():
        warn.append(f"{int((vdf['pts_offset_s'].fillna(0) != 0).sum())} videos with first-frame PTS offset "
                    f"(subtracted)")
    if "cache_tau_max_absdiff" in vdf and (vdf["cache_tau_max_absdiff"].fillna(0) > 1e-4).any():
        warn.append(f"{int((vdf['cache_tau_max_absdiff'].fillna(0) > 1e-4).sum())} positives: cached tau != "
                    f"recomputed nominal tau")

    vtl = vt[vt.kind == "lead"] if len(vt) else vt
    vtd = vt[(vt.kind == "dense") & (vt.target == 1)] if len(vt) else vt
    timing = {"val_videos": int(vt["vid"].nunique()) if len(vt) else 0,
              "val_positives": int(vt[vt.target == 1]["vid"].nunique()) if len(vt) else 0,
              "lead_future_frames": int(vtl["future_frame"].sum()) if len(vtl) else 0,
              "dense_future_frames": int(vtd["future_frame"].sum()) if len(vtd) else 0,
              "dense_max_abs_dev_frames": float(vtd["dev_frames"].abs().max()) if len(vtd) else None,
              "dense_windows_dev_gt_1_frame": int((vtd["dev_frames"].abs() > 1.0).sum()) if len(vtd) else 0}
    complete = (not limited and prep is not None and not unverifiable and not bad
                and len(vdf) == pc.N_TRAIN_CSV and n_scan_tr == pc.N_TRAIN and n_scan_va == pc.N_VAL)
    summary = {
        "version": pc.V4_VERSION, "pass": not fatal, "complete": bool(complete), "limited": limited,
        "synthetic": pc.SYNTHETIC,
        "fatal": fatal, "warnings": warn, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "inputs": {"train_csv": csv_path, "train_csv_sha256": csv_sha,
                   "split_manifest_sha256": sha256_file(a.split_manifest),
                   "id_map_sha256": sha256_file(a.id_map),
                   "train_index_sha256": sha256_file(tr_idx_path), "val_index_sha256": sha256_file(va_idx_path),
                   "prepare_report_sha256": sha256_file(a.prepare_report) if prep is not None else None,
                   "hf_revision": (prep or {}).get("revision")},
        "counts": {"csv_rows": int(len(df)), "csv_pos": int((df.target == 1).sum()),
                   "train": len(man["train_ids"]), "val": len(man["val_ids"]),
                   "probed": int(len(vdf)), "train_scanned": n_scan_tr, "val_scanned": n_scan_va},
        "alert_event_gap": {"train": gap_distribution(df, set(man["train_ids"])),
                            "val": gap_distribution(df, set(man["val_ids"]))},
        "val_timing": timing,
        # legacy keys read by v3 consumers
        "probed_videos": int(len(vdf)), "scanned_cache": n_scan_tr,
    }
    pc.write_json_atomic(os.path.join(a.output_dir, "audit_summary.json"), summary)
    lines = [f"AUDIT {'PASS' if summary['pass'] else 'FAIL'} complete={summary['complete']} "
             f"({time.time()-t0:.0f}s)", f"counts: {summary['counts']}", f"val timing: {timing}",
             "", "FATAL:"] + [f"  - {x}" for x in fatal] + ["", "WARNINGS:"] + [f"  - {x}" for x in warn]
    with open(os.path.join(a.output_dir, "audit_report.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    sys.exit(0 if summary["pass"] else 2)


if __name__ == "__main__":
    main()
