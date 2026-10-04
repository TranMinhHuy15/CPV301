"""
re11 -- READ-ONLY data / split / cache audit before Progressive RiskProp
(RQ3 new). Writes nothing outside --output-dir; never touches raw data,
caches or checkpoints.

Fatal checks (exit code 2, audit_summary.json "pass": false):
  F1  train.csv sha256 == split manifest train_csv_sha256
  F2  re-computing the seed-42 stratified split (cell30 rule) reproduces the
      manifest train_ids / val_ids exactly
  F3  train and val ids are disjoint
  F4  RiskProp sequence cache train_index keys == manifest train_ids;
      nexar_cache_5f val_index keys == manifest val_ids (same order)
  F5  every positive has a finite time_of_event >= 0; every negative has none
  F6  (--probe-videos) positives: time_of_event <= decoded duration; avg fps
      finite and > 0; video opens
  F7  (--id-map) data/train/<id>.mp4 resolves to the same HF file as recorded
      in results/repro/id_to_hf_source.csv (ID-by-enumeration risk of cell09)
  F8  (--probe-videos) PTS of every frame finite and strictly increasing, first
      PTS not negative (otherwise tau cannot be computed reliably)
Warnings (reported, not fatal): time_of_alert missing / after event, event
earlier than 6 s (12-snippet clamp -> duplicate endpoints), first-frame PTS
!= 0 (re12 subtracts the offset), PTS far from index/avg_fps (variable frame
rate), videos without PTS, SUSPECT all-zero cached snippets (not proof of a
decode failure -- re12 re-decodes and classifies them), cache tau !=
recomputed nominal tau, episode-level leakage (NOT verifiable: no episode key).

Usage (vast.ai layout):
  python reeval/re11_audit_progress_data.py \
      --data-dir data/nexar_kaggle_style \
      --split-manifest results/repro/split_manifest_seed42.json \
      --cache-root data/nexar_cache_riskprop --cache-5f data/nexar_cache_5f \
      --id-map results/repro/id_to_hf_source.csv \
      --probe-videos --scan-cache --output-dir outputs_progress/audit
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

SEED = 42
NUM_FRAMES, SAMPLE_FPS, SEQ_LEN, STRIDE_SEC = 5, 10, 12, 0.5   # = cell30
GAP_BINS = [-np.inf, 0.5, 1.0, 1.5, 2.0, np.inf]
GAP_LABELS = ["<=0.5", "0.5-1.0", "1.0-1.5", "1.5-2.0", ">2.0"]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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
    tr, va = train_test_split(df, test_size=0.2, stratify=make_stratify_key(df),
                              random_state=SEED)
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


def check_metadata(df):
    """F5 + metadata warnings. Returns (fatal, warnings, per-video dict)."""
    fatal, warn, info = [], [], {}
    pos = df[df["target"] == 1]
    neg = df[df["target"] == 0]
    bad_pos = pos[~np.isfinite(pos["time_of_event"].astype(float)) |
                  (pos["time_of_event"].astype(float) < 0)]
    if len(bad_pos):
        fatal.append(f"F5: {len(bad_pos)} positives with missing/negative time_of_event: "
                     f"{[vid_str(x) for x in bad_pos['id'][:10]]}")
    bad_neg = neg[neg["time_of_event"].notna()]
    if len(bad_neg):
        fatal.append(f"F5: {len(bad_neg)} negatives with a time_of_event")
    no_alert = pos[pos["time_of_alert"].isna()]
    if len(no_alert):
        warn.append(f"{len(no_alert)} positives without time_of_alert")
    late_alert = pos[pos["time_of_alert"] > pos["time_of_event"]]
    if len(late_alert):
        warn.append(f"{len(late_alert)} positives with time_of_alert > time_of_event")
    early = pos[pos["time_of_event"] < (SEQ_LEN - 1) * STRIDE_SEC + 0.5]
    if len(early):
        warn.append(f"{len(early)} positives with time_of_event < 6.0 s -> 12-snippet clamp "
                    f"gives duplicate endpoints (handled by the CPS mask): "
                    f"{[vid_str(x) for x in early['id']]}")
    warn.append("Episode-level leakage NOT verifiable: Nexar metadata has no episode/"
                "source-drive key; split is stratified by label + alert gap only.")
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
    out["n_pos"] = int(len(sub))
    out["gap_gt_1.5"] = int((gap > 1.5).sum())
    out["gap_gt_2.0"] = int((gap > 2.0).sum())
    out["gap_median"] = float(np.nanmedian(gap)) if len(gap) else None
    return out


def probe_video(path):
    from decord import VideoReader, cpu
    vr = VideoReader(path, ctx=cpu(0))
    n, fps = len(vr), float(vr.get_avg_fps())
    rec = {"frames": n, "avg_fps": fps, "duration_avg": n / fps if fps > 0 else float("nan")}
    try:
        ts = np.asarray(vr.get_frame_timestamp(np.arange(n)), dtype=np.float64)[:, 0]
        rec["pts_available"] = True
        rec["pts_first"] = float(ts[0])
        rec["pts_last"] = float(ts[-1])
        rec["pts_finite"] = bool(np.all(np.isfinite(ts)))
        rec["pts_nonincreasing"] = int((np.diff(ts) <= 0).sum())
        rec["pts_max_dev_from_avgfps_s"] = float(np.max(np.abs(ts - np.arange(n) / fps)))
    except Exception as e:  # decord build without timestamps
        rec["pts_available"] = False
        rec["pts_error"] = str(e)[:120]
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--split-manifest", required=True)
    ap.add_argument("--cache-root", required=True, help="nexar_cache_riskprop (cell30)")
    ap.add_argument("--cache-5f", required=True, help="nexar_cache_5f (val_index.json, re01)")
    ap.add_argument("--id-map", default="", help="results/repro/id_to_hf_source.csv")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--probe-videos", action="store_true", help="open every video (F6, PTS/VFR)")
    ap.add_argument("--scan-cache", action="store_true",
                    help="load every train .pt: zero snippets, tau vs nominal")
    ap.add_argument("--limit-videos", type=int, default=0, help="probe/scan only the first N")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    t0 = time.time()
    fatal, warn = [], []

    csv_path = os.path.join(args.data_dir, "train.csv")
    for p in (csv_path, args.split_manifest, os.path.join(args.cache_root, "train_index.json"),
              os.path.join(args.cache_5f, "val_index.json")):
        if not os.path.exists(p):
            sys.exit(f"[FATAL] missing input {p}")
    df = pd.read_csv(csv_path)
    with open(args.split_manifest) as f:
        man = json.load(f)

    # F1
    csv_sha = sha256_file(csv_path)
    if csv_sha != man["train_csv_sha256"]:
        fatal.append(f"F1: train.csv sha256 {csv_sha[:16]} != manifest {man['train_csv_sha256'][:16]}")
    # F2 / F3
    tr_ids, va_ids = recompute_split(df)
    if tr_ids != man["train_ids"] or va_ids != man["val_ids"]:
        fatal.append("F2: recomputed seed-42 split differs from the manifest "
                     f"(train equal={tr_ids == man['train_ids']}, val equal={va_ids == man['val_ids']})")
    inter = set(man["train_ids"]) & set(man["val_ids"])
    if inter:
        fatal.append(f"F3: {len(inter)} ids in both train and val")
    # F4
    with open(os.path.join(args.cache_root, "train_index.json")) as f:
        tr_index = json.load(f)
    with open(os.path.join(args.cache_5f, "val_index.json")) as f:
        va_index = json.load(f)
    if set(tr_index) != set(man["train_ids"]):
        fatal.append(f"F4: RiskProp cache has {len(tr_index)} ids; "
                     f"{len(set(man['train_ids']) - set(tr_index))} manifest train ids missing, "
                     f"{len(set(tr_index) - set(man['train_ids']))} extra")
    if list(va_index) != man["val_ids"]:
        fatal.append("F4: nexar_cache_5f val_index order/ids != manifest val_ids")
    # F5 + metadata warnings
    f5, w5, meta = check_metadata(df)
    fatal += f5
    warn += w5
    # F7
    idmap_status = "not checked"
    if args.id_map:
        m = pd.read_csv(args.id_map)
        bad, unverifiable = [], 0
        for _, r in m.iterrows():
            p = os.path.join(args.data_dir, "train", r["id_file"])
            if not os.path.islink(p):
                unverifiable += 1
                continue
            real = os.path.realpath(p).replace("\\", "/")
            if not real.endswith(r["hf_source"]):
                bad.append(r["id_file"])
        if bad:
            fatal.append(f"F7: {len(bad)} ids point to a different HF file than recorded: {bad[:10]}")
        idmap_status = f"checked {len(m)} rows, mismatched={len(bad)}, not-a-symlink={unverifiable}"
        if unverifiable:
            warn.append(f"F7: {unverifiable} files are not symlinks -> HF mapping unverifiable")

    # per-video probing / cache scan
    rows = []
    ids = man["train_ids"] + man["val_ids"]
    if args.limit_videos:
        ids = ids[:args.limit_videos]
    if args.probe_videos or args.scan_cache:
        import torch
        for i, vid in enumerate(ids):
            rec = {"vid": vid, "split": "train" if vid in tr_index else "val", **meta.get(vid, {})}
            path = os.path.join(args.data_dir, "train", vid + ".mp4")
            if args.probe_videos:
                try:
                    rec.update(probe_video(path))
                    if rec["target"] == 1 and rec["time_of_event"] is not None:
                        rec["event_beyond_duration"] = bool(rec["time_of_event"] > rec["duration_avg"])
                        if rec["event_beyond_duration"]:
                            fatal.append(f"F6: {vid} time_of_event {rec['time_of_event']:.3f} > "
                                         f"duration {rec['duration_avg']:.3f}")
                    if not (np.isfinite(rec["avg_fps"]) and rec["avg_fps"] > 0):
                        fatal.append(f"F6: {vid} invalid avg fps {rec['avg_fps']}")
                    if rec.get("pts_available"):
                        if not rec["pts_finite"] or rec["pts_nonincreasing"] > 0:
                            fatal.append(f"F8: {vid} PTS not finite/strictly increasing "
                                         f"({rec['pts_nonincreasing']} non-increasing steps)")
                        if rec["pts_first"] < -1e-3:
                            fatal.append(f"F8: {vid} negative first PTS {rec['pts_first']:.4f}")
                except Exception as e:
                    rec["open_error"] = str(e)[:120]
                    fatal.append(f"F6: {vid} cannot be opened: {str(e)[:80]}")
            if args.scan_cache and vid in tr_index:
                d = torch.load(os.path.join(args.cache_root, "train", tr_index[vid]),
                               weights_only=False)
                fr = d["frames"]
                zero = (fr.reshape(fr.shape[0], -1).amax(dim=1) == 0).numpy()
                rec["cache_zero_snippets"] = int(zero.sum())
                rec["cache_shape"] = "x".join(map(str, fr.shape))
                if "duration_avg" in rec and rec["target"] == 1:
                    nom = nominal_t_obs(1, rec["time_of_event"], rec["duration_avg"])
                    tau_nom = np.maximum(0.0, rec["time_of_event"] - np.array(nom))
                    rec["cache_tau_max_absdiff"] = float(np.max(np.abs(tau_nom - d["tau"].numpy())))
                    rec["n_duplicate_t_obs"] = int(SEQ_LEN - len(np.unique(np.round(nom, 6))))
            rows.append(rec)
            if (i + 1) % 100 == 0:
                print(f"  [audit] {i+1}/{len(ids)} videos ({time.time()-t0:.0f}s)")
    vdf = pd.DataFrame(rows)
    if len(vdf):
        vdf.to_csv(os.path.join(args.output_dir, "audit_videos.csv"), index=False)
        if "cache_zero_snippets" in vdf and vdf["cache_zero_snippets"].sum() > 0:
            warn.append(f"{int((vdf['cache_zero_snippets'] > 0).sum())} train videos have SUSPECT "
                        f"all-zero cached snippets ({int(vdf['cache_zero_snippets'].sum())} snippets): "
                        f"decode failure OR genuinely black frames -- re12 re-decodes and classifies")
        if "cache_tau_max_absdiff" in vdf and (vdf["cache_tau_max_absdiff"] > 1e-4).any():
            warn.append(f"{int((vdf['cache_tau_max_absdiff'] > 1e-4).sum())} positives: cached tau "
                        f"!= recomputed nominal tau")
        if "pts_max_dev_from_avgfps_s" in vdf:
            dev = vdf["pts_max_dev_from_avgfps_s"].astype(float)
            n_vfr = int((dev > 1.0 / 30).sum())
            if n_vfr:
                warn.append(f"{n_vfr} videos: PTS deviates > 1 frame (1/30 s) from index/avg_fps "
                            f"(variable frame rate?) -> use PTS in the sidecar")
        if "pts_first" in vdf:
            n_off = int((vdf["pts_first"].astype(float).abs() > 1e-3).sum())
            if n_off:
                warn.append(f"{n_off} videos: first-frame PTS != 0 (re12 subtracts the offset)")
        if "pts_available" in vdf and not vdf["pts_available"].all():
            warn.append(f"{int((~vdf['pts_available'].astype(bool)).sum())} videos without PTS")

    summary = {
        "pass": len(fatal) == 0,
        "fatal": fatal, "warnings": warn,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "inputs": {"train_csv": csv_path, "train_csv_sha256": csv_sha,
                   "split_manifest": args.split_manifest,
                   "split_manifest_sha256": sha256_file(args.split_manifest),
                   "cache_root": args.cache_root, "cache_5f": args.cache_5f,
                   "train_index_sha256": sha256_file(os.path.join(args.cache_root, "train_index.json")),
                   "val_index_sha256": sha256_file(os.path.join(args.cache_5f, "val_index.json"))},
        "counts": {"csv_rows": int(len(df)), "csv_pos": int((df.target == 1).sum()),
                   "csv_neg": int((df.target == 0).sum()),
                   "train": len(man["train_ids"]), "val": len(man["val_ids"]),
                   "train_cache": len(tr_index), "val_cache": len(va_index)},
        "alert_event_gap": {"train": gap_distribution(df, set(man["train_ids"])),
                            "val": gap_distribution(df, set(man["val_ids"]))},
        "id_map": idmap_status,
        "probed_videos": int(len(vdf)) if args.probe_videos else 0,
        "scanned_cache": int(vdf["cache_zero_snippets"].notna().sum())
        if (args.scan_cache and "cache_zero_snippets" in vdf) else 0,
        "limited": bool(args.limit_videos),
    }
    with open(os.path.join(args.output_dir, "audit_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    lines = [f"AUDIT {'PASS' if summary['pass'] else 'FAIL'}  ({time.time()-t0:.0f}s)",
             f"counts: {summary['counts']}",
             f"alert-event gap (train): {summary['alert_event_gap']['train']}",
             f"alert-event gap (val):   {summary['alert_event_gap']['val']}",
             f"id map: {idmap_status}", "", "FATAL:"] + [f"  - {x}" for x in fatal] + \
            ["", "WARNINGS:"] + [f"  - {x}" for x in warn]
    with open(os.path.join(args.output_dir, "audit_report.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    sys.exit(0 if summary["pass"] else 2)


if __name__ == "__main__":
    main()
