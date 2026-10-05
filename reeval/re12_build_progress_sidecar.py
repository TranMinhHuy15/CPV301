"""
re12 -- Build the Progressive-RiskProp SIDECAR for the existing RiskProp
train cache (cell30, nexar_cache_riskprop). The cache itself is NOT
modified and no pixels are duplicated: the sidecar only stores, per video and
per snippet, the frame index / timestamp that cell30 actually used, the
actual time-to-event tau, and whether the snippet is CPS-valid (+ reason).

For every train video:
  * re-derive cell30's 12 nominal end times and frame indices with the same
    avg-fps rule (int(t_obs * fps), clipped to the last frame);
  * TIMING (v2): read the PTS of EVERY frame (decord get_frame_timestamp, frame
    start time, no pixel decoding). The video is rejected (fatal) if the PTS
    are not strictly increasing. If the first frame's PTS is not 0 the offset
    is subtracted (time_of_event is measured from the first frame) and
    recorded per video (pts_offset_s). Endpoint time = PTS(end frame) - offset.
    Fallback without PTS: end_frame / avg_fps, flagged; such a sidecar is not
    usable for a full run unless --allow-index-timing;
  * tau_actual = time_of_event - endpoint time;
  * ZERO SNIPPETS (v2): an all-zero cached snippet is SUSPECT, not proof of a
    decode failure. Every zero snippet is re-decoded and classified as
    decode_fail / black (decoded pixels are zero too) / cache_only (video
    decodes fine, cache holds zeros). All are excluded from CPS and counted;
  * PIXEL CHECK (v2): --verify-pixels N re-decodes N videos (balanced pos/neg,
    deterministic) exactly like cell30 and requires identical frames (suspect
    zero snippets are classified above instead). A full run requires
    >= MIN_PIXEL_VERIFY verified videos, all identical, and no cache_only zeros
    unless --accept-cache-zeros is given (recorded in the sidecar meta);
  * PROVENANCE (v2): the sidecar meta records sha256 of the split manifest, of
    train_index.json and a digest of every cached .pt file (cache_digest);
    re14 recomputes them and refuses to train on a different split/cache;
  * validity rules: re13.snippet_records (anchors k=0 / k=11 excluded,
    decode / zero, non-finite, tau <= 0, duplicate endpoints, non-monotonic).

Refuses to run unless --audit points to a PASSING re11 audit_summary.json
(override --allow-failed-audit marks the sidecar "usable_for_full_run": false).

Usage:
  python reeval/re12_build_progress_sidecar.py --audit outputs_progress/audit/audit_summary.json \
      --data-dir data/nexar_kaggle_style --split-manifest results/repro/split_manifest_seed42.json \
      --cache-root data/nexar_cache_riskprop --output-dir outputs_progress/sidecar --verify-pixels 24
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from re11_audit_progress_data import nominal_t_obs, sha256_file, vid_str  # noqa: E402
import progress_common as pc  # noqa: E402
from re13_progress_supervision import REASONS, snippet_records  # noqa: E402

NUM_FRAMES, SAMPLE_FPS, SIZE, SEQ_LEN = 5, 10, 224, 12   # = cell30
MIN_PIXEL_VERIFY = 24          # videos that must be re-decoded and identical for a full run
PTS_OFFSET_TOL = 1e-3          # s; first-frame PTS above this is treated as an offset


def cell30_indices(t_obs, fps, total):
    """Verbatim cell30.indices_for_t_obs."""
    end_frame = min(int(t_obs * fps), total - 1)
    step = max(1, int(round(fps / SAMPLE_FPS)))
    idx = [end_frame - (NUM_FRAMES - 1 - k) * step for k in range(NUM_FRAMES)]
    return np.clip(idx, 0, total - 1).astype(int)


def decode_like_cell30(vr, indices):
    from PIL import Image
    frames = vr.get_batch(indices).asnumpy()
    return np.stack([np.array(Image.fromarray(f).resize((SIZE, SIZE))) for f in frames]).astype(np.uint8)


def choose_subset(train_ids, targets, n):
    """Deterministic: alternate positives / negatives in sorted id order."""
    pos = sorted(v for v in train_ids if targets[v] == 1)
    neg = sorted(v for v in train_ids if targets[v] == 0)
    out = []
    for a, b in zip(pos, neg):
        out += [a, b]
    out += pos[len(neg):] + neg[len(pos):]
    return out[:n]


def check_pts(all_ts, end_frames, fps):
    """PTS of all frames -> (end times relative to the first frame, info, error or None).
    Same policy as re11.pts_policy (single source of truth)."""
    from re11_audit_progress_data import pts_policy
    rel, info, err = pts_policy(all_ts, fps)
    if err:
        return None, info, err
    return rel[np.asarray(end_frames)], info, None


def classify_zero(vr, indices):
    """Re-decode one suspect all-zero snippet -> 'decode_fail' | 'black' | 'cache_only'."""
    try:
        px = decode_like_cell30(vr, indices)
    except Exception:
        return "decode_fail"
    return "black" if int(px.max()) == 0 else "cache_only"


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cache_digest(cache_root, tr_index, ids):
    """Digest of the cached .pt files of `ids` (order-independent)."""
    per = {v: file_sha256(os.path.join(cache_root, "train", tr_index[v])) for v in ids}
    h = hashlib.sha256()
    for v in sorted(per):
        h.update(f"{v}:{per[v]}\n".encode())
    return h.hexdigest(), per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--split-manifest", required=True)
    ap.add_argument("--cache-root", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--limit-videos", type=int, default=0)
    ap.add_argument("--verify-pixels", type=int, default=MIN_PIXEL_VERIFY,
                    help=f"re-decode N videos (balanced) and require identical pixels "
                         f"(full run needs >= {MIN_PIXEL_VERIFY})")
    ap.add_argument("--pixel-tolerance", type=float, default=0.0,
                    help="max mean |decoded - cached| (uint8 units) per snippet; 0 = bit-identical")
    ap.add_argument("--allow-index-timing", action="store_true",
                    help="DEV ONLY: continue when a video has no PTS (endpoint = frame / avg fps); "
                         "such a sidecar is never usable for a full run")
    ap.add_argument("--accept-cache-zeros", action="store_true",
                    help="DEV ONLY: continue despite cache_only zero snippets; the sidecar stays "
                         "NOT usable for a full run (rebuild the cell30 cache instead)")
    ap.add_argument("--allow-failed-audit", action="store_true")
    args = ap.parse_args()
    t0 = time.time()

    with open(args.audit) as f:
        audit = json.load(f)
    audit_ok = bool(audit.get("pass")) and bool(audit.get("complete")) and not audit.get("limited", False)
    if not audit_ok and not args.allow_failed_audit:
        sys.exit(f"[REFUSE] audit {args.audit} did not pass (or was limited). Fix the data first; "
                 f"--allow-failed-audit only for debugging.")
    from decord import VideoReader, cpu
    import torch

    df = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    df["vid"] = df["id"].apply(vid_str)
    rows = df.set_index("vid")
    with open(args.split_manifest) as f:
        man = json.load(f)
    with open(os.path.join(args.cache_root, "train_index.json")) as f:
        tr_index = json.load(f)
    targets = {v: int(rows.loc[v, "target"]) for v in man["train_ids"]}
    ids = man["train_ids"] if not args.limit_videos else choose_subset(
        man["train_ids"], targets, args.limit_videos)
    verify_set = set(choose_subset(ids, targets, args.verify_pixels)) if args.verify_pixels else set()
    abnormal = set()     # added to the pixel check below: zero snippets, clamps, PTS offsets
    os.makedirs(args.output_dir, exist_ok=True)

    print("hashing cached .pt files ...")
    digest, per_file = cache_digest(args.cache_root, tr_index, ids)

    videos, flat, fatal, warn, pix = {}, [], [], [], []
    zero_counts = {"decode_fail": 0, "black": 0, "cache_only": 0}
    n_offset = 0
    for i, vid in enumerate(ids):
        row = rows.loc[vid]
        is_pos = int(row["target"]) == 1
        toe = float(row["time_of_event"]) if is_pos and not pd.isna(row["time_of_event"]) else float("nan")
        path = os.path.join(args.data_dir, "train", vid + ".mp4")
        vr = VideoReader(path, ctx=cpu(0))
        total, fps = len(vr), float(vr.get_avg_fps())
        t_nom = nominal_t_obs(1 if is_pos else 0, toe if is_pos else None, total / fps)
        idx = [cell30_indices(t, fps, total) for t in t_nom]
        end_frames = np.array([int(x[-1]) for x in idx])
        try:
            all_ts = np.asarray(vr.get_frame_timestamp(np.arange(total)), dtype=np.float64)[:, 0]
            timing = "pts"
        except Exception:
            all_ts = None
            timing = "index_over_avg_fps"
        if all_ts is not None:
            ts, pts_info, err = check_pts(all_ts, end_frames, fps)
            if err:
                fatal.append(f"{vid}: {err}")
                ts = np.full(SEQ_LEN, np.nan)      # -> nonfinite_time, never supervised
            n_offset += int(pts_info["pts_offset_s"] != 0.0)
        else:
            ts, pts_info = end_frames / fps, {}
        d = torch.load(os.path.join(args.cache_root, "train", tr_index[vid]), weights_only=False)
        fr = d["frames"].numpy()
        if fr.shape[0] != SEQ_LEN:
            fatal.append(f"{vid}: cache has {fr.shape[0]} snippets, expected {SEQ_LEN}")
        zero = fr.reshape(fr.shape[0], -1).max(axis=1) == 0
        zero_kind = [None] * SEQ_LEN
        nominal_clamped = [abs(t - 0.4) < 1e-9 for t in t_nom] if is_pos else [False] * SEQ_LEN
        if zero.any() or any(nominal_clamped) or pts_info.get("pts_offset_s"):
            abnormal.add(vid)
        for k in np.where(zero)[0]:
            zero_kind[k] = classify_zero(vr, idx[k])
            zero_counts[zero_kind[k]] += 1
        decode_ok = np.array([zk != "decode_fail" for zk in zero_kind])
        if vid in verify_set or (args.verify_pixels and vid in abnormal):
            match, mad = [], []
            for k in range(SEQ_LEN):
                if zero_kind[k] is not None:
                    match.append(True)             # suspect zeros are classified/counted separately
                    continue
                try:
                    px = decode_like_cell30(vr, idx[k])
                except Exception:
                    match.append(False)
                    continue
                m_abs = float(np.abs(px.astype(np.int16) - fr[k].astype(np.int16)).mean())
                mad.append(m_abs)
                match.append(m_abs <= args.pixel_tolerance)
            pix.append({"vid": vid, "target": int(is_pos), "pixel_match_all": all(match),
                        "n_mismatch": int(len(match) - sum(match)),
                        "max_mean_abs_diff": max(mad) if mad else None})
            if not all(match):
                fatal.append(f"{vid}: decoded pixels != cached pixels for "
                             f"{len(match)-sum(match)} snippets (sidecar would not describe the cache)")
        rec = snippet_records(is_pos, toe if is_pos else None, end_frames, ts, decode_ok, zero,
                              zero_kind=zero_kind)
        tau_cache = d["tau"].numpy().astype(float).tolist()
        videos[vid] = {
            "target": int(is_pos), "time_of_event": None if not is_pos else toe,
            "avg_fps": fps, "frames": total, "timing_source": timing, **pts_info,
            "cache_sha256": per_file[vid],
            "t_obs_nominal": [float(x) for x in t_nom],
            "end_frame": end_frames.tolist(), "end_time": [float(x) for x in ts],
            "tau_cache": tau_cache,
            "tau_actual": [float(x) if np.isfinite(x) else None for x in rec["tau_actual"]],
            "cps_mask": [bool(x) for x in rec["cps_mask"]], "reason": rec["reason"],
            "zero_kind": zero_kind, "nominal_clamped": nominal_clamped,
            "pixel_verified": vid in verify_set or (bool(args.verify_pixels) and vid in abnormal),
        }
        for k in range(SEQ_LEN):
            flat.append({"vid": vid, "k": k, "target": int(is_pos), "t_obs_nominal": t_nom[k],
                         "end_frame": int(end_frames[k]), "end_time": float(ts[k]),
                         "tau_cache": tau_cache[k], "tau_actual": rec["tau_actual"][k],
                         "cps_mask": bool(rec["cps_mask"][k]), "reason": rec["reason"][k],
                         "zero_kind": zero_kind[k] or "", "nominal_clamped": bool(nominal_clamped[k])})
        if (i + 1) % 100 == 0:
            print(f"  [sidecar] {i+1}/{len(ids)} videos ({time.time()-t0:.0f}s)")

    if n_offset:
        warn.append(f"{n_offset} videos: first-frame PTS != 0 -> offset subtracted (pts_offset_s)")
    if zero_counts["cache_only"]:
        warn.append(f"{zero_counts['cache_only']} cached snippets are zero although the video decodes "
                    f"-> cache corrupted/stale; B/P/S train on these zeros (consider rebuilding cell30)")
    if zero_counts["black"]:
        warn.append(f"{zero_counts['black']} snippets are genuinely black (decoded pixels are zero)")
    n_index_timing = sum(v["timing_source"] != "pts" for v in videos.values())
    if n_index_timing:
        warn.append(f"{n_index_timing} videos without PTS (endpoint = frame / avg fps)")

    fl = pd.DataFrame(flat)
    fl.to_csv(os.path.join(args.output_dir, "sidecar_snippets.csv"), index=False)
    pos = fl[fl.target == 1]
    valid = pos[pos.cps_mask]
    per_vid_valid = pos.groupby("vid")["cps_mask"].sum()
    diff = (pos["tau_actual"] - pos["tau_cache"]).astype(float)
    diff = diff[np.isfinite(diff) & (pos["tau_cache"] > 0)]
    n_ok_pix = sum(p["pixel_match_all"] for p in pix)
    pix_pos = sum(p["target"] == 1 for p in pix)
    pixel_ok = (len(pix) >= MIN_PIXEL_VERIFY and n_ok_pix == len(pix) and pix_pos > 0 and pix_pos < len(pix))
    timing_ok = n_index_timing == 0            # --allow-index-timing never makes a sidecar usable
    zeros_ok = zero_counts["cache_only"] == 0  # --accept-cache-zeros never makes a sidecar usable
    usable = len(fatal) == 0 and not args.limit_videos and audit_ok and pixel_ok and timing_ok and zeros_ok
    why_not = [m for c, m in [(len(fatal) == 0, "fatal issues"), (not args.limit_videos, "limited"),
                              (audit_ok, "audit not passed"),
                              (pixel_ok, f"pixel check {n_ok_pix}/{len(pix)} (need >= {MIN_PIXEL_VERIFY}, "
                                         f"all identical, both labels)"),
                              (timing_ok, "videos without PTS"),
                              (zeros_ok, f"{zero_counts['cache_only']} cache-only zero snippets "
                                         f"(rebuild the cell30 cache)")] if not c]
    summary = {
        "pass": len(fatal) == 0,
        "complete": not args.limit_videos,
        "usable_for_full_run": usable, "not_usable_because": why_not,
        "fatal": fatal, "warnings": warn,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "audit": args.audit, "audit_pass": audit_ok, "audit_sha256": sha256_file(args.audit),
        "version": pc.V4_VERSION, "synthetic": pc.SYNTHETIC,
        "cps_coverage_note": "positives with 0 CPS-valid snippets are still trained with BCE/FFR/AMC",
        "nominal_clamped_snippets_positive": int(pos["nominal_clamped"].sum()),
        "abnormal_videos_pixel_checked": sorted(v for v in abnormal if v in {p["vid"] for p in pix}),
        "n_videos": len(ids), "n_pos": int((fl.groupby("vid").target.first() == 1).sum()),
        "reason_counts_positive": {r: int((pos.reason == r).sum()) for r in REASONS},
        "zero_snippets_all_videos": zero_counts,
        "videos_with_pts_offset": n_offset,
        "valid_snippets_per_positive": {str(k): int(v) for k, v in
                                        per_vid_valid.value_counts().sort_index().items()},
        "positives_without_valid_snippet": int((per_vid_valid == 0).sum()),
        "valid_tau_coverage": {"tau<1.5": int((valid.tau_actual < 1.5).sum()),
                               "1.5<=tau<2.0": int(((valid.tau_actual >= 1.5) & (valid.tau_actual < 2.0)).sum()),
                               "tau>=2.0": int((valid.tau_actual >= 2.0).sum())},
        "tau_actual_minus_cache_s": {"mean": float(diff.mean()) if len(diff) else None,
                                     "max_abs": float(diff.abs().max()) if len(diff) else None},
        "timing_sources": {k: int(v) for k, v in
                           pd.Series([v["timing_source"] for v in videos.values()]).value_counts().items()},
        "pixel_verification": {"n_verified": len(pix), "n_identical": int(n_ok_pix),
                               "required_for_full_run": MIN_PIXEL_VERIFY, "videos": pix},
    }
    out = {"meta": {**{k: summary[k] for k in ("created", "complete", "usable_for_full_run", "audit")},
                    "split_manifest_sha256": sha256_file(args.split_manifest),
                    "train_index_sha256": sha256_file(os.path.join(args.cache_root, "train_index.json")),
                    "cache_digest": digest, "cache_digest_ids": len(ids),
                    "pixel_verified": len(pix), "pixel_identical": int(n_ok_pix),
                    "pixel_positive": int(pix_pos), "audit_sha256": sha256_file(args.audit),
                    "synthetic": pc.SYNTHETIC,
                    "version": pc.V4_VERSION,
                    "zero_snippets": zero_counts, "accepted_cache_zeros": bool(args.accept_cache_zeros),
                    "rules": "re13.snippet_records v2: anchors k=0,k=11 excluded; decode / suspect-zero "
                             "(re-decoded); PTS relative to first frame, strictly increasing; tau<=0; "
                             "duplicate endpoints (keep latest); non-monotonic"},
           "videos": videos}
    side_path = os.path.join(args.output_dir, "sidecar.json")
    with open(side_path, "w") as f:
        json.dump(out, f)
    summary["sidecar_sha256"] = sha256_file(side_path)
    with open(os.path.join(args.output_dir, "sidecar_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(json.dumps({k: v for k, v in summary.items() if k != "pixel_verification"}, indent=2, default=float))
    print(f"pixel verification: {n_ok_pix}/{len(pix)} videos identical (full run needs >= {MIN_PIXEL_VERIFY})")
    print(f"-> {side_path} ({time.time()-t0:.0f}s)")
    sys.exit(0 if summary["pass"] else 2)


if __name__ == "__main__":
    main()
