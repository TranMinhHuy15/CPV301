"""
RQ1b -- build the Nexar data in the layout the authors' RiskProp code reads:

    <out-root>/annotations.csv
    <out-root>/train_raw_frames/<video_id>/<frame:06d>.jpg

Input is the Kaggle-style copy made by cell09_prepare_hf_data.py
(<data-dir>/train.csv and <data-dir>/train/<id>.mp4).

Frames are taken at 10 fps: frame k is the source frame closest to time
k / 10 s (k = 0 .. floor(duration * 10) - 1). Event and alert times are put on
the same grid, so frame numbers and seconds always agree:

    accident_frame       = round(time_of_event * 10)
    abnormal_start_frame = round(time_of_alert * 10)

Frames are stored with the short side resized to 256 px (aspect ratio kept) to
save disk space. The authors' training pipeline then applies RandomResizedCrop
and resizes to 224x224 without keeping the aspect ratio, as in their config.

annotations.csv columns (the authors' loader reads them by position):
    0 video_id | 1 is_test | 2 total_frames (10 fps) | 3 fps_source (info only)
    4 abnormal_start_frame | 5 accident_frame | 6 target

Usage:
    python riskprop_full/prepare_nexar_10fps.py \
        --data-dir /workspace/CPV301/data/nexar_kaggle_style \
        --out-root /workspace/RiskProp/data/nexar-collision-prediction --workers 8
"""
import argparse
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

TARGET_FPS = 10


def frame_plan(total, fps_src):
    duration = total / fps_src
    n = int(math.floor(duration * TARGET_FPS + 1e-6))
    idx = [min(int(round(k / TARGET_FPS * fps_src)), total - 1) for k in range(n)]
    return n, idx, duration


def extract_one(args):
    vid5, mp4, out_dir, short_side, quality = args
    import cv2
    from decord import VideoReader, cpu
    try:
        vr = VideoReader(mp4, ctx=cpu(0))
        total, fps_src = len(vr), float(vr.get_avg_fps())
    except Exception as e:  # noqa: BLE001
        return vid5, None, f"open failed: {e}"
    n, idx, duration = frame_plan(total, fps_src)
    vdir = os.path.join(out_dir, vid5)
    done_flag = os.path.join(vdir, ".done")
    if os.path.exists(done_flag):
        return vid5, dict(n=n, fps_src=fps_src, total_src=total, duration=duration), "skipped"
    os.makedirs(vdir, exist_ok=True)
    try:
        for s in range(0, n, 64):
            chunk = idx[s:s + 64]
            frames = vr.get_batch(chunk).asnumpy()  # (c, H, W, 3) RGB
            for j, fr in enumerate(frames):
                h, w = fr.shape[:2]
                scale = short_side / min(h, w)
                fr = cv2.resize(fr, (int(round(w * scale)), int(round(h * scale))),
                                interpolation=cv2.INTER_AREA)
                cv2.imwrite(os.path.join(vdir, f"{s + j:06d}.jpg"),
                            cv2.cvtColor(fr, cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, quality])
    except Exception as e:  # noqa: BLE001
        return vid5, None, f"decode failed: {e}"
    open(done_flag, "w").close()
    return vid5, dict(n=n, fps_src=fps_src, total_src=total, duration=duration), "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/workspace/CPV301/data/nexar_kaggle_style")
    ap.add_argument("--out-root", default="/workspace/RiskProp/data/nexar-collision-prediction")
    ap.add_argument("--short-side", type=int, default=256)
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="only the first N videos (testing)")
    args = ap.parse_args()

    df = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    df["vid5"] = df["id"].apply(lambda x: str(int(x)).zfill(5))
    if args.limit:
        df = df.head(args.limit)
    frames_root = os.path.join(args.out_root, "train_raw_frames")
    os.makedirs(frames_root, exist_ok=True)

    jobs = [(r.vid5, os.path.join(args.data_dir, "train", r.vid5 + ".mp4"), frames_root,
             args.short_side, args.quality) for r in df.itertuples()]
    info, failed = {}, {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(extract_one, j) for j in jobs]
        for i, fu in enumerate(as_completed(futs), 1):
            vid5, meta, status = fu.result()
            if meta is None:
                failed[vid5] = status
            else:
                info[vid5] = meta
            if i % 100 == 0 or i == len(futs):
                print(f"  {i}/{len(futs)} videos ({time.time() - t0:.0f}s), failed={len(failed)}")

    rows, clamps = [], []
    for r in df.itertuples():
        if r.vid5 not in info:
            continue
        n = info[r.vid5]["n"]
        target = int(r.target)
        acc = abn = np.nan
        if target == 1:
            acc = int(round(float(r.time_of_event) * TARGET_FPS))
            abn = int(round(float(r.time_of_alert) * TARGET_FPS)) if not pd.isna(r.time_of_alert) else 0
            if acc > n - 1:
                clamps.append((r.vid5, "accident_frame", acc, n - 1))
                acc = n - 1
            abn = max(0, min(abn, acc))
        rows.append([r.vid5, 0, n, round(info[r.vid5]["fps_src"], 3), abn, acc, target])
    ann = pd.DataFrame(rows, columns=["video_id", "is_test", "total_frames", "fps_source",
                                      "abnormal_start_frame", "accident_frame", "target"])
    ann["abnormal_start_frame"] = ann["abnormal_start_frame"].astype("Int64")
    ann["accident_frame"] = ann["accident_frame"].astype("Int64")
    ann.to_csv(os.path.join(args.out_root, "annotations.csv"), index=False)

    fps_vals = pd.Series([v["fps_src"] for v in info.values()])
    meta = {
        "n_videos_in_train_csv": int(len(df)), "n_extracted": len(info), "failed": failed,
        "target_fps": TARGET_FPS, "short_side": args.short_side, "jpeg_quality": args.quality,
        "source_fps_counts": {str(k): int(v) for k, v in fps_vals.round(2).value_counts().items()},
        "clamped": clamps, "positives": int(ann.target.sum()), "negatives": int((ann.target == 0).sum()),
        "rule": "frame k = source frame nearest to k/10 s; accident_frame = round(time_of_event*10); "
                "abnormal_start_frame = round(time_of_alert*10)",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(args.out_root, "prepare_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nannotations.csv: {len(ann)} videos ({meta['positives']} pos / {meta['negatives']} neg)")
    print(f"source fps: {meta['source_fps_counts']}")
    print(f"failed: {len(failed)} | accident_frame clamped: {len(clamps)}")
    pos = ann[ann.target == 1].head(3)
    for r in pos.itertuples():
        toe = float(df.loc[df.vid5 == r.video_id, "time_of_event"].iloc[0])
        print(f"  check {r.video_id}: time_of_event={toe:.3f}s -> accident_frame={r.accident_frame} "
              f"(= {int(r.accident_frame) / TARGET_FPS:.1f}s), total_frames={r.total_frames}")


if __name__ == "__main__":
    main()
