"""
re01 -- Re-create the internal VALIDATION cache (300 videos x 3 lead times)
exactly as cell10_precache_5f.py did, and save a split manifest.

Only the validation part is rebuilt: the 24 trained checkpoints already
exist, so the train cache is not needed for re-evaluation.

The split / window logic below is copied verbatim from cell10 (same
stratify key, same train_test_split(test_size=0.2, random_state=42), same
causal-window indices, same PIL resize). Do NOT edit it -- the whole point
is to reproduce the old validation set bit-for-bit. Whether it really is
identical is verified later by re03 (legacy check against the eval txt
files already on GitHub).

Usage (on the GPU/CPU box, after cell09_prepare_hf_data.py):
    python reeval/re01_precache_val.py
    python reeval/re01_precache_val.py --data-dir ... --cache-dir ...
"""
import argparse
import hashlib
import json
import os
import platform
import time

import numpy as np
import pandas as pd
import sklearn
import torch
from decord import VideoReader, cpu
from PIL import Image
from sklearn.model_selection import train_test_split

# ---- constants copied from cell10_precache_5f.py (do not change) ----
SEED = 42
NUM_FRAMES = 5
SIZE = 224
SAMPLE_FPS = 10
VAL_LEAD_TIMES = [0.5, 1.0, 1.5]


def default_data_dir():
    for d in ("/kaggle/input/competitions/nexar-collision-prediction",
              "/kaggle/input/nexar-collision-prediction",
              "/workspace/CPV301/data/nexar_kaggle_style"):
        if os.path.exists(d):
            return d
    return "/workspace/CPV301/data/nexar_kaggle_style"


def make_stratify_key(df):
    """Verbatim from cell10."""
    key = pd.Series("neg", index=df.index, dtype=object)
    pos_mask = df["target"] == 1
    alert_to_event = (
        df.loc[pos_mask, "time_of_event"]
        - df.loc[pos_mask, "time_of_alert"])
    bins = [-np.inf, 0.5, 1.0, 1.5, 2.0, np.inf]
    labels = ["b1", "b2", "b3", "b4", "b5"]
    bin_labels = pd.cut(alert_to_event, bins=bins, labels=labels)
    key.loc[pos_mask] = "pos_" + bin_labels.astype(str)
    return key


def get_val_indices(row, fps, total_frames, lead_time):
    """cell10.get_causal_indices with is_train=False (no randomness)."""
    target = int(row["target"])
    duration = total_frames / fps
    window_sec = (NUM_FRAMES - 1) / SAMPLE_FPS
    min_t_obs = window_sec

    if target == 1:
        toe = row["time_of_event"]
        if pd.isna(toe):
            tau = float("inf")
            t_obs = duration * 0.5
        else:
            lt = lead_time
            t_obs = max(min_t_obs, toe - lt)
            t_obs = min(t_obs, toe - 0.1)
            t_obs = max(min_t_obs, t_obs)
            tau = toe - t_obs
    else:
        tau = float("inf")
        t_obs = duration * 0.5

    end_frame = min(int(t_obs * fps), total_frames - 1)
    frame_step = max(1, int(round(fps / SAMPLE_FPS)))
    indices = [end_frame - (NUM_FRAMES - 1 - k) * frame_step
               for k in range(NUM_FRAMES)]
    indices = np.clip(indices, 0, total_frames - 1).astype(int)
    return indices, tau


def extract_frames(vr, indices, size=SIZE):
    """Verbatim from cell10."""
    try:
        frames = vr.get_batch(indices).asnumpy()
    except Exception as e:
        print(f"  [WARN] frame extract failed: {e}")
        return np.zeros((len(indices), size, size, 3), dtype=np.uint8)
    imgs = [np.array(Image.fromarray(f).resize((size, size))) for f in frames]
    return np.stack(imgs).astype(np.uint8)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=default_data_dir())
    ap.add_argument("--cache-dir", default="/workspace/CPV301/data/nexar_cache_5f")
    ap.add_argument("--overwrite", action="store_true",
                    help="re-extract windows even if the .pt file exists")
    args = ap.parse_args()

    csv_path = os.path.join(args.data_dir, "train.csv")
    video_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.cache_dir, "val")
    os.makedirs(val_dir, exist_ok=True)

    full_df = pd.read_csv(csv_path)
    strat_key = make_stratify_key(full_df)
    train_df, val_df = train_test_split(
        full_df, test_size=0.2, stratify=strat_key, random_state=SEED)
    print(f"Train: {len(train_df)} videos | Val: {len(val_df)} videos")

    def ids(df):
        return [str(int(x)).zfill(5) for x in df["id"]]

    manifest = {
        "seed": SEED,
        "test_size": 0.2,
        "stratify": "target + alert-to-event bin (cell10.make_stratify_key)",
        "train_csv_sha256": sha256_file(csv_path),
        "n_rows_train_csv": int(len(full_df)),
        "number_of_train_videos": int(len(train_df)),
        "number_of_val_videos": int(len(val_df)),
        "val_positive_count": int((val_df["target"] == 1).sum()),
        "val_negative_count": int((val_df["target"] == 0).sum()),
        "val_strat_counts": strat_key.loc[val_df.index].value_counts().to_dict(),
        "train_ids": ids(train_df),
        "val_ids": ids(val_df),          # order == val_index.json order
        "versions": {"python": platform.python_version(),
                     "numpy": np.__version__, "pandas": pd.__version__,
                     "sklearn": sklearn.__version__},
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    man_path = os.path.join(args.cache_dir, "split_manifest_seed42.json")
    with open(man_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Split manifest -> {man_path}")
    print(f"  val positives={manifest['val_positive_count']} "
          f"negatives={manifest['val_negative_count']}")

    t0 = time.time()
    val_index, skipped = {}, 0
    for i, (_, row) in enumerate(val_df.iterrows()):
        vid_id = str(int(row["id"])).zfill(5)
        files = {str(l): f"{vid_id}_lead{l}.pt" for l in VAL_LEAD_TIMES}
        if not args.overwrite and all(
                os.path.exists(os.path.join(val_dir, fn)) for fn in files.values()):
            val_index[vid_id] = files
            continue
        vid_path = os.path.join(video_dir, vid_id + ".mp4")
        try:
            vr = VideoReader(vid_path, ctx=cpu(0))
            total, fps = len(vr), vr.get_avg_fps()
        except Exception as e:
            print(f"  [SKIP] {vid_path}: {e}")
            skipped += 1
            continue
        for lead in VAL_LEAD_TIMES:
            indices, tau = get_val_indices(row, fps, total, lead)
            frames = extract_frames(vr, indices)
            torch.save({"frames": torch.from_numpy(frames), "tau": tau,
                        "target": float(row["target"])},
                       os.path.join(val_dir, files[str(lead)]))
        val_index[vid_id] = files
        if (i + 1) % 50 == 0:
            print(f"  [val] {i+1}/{len(val_df)} videos cached...")

    with open(os.path.join(args.cache_dir, "val_index.json"), "w") as f:
        json.dump(val_index, f)
    print(f"\nVal cache done: {len(val_index)} videos x {len(VAL_LEAD_TIMES)} "
          f"lead times (skipped={skipped}) in {(time.time()-t0)/60:.1f} min")
    if skipped:
        print("WARNING: some videos were skipped -> the old eval had "
              "300 videos; re03 legacy check will tell whether it still matches.")


if __name__ == "__main__":
    main()
