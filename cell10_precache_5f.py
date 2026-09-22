"""
Pre-cache causal windows to .pt tensors -- tach decode video ra khoi
vong train de GPU khong phai cho CPU doc file .mp4.

Train : K=3 window ngau nhien/video (giu augmentation causal cua v3)
Val   : 3 window co dinh tai lead_time = 0.5s / 1.0s / 1.5s
        (khop dung protocol multi-horizon eval, Muc 5.5)
"""
import os
import sys
import json
import time
import random
import numpy as np
import pandas as pd
import torch
from decord import VideoReader, cpu
from PIL import Image
from sklearn.model_selection import train_test_split

# ============ CONFIG ============
DATA_DIR    = "/kaggle/input/competitions/nexar-collision-prediction"
if not os.path.exists(DATA_DIR):
    DATA_DIR    = "/kaggle/input/nexar-collision-prediction"
CACHE_DIR   = "/kaggle/working/data/nexar_cache_5f"
SEED        = 42
NUM_FRAMES  = 5        # proposal Muc 5.5: causal 5-frame snippet (RQ1 shared protocol)
SIZE        = 224
SAMPLE_FPS  = 10
K_TRAIN_WINDOWS = 3     # restored to v3 default -- 5-frame cache is much smaller
VAL_LEAD_TIMES  = [0.5, 1.0, 1.5]
# ================================

random.seed(SEED)
np.random.seed(SEED)

TRAIN_DIR = os.path.join(CACHE_DIR, "train")
VAL_DIR = os.path.join(CACHE_DIR, "val")
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)


def make_stratify_key(df):
    """Combined stratify key: target + alert-to-event duration bin
    (Proposal Muc 5.2/5.5). Negative videos -> "neg"."""
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


full_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
strat_key = make_stratify_key(full_df)
train_df, val_df = train_test_split(
    full_df, test_size=0.2, stratify=strat_key, random_state=SEED)
video_dir = os.path.join(DATA_DIR, "train")

print(f"Train: {len(train_df)} videos | Val: {len(val_df)} videos")


def get_causal_indices(row, fps, total_frames, lead_time=None, is_train=True):
    """Same causal-window logic as cell11_dataset.NexarDataset (v3)."""
    target = int(row["target"])
    duration = total_frames / fps
    window_sec = (NUM_FRAMES - 1) / SAMPLE_FPS
    min_t_obs = window_sec

    if target == 1:
        toe = row["time_of_event"]
        if pd.isna(toe):
            tau = float("inf")
            t_obs = (duration * 0.5 if not is_train else
                      random.uniform(min_t_obs, max(min_t_obs, duration - 0.5)))
        else:
            lt = lead_time if lead_time is not None else (
                random.uniform(0.2, 2.0) if is_train else 1.0)
            t_obs = max(min_t_obs, toe - lt)
            t_obs = min(t_obs, toe - 0.1)
            t_obs = max(min_t_obs, t_obs)
            tau = toe - t_obs
    else:
        tau = float("inf")
        t_obs = (random.uniform(min_t_obs, max(min_t_obs, duration - 0.5))
                  if is_train else duration * 0.5)

    end_frame = min(int(t_obs * fps), total_frames - 1)
    frame_step = max(1, int(round(fps / SAMPLE_FPS)))
    indices = [end_frame - (NUM_FRAMES - 1 - k) * frame_step
               for k in range(NUM_FRAMES)]
    indices = np.clip(indices, 0, total_frames - 1).astype(int)
    return indices, tau


def extract_frames(vr, indices, size=SIZE):
    try:
        frames = vr.get_batch(indices).asnumpy()
    except Exception as e:
        print(f"  [WARN] frame extract failed: {e}")
        return np.zeros((len(indices), size, size, 3), dtype=np.uint8)
    imgs = [np.array(Image.fromarray(f).resize((size, size))) for f in frames]
    return np.stack(imgs).astype(np.uint8)  # (T,H,W,3) uint8


t0 = time.time()

# --- TRAIN: K random windows / video ---
train_index = {}
skipped = 0
for i, (_, row) in enumerate(train_df.iterrows()):
    vid_id = str(int(row["id"])).zfill(5)
    vid_path = os.path.join(video_dir, vid_id + ".mp4")
    try:
        vr = VideoReader(vid_path, ctx=cpu(0))
        total, fps = len(vr), vr.get_avg_fps()
    except Exception as e:
        print(f"  [SKIP] {vid_path}: {e}")
        skipped += 1
        continue

    files = []
    for w in range(K_TRAIN_WINDOWS):
        indices, tau = get_causal_indices(
            row, fps, total, lead_time=None, is_train=True)
        frames = extract_frames(vr, indices)
        out_path = os.path.join(TRAIN_DIR, f"{vid_id}_w{w}.pt")
        torch.save({"frames": torch.from_numpy(frames), "tau": tau,
                    "target": float(row["target"])}, out_path)
        files.append(os.path.basename(out_path))
    train_index[vid_id] = files

    if (i + 1) % 200 == 0:
        print(f"  [train] {i+1}/{len(train_df)} videos cached...")

with open(os.path.join(CACHE_DIR, "train_index.json"), "w") as f:
    json.dump(train_index, f)

print(f"\nTrain cache done: {len(train_index)} videos x "
      f"{K_TRAIN_WINDOWS} windows (skipped={skipped})")

# --- VAL: 3 fixed lead times / video ---
val_index = {}
skipped_val = 0
for i, (_, row) in enumerate(val_df.iterrows()):
    vid_id = str(int(row["id"])).zfill(5)
    vid_path = os.path.join(video_dir, vid_id + ".mp4")
    try:
        vr = VideoReader(vid_path, ctx=cpu(0))
        total, fps = len(vr), vr.get_avg_fps()
    except Exception as e:
        print(f"  [SKIP] {vid_path}: {e}")
        skipped_val += 1
        continue

    files = {}
    for lead in VAL_LEAD_TIMES:
        indices, tau = get_causal_indices(
            row, fps, total, lead_time=lead, is_train=False)
        frames = extract_frames(vr, indices)
        out_path = os.path.join(VAL_DIR, f"{vid_id}_lead{lead}.pt")
        torch.save({"frames": torch.from_numpy(frames), "tau": tau,
                    "target": float(row["target"])}, out_path)
        files[str(lead)] = os.path.basename(out_path)
    val_index[vid_id] = files

    if (i + 1) % 100 == 0:
        print(f"  [val] {i+1}/{len(val_df)} videos cached...")

with open(os.path.join(CACHE_DIR, "val_index.json"), "w") as f:
    json.dump(val_index, f)

elapsed = time.time() - t0
print(f"\nVal cache done: {len(val_index)} videos x "
      f"{len(VAL_LEAD_TIMES)} lead times (skipped={skipped_val})")
print(f"\nTotal precache time: {elapsed/60:.1f} min")
os.system(f"du -sh {CACHE_DIR}")
