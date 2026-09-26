"""
Pre-cache SEQUENCE of 12 causal 5-frame snippets/video (stride 0.5s) for
RiskProp (Zou et al., CVPR 2026 -- "RiskProp: Collision-Anchored
Self-Supervised Risk Propagation"). Unlike TOP/AdaLEA's independent
random windows, RiskProp's Future-Frame Regularization (FFR) and Adaptive
Monotonic Constraint (AMC) losses need an ORDERED chain of snippets
within the same video (known relative Δt between snippets).

TRAIN only -- val reuses the EXISTING nexar_cache_5f/val cache as-is
(fixed 0.5/1.0/1.5s lead-time windows, same format already used by
TOP-5f/AdaLEA-5f eval, no sequence needed at eval time).

Positive video: 12 snippets end near collision (toe), spaced 0.5s apart
  going backward -> covers last ~6s before collision.
Negative video: 12 snippets centered on video midpoint, spaced 0.5s apart
  (no collision anchor to align to; paper trains negatives w/ plain BCE
  over the whole clip, FFR/AMC disabled for these videos at loss time).

Same train/val split (SEED=42, same stratify key) as nexar_cache_5f, so
RiskProp trains on the IDENTICAL 1200 train videos as TOP/AdaLEA (RQ1
fairness).
"""
import os
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
if not os.path.exists(DATA_DIR):
    # vast.ai fallback -- data prepared from HuggingFace (nexar-ai/nexar_collision_prediction)
    # into Kaggle-style layout by cell09_prepare_hf_data.py. See chat 2026-09-23.
    DATA_DIR    = "/workspace/CPV301/data/nexar_kaggle_style"
CACHE_DIR   = "/kaggle/working/data/nexar_cache_riskprop"
SEED        = 42
NUM_FRAMES  = 5          # proposal Muc 5.5: causal 5-frame snippet (RQ1 shared protocol)
SIZE        = 224
SAMPLE_FPS  = 10
SEQ_LEN     = 12         # snippets/video (team decision: stride 0.5s option)
STRIDE_SEC  = 0.5
# ================================

random.seed(SEED)
np.random.seed(SEED)

TRAIN_DIR = os.path.join(CACHE_DIR, "train")
os.makedirs(TRAIN_DIR, exist_ok=True)


def make_stratify_key(df):
    """Identical to nexar_cache_5f's split logic -- keeps RiskProp on the
    SAME train/val partition as TOP/AdaLEA."""
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

print(f"RiskProp sequence precache -- Train: {len(train_df)} videos "
      f"(val untouched, reuses nexar_cache_5f/val)")


def get_sequence_t_obs(row, duration):
    """Return SEQ_LEN t_obs values (seconds), ordered earliest->latest,
    plus per-snippet tau (time-to-collision; inf for negatives)."""
    target = int(row["target"])
    window_sec = (NUM_FRAMES - 1) / SAMPLE_FPS   # 0.4s min span for 1 window
    min_t_obs = window_sec

    if target == 1:
        toe = row["time_of_event"]
        if pd.isna(toe):
            toe = duration - 0.5  # fallback, should not normally happen
        t_last = max(min_t_obs, min(toe - 0.1, duration - 0.1))
        t_obs_list = [max(min_t_obs, t_last - (SEQ_LEN - 1 - k) * STRIDE_SEC)
                      for k in range(SEQ_LEN)]
        tau_list = [max(0.0, toe - t) for t in t_obs_list]
    else:
        center = duration * 0.5
        t_obs_list = [np.clip(center - (SEQ_LEN / 2 - 0.5 - k) * STRIDE_SEC,
                               min_t_obs, max(min_t_obs, duration - 0.1))
                      for k in range(SEQ_LEN)]
        tau_list = [float("inf")] * SEQ_LEN
    return t_obs_list, tau_list


def indices_for_t_obs(t_obs, fps, total_frames):
    end_frame = min(int(t_obs * fps), total_frames - 1)
    frame_step = max(1, int(round(fps / SAMPLE_FPS)))
    idx = [end_frame - (NUM_FRAMES - 1 - k) * frame_step for k in range(NUM_FRAMES)]
    return np.clip(idx, 0, total_frames - 1).astype(int)


def extract_frames(vr, indices, size=SIZE):
    try:
        frames = vr.get_batch(indices).asnumpy()
    except Exception as e:
        print(f"  [WARN] frame extract failed: {e}")
        return np.zeros((len(indices), size, size, 3), dtype=np.uint8)
    imgs = [np.array(Image.fromarray(f).resize((size, size))) for f in frames]
    return np.stack(imgs).astype(np.uint8)


t0 = time.time()
train_index = {}
skipped = 0

for i, (_, row) in enumerate(train_df.iterrows()):
    vid_id = str(int(row["id"])).zfill(5)
    vid_path = os.path.join(video_dir, vid_id + ".mp4")
    try:
        vr = VideoReader(vid_path, ctx=cpu(0))
        total, fps = len(vr), vr.get_avg_fps()
        duration = total / fps
    except Exception as e:
        print(f"  [SKIP] {vid_path}: {e}")
        skipped += 1
        continue

    t_obs_list, tau_list = get_sequence_t_obs(row, duration)
    seq_frames = np.zeros((SEQ_LEN, NUM_FRAMES, SIZE, SIZE, 3), dtype=np.uint8)
    for k, t_obs in enumerate(t_obs_list):
        idx = indices_for_t_obs(t_obs, fps, total)
        seq_frames[k] = extract_frames(vr, idx)

    out_path = os.path.join(TRAIN_DIR, f"{vid_id}.pt")
    torch.save({
        "frames": torch.from_numpy(seq_frames),        # (SEQ_LEN, 5, 224, 224, 3) uint8
        "tau": torch.tensor(tau_list, dtype=torch.float32),  # (SEQ_LEN,)
        "target": float(row["target"]),
        "dt": STRIDE_SEC,
    }, out_path)
    train_index[vid_id] = os.path.basename(out_path)

    if (i + 1) % 200 == 0:
        print(f"  [train] {i+1}/{len(train_df)} videos cached...")

with open(os.path.join(CACHE_DIR, "train_index.json"), "w") as f:
    json.dump(train_index, f)

elapsed = time.time() - t0
print(f"\nRiskProp train cache done: {len(train_index)} videos x "
      f"{SEQ_LEN} snippets (stride {STRIDE_SEC}s), skipped={skipped}")
print(f"Total precache time: {elapsed/60:.1f} min")
os.system(f"du -sh {CACHE_DIR}")
print("\n(Val untouched -- RiskProp eval reuses /kaggle/working/data/nexar_cache_5f/val)")
