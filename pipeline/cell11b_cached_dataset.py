"""
TOP cached dataset -- reads the SAME cache tensors produced by
cell10_precache_5f.py (nexar_cache_5f, 5-frame causal windows). Converts
each window's tau (time-to-accident, seconds; inf for negatives) into a
20-dim cumulative binary label vector: label[h] = 1 if the collision
happens within (h+1)*0.1 seconds from the observed window (h = 0..19,
bin width 0.1s, covering 0.1s..2.0s) -- matches cell13_model.TOPModel's
20-horizon output and cell16_eval_cached_5f.py's LEAD_TO_HIDX mapping.
"""
import os
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

NUM_HORIZONS = 20
HORIZON_STEP = 0.1


def _to_tensor(data):
    frames = data["frames"].numpy().astype(np.float32) / 255.0
    frames = (frames - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(frames).float().permute(3, 0, 1, 2)


def _cumulative_labels(tau, num_horizons=NUM_HORIZONS, step=HORIZON_STEP):
    labels = torch.zeros(num_horizons, dtype=torch.float32)
    tau = float("inf") if tau is None else float(tau)
    if tau != float("inf"):
        for h in range(num_horizons):
            t_h = (h + 1) * step
            if tau <= t_h:
                labels[h] = 1.0
    return labels


class CachedTrainDataset(Dataset):
    def __init__(self, cache_dir):
        self.cache_dir = os.path.join(cache_dir, "train")
        with open(os.path.join(cache_dir, "train_index.json")) as f:
            self.index = json.load(f)
        self.vid_ids = list(self.index.keys())

    def __len__(self):
        return len(self.vid_ids)

    def __getitem__(self, idx):
        vid_id = self.vid_ids[idx]
        chosen = random.choice(self.index[vid_id])
        data = torch.load(os.path.join(self.cache_dir, chosen), weights_only=False)
        tensor = _to_tensor(data)
        labels = _cumulative_labels(data["tau"])
        target = torch.tensor(data["target"], dtype=torch.float32)
        return tensor, labels, target


class CachedValDataset(Dataset):
    def __init__(self, cache_dir, lead_time):
        self.cache_dir = os.path.join(cache_dir, "val")
        self.lead_time = str(lead_time)
        with open(os.path.join(cache_dir, "val_index.json")) as f:
            self.index = json.load(f)
        self.vid_ids = list(self.index.keys())

    def __len__(self):
        return len(self.vid_ids)

    def __getitem__(self, idx):
        vid_id = self.vid_ids[idx]
        fname = self.index[vid_id][self.lead_time]
        data = torch.load(os.path.join(self.cache_dir, fname), weights_only=False)
        tensor = _to_tensor(data)
        labels = _cumulative_labels(data["tau"])
        target = torch.tensor(data["target"], dtype=torch.float32)
        return tensor, labels, target
