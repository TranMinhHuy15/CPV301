"""
AdaLEA cached dataset -- reads the SAME cache tensors produced for the
shared RQ1 protocol (nexar_cache_5f, 5-frame causal windows), but returns
raw tau (time-to-accident, seconds; inf for negatives) instead of the
20-horizon cumulative labels TOP/RiskProp use. No new precache needed.
"""
import os
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def _to_tensor(data):
    frames = data["frames"].numpy().astype(np.float32) / 255.0
    frames = (frames - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(frames).float().permute(3, 0, 1, 2)


class AdaLEATrainDataset(Dataset):
    """Same K-random-window-per-video sampling as CachedTrainDataset."""

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
        tau = data["tau"]
        tau = float("inf") if tau is None else float(tau)
        target = torch.tensor(data["target"], dtype=torch.float32)
        return tensor, torch.tensor(tau, dtype=torch.float32), target


class AdaLEAValDataset(Dataset):
    """Same fixed-lead-time sampling as CachedValDataset (0.5/1.0/1.5s)."""

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
        tau = data["tau"]
        tau = float("inf") if tau is None else float(tau)
        target = torch.tensor(data["target"], dtype=torch.float32)
        return tensor, torch.tensor(tau, dtype=torch.float32), target
