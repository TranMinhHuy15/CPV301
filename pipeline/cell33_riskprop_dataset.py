"""
RiskProp datasets.
Train: reads the NEW sequence cache (nexar_cache_riskprop/train, 12
       ordered 5-frame snippets/video, stride 0.5s) built by cell30.
Val  : reuses AdaLEAValDataset's exact cache-reading logic (SAME
       nexar_cache_5f/val used by TOP-5f/AdaLEA-5f, fixed 0.5/1.0/1.5s
       lead-time windows) -- imported directly, no duplicate code.
"""
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def _snippet_to_tensor(frames_np):
    """frames_np: (5,H,W,3) uint8 -> (3,5,H,W) float normalized tensor."""
    f = frames_np.astype(np.float32) / 255.0
    f = (f - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(f).float().permute(3, 0, 1, 2)


class RiskPropTrainDataset(Dataset):
    """Returns 1 item = 1 video's full N-snippet sequence."""

    def __init__(self, cache_dir):
        self.cache_dir = os.path.join(cache_dir, "train")
        with open(os.path.join(cache_dir, "train_index.json")) as f:
            self.index = json.load(f)
        self.vid_ids = list(self.index.keys())

    def __len__(self):
        return len(self.vid_ids)

    def __getitem__(self, idx):
        vid_id = self.vid_ids[idx]
        fname = self.index[vid_id]
        data = torch.load(os.path.join(self.cache_dir, fname), weights_only=False)
        frames = data["frames"].numpy()          # (N, 5, H, W, 3) uint8
        seq = torch.stack([_snippet_to_tensor(frames[k]) for k in range(frames.shape[0])])
        # seq: (N, 3, 5, H, W)
        tau = data["tau"]                          # (N,) tensor, inf for negatives
        target = torch.tensor(data["target"], dtype=torch.float32)
        dt = float(data.get("dt", 0.5))
        return seq, tau, target, dt


# Val dataset: re-export AdaLEAValDataset unchanged (identical cache format,
# identical single-snippet (3,5,H,W) shape -- RiskPropModel.forward already
# handles that shape via its x.dim()==5 branch).
from cell22_adalea_dataset import AdaLEAValDataset as RiskPropValDataset  # noqa: E402,F401
