"""
Multi-horizon evaluation doc tu cache .pt (3 window co dinh
0.5s/1.0s/1.5s da precache) -- cung logic metric nhu cell16_eval.py
ban decord, chi khac nguon du lieu.
"""
import os, sys, time
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import (
    average_precision_score, roc_auc_score, roc_curve, auc)

sys.path.insert(0, "/kaggle/working")
from cell11b_cached_dataset import CachedValDataset
from cell13_model import TOPModel

CACHE_DIR   = "/kaggle/working/data/nexar_cache_5f"
OUTPUT_DIR  = "/kaggle/working/outputs_5f"
CKPT_PATH   = os.path.join(OUTPUT_DIR, "best_cached.pth")
BATCH_SIZE  = 8
NUM_WORKERS = 4
LEAD_TO_HIDX = {0.5: 4, 1.0: 9, 1.5: 14}
TARGET_FAR = 0.1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Loading {CKPT_PATH}...")
model = TOPModel().to(device)
ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
model.load_state_dict(ckpt["model"])
model.eval()
print(f"  Epoch: {ckpt['epoch']}, Val Loss: {ckpt['best_val_loss']:.4f}")


def run_inference(lead_time):
    ds = CachedValDataset(CACHE_DIR, lead_time=lead_time)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS)
    all_probs, all_targets = [], []
    with torch.no_grad():
        for frames, _, targets in loader:
            logits = model(frames.to(device))
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu())
            all_targets.append(targets.float())
    return torch.cat(all_probs).numpy(), torch.cat(all_targets).numpy()


def low_far_metrics(targets, scores, target_far=TARGET_FAR):
    fpr, tpr, thr = roc_curve(targets, scores)
    mask = fpr <= target_far
    if np.any(mask):
        fpr_low = np.append(fpr[mask], target_far)
        tpr_low = np.append(tpr[mask], np.interp(target_far, fpr, tpr))
        mauc = auc(fpr_low, tpr_low) / target_far
        idx = np.where(fpr <= target_far)[0][-1]
        return mauc, tpr[idx], thr[idx]
    return 0.0, 0.0, 1.0


print(f"\n{'='*65}\nRunning multi-horizon evaluation (cached)...\n{'='*65}")
results = {}
t0 = time.time()
for lead_time, h_idx in LEAD_TO_HIDX.items():
    probs, targets = run_inference(lead_time)
    score_matched = probs[:, h_idx]
    score_max = probs.max(axis=1)
    ap = average_precision_score(targets, score_matched)
    mauc01, recall01, thr01 = low_far_metrics(targets, score_matched)
    results[lead_time] = dict(
        AP=ap, mAUC01=mauc01, Recall01=recall01, Thresh01=thr01,
        score_matched=score_matched, score_max=score_max, targets=targets)
    print(f"  [{lead_time:.1f}s] AP={ap:.4f}  mAUC@0.1={mauc01:.4f}  "
          f"Recall@FAR<=0.1={recall01:.4f}")
print(f"  Inference time: {time.time()-t0:.0f}s")

proposal_map = np.mean([results[l]["AP"] for l in LEAD_TO_HIDX])
video_auc = roc_auc_score(results[1.0]["targets"], results[1.0]["score_max"])

targets_ref = results[1.0]["targets"]
pos_idx = np.where(targets_ref == 1)[0]
tta_values = []
for i in pos_idx:
    detected = [l for l in LEAD_TO_HIDX
                if results[l]["score_matched"][i] >= results[l]["Thresh01"]]
    if detected:
        tta_values.append(max(detected))
mtta = np.mean(tta_values) if tta_values else 0.0
coverage = len(tta_values) / max(1, len(pos_idx))

print(f"\n{'='*65}")
print("  TOP BASELINE v5-5f (cached) — FINAL EVALUATION RESULTS (RQ1)")
print(f"{'='*65}")
print(f" Proposal mAP : {proposal_map:.4f}")
for l in LEAD_TO_HIDX:
    print(f"   AP@{l:.1f}s : {results[l]['AP']:.4f}")
print(f" Video-level AUC : {video_auc:.4f}")
print(f" mTTA@FAR<=0.1   : {mtta:.3f}s (coverage={coverage*100:.1f}%)")
# --- Paper-style table (TOP paper, Table 1 format) ---
mauc_mean = np.mean([results[l]["mAUC01"] for l in LEAD_TO_HIDX])
mtta_all = mtta * coverage  # undetected positives counted as TTA=0
print("\n Paper-style (TOP Table 1 format, FAR<=0.1):")
print(f"   AUC^0.1@0.5s={results[0.5]['mAUC01']:.4f}  AUC^0.1@1.0s={results[1.0]['mAUC01']:.4f}  "
      f"AUC^0.1@1.5s={results[1.5]['mAUC01']:.4f}")
print(f"   mAUC^0.1 (mean 3 horizons) = {mauc_mean:.4f}")
print(f"   mTTA^0.1 (undetected=0)    = {mtta_all:.3f}s  | detected-only = {mtta:.3f}s")
with open(os.path.join(OUTPUT_DIR, "eval_paper_style.txt"), "w") as f:
    f.write(f"AUC01@0.5s={results[0.5]['mAUC01']:.4f}\nAUC01@1.0s={results[1.0]['mAUC01']:.4f}\n"
            f"AUC01@1.5s={results[1.5]['mAUC01']:.4f}\nmAUC01={mauc_mean:.4f}\n"
            f"mTTA01_all={mtta_all:.3f}\nmTTA01_detected={mtta:.3f}\ncoverage={coverage:.3f}\n")

print(f"{'='*65}")

results_path = os.path.join(OUTPUT_DIR, "eval_results_rq1_cached.txt")
with open(results_path, "w") as f:
    f.write("TOP Baseline v5-5f (cached) - Evaluation Results (RQ1)\n")
    f.write(f"Proposal mAP: {proposal_map:.4f}\n")
    for l in LEAD_TO_HIDX:
        f.write(f"  AP@{l:.1f}s: {results[l]['AP']:.4f}\n")
    f.write(f"Video AUC: {video_auc:.4f}\n")
    f.write(f"mTTA@FAR<=0.1: {mtta:.3f}s (coverage={coverage*100:.1f}%)\n")
print(f"\nResults saved to: {results_path}")