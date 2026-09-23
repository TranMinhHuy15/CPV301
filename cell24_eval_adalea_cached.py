"""
AdaLEA multi-horizon evaluation -- same metric definitions/format as
TOP's cell16_eval_cached.py (for direct RQ1 comparability), reading the
SAME cached val windows (nexar_cache_5f, 0.5/1.0/1.5s fixed lead times).
AdaLEA's model outputs one score per window (no horizon indexing needed
-- the lead time is already baked into which window is loaded).
"""
import os, sys, time
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import (
    average_precision_score, roc_auc_score, roc_curve, auc)

sys.path.insert(0, "/kaggle/working")
from cell22_adalea_dataset import AdaLEAValDataset
from cell20_model_adalea import AdaLEAModel

CACHE_DIR   = "/kaggle/working/data/nexar_cache_5f"
OUTPUT_DIR  = "/kaggle/working/outputs_adalea"
# Multi-seed support: same ADALEA_SEED env var as cell23_train_adalea_cached.py.
SEED        = int(os.environ.get("ADALEA_SEED", "42"))
SUFFIX      = f"_seed{SEED}"
CKPT_PATH   = os.path.join(OUTPUT_DIR, f"best_adalea{SUFFIX}.pth")
BATCH_SIZE  = 8
NUM_WORKERS = 4
LEAD_TIMES  = [0.5, 1.0, 1.5]
TARGET_FAR  = 0.1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"[seed={SEED}] Loading {CKPT_PATH}...")
model = AdaLEAModel().to(device)
ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
model.load_state_dict(ckpt["model"])
model.eval()
print(f"  Epoch: {ckpt['epoch']}, Val Loss: {ckpt['best_val_loss']:.4f}, "
      f"Phi(final)={ckpt.get('phi', 0.0):.3f}s")


def run_inference(lead_time):
    ds = AdaLEAValDataset(CACHE_DIR, lead_time=lead_time)
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
        return mauc, tpr[idx], thr[idx], fpr[idx]
    return 0.0, 0.0, 1.0, 0.0


print(f"\n{'='*65}\nRunning multi-horizon evaluation (AdaLEA, cached)...\n{'='*65}")
results = {}
t0 = time.time()
for lead_time in LEAD_TIMES:
    scores, targets = run_inference(lead_time)
    ap = average_precision_score(targets, scores)
    mauc01, recall01, thr01, far01 = low_far_metrics(targets, scores)
    results[lead_time] = dict(
        AP=ap, mAUC01=mauc01, Recall01=recall01, Thresh01=thr01, FAR01=far01,
        scores=scores, targets=targets)
    print(f"  [{lead_time:.1f}s] AP={ap:.4f}  mAUC@0.1={mauc01:.4f}  "
          f"Recall@FAR<=0.1={recall01:.4f}")
print(f"  Inference time: {time.time()-t0:.0f}s")

proposal_map = np.mean([results[l]["AP"] for l in LEAD_TIMES])
video_auc = roc_auc_score(results[1.0]["targets"], results[1.0]["scores"])

targets_ref = results[1.0]["targets"]
pos_idx = np.where(targets_ref == 1)[0]
tta_values = []
for i in pos_idx:
    detected = [l for l in LEAD_TIMES
                if results[l]["scores"][i] >= results[l]["Thresh01"]]
    if detected:
        tta_values.append(max(detected))
mtta = np.mean(tta_values) if tta_values else 0.0
coverage = len(tta_values) / max(1, len(pos_idx))

print(f"\n{'='*65}")
print(f"  ADALEA BASELINE (cached, 5-frame, seed={SEED}) -- FINAL EVALUATION RESULTS (RQ1)")
print(f"{'='*65}")
print(f" Proposal mAP : {proposal_map:.4f}")
for l in LEAD_TIMES:
    print(f"   AP@{l:.1f}s : {results[l]['AP']:.4f}")
print(f" Video-level AUC : {video_auc:.4f}")
print(f" mTTA@FAR<=0.1   : {mtta:.3f}s (coverage={coverage*100:.1f}%)")
for l in LEAD_TIMES:
    print(f"   Recall@FAR<=0.1@{l:.1f}s : {results[l]['Recall01']:.4f}  "
          f"(actual FAR={results[l]['FAR01']:.4f}, threshold={results[l]['Thresh01']:.4f})")

mauc_mean = np.mean([results[l]["mAUC01"] for l in LEAD_TIMES])
mtta_all = mtta * coverage
print("\n Paper-style (TOP Table 1 format, FAR<=0.1):")
print(f"   AUC^0.1@0.5s={results[0.5]['mAUC01']:.4f}  AUC^0.1@1.0s={results[1.0]['mAUC01']:.4f}  "
      f"AUC^0.1@1.5s={results[1.5]['mAUC01']:.4f}")
print(f"   mAUC^0.1 (mean 3 horizons) = {mauc_mean:.4f}")
print(f"   mTTA^0.1 (undetected=0)    = {mtta_all:.3f}s  | detected-only = {mtta:.3f}s")
print(f"{'='*65}")

results_path = os.path.join(OUTPUT_DIR, f"eval_results_rq1_adalea5f{SUFFIX}.txt")
with open(results_path, "w") as f:
    f.write(f"AdaLEA Baseline (cached, 5-frame, seed={SEED}) - Evaluation Results (RQ1)\n")
    f.write(f"Proposal mAP: {proposal_map:.4f}\n")
    for l in LEAD_TIMES:
        f.write(f"  AP@{l:.1f}s: {results[l]['AP']:.4f}\n")
    f.write(f"Video AUC: {video_auc:.4f}\n")
    f.write(f"mTTA@FAR<=0.1: {mtta:.3f}s (coverage={coverage*100:.1f}%)\n")
    for l in LEAD_TIMES:
        f.write(f"Recall@FAR<=0.1@{l:.1f}s: {results[l]['Recall01']:.4f}\n")
    for l in LEAD_TIMES:
        f.write(f"ActualFAR@{l:.1f}s: {results[l]['FAR01']:.4f}\n")
    for l in LEAD_TIMES:
        f.write(f"Threshold@{l:.1f}s: {results[l]['Thresh01']:.4f}\n")
    f.write(f"mAUC01@0.5s={results[0.5]['mAUC01']:.4f}\nmAUC01@1.0s={results[1.0]['mAUC01']:.4f}\n"
            f"mAUC01@1.5s={results[1.5]['mAUC01']:.4f}\nmAUC01_mean={mauc_mean:.4f}\n"
            f"mTTA01_all={mtta_all:.3f}\nmTTA01_detected={mtta:.3f}\ncoverage={coverage:.3f}\n")
print(f"\nResults saved to: {results_path}")
