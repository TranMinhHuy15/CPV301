"""
AdaLEA training loop -- reads from the SAME cache as TOP-5f
(nexar_cache_5f, shared RQ1 input protocol). Matched-budget vs TOP:
SlowOnly-R50, SGD lr=0.01, momentum 0.9, wd=1e-4, step decay @20/40,
50 epochs, batch 8, AMP, auto-resume. Only the loss (AdaLEA adaptive
curriculum) and output head (single score) differ.

Phi update each epoch: run inference on ALL 3 fixed val lead-times
(0.5/1.0/1.5s), measure ATTC (farthest lead each positive is confidently
detected at, FAR<=0.1), EMA-update Phi, clamp to [0.3, 2.0]s.
"""
import os, sys, time, csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

sys.path.insert(0, "/kaggle/working")
from cell22_adalea_dataset import AdaLEATrainDataset, AdaLEAValDataset
from cell20_model_adalea import AdaLEAModel
from cell21_adalea_loss import adalea_loss, measure_attc, update_phi, PHI_INIT

# ============ CONFIG (matched budget vs TOP) ============
EPOCHS       = 50
BATCH_SIZE   = 8
LR           = 0.01
MOMENTUM     = 0.9
WEIGHT_DECAY = 1e-4
NUM_WORKERS  = 4
CACHE_DIR    = "/kaggle/working/data/nexar_cache_5f"   # shared w/ TOP-5f
OUTPUT_DIR   = "/kaggle/working/outputs_adalea"
# Multi-seed support: SEED overridable via ADALEA_SEED env var (falls back
# to 42), same pattern as cell34_train_riskprop_cached.py's RISKPROP_SEED.
SEED         = int(os.environ.get("ADALEA_SEED", "42"))
SUFFIX       = f"_seed{SEED}"
LEAD_TIMES   = [0.5, 1.0, 1.5]
# ==========================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device("cuda")
torch.manual_seed(SEED)
import numpy as np
np.random.seed(SEED)
print(f"[seed={SEED}] Training AdaLEA-5f baseline")

train_ds = AdaLEATrainDataset(CACHE_DIR)
val_ds_main = AdaLEAValDataset(CACHE_DIR, lead_time=1.0)  # for val_loss tracking

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
val_loader_main = DataLoader(
    val_ds_main, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True)
val_loaders_all = {
    l: DataLoader(AdaLEAValDataset(CACHE_DIR, lead_time=l),
                  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    for l in LEAD_TIMES
}

print(f"Train: {len(train_ds)} videos, {len(train_loader)} batches/epoch")
print(f"Val  : {len(val_ds_main)} videos, {len(val_loader_main)} batches/epoch")

model = AdaLEAModel().to(device)
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

optimizer = torch.optim.SGD(
    model.parameters(), lr=LR,
    momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

def lr_lambda(epoch):
    if epoch >= 40: return 0.01
    if epoch >= 20: return 0.1
    return 1.0

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
scaler = GradScaler()

start_epoch = 0
best_val_loss = float("inf")
phi = PHI_INIT
resume_path = os.path.join(OUTPUT_DIR, f"latest_adalea{SUFFIX}.pth")

if os.path.exists(resume_path):
    ckpt = torch.load(resume_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    scaler.load_state_dict(ckpt["scaler"])
    start_epoch = ckpt["epoch"]
    best_val_loss = ckpt["best_val_loss"]
    phi = ckpt.get("phi", PHI_INIT)
    print(f"\n[seed={SEED}] Resumed from epoch {start_epoch}, "
          f"best_val_loss={best_val_loss:.4f}, phi={phi:.3f}")

log_path = os.path.join(OUTPUT_DIR, f"training_log_adalea{SUFFIX}.csv")
if start_epoch == 0:
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch", "train_loss", "val_loss", "lr", "phi",
             "n_detected", "time_sec", "best_val_loss"])

print(f"\n{'='*50}\n[seed={SEED}] Training AdaLEA (cached, 5-frame): epoch {start_epoch} -> {EPOCHS-1}\n{'='*50}\n")

total_start = time.time()


@torch.no_grad()
def run_val_inference(loader):
    model.eval()
    all_scores, all_targets = [], []
    for videos, _, targets in loader:
        logits = model(videos.to(device))
        all_scores.append(torch.sigmoid(logits).cpu().numpy())
        all_targets.append(targets.numpy())
    import numpy as np
    return np.concatenate(all_scores), np.concatenate(all_targets)


for epoch in range(start_epoch, EPOCHS):
    model.train()
    train_loss = 0.0
    t0 = time.time()
    phi_for_this_epoch = phi

    for videos, tau, targets in train_loader:
        videos = videos.to(device, non_blocking=True)
        tau = tau.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        with autocast(device_type="cuda"):
            logits = model(videos)
            loss = adalea_loss(logits, tau, targets, phi_for_this_epoch)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item()

    # --- val_loss (tracked at lead=1.0s, same alpha/phi as this epoch's training) ---
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for videos, tau, targets in val_loader_main:
            videos = videos.to(device, non_blocking=True)
            tau = tau.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with autocast(device_type="cuda"):
                logits = model(videos)
                loss = adalea_loss(logits, tau, targets, phi_for_this_epoch)
            val_loss += loss.item()

    # --- ATTC measurement across all 3 lead-times -> EMA-update Phi ---
    lead_results = {l: run_val_inference(val_loaders_all[l]) for l in LEAD_TIMES}
    attc_measured, n_detected = measure_attc(lead_results)
    phi = update_phi(phi, attc_measured)

    scheduler.step()
    avg_train = train_loss / len(train_loader)
    avg_val = val_loss / len(val_loader_main)
    elapsed = time.time() - t0
    lr_now = optimizer.param_groups[0]["lr"]

    attc_str = f"{attc_measured:.3f}" if attc_measured is not None else "None"
    print(f"[seed={SEED}] Epoch {epoch:2d}: train={avg_train:.4f}, val={avg_val:.4f}, "
          f"lr={lr_now:.6f}, phi={phi_for_this_epoch:.3f}s, "
          f"attc={attc_str}s (n_det={n_detected}) -> phi_next={phi:.3f}s, "
          f"time={elapsed:.1f}s")

    with open(log_path, "a", newline="") as f:
        csv.writer(f).writerow(
            [epoch, f"{avg_train:.6f}", f"{avg_val:.6f}", f"{lr_now:.6f}",
             f"{phi_for_this_epoch:.4f}", n_detected, f"{elapsed:.1f}", f"{best_val_loss:.6f}"])

    ckpt_dict = {
        "epoch": epoch + 1, "model": model.state_dict(),
        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(), "best_val_loss": best_val_loss, "phi": phi,
        "seed": SEED,
    }

    if avg_val < best_val_loss:
        best_val_loss = avg_val
        ckpt_dict["best_val_loss"] = best_val_loss
        torch.save(ckpt_dict, os.path.join(OUTPUT_DIR, f"best_adalea{SUFFIX}.pth"))
        print(f"  * [seed={SEED}] Saved best_adalea{SUFFIX}.pth (val_loss={best_val_loss:.4f})")

    torch.save(ckpt_dict, resume_path)

total_time = time.time() - total_start
print(f"\n{'='*50}")
print(f"[seed={SEED}] Training complete! Total: {total_time/60:.1f} min ({total_time/3600:.2f}h)")
print(f"[seed={SEED}] Best val_loss: {best_val_loss:.4f} | Final phi: {phi:.3f}s")
print(f"{'='*50}")
