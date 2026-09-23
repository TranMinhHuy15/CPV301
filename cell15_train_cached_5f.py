"""
Training loop doc tu cache .pt (nhanh hon nhieu so voi decode video
truc tiep). Giu nguyen toan bo config TOP paper: SGD lr=0.01, step
decay @20/40, 50 epoch, pos_weight=10, AMP, auto-resume. Chi khac
DataLoader (doc tensor cache) va batch_size (8 thay vi 4, vi khong
con bi nghen boi decode CPU).
"""
import os, sys, time, csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

sys.path.insert(0, "/kaggle/working")
from cell11b_cached_dataset import CachedTrainDataset, CachedValDataset
from cell13_model import TOPModel

# ============ CONFIG ============
EPOCHS       = 50
BATCH_SIZE   = 8
LR           = 0.01
MOMENTUM     = 0.9
WEIGHT_DECAY = 1e-4
POS_WEIGHT   = 10.0
NUM_WORKERS  = 4
CACHE_DIR    = "/kaggle/working/data/nexar_cache_5f"
OUTPUT_DIR   = "/kaggle/working/outputs_5f"
# Multi-seed support: SEED overridable via TOP_SEED env var (falls back to
# 42), same pattern as cell34_train_riskprop_cached.py's RISKPROP_SEED --
# so this script can be run 3x (seeds 42/43/44) without edits, each run
# writing to its own best_cached_seed{N}.pth / log file.
SEED         = int(os.environ.get("TOP_SEED", "42"))
SUFFIX       = f"_seed{SEED}"
# ================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device("cuda")
torch.manual_seed(SEED)
import numpy as np
np.random.seed(SEED)
print(f"[seed={SEED}] Training TOP-5f baseline")

train_ds = CachedTrainDataset(CACHE_DIR)
val_ds = CachedValDataset(CACHE_DIR, lead_time=1.0)

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
val_loader = DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True)

print(f"Train: {len(train_ds)} videos, {len(train_loader)} batches/epoch")
print(f"Val  : {len(val_ds)} videos, {len(val_loader)} batches/epoch")

model = TOPModel().to(device)
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

criterion = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor([POS_WEIGHT], device=device))
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
resume_path = os.path.join(OUTPUT_DIR, f"latest_cached{SUFFIX}.pth")

if os.path.exists(resume_path):
    ckpt = torch.load(resume_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    scaler.load_state_dict(ckpt["scaler"])
    start_epoch = ckpt["epoch"]
    best_val_loss = ckpt["best_val_loss"]
    print(f"\n✅ [seed={SEED}] Resumed from epoch {start_epoch}, "
          f"best_val_loss={best_val_loss:.4f}")

log_path = os.path.join(OUTPUT_DIR, f"training_log_top5f{SUFFIX}.csv")
if start_epoch == 0:
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch", "train_loss", "val_loss", "lr",
             "time_sec", "best_val_loss"])

print(f"\n{'='*50}")
print(f"[seed={SEED}] Training (cached): epoch {start_epoch} -> {EPOCHS-1}")
print(f"{'='*50}\n")

total_start = time.time()

for epoch in range(start_epoch, EPOCHS):
    model.train()
    train_loss = 0.0
    t0 = time.time()

    for i, (videos, labels, targets) in enumerate(train_loader):
        videos = videos.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        with autocast(device_type="cuda"):
            logits = model(videos)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item()

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for videos, labels, targets in val_loader:
            videos = videos.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with autocast(device_type="cuda"):
                logits = model(videos)
                loss = criterion(logits, labels)
            val_loss += loss.item()

    scheduler.step()
    avg_train = train_loss / len(train_loader)
    avg_val = val_loss / len(val_loader)
    elapsed = time.time() - t0
    lr_now = optimizer.param_groups[0]["lr"]

    print(f"[seed={SEED}] Epoch {epoch:2d}: train={avg_train:.4f}, val={avg_val:.4f}, "
          f"lr={lr_now:.6f}, time={elapsed:.1f}s")

    with open(log_path, "a", newline="") as f:
        csv.writer(f).writerow(
            [epoch, f"{avg_train:.6f}", f"{avg_val:.6f}",
             f"{lr_now:.6f}", f"{elapsed:.1f}", f"{best_val_loss:.6f}"])

    if avg_val < best_val_loss:
        best_val_loss = avg_val
        torch.save({
            "epoch": epoch + 1, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(), "best_val_loss": best_val_loss,
            "seed": SEED,
        }, os.path.join(OUTPUT_DIR, f"best_cached{SUFFIX}.pth"))
        print(f"  ★ [seed={SEED}] Saved best_cached{SUFFIX}.pth (val_loss={best_val_loss:.4f})")

    torch.save({
        "epoch": epoch + 1, "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(), "best_val_loss": best_val_loss,
        "seed": SEED,
    }, resume_path)

total_time = time.time() - total_start
print(f"\n{'='*50}")
print(f"[seed={SEED}] Training complete! Total: {total_time/60:.1f} min "
      f"({total_time/3600:.2f}h)")
print(f"[seed={SEED}] Best val_loss: {best_val_loss:.4f}")
print(f"{'='*50}")
