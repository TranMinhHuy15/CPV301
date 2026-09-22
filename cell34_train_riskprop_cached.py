"""
RiskProp training loop -- matched-budget vs TOP/AdaLEA where the budget
definition (epochs/optimizer/schedule) allows: SlowOnly-R50, SGD lr=0.01,
momentum 0.9, wd=1e-4, step decay @20/40, 50 epochs, AMP, auto-resume.

Per-step compute is heavier than TOP/AdaLEA by design (each "batch" is
BATCH_SIZE videos x SEQ_LEN=12 snippets = BATCH_SIZE*12 clips through the
backbone, vs TOP/AdaLEA's BATCH_SIZE single clips) -- this is the team's
accepted cost of giving RiskProp's FFR/AMC losses a real temporal chain
to operate on (see cell30/cell32 docstrings). Expect ~4x TOP-5f's
per-epoch wall time.

Train reads the NEW sequence cache (nexar_cache_riskprop). Val reads the
EXISTING nexar_cache_5f/val (single fixed-lead-time windows, same as
TOP-5f/AdaLEA-5f) -- val_loss here is a simplified plain-BCE proxy (no
FFR/AMC possible on isolated single windows); used only for checkpoint
selection, not part of the RQ1 metric table (eval script computes that).
"""
import os, sys, time, csv
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import torch.nn.functional as F

sys.path.insert(0, "/kaggle/working")
from cell33_riskprop_dataset import RiskPropTrainDataset, RiskPropValDataset
from cell31_model_riskprop import RiskPropModel
from cell32_riskprop_loss import riskprop_loss, COLLISION_WEIGHT, NEG_WEIGHT

# ============ CONFIG (matched budget vs TOP/AdaLEA) ============
EPOCHS       = 50
BATCH_SIZE   = 2          # videos/batch -> 2*SEQ_LEN=24 clips/forward
VAL_BATCH    = 8          # single-snippet val loader, same as TOP/AdaLEA
LR           = 0.01
MOMENTUM     = 0.9
WEIGHT_DECAY = 1e-4
NUM_WORKERS  = 4
CACHE_DIR_SEQ = "/kaggle/working/data/nexar_cache_riskprop"   # train
CACHE_DIR_VAL = "/kaggle/working/data/nexar_cache_5f"          # val (shared)
OUTPUT_DIR   = "/kaggle/working/outputs_riskprop"
SEED         = 42
USE_FFR      = True   # RQ1 full model. RQ2 ablation later flips these.
USE_AMC      = True
PAIRING_MODE = "random"  # paper default. RQ3 later adds "fixed".
# ==========================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device("cuda")
torch.manual_seed(SEED)

train_ds = RiskPropTrainDataset(CACHE_DIR_SEQ)
val_ds = RiskPropValDataset(CACHE_DIR_VAL, lead_time=1.0)

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
val_loader = DataLoader(
    val_ds, batch_size=VAL_BATCH, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True)

print(f"Train: {len(train_ds)} videos, {len(train_loader)} batches/epoch "
      f"({BATCH_SIZE}x12={BATCH_SIZE*12} clips/batch)")
print(f"Val  : {len(val_ds)} videos, {len(val_loader)} batches/epoch (single-snippet)")

model = RiskPropModel().to(device)
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

optimizer = torch.optim.SGD(
    model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

def lr_lambda(epoch):
    if epoch >= 40: return 0.01
    if epoch >= 20: return 0.1
    return 1.0

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
scaler = GradScaler()

start_epoch = 0
best_val_loss = float("inf")
resume_path = os.path.join(OUTPUT_DIR, "latest_riskprop.pth")

if os.path.exists(resume_path):
    ckpt = torch.load(resume_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    scaler.load_state_dict(ckpt["scaler"])
    start_epoch = ckpt["epoch"]
    best_val_loss = ckpt["best_val_loss"]
    print(f"\nResumed from epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")

log_path = os.path.join(OUTPUT_DIR, "training_log_riskprop.csv")
if start_epoch == 0:
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch", "train_loss", "train_bce", "train_reg", "train_mono",
             "val_loss", "lr", "time_sec", "best_val_loss"])

print(f"\n{'='*50}\nTraining RiskProp (cached, FFR={USE_FFR} AMC={USE_AMC} "
      f"pairing={PAIRING_MODE}): epoch {start_epoch} -> {EPOCHS-1}\n{'='*50}\n")

total_start = time.time()

for epoch in range(start_epoch, EPOCHS):
    model.train()
    train_loss = train_bce = train_reg = train_mono = 0.0
    t0 = time.time()

    for seq, tau, targets, dt in train_loader:
        seq = seq.to(device, non_blocking=True)          # (B,N,3,5,H,W)
        targets = targets.to(device, non_blocking=True)
        dt0 = float(dt[0])

        optimizer.zero_grad()
        with autocast(device_type="cuda"):
            logits = model(seq)                            # (B,N)
            loss, parts = riskprop_loss(
                logits, targets, dt=dt0, use_ffr=USE_FFR, use_amc=USE_AMC,
                pairing_mode=PAIRING_MODE)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item()
        train_bce += parts["bce"]; train_reg += parts["reg"]; train_mono += parts["mono"]

    # --- val_loss: plain weighted BCE on isolated single-snippet windows ---
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for frames, _, v_targets in val_loader:
            frames = frames.to(device, non_blocking=True)
            v_targets = v_targets.to(device, non_blocking=True)
            with autocast(device_type="cuda"):
                v_logits = model(frames)                    # (B,) single-snippet path
                w = torch.where(v_targets == 1,
                                 torch.full_like(v_targets, COLLISION_WEIGHT),
                                 torch.full_like(v_targets, NEG_WEIGHT))
                bce = F.binary_cross_entropy_with_logits(
                    v_logits, v_targets, weight=w)
            val_loss += bce.item()

    scheduler.step()
    n_tb = len(train_loader)
    avg_train = train_loss / n_tb
    avg_val = val_loss / len(val_loader)
    elapsed = time.time() - t0
    lr_now = optimizer.param_groups[0]["lr"]

    print(f"Epoch {epoch:2d}: train={avg_train:.4f} "
          f"(bce={train_bce/n_tb:.4f} reg={train_reg/n_tb:.4f} mono={train_mono/n_tb:.4f}), "
          f"val={avg_val:.4f}, lr={lr_now:.6f}, time={elapsed:.1f}s")

    with open(log_path, "a", newline="") as f:
        csv.writer(f).writerow(
            [epoch, f"{avg_train:.6f}", f"{train_bce/n_tb:.6f}", f"{train_reg/n_tb:.6f}",
             f"{train_mono/n_tb:.6f}", f"{avg_val:.6f}", f"{lr_now:.6f}",
             f"{elapsed:.1f}", f"{best_val_loss:.6f}"])

    ckpt_dict = {
        "epoch": epoch + 1, "model": model.state_dict(),
        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(), "best_val_loss": best_val_loss,
    }

    if avg_val < best_val_loss:
        best_val_loss = avg_val
        ckpt_dict["best_val_loss"] = best_val_loss
        torch.save(ckpt_dict, os.path.join(OUTPUT_DIR, "best_riskprop.pth"))
        print(f"  * Saved best_riskprop.pth (val_loss={best_val_loss:.4f})")

    torch.save(ckpt_dict, os.path.join(OUTPUT_DIR, "latest_riskprop.pth"))

total_time = time.time() - total_start
print(f"\n{'='*50}")
print(f"Training complete! Total: {total_time/60:.1f} min ({total_time/3600:.2f}h)")
print(f"Best val_loss: {best_val_loss:.4f}")
print(f"{'='*50}")
