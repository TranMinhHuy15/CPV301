"""
re09 -- RQ3 secondary sensitivity: FixedLag-RiskProp with tau = 0.5 s and 1.5 s
(proposal Sec. 5.3: "Secondary sensitivity runs use tau = 0.5 and 1.5 seconds
and will not be promoted as separate contributions"; objectives: "...including
sensitivity at 0.5 and 1.5 seconds").

COPY of cell34_train_riskprop_cached.py (v3, matched budget), NOT an edit --
cell34/cell32 stay untouched. Only differences:
  * pairing is always "fixed" (both FFR and AMC on, like the tau = 1.0 s runs);
  * the fixed AMC gap comes from env RISKPROP_TAU (seconds) and is passed to
    cell32.riskprop_loss(..., fixed_gap=round(tau / dt)). With the 0.5 s snippet
    stride: tau 0.5 -> 1 step, tau 1.0 -> 2 steps (= cell32 default, already
    trained as "RiskProp fixed"), tau 1.5 -> 3 steps.
Everything else (epochs, LR schedule, batch, checkpoint rule, lambdas, pair
count, weights) is identical to cell34 -- proposal Sec. 5.4 "Controls".

Usage:
    RISKPROP_SEED=42 RISKPROP_TAU=0.5 python reeval/re09_train_rq3_sensitivity.py
    (or run everything: bash reeval/run_rq3_sens.sh)
Outputs (same folder as cell34): best_/latest_riskprop_fixed_seed{S}_tau{T}.pth,
training_log_riskprop_fixed_seed{S}_tau{T}.csv
"""
import os, sys, time, csv
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from sklearn.metrics import average_precision_score
import torch.nn.functional as F

sys.path.insert(0, "/workspace/CPV301")
from cell33_riskprop_dataset import RiskPropTrainDataset, RiskPropValDataset
from cell31_model_riskprop import RiskPropModel
from cell32_riskprop_loss import riskprop_loss, COLLISION_WEIGHT, NEG_WEIGHT

# ============ CONFIG (identical values to cell34 v3 -- do not change) ============
EPOCHS        = 50
BATCH_SIZE    = 2
VAL_BATCH     = 8
LR            = 0.01
MOMENTUM      = 0.9
WEIGHT_DECAY  = 1e-4
NUM_WORKERS   = 4
CACHE_DIR_SEQ = "/workspace/CPV301/data/nexar_cache_riskprop"
CACHE_DIR_VAL = "/workspace/CPV301/data/nexar_cache_5f"

# ---- RQ3-sensitivity-specific: fixed pairing with a configurable lag ----
USE_FFR = True
USE_AMC = True
PAIRING_MODE = "fixed"
TAU = float(os.environ.get("RISKPROP_TAU", "nan"))
if TAU not in (0.5, 1.5):
    sys.exit("Set RISKPROP_TAU to 0.5 or 1.5 (tau = 1.0 s is already trained "
             "as 'RiskProp fixed' by cell34).")

SEED = int(os.environ.get("RISKPROP_SEED", "42"))
OUTPUT_DIR = "/workspace/CPV301/outputs_riskprop"
SUFFIX = f"_{PAIRING_MODE}_seed{SEED}_tau{TAU}"
# ==========================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device("cuda")
torch.manual_seed(SEED)
np.random.seed(SEED)

train_ds = RiskPropTrainDataset(CACHE_DIR_SEQ)
val_ds = RiskPropValDataset(CACHE_DIR_VAL, lead_time=1.0)

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
val_loader = DataLoader(
    val_ds, batch_size=VAL_BATCH, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True)

print(f"[seed={SEED}] Train: {len(train_ds)} videos, {len(train_loader)} batches/epoch "
      f"({BATCH_SIZE}x12={BATCH_SIZE*12} clips/batch)")
print(f"[seed={SEED}] Val  : {len(val_ds)} videos, {len(val_loader)} batches/epoch (single-snippet)")

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
resume_path = os.path.join(OUTPUT_DIR, f"latest_riskprop{SUFFIX}.pth")

if os.path.exists(resume_path):
    ckpt = torch.load(resume_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    scaler.load_state_dict(ckpt["scaler"])
    start_epoch = ckpt["epoch"]
    best_val_loss = ckpt.get("best_val_loss", float("inf"))
    print(f"\nResumed from epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")

log_path = os.path.join(OUTPUT_DIR, f"training_log_riskprop{SUFFIX}.csv")
if start_epoch == 0:
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch", "train_loss", "train_bce", "train_reg", "train_mono",
             "val_loss", "val_mAP", "lr", "time_sec", "best_val_loss"])

print(f"\n{'='*50}\n[seed={SEED}] RQ3 sensitivity -- FixedLag tau={TAU}s "
      f"(FFR={USE_FFR} AMC={USE_AMC} pairing={PAIRING_MODE}, LR={LR}, "
      f"matched-budget, identical to cell34 otherwise): "
      f"epoch {start_epoch} -> {EPOCHS-1}\n{'='*50}\n")

total_start = time.time()

for epoch in range(start_epoch, EPOCHS):
    model.train()
    train_loss = train_bce = train_reg = train_mono = 0.0
    t0 = time.time()

    for seq, tau, targets, dt in train_loader:
        seq = seq.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        dt0 = float(dt[0])
        gap = max(1, int(round(TAU / dt0)))

        optimizer.zero_grad()
        with autocast(device_type="cuda"):
            logits = model(seq)
            loss, parts = riskprop_loss(
                logits, targets, dt=dt0, use_ffr=USE_FFR, use_amc=USE_AMC,
                pairing_mode=PAIRING_MODE, fixed_gap=gap)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item()
        train_bce += parts["bce"]; train_reg += parts["reg"]; train_mono += parts["mono"]

    model.eval()
    val_loss = 0.0
    val_probs, val_targets = [], []
    with torch.no_grad():
        for frames, _, v_targets in val_loader:
            frames = frames.to(device, non_blocking=True)
            v_targets = v_targets.to(device, non_blocking=True)
            with autocast(device_type="cuda"):
                v_logits = model(frames)
                w = torch.where(v_targets == 1,
                                 torch.full_like(v_targets, COLLISION_WEIGHT),
                                 torch.full_like(v_targets, NEG_WEIGHT))
                bce = F.binary_cross_entropy_with_logits(
                    v_logits, v_targets, weight=w)
            val_loss += bce.item()
            val_probs.append(torch.sigmoid(v_logits).float().cpu())
            val_targets.append(v_targets.float().cpu())

    val_probs_np = torch.cat(val_probs).numpy()
    val_targets_np = torch.cat(val_targets).numpy()
    val_map = average_precision_score(val_targets_np, val_probs_np)

    scheduler.step()
    n_tb = len(train_loader)
    avg_train = train_loss / n_tb
    avg_val = val_loss / len(val_loader)
    elapsed = time.time() - t0
    lr_now = optimizer.param_groups[0]["lr"]

    print(f"[seed={SEED}] Epoch {epoch:2d}: train={avg_train:.4f} "
          f"(bce={train_bce/n_tb:.4f} reg={train_reg/n_tb:.4f} mono={train_mono/n_tb:.4f}), "
          f"val_loss={avg_val:.4f}, val_mAP={val_map:.4f}, lr={lr_now:.6f}, time={elapsed:.1f}s")

    with open(log_path, "a", newline="") as f:
        csv.writer(f).writerow(
            [epoch, f"{avg_train:.6f}", f"{train_bce/n_tb:.6f}", f"{train_reg/n_tb:.6f}",
             f"{train_mono/n_tb:.6f}", f"{avg_val:.6f}", f"{val_map:.6f}", f"{lr_now:.6f}",
             f"{elapsed:.1f}", f"{best_val_loss:.6f}"])

    ckpt_dict = {
        "epoch": epoch + 1, "model": model.state_dict(),
        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(), "best_val_loss": best_val_loss, "seed": SEED,
    }

    if avg_val < best_val_loss:
        best_val_loss = avg_val
        ckpt_dict["best_val_loss"] = best_val_loss
        torch.save(ckpt_dict, os.path.join(OUTPUT_DIR, f"best_riskprop{SUFFIX}.pth"))
        print(f"  * Saved best_riskprop{SUFFIX}.pth (val_loss={best_val_loss:.4f})")

    torch.save(ckpt_dict, os.path.join(OUTPUT_DIR, f"latest_riskprop{SUFFIX}.pth"))

total_time = time.time() - total_start
print(f"\n{'='*50}")
print(f"[seed={SEED}] Training complete! Total: {total_time/60:.1f} min ({total_time/3600:.2f}h)")
print(f"Best val_loss: {best_val_loss:.4f}")
print(f"{'='*50}")
