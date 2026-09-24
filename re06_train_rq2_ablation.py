"""
re06 -- RQ2's 2x2 RiskProp ablation (FFR x AMC), conditions A/B/C.

This is a COPY of cell34_train_riskprop_cached.py (v3, matched-budget),
NOT an edit of it -- cell34 is kept exactly as it was for RQ1/RQ3
reproducibility (the "from start to end" code trail). The only change
here: USE_FFR / USE_AMC are read from env vars instead of hardcoded, so
the same file can produce all 3 new RQ2 conditions without any further
editing. Everything else (LR, epochs, optimizer, checkpoint rule, pair
count, cell32's lambdas) is byte-identical to cell34 -- required by
proposal Sec 5.4 ("Controls": all paired RQ2 runs keep the backbone,
split, batch construction, optimizer schedule, pair count, and stopping
rule unchanged).

Condition D (both FFR and AMC on) is NOT run from this file -- it is
already trained (3 seeds) as "RiskProp random" for RQ1/RQ3. Re-running
it here would just reproduce those same checkpoints at extra GPU cost.

Usage (default pairing=random, since RQ2 always ablates the random-offset
AMC per proposal Sec 5.4; PAIRING_MODE is still overridable if ever
needed):
    # A: neither
    RISKPROP_SEED=42 RISKPROP_USE_FFR=0 RISKPROP_USE_AMC=0 \
        python reeval/re06_train_rq2_ablation.py
    # B: FFR-only
    RISKPROP_SEED=42 RISKPROP_USE_FFR=1 RISKPROP_USE_AMC=0 \
        python reeval/re06_train_rq2_ablation.py
    # C: AMC-only (random)
    RISKPROP_SEED=42 RISKPROP_USE_FFR=0 RISKPROP_USE_AMC=1 \
        python reeval/re06_train_rq2_ablation.py
Repeat each for RISKPROP_SEED in {42,43,44} -- 9 runs total.

Output files go to the SAME /workspace/CPV301/outputs_riskprop/ directory
as cell34, with filenames tagged by condition (_ffronly / _amconly /
_neither) so they never collide with the existing "both" (D) checkpoints.
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

# ---- RQ2-specific: FFR/AMC toggles via env var (this is the only real
# difference from cell34, which hardcodes both True) ----
def _env_flag(name, default):
    v = os.environ.get(name)
    if v is None:
        return default
    return v not in ("0", "false", "False", "")

USE_FFR = _env_flag("RISKPROP_USE_FFR", True)
USE_AMC = _env_flag("RISKPROP_USE_AMC", True)
if USE_FFR and USE_AMC:
    sys.exit("USE_FFR=USE_AMC=True is condition D -- already trained as "
             "'RiskProp random' for RQ1/RQ3. Use those checkpoints instead "
             "of re-running this script for that condition.")

# RQ2 ablates the RANDOM-offset AMC specifically (proposal Sec 5.4/5.3);
# default here is "random" (cell34's default is "fixed" -- different
# scripts, different defaults, both driven by the same env var name).
PAIRING_MODE = os.environ.get("RISKPROP_PAIRING", "random")

SEED = int(os.environ.get("RISKPROP_SEED", "42"))
OUTPUT_DIR = "/workspace/CPV301/outputs_riskprop"

_cond_tag = "_ffronly" if USE_FFR else ("_amconly" if USE_AMC else "_neither")
SUFFIX = f"_{PAIRING_MODE}_seed{SEED}{_cond_tag}"
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

_cond_label = {"_ffronly": "B: FFR-only", "_amconly": "C: AMC-only (random)",
               "_neither": "A: neither"}[_cond_tag]
print(f"\n{'='*50}\n[seed={SEED}] RQ2 ablation -- condition {_cond_label} "
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

        optimizer.zero_grad()
        with autocast(device_type="cuda"):
            logits = model(seq)
            loss, parts = riskprop_loss(
                logits, targets, dt=dt0, use_ffr=USE_FFR, use_amc=USE_AMC,
                pairing_mode=PAIRING_MODE)

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
