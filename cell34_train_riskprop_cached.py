"""
RiskProp training loop -- matched-budget vs TOP/AdaLEA where the budget
definition (epochs/optimizer/schedule) allows: SlowOnly-R50, SGD, step
decay @20/40, up to 50 epochs, AMP, auto-resume.

Per-step compute is heavier than TOP/AdaLEA by design (each "batch" is
BATCH_SIZE videos x SEQ_LEN=12 snippets = BATCH_SIZE*12 clips through the
backbone, vs TOP/AdaLEA's BATCH_SIZE single clips) -- this is the team's
accepted cost of giving RiskProp's FFR/AMC losses a real temporal chain
to operate on (see cell30/cell32 docstrings). Expect ~4x TOP-5f's
per-epoch wall time.

Train reads the NEW sequence cache (nexar_cache_riskprop). Val reads the
EXISTING nexar_cache_5f/val (single fixed-lead-time windows, same as
TOP-5f/AdaLEA-5f).

---
v2 UPDATE (post-RQ1 optimization pass, per partner feedback
"KE HOACH KHAC PHUC VA TOI UU RISKPROP" -- goal: RiskProp > AdaLEA on
mAP and mAUC@0.1 while keeping mTTA/coverage/FAR at least as good):
- LR: 0.01 -> 0.002 (lower LR for a more stable training curve; v1 showed
  a transient gradient-instability spike around epoch 21-22).
- Added gradient clipping (max norm 1.0) -- v1 had none, likely
  contributing to that instability spike.
- Checkpoint selection AND early stopping now driven by validation mAP
  (average_precision_score on the val set, computed once per epoch),
  not plain val BCE loss -- directly optimizes for the metric RQ1 is
  judged on, instead of a proxy.
- Early stopping: patience=7 epochs with no val-mAP improvement.
- PAIRING_MODE default: "random" -> "fixed" (see cell32 v2 changes).
- Multi-seed support: SEED, and all output filenames, can be overridden
  via the RISKPROP_SEED environment variable (falls back to 42), so the
  same script can be run 3x (seeds 42/43/44) without edits, each run
  writing to its own best_riskprop_seed{N}.pth / log file, keeping v1's
  original best_riskprop.pth/training_log_riskprop.csv untouched as the
  "before optimization" reference point.
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

# ============ CONFIG (v2, post-optimization -- see docstring above) ============
EPOCHS        = 50
BATCH_SIZE    = 2          # videos/batch -> 2*SEQ_LEN=24 clips/forward
VAL_BATCH     = 8          # single-snippet val loader, same as TOP/AdaLEA
LR            = 0.002      # was 0.01
MOMENTUM      = 0.9
WEIGHT_DECAY  = 1e-4
GRAD_CLIP_NORM = 1.0       # new in v2 -- none in v1
EARLY_STOP_PATIENCE = 7    # new in v2 -- epochs with no val-mAP improvement
NUM_WORKERS   = 4
CACHE_DIR_SEQ = "/workspace/CPV301/data/nexar_cache_riskprop"   # train
CACHE_DIR_VAL = "/workspace/CPV301/data/nexar_cache_5f"          # val (shared)
USE_FFR       = True   # RQ1 full model. RQ2 ablation later flips these.
USE_AMC       = True
PAIRING_MODE  = "fixed"  # was "random" in v1; see cell32 v2 (FIXED_LAG_SEC=1.0)

SEED = int(os.environ.get("RISKPROP_SEED", "42"))
OUTPUT_DIR = "/workspace/CPV301/outputs_riskprop"
SUFFIX = f"_seed{SEED}"
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
best_val_map = -1.0
epochs_no_improve = 0
resume_path = os.path.join(OUTPUT_DIR, f"latest_riskprop{SUFFIX}.pth")

if os.path.exists(resume_path):
    ckpt = torch.load(resume_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    scaler.load_state_dict(ckpt["scaler"])
    start_epoch = ckpt["epoch"]
    best_val_map = ckpt.get("best_val_map", -1.0)
    epochs_no_improve = ckpt.get("epochs_no_improve", 0)
    print(f"\nResumed from epoch {start_epoch}, best_val_mAP={best_val_map:.4f}")

log_path = os.path.join(OUTPUT_DIR, f"training_log_riskprop{SUFFIX}.csv")
if start_epoch == 0:
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch", "train_loss", "train_bce", "train_reg", "train_mono",
             "val_loss", "val_mAP", "lr", "time_sec", "best_val_mAP"])

print(f"\n{'='*50}\n[seed={SEED}] Training RiskProp v2 (cached, FFR={USE_FFR} AMC={USE_AMC} "
      f"pairing={PAIRING_MODE}, LR={LR}, grad_clip={GRAD_CLIP_NORM}, "
      f"early_stop_patience={EARLY_STOP_PATIENCE}): epoch {start_epoch} -> {EPOCHS-1}\n{'='*50}\n")

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
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item()
        train_bce += parts["bce"]; train_reg += parts["reg"]; train_mono += parts["mono"]

    # --- val: plain weighted BCE loss + val mAP (average_precision_score) ---
    model.eval()
    val_loss = 0.0
    val_probs, val_targets = [], []
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
             f"{elapsed:.1f}", f"{best_val_map:.6f}"])

    ckpt_dict = {
        "epoch": epoch + 1, "model": model.state_dict(),
        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(), "best_val_map": best_val_map,
        "epochs_no_improve": epochs_no_improve, "seed": SEED,
    }

    if val_map > best_val_map:
        best_val_map = val_map
        epochs_no_improve = 0
        ckpt_dict["best_val_map"] = best_val_map
        ckpt_dict["epochs_no_improve"] = epochs_no_improve
        torch.save(ckpt_dict, os.path.join(OUTPUT_DIR, f"best_riskprop{SUFFIX}.pth"))
        print(f"  * Saved best_riskprop{SUFFIX}.pth (val_mAP={best_val_map:.4f})")
    else:
        epochs_no_improve += 1
        ckpt_dict["epochs_no_improve"] = epochs_no_improve

    torch.save(ckpt_dict, os.path.join(OUTPUT_DIR, f"latest_riskprop{SUFFIX}.pth"))

    if epochs_no_improve >= EARLY_STOP_PATIENCE:
        print(f"\n[seed={SEED}] Early stopping at epoch {epoch}: "
              f"no val_mAP improvement for {EARLY_STOP_PATIENCE} epochs "
              f"(best_val_mAP={best_val_map:.4f}).")
        break

total_time = time.time() - total_start
print(f"\n{'='*50}")
print(f"[seed={SEED}] Training complete! Total: {total_time/60:.1f} min ({total_time/3600:.2f}h)")
print(f"Best val_mAP: {best_val_map:.4f}")
print(f"{'='*50}")
