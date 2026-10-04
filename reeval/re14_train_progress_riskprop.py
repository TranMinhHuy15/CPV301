"""
re14 -- Train Progressive RiskProp conditions (RQ3 new). NEW file; cell31 /
cell32 / cell33 / cell34 are imported, never edited.

Conditions (all share backbone, cache, split, 12 snippets, transforms,
optimizer, LR schedule, epochs, batch, seeds, checkpoint rule, pairing):
    B   L_BCE + 0.5 L_FFR + 0.5 L_AMC                 (matched RiskProp baseline)
    P   B + lambda * L_CPS(continuous target)         (PRE-ACT-inspired)
    S   B + lambda * L_CPS(binary 1[0<tau<H])         (dense-binary control)
    F   L_BCE + 0.5 L_FFR                             (optional, AMC off)
    FP  F + lambda * L_CPS(continuous)                (optional)
Base loss = cell32.riskprop_loss called unchanged (bit-identical to cell34 /
re09 for the same logits and RNG state). CPS = re13.cps_loss on sigmoid(z),
computed in fp32 outside the base loss; it consumes no RNG, so B/P/S see the
same batch order and the same AMC pair draws for a given seed.

The model only receives the video tensor `seq`; tau / time_of_event /
masks are used outside forward() for the loss only.

Modes
  dev (default)  : any of --max-steps / --max-epochs / --limit-videos /
                   --limit-val / --model dummy allowed; output goes to
                   <output-root>/dev/<run>/ and is marked research_result=false.
  --full-run     : refuses unless the audit passed (complete, probed, cache
                   scanned), the sidecar is complete and usable, cache ids ==
                   split manifest, no limits, 50 epochs, real model, and the
                   arguments equal the locked design in --lock-file.

Example (smoke, 1 step):
  python reeval/re14_train_progress_riskprop.py --condition P --seed 42 \
     --pairing-mode fixed --horizon-sec 2.0 --alpha 3 --lambda-prog 1.0 \
     --cache-root data/nexar_cache_riskprop --cache-5f data/nexar_cache_5f \
     --sidecar outputs_progress/sidecar/sidecar.json \
     --split-manifest results/repro/split_manifest_seed42.json \
     --output-root outputs_progress --max-steps 1 --limit-val 16
"""
import argparse
import csv
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

CONDITIONS = {   # name -> (cps mode, use_ffr, use_amc)
    "B": ("none", True, True),
    "P": ("continuous", True, True),
    "S": ("binary", True, True),
    "F": ("none", True, False),
    "FP": ("continuous", True, False),
}
FULL_EPOCHS = 50
MIN_PIXEL_VERIFY = 24          # = re12.MIN_PIXEL_VERIFY
BATCH_SIZE, VAL_BATCH = 2, 8
LR, MOMENTUM, WEIGHT_DECAY = 0.01, 0.9, 1e-4          # = cell34 / re09


def sha256_file(path, head=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        if head:
            h.update(f.read(head))
        else:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


def git_info(repo):
    def run(*a):
        try:
            return subprocess.check_output(["git", "-C", repo, *a], stderr=subprocess.DEVNULL,
                                           text=True).strip()
        except Exception:
            return None
    status = run("status", "--porcelain") or ""
    return {"sha": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "uncommitted_files": [ln[3:] for ln in status.splitlines() if ln.strip()]}


def run_name(cond, pairing, horizon, alpha, lam, seed, tag=""):
    """Baselines (B, F) have no CPS term -> name independent of H/alpha/lambda,
    so the H = 1.5 s sensitivity run of P is compared with the same B."""
    if CONDITIONS[cond][0] == "none":
        name = f"{cond}_{pairing}_seed{seed}"
    else:
        name = f"{cond}_{pairing}_H{horizon:g}_a{alpha:g}_lam{lam:g}_seed{seed}"
    return f"{name}_{tag}" if tag else name


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True, choices=list(CONDITIONS))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--pairing-mode", required=True, choices=["fixed", "random"])
    ap.add_argument("--horizon-sec", type=float, default=2.0)
    ap.add_argument("--alpha", type=float, default=3.0)
    ap.add_argument("--lambda-prog", type=float, default=None)
    ap.add_argument("--repo-dir", default=os.path.dirname(HERE))
    ap.add_argument("--cache-root", required=True, help="nexar_cache_riskprop (cell30)")
    ap.add_argument("--cache-5f", required=True, help="nexar_cache_5f (val windows, re01)")
    ap.add_argument("--sidecar", default="", help="re12 sidecar.json (required for CPS conditions)")
    ap.add_argument("--split-manifest", required=True)
    ap.add_argument("--audit", default="", help="re11 audit_summary.json (required for --full-run)")
    ap.add_argument("--lock-file", default="", help="locked design JSON (required for --full-run)")
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--max-epochs", type=int, default=FULL_EPOCHS)
    ap.add_argument("--max-steps", type=int, default=0, help="stop after N optimizer steps (0 = off)")
    ap.add_argument("--limit-videos", type=int, default=0)
    ap.add_argument("--limit-val", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--model", default="riskprop", choices=["riskprop", "dummy"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--full-run", action="store_true")
    a = ap.parse_args(argv)
    mode = CONDITIONS[a.condition][0]
    if mode == "none":
        if a.lambda_prog not in (None, 0.0):
            ap.error(f"condition {a.condition} has no CPS term; do not pass --lambda-prog")
        a.lambda_prog = 0.0
    else:
        if a.lambda_prog is None or not a.lambda_prog > 0:
            ap.error(f"condition {a.condition} needs an explicit --lambda-prog > 0 (no default; "
                     f"PRE-ACT's 10 is not transferable)")
        if not a.sidecar:
            ap.error(f"condition {a.condition} needs --sidecar (actual endpoint tau + mask)")
    if a.horizon_sec <= 0 or a.alpha <= 0:
        ap.error("horizon and alpha must be > 0")
    return a


# ----------------------------------------------------------------------------
# Gates
# ----------------------------------------------------------------------------
def sidecar_binding_errors(a, sm, full_cache_digest=False):
    """The sidecar must describe THIS split manifest and THIS cache."""
    errs = []
    if sm.get("split_manifest_sha256") != sha256_file(a.split_manifest):
        errs.append("sidecar was built for a different split manifest")
    tr_path = os.path.join(a.cache_root, "train_index.json")
    if sm.get("train_index_sha256") != sha256_file(tr_path):
        errs.append("sidecar was built for a different cache train_index.json")
    if full_cache_digest:
        sys.path.insert(0, HERE)
        from re12_build_progress_sidecar import cache_digest
        man = json.load(open(a.split_manifest))
        tr = json.load(open(tr_path))
        if set(tr) != set(man["train_ids"]):
            errs.append("cache ids != manifest train ids (cannot digest)")
        else:
            print("verifying cache digest (hashing every cached .pt) ...")
            dig, _ = cache_digest(a.cache_root, tr, man["train_ids"])
            if sm.get("cache_digest") != dig or sm.get("cache_digest_ids") != len(man["train_ids"]):
                errs.append("cached .pt files changed since the sidecar was built (cache_digest mismatch)")
    return errs


def run_fingerprint(cfg, manifest):
    """Everything that must be identical for a resume to continue the same run."""
    keys = ["condition", "cps_mode", "use_ffr", "use_amc", "pairing_mode", "horizon_sec", "alpha",
            "lambda_prog", "seed", "batch_size", "lr", "momentum", "weight_decay", "model",
            "limit_videos", "run_name"]
    fp = {k: cfg[k] for k in keys}
    fp.update({k: manifest["inputs"][k] for k in ("split_manifest_sha256", "train_index_sha256",
                                                  "val_index_sha256", "sidecar_sha256", "lock_sha256")})
    return fp


def full_run_gate(a, sidecar_meta):
    """Return list of reasons to refuse a full run (empty = OK)."""
    why = []
    if a.model != "riskprop":
        why.append("--model must be riskprop")
    if a.max_steps or a.limit_videos or a.limit_val or a.max_epochs != FULL_EPOCHS or a.tag:
        why.append("limits/tag not allowed (max-steps, limit-videos, limit-val, max-epochs != 50, tag)")
    if not a.audit or not os.path.exists(a.audit):
        why.append("--audit missing")
    else:
        au = json.load(open(a.audit))
        if not au.get("pass") or au.get("limited") or not au.get("probed_videos") or not au.get("scanned_cache"):
            why.append("audit not passed / limited / videos not probed / cache not scanned")
    if not a.sidecar:
        why.append("sidecar missing (required for every condition in a full run: it binds the run "
                   "to the verified split/cache)")
    else:
        sm = sidecar_meta or {}
        if not sm.get("usable_for_full_run"):
            why.append("sidecar not complete/usable (re12 limited, fatal issues, pixel check or PTS)")
        if sm.get("pixel_verified", 0) < MIN_PIXEL_VERIFY or sm.get("pixel_identical") != sm.get("pixel_verified"):
            why.append(f"pixel check: {sm.get('pixel_identical')}/{sm.get('pixel_verified')} identical "
                       f"(need >= {MIN_PIXEL_VERIFY}, all identical)")
        why += sidecar_binding_errors(a, sm, full_cache_digest=True)
    if not a.lock_file or not os.path.exists(a.lock_file):
        why.append("--lock-file missing (design not locked)")
    else:
        lk = json.load(open(a.lock_file))
        if not lk.get("confirmed"):
            why.append("lock file not confirmed by the team")
        if lk.get("pairing_mode") != a.pairing_mode:
            why.append(f"pairing {a.pairing_mode} != locked {lk.get('pairing_mode')}")
        if a.condition not in lk.get("conditions", []):
            why.append(f"condition {a.condition} not in locked conditions")
        if a.seed not in lk.get("seeds", []):
            why.append(f"seed {a.seed} not in locked seeds")
        allowed_h = [lk.get("horizon_sec")] + list(lk.get("sensitivity_horizons", []))
        if a.horizon_sec not in allowed_h:
            why.append(f"horizon {a.horizon_sec} not locked ({allowed_h})")
        if a.horizon_sec != lk.get("horizon_sec") and a.condition not in lk.get("sensitivity_conditions", []):
            why.append(f"sensitivity horizon only for {lk.get('sensitivity_conditions', [])}")
        if lk.get("alpha") != a.alpha:
            why.append("alpha != locked")
        if CONDITIONS[a.condition][0] != "none" and lk.get("lambda_prog") != a.lambda_prog:
            why.append(f"lambda {a.lambda_prog} != locked {lk.get('lambda_prog')}")
    man = json.load(open(a.split_manifest))
    tr = json.load(open(os.path.join(a.cache_root, "train_index.json")))
    va = json.load(open(os.path.join(a.cache_5f, "val_index.json")))
    if set(tr) != set(man["train_ids"]):
        why.append("train cache ids != split manifest train_ids")
    if list(va) != man["val_ids"]:
        why.append("val cache ids != split manifest val_ids")
    if set(man["train_ids"]) & set(man["val_ids"]):
        why.append("train/val overlap")
    return why


# ----------------------------------------------------------------------------
# Data / model
# ----------------------------------------------------------------------------
def load_sidecar(path):
    if not path:
        return None, None
    with open(path) as f:
        sc = json.load(f)
    return sc["videos"], sc["meta"]


def make_train_dataset(cache_root, sidecar_videos, limit_videos):
    import torch
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "pipeline"))
    from cell33_riskprop_dataset import RiskPropTrainDataset

    class ProgressTrainDataset(RiskPropTrainDataset):
        """cell33 item + (cps_tau, cps_mask) looked up from the sidecar."""

        def __init__(self, root, side, ids=None):
            super().__init__(root)
            self.side = side
            if ids is not None:
                self.vid_ids = list(ids)

        def __getitem__(self, idx):
            seq, tau, target, dt = super().__getitem__(idx)
            vid = self.vid_ids[idx]
            n = seq.shape[0]
            if self.side is None:
                ctau = torch.full((n,), float("inf"))
                cmask = torch.zeros(n, dtype=torch.bool)
            else:
                if vid not in self.side:
                    raise KeyError(f"video {vid} missing from sidecar")
                s = self.side[vid]
                if int(s["target"]) != int(target.item()):
                    raise ValueError(f"{vid}: sidecar target != cache target")
                ctau = torch.tensor([float("inf") if v is None else float(v) for v in s["tau_actual"]])
                cmask = torch.tensor(s["cps_mask"], dtype=torch.bool)
                if len(ctau) != n:
                    raise ValueError(f"{vid}: sidecar has {len(ctau)} snippets, cache {n}")
            return seq, tau, target, dt, ctau, cmask

    ds = ProgressTrainDataset(cache_root, sidecar_videos)
    if limit_videos:
        if sidecar_videos is not None:
            sys.path.insert(0, HERE)
            from re12_build_progress_sidecar import choose_subset
            pool = [v for v in ds.vid_ids if v in sidecar_videos]
            ids = choose_subset(pool, {v: sidecar_videos[v]["target"] for v in pool}, limit_videos)
        else:
            ids = sorted(ds.vid_ids)[:limit_videos]
        ds.vid_ids = ids
    return ds


def build_model(kind):
    import torch.nn as nn
    if kind == "riskprop":
        sys.path.insert(0, os.path.join(os.path.dirname(HERE), "pipeline"))
        from cell31_model_riskprop import RiskPropModel
        return RiskPropModel()

    class DummyRiskProp(nn.Module):
        """CPU pipeline smoke only -- same I/O contract as RiskPropModel."""

        def __init__(self):
            super().__init__()
            self.conv = nn.Conv3d(3, 4, kernel_size=1)
            self.head = nn.Linear(4, 1)

        def forward_snippet(self, x):
            return self.head(self.conv(x).mean(dim=(2, 3, 4))).squeeze(-1)

        def forward(self, x):
            if x.dim() == 5:
                return self.forward_snippet(x)
            B, N = x.shape[:2]
            return self.forward_snippet(x.reshape(B * N, *x.shape[2:])).view(B, N)
    return DummyRiskProp()


def forward_losses(model, seq, targets, dt0, cps_tau, cps_mask, cfg):
    """One training forward. The model sees ONLY `seq`."""
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "pipeline"))
    sys.path.insert(0, HERE)
    from cell32_riskprop_loss import riskprop_loss
    from re13_progress_supervision import cps_loss
    logits = model(seq)
    base, parts = riskprop_loss(logits, targets, dt=dt0, use_ffr=cfg["use_ffr"],
                                use_amc=cfg["use_amc"], pairing_mode=cfg["pairing_mode"])
    cps, cst = cps_loss(logits, cps_tau, cps_mask, cfg["cps_mode"],
                        horizon=cfg["horizon_sec"], alpha=cfg["alpha"])
    total = base + cfg["lambda_prog"] * cps
    parts = dict(parts, **cst, base=float(base.detach()), total=float(total.detach()))
    return total, parts


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main(argv=None):
    a = parse_args(argv)
    side_videos, side_meta = load_sidecar(a.sidecar)
    if a.full_run:
        why = full_run_gate(a, side_meta)
        if why:
            print("[REFUSE full run]\n  - " + "\n  - ".join(why))
            sys.exit(3)
    elif side_meta is not None:
        errs = sidecar_binding_errors(a, side_meta)
        if errs:
            print("[REFUSE] sidecar does not match the data:\n  - " + "\n  - ".join(errs))
            sys.exit(3)
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Subset
    from sklearn.metrics import average_precision_score
    sys.path.insert(0, os.path.join(a.repo_dir, "pipeline"))
    from cell32_riskprop_loss import COLLISION_WEIGHT, NEG_WEIGHT
    from cell33_riskprop_dataset import RiskPropValDataset

    cps_mode, use_ffr, use_amc = CONDITIONS[a.condition]
    name = run_name(a.condition, a.pairing_mode, a.horizon_sec, a.alpha, a.lambda_prog, a.seed, a.tag)
    out_dir = os.path.join(a.output_root, "runs" if a.full_run else "dev", name)
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device)
    use_amp = device.type == "cuda"

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    train_ds = make_train_dataset(a.cache_root, side_videos if cps_mode != "none" or a.sidecar else None,
                                  a.limit_videos)
    val_ds = RiskPropValDataset(a.cache_5f, lead_time=1.0)
    if a.limit_val:
        val_ds = Subset(val_ds, list(range(min(a.limit_val, len(val_ds)))))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=a.num_workers,
                              pin_memory=use_amp, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=VAL_BATCH, shuffle=False, num_workers=a.num_workers,
                            pin_memory=use_amp)

    cfg = {"condition": a.condition, "cps_mode": cps_mode, "use_ffr": use_ffr, "use_amc": use_amc,
           "pairing_mode": a.pairing_mode, "horizon_sec": a.horizon_sec, "alpha": a.alpha,
           "lambda_prog": a.lambda_prog, "seed": a.seed, "epochs": a.max_epochs, "max_steps": a.max_steps,
           "batch_size": BATCH_SIZE, "lr": LR, "momentum": MOMENTUM, "weight_decay": WEIGHT_DECAY,
           "lr_decay_epochs": [20, 40], "checkpoint_rule_train": "best = lowest val BCE @1.0s lead; latest = last epoch",
           "cps_boundary": "tau<=0 excluded (PRE-ACT assigns 1); anchors k=0,k=11 excluded",
           "cps_loss": "smooth_l1(beta=1) on sigmoid(z), per-video mean then mean over videos",
           "full_run": a.full_run, "research_result": a.full_run, "model": a.model,
           "limit_videos": a.limit_videos, "limit_val": a.limit_val, "run_name": name}
    manifest = {"config": cfg, "git": git_info(a.repo_dir), "host": platform.node(),
                "python": platform.python_version(), "torch": torch.__version__,
                "device": torch.cuda.get_device_name(0) if use_amp else "cpu",
                "inputs": {"split_manifest_sha256": sha256_file(a.split_manifest),
                           "train_index_sha256": sha256_file(os.path.join(a.cache_root, "train_index.json")),
                           "val_index_sha256": sha256_file(os.path.join(a.cache_5f, "val_index.json")),
                           "sidecar_sha256": sha256_file(a.sidecar) if a.sidecar else None,
                           "audit": a.audit or None, "lock_file": a.lock_file or None,
                           "lock_sha256": sha256_file(a.lock_file) if a.lock_file else None},
                "n_train_videos": len(train_ds), "n_val_videos": len(val_ds),
                "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    print(json.dumps(manifest, indent=2, default=str))

    model = build_model(a.model).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda e: 0.01 if e >= 40 else (0.1 if e >= 20 else 1.0))
    scaler = torch.amp.GradScaler(enabled=use_amp)

    start_epoch, best_val, steps = 0, float("inf"), 0
    latest = os.path.join(out_dir, "latest.pth")
    if os.path.exists(latest):
        ck = torch.load(latest, map_location=device, weights_only=False)
        old_fp, new_fp = ck.get("fingerprint"), run_fingerprint(cfg, manifest)
        if old_fp != new_fp:
            diff = sorted(k for k in set(new_fp) | set(old_fp or {})
                          if (old_fp or {}).get(k) != new_fp.get(k))
            sys.exit(f"[REFUSE resume] {latest} was trained with a different config/data: {diff}. "
                     f"Use a new --output-root/--tag or move the old run away; nothing was overwritten.")
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, best_val, steps = ck["epoch"], ck.get("best_val_loss", best_val), ck.get("steps", 0)
        if "rng" in ck:
            torch.set_rng_state(ck["rng"]["torch"])
            np.random.set_state(ck["rng"]["numpy"])
            random.setstate(ck["rng"]["python"])
            if use_amp and ck["rng"].get("cuda") is not None:
                torch.cuda.set_rng_state_all(ck["rng"]["cuda"])
        print(f"Resumed {name} at epoch {start_epoch} (best_val_loss={best_val:.4f})")
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    log_path = os.path.join(out_dir, "train_log.csv")
    cols = ["epoch", "steps", "train_total", "train_base", "train_bce", "train_reg", "train_mono",
            "train_cps", "cps_videos", "cps_snippets", "val_loss", "val_AP_1.0s", "lr", "time_sec",
            "best_val_loss"]
    if start_epoch == 0 or not os.path.exists(log_path):
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(cols)

    stop = False
    t_all = time.time()
    epoch = start_epoch - 1
    for epoch in range(start_epoch, a.max_epochs):
        model.train()
        acc = {k: 0.0 for k in ("total", "base", "bce", "reg", "mono", "cps", "cps_videos", "cps_snippets")}
        nb = 0
        t0 = time.time()
        for seq, tau, targets, dt, ctau, cmask in train_loader:
            seq = seq.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            ctau, cmask = ctau.to(device), cmask.to(device)
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss, parts = forward_losses(model, seq, targets, float(dt[0]), ctau, cmask, cfg)
            if not torch.isfinite(loss):
                sys.exit(f"[FATAL] non-finite loss at epoch {epoch} step {steps}: {parts}")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            for k in acc:
                acc[k] += parts[k]
            nb += 1
            steps += 1
            if a.max_steps and steps >= a.max_steps:
                stop = True
                break

        model.eval()
        vloss, probs, tg = 0.0, [], []
        with torch.no_grad():
            for frames, _, vt in val_loader:
                frames, vt = frames.to(device), vt.to(device)
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    vl = model(frames)
                    w = torch.where(vt == 1, torch.full_like(vt, COLLISION_WEIGHT), torch.full_like(vt, NEG_WEIGHT))
                    vloss += F.binary_cross_entropy_with_logits(vl.float(), vt.float(), weight=w.float()).item()
                probs.append(torch.sigmoid(vl.float()).cpu())
                tg.append(vt.float().cpu())
        p, y = torch.cat(probs).numpy(), torch.cat(tg).numpy()
        vap = float(average_precision_score(y, p)) if 0 < y.sum() < len(y) else float("nan")
        avg_val = vloss / max(1, len(val_loader))
        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]
        m = {k: v / max(1, nb) for k, v in acc.items()}
        print(f"[{name}] epoch {epoch:2d} steps={steps} total={m['total']:.4f} base={m['base']:.4f} "
              f"(bce={m['bce']:.4f} reg={m['reg']:.4f} mono={m['mono']:.4f}) cps={m['cps']:.4f} "
              f"val_loss={avg_val:.4f} val_AP@1.0={vap:.4f} lr={lr_now:.5f} ({time.time()-t0:.0f}s)")

        ck = {"epoch": epoch + 1, "steps": steps, "model": model.state_dict(),
              "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
              "scaler": scaler.state_dict(), "best_val_loss": best_val, "seed": a.seed,
              "config": cfg, "manifest": manifest, "fingerprint": run_fingerprint(cfg, manifest),
              "rng": {"torch": torch.get_rng_state(), "numpy": np.random.get_state(),
                      "python": random.getstate(),
                      "cuda": torch.cuda.get_rng_state_all() if use_amp else None}}
        if avg_val < best_val:
            best_val = avg_val
            ck["best_val_loss"] = best_val
            torch.save(ck, os.path.join(out_dir, "best.pth"))
            print(f"  * best.pth (val_loss={best_val:.4f})")
        torch.save(ck, latest)
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, steps, f"{m['total']:.6f}", f"{m['base']:.6f}", f"{m['bce']:.6f}",
                                    f"{m['reg']:.6f}", f"{m['mono']:.6f}", f"{m['cps']:.6f}",
                                    f"{m['cps_videos']:.2f}", f"{m['cps_snippets']:.2f}", f"{avg_val:.6f}",
                                    f"{vap:.6f}", f"{lr_now:.6f}", f"{time.time()-t0:.1f}", f"{best_val:.6f}"])
        if stop:
            print(f"[{name}] --max-steps {a.max_steps} reached -> stop (dev run, not a result)")
            break

    manifest["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    manifest["train_hours"] = (time.time() - t_all) / 3600
    manifest["completed_epochs"] = epoch + 1
    manifest["best_val_loss"] = best_val
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"[{name}] done -> {out_dir}")


if __name__ == "__main__":
    main()
