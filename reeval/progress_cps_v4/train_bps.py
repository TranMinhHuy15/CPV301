"""
train_bps (v4) -- Train Progressive RiskProp conditions (RQ3). riskprop_model / riskprop_loss_ffr_amc /
riskprop_dataset are imported, never edited.

Conditions (same backbone, cache, split, 12 snippets, transforms, optimizer,
LR schedule, epochs, batch, seeds, checkpoint rule, pairing):
    B   L_BCE + 0.5 L_FFR + 0.5 L_AMC                 (matched RiskProp baseline)
    P   B + lambda * L_CPS(continuous target)         (PRE-ACT-inspired)
    S   B + lambda * L_CPS(binary 1[0<tau<H])         (dense-binary control)
    F / FP  optional AMC-off variants (not part of main v4)
Base loss = riskprop_loss_ffr_amc.riskprop_loss unchanged. CPS = supervision_targets.cps_loss on sigmoid(z)
in fp32; it consumes no RNG, so for one seed B/P/S share initialisation, batch
order and AMC pair draws (verified by the init digest stored per run).
The model only receives the RGB tensor `seq`.

v4 changes
  * fingerprint = config + code digest (common.SOURCE_FILES) + init
    digest (model state right after construction, i.e. pretrained weights +
    seeded head) + split/index/val-cache/sidecar/audit/lock hashes (+ train
    cache content digest on a full run). Resume, completion check and
    eval_val all compare it; any difference -> refuse.
  * checkpoints are loaded on CPU and RNG states restored with .cpu()
    (a CUDA map_location broke torch.set_rng_state); resume is at epoch
    boundaries only (no bitwise mid-epoch continuation).
  * checkpoints are written atomically (tmp -> fsync -> reload -> replace);
    COMPLETE.json is written only after the final checkpoint re-loads, and
    holds the fingerprint + checkpoint hashes. `--check-complete` exits 0 only
    when that marker matches the current fingerprint (used by the run script).
  * --full-run requires CUDA, the complete audit, the usable sidecar (pixel
    check, PTS, no cache-only zeros, cache digest re-verified), and a LOCK whose
    design and fingerprints (code, sidecar, audit, val cache) equal the current
    ones. Budget fixed at 50 epochs.
  * logs per epoch: AMP skipped optimizer steps, CPS coverage, and (first batch
    of each epoch, diagnostic only) gradient norms of the base and CPS terms at
    the classification head.
"""
import argparse
import csv
import json
import os
import random
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as pc  # noqa: E402

CONDITIONS = {   # name -> (cps mode, use_ffr, use_amc)
    "B": ("none", True, True),
    "P": ("continuous", True, True),
    "S": ("binary", True, True),
    "F": ("none", True, False),
    "FP": ("continuous", True, False),
}
FULL_EPOCHS = 50
MIN_PIXEL_VERIFY = 24          # = build_sidecar.MIN_PIXEL_VERIFY
BATCH_SIZE, VAL_BATCH = 2, 8
LR, MOMENTUM, WEIGHT_DECAY = 0.01, 0.9, 1e-4          # = riskprop_train / train_tau_sweep
sha256_file = pc.sha256_file
git_info = pc.git_info


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
    ap.add_argument("--repo-dir", default=os.path.dirname(os.path.dirname(HERE)))
    ap.add_argument("--cache-root", required=True, help="nexar_cache_riskprop (riskprop_precache_snippet_sequence)")
    ap.add_argument("--cache-5f", required=True, help="nexar_cache_5f (val windows, precache_val)")
    ap.add_argument("--sidecar", default="", help="build_sidecar sidecar.json (required for CPS conditions)")
    ap.add_argument("--split-manifest", required=True)
    ap.add_argument("--audit", default="", help="audit_data audit_summary.json (required for --full-run)")
    ap.add_argument("--lock-file", default="", help="lock_config LOCK.json (required for --full-run)")
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
    ap.add_argument("--stop-after-epochs", type=int, default=0,
                    help="DEV ONLY: end this invocation after N epochs without COMPLETE.json "
                         "(simulated interruption for the resume test)")
    ap.add_argument("--check-complete", action="store_true",
                    help="exit 0 iff COMPLETE.json exists and matches the current fingerprint; no training")
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
    if a.full_run and a.stop_after_epochs:
        ap.error("--stop-after-epochs is dev only")
    return a


# ----------------------------------------------------------------------------
# Gates and fingerprints
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
        from build_sidecar import cache_digest
        man = json.load(open(a.split_manifest))
        tr = json.load(open(tr_path))
        if set(tr) != set(man["train_ids"]):
            errs.append("cache ids != manifest train ids (cannot digest)")
        else:
            print("verifying train cache content digest (hashing every cached .pt) ...")
            dig, _ = cache_digest(a.cache_root, tr, man["train_ids"])
            if sm.get("cache_digest") != dig or sm.get("cache_digest_ids") != len(man["train_ids"]):
                errs.append("cached .pt files changed since the sidecar was built (cache_digest mismatch)")
    return errs


def run_fingerprint(cfg, manifest):
    """Everything that must be identical for a resume / completion / evaluation
    to refer to the same run."""
    keys = ["condition", "cps_mode", "use_ffr", "use_amc", "pairing_mode", "horizon_sec", "alpha",
            "lambda_prog", "seed", "batch_size", "lr", "momentum", "weight_decay", "model",
            "limit_videos", "run_name", "epochs_budget", "full_run"]
    fp = {k: cfg.get(k) for k in keys}
    fp.update({k: manifest["inputs"].get(k) for k in (
        "split_manifest_sha256", "train_index_sha256", "val_index_sha256", "val_cache_digest",
        "train_cache_digest", "sidecar_sha256", "audit_sha256", "lock_sha256")})
    fp["code_digest"] = manifest.get("code", {}).get("digest")
    fp["init_state_sha256"] = manifest.get("init_state_sha256")
    return fp


def lock_errors(a, lk, current):
    """Design + fingerprint comparison against LOCK.json (v4 schema)."""
    why = []
    if not lk.get("confirmed"):
        why.append("lock file not confirmed by the team")
    missing = [k for k in ("primary_contrast", "primary_metric", "ni_metric", "ni_margin", "ps_metric",
                           "success_rule") if lk.get(k) in (None, "")]
    if missing:
        why.append(f"lock file lacks the pre-registered decision fields {missing}")
    if lk.get("pairing_mode") != a.pairing_mode:
        why.append(f"pairing {a.pairing_mode} != locked {lk.get('pairing_mode')}")
    if a.condition not in lk.get("conditions", []) and not (
            a.condition in lk.get("sensitivity_conditions", []) and a.horizon_sec in lk.get("sensitivity_horizons", [])):
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
    if lk.get("epochs", FULL_EPOCHS) != FULL_EPOCHS:
        why.append("locked epoch budget != 50")
    fps = lk.get("fingerprints")
    if not fps:
        why.append("LOCK has no fingerprints (create it with lock_config.py)")
        return why
    checks = [("code", (fps.get("code") or {}).get("digest"), current.get("code_digest")),
              ("sidecar", fps.get("sidecar_sha256"), current.get("sidecar_sha256")),
              ("audit", fps.get("audit_sha256"), current.get("audit_sha256")),
              ("val cache", fps.get("val_cache_digest"), current.get("val_cache_digest")),
              ("split manifest", fps.get("split_manifest_sha256"), current.get("split_manifest_sha256")),
              ("train index", fps.get("train_index_sha256"), current.get("train_index_sha256")),
              ("train cache", fps.get("train_cache_digest"), current.get("train_cache_digest"))]
    for label, locked, now in checks:
        if locked != now:
            why.append(f"{label} differs from LOCK (locked {str(locked)[:12]}, now {str(now)[:12]})")
    return why


def full_run_gate(a, sidecar_meta, current=None):
    """Return list of reasons to refuse a full run (empty = OK). `current` =
    current input hashes (manifest['inputs'] + code digest) for the LOCK check."""
    why = []
    if a.model != "riskprop":
        why.append("--model must be riskprop")
    if a.max_steps or a.limit_videos or a.limit_val or a.max_epochs != FULL_EPOCHS or a.tag:
        why.append("limits/tag not allowed (max-steps, limit-videos, limit-val, max-epochs != 50, tag)")
    if not a.audit or not os.path.exists(a.audit):
        why.append("--audit missing")
    else:
        au = json.load(open(a.audit))
        if not au.get("pass") or au.get("limited") or not au.get("complete", False):
            why.append("audit not passed / limited / not complete (audit_data v4 with --prepare-report)")
    if not a.sidecar:
        why.append("sidecar missing (required for every condition in a full run: it binds the run "
                   "to the verified split/cache)")
    else:
        sm = sidecar_meta or {}
        if not sm.get("usable_for_full_run"):
            why.append("sidecar not complete/usable (build_sidecar limited, fatal issues, pixel check, PTS, cache zeros)")
        if sm.get("pixel_verified", 0) < MIN_PIXEL_VERIFY or sm.get("pixel_identical") != sm.get("pixel_verified"):
            why.append(f"pixel check: {sm.get('pixel_identical')}/{sm.get('pixel_verified')} identical "
                       f"(need >= {MIN_PIXEL_VERIFY}, all identical)")
        if a.audit and os.path.exists(a.audit) and sm.get("audit_sha256") not in (None, sha256_file(a.audit)):
            why.append("sidecar was built from a different audit file")
        why += sidecar_binding_errors(a, sm, full_cache_digest=True)
    if not a.lock_file or not os.path.exists(a.lock_file):
        why.append("--lock-file missing (design not locked; run lock_config.py)")
    else:
        why += lock_errors(a, json.load(open(a.lock_file)), current or {})
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
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "pipeline"))
    from riskprop_dataset import RiskPropTrainDataset

    class ProgressTrainDataset(RiskPropTrainDataset):
        """riskprop_dataset item + (cps_tau, cps_mask) looked up from the sidecar."""

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
            from build_sidecar import choose_subset
            pool = [v for v in ds.vid_ids if v in sidecar_videos]
            ids = choose_subset(pool, {v: sidecar_videos[v]["target"] for v in pool}, limit_videos)
        else:
            ids = sorted(ds.vid_ids)[:limit_videos]
        ds.vid_ids = ids
    return ds


def build_model(kind):
    import torch.nn as nn
    if kind == "riskprop":
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "pipeline"))
        from riskprop_model import RiskPropModel
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


def forward_losses(model, seq, targets, dt0, cps_tau, cps_mask, cfg, return_terms=False):
    """One training forward. The model sees ONLY `seq`."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "pipeline"))
    from riskprop_loss_ffr_amc import riskprop_loss
    from supervision_targets import cps_loss
    logits = model(seq)
    base, parts = riskprop_loss(logits, targets, dt=dt0, use_ffr=cfg["use_ffr"],
                                use_amc=cfg["use_amc"], pairing_mode=cfg["pairing_mode"])
    cps, cst = cps_loss(logits, cps_tau, cps_mask, cfg["cps_mode"],
                        horizon=cfg["horizon_sec"], alpha=cfg["alpha"])
    total = base + cfg["lambda_prog"] * cps
    parts = dict(parts, **cst, base=float(base.detach()), total=float(total.detach()))
    if return_terms:
        return total, parts, base, cps
    return total, parts


def head_grad_norms(model, base, cps, lam):
    """Diagnostic: ||d base / d head|| and ||d lam*cps / d head|| (no update)."""
    import torch
    head = [p for p in model.head.parameters() if p.requires_grad]
    gb = torch.autograd.grad(base, head, retain_graph=True, allow_unused=True)
    out = {"gradnorm_base_head": float(torch.sqrt(sum((g.float() ** 2).sum() for g in gb if g is not None)))}
    if lam and cps.requires_grad:
        gc = torch.autograd.grad(lam * cps, head, retain_graph=True, allow_unused=True)
        out["gradnorm_cps_head"] = float(torch.sqrt(sum((g.float() ** 2).sum() for g in gc if g is not None)))
    else:
        out["gradnorm_cps_head"] = 0.0
    return out


def build_inputs(a, side_meta, full):
    """Input hashes recorded in the manifest / fingerprint."""
    inp = {"split_manifest_sha256": sha256_file(a.split_manifest),
           "train_index_sha256": sha256_file(os.path.join(a.cache_root, "train_index.json")),
           "val_index_sha256": sha256_file(os.path.join(a.cache_5f, "val_index.json")),
           "sidecar_sha256": sha256_file(a.sidecar) if a.sidecar else None,
           "audit_sha256": sha256_file(a.audit) if a.audit and os.path.exists(a.audit) else None,
           "lock_sha256": sha256_file(a.lock_file) if a.lock_file and os.path.exists(a.lock_file) else None,
           "audit": a.audit or None, "lock_file": a.lock_file or None,
           # full runs bind the content of the caches; dev runs only their index files
           "val_cache_digest": pc.val_cache_digest(a.cache_5f) if full else None,
           "train_cache_digest": (side_meta or {}).get("cache_digest") if full else None}
    return inp


def complete_ok(out_dir, fp):
    """COMPLETE.json present, same fingerprint, checkpoints present with the recorded hashes."""
    p = os.path.join(out_dir, "COMPLETE.json")
    if not os.path.exists(p):
        return False, "no COMPLETE.json"
    c = json.load(open(p))
    if c.get("fingerprint") != fp:
        diff = sorted(k for k in set(fp) | set(c.get("fingerprint") or {})
                      if (c.get("fingerprint") or {}).get(k) != fp.get(k))
        return False, f"fingerprint differs: {diff}"
    for kind in ("best", "latest"):
        f = os.path.join(out_dir, f"{kind}.pth")
        if not os.path.exists(f) or sha256_file(f) != c.get("checkpoints", {}).get(kind):
            return False, f"{kind}.pth missing or changed"
    return True, "complete"


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main(argv=None):
    a = parse_args(argv)
    side_videos, side_meta = load_sidecar(a.sidecar)
    import torch
    full = a.full_run
    if side_meta is not None:
        errs = sidecar_binding_errors(a, side_meta)
        if errs:
            print("[REFUSE] sidecar does not match the data:\n  - " + "\n  - ".join(errs))
            sys.exit(3)

    cps_mode, use_ffr, use_amc = CONDITIONS[a.condition]
    name = run_name(a.condition, a.pairing_mode, a.horizon_sec, a.alpha, a.lambda_prog, a.seed, a.tag)
    out_dir = os.path.join(a.output_root, "runs" if full else "dev", name)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device)
    if full and device.type != "cuda":
        print("[REFUSE full run] CUDA is required for a full run (torch.cuda.is_available() is False)")
        sys.exit(3)
    use_amp = device.type == "cuda"

    cfg = {"condition": a.condition, "cps_mode": cps_mode, "use_ffr": use_ffr, "use_amc": use_amc,
           "pairing_mode": a.pairing_mode, "horizon_sec": a.horizon_sec, "alpha": a.alpha,
           "lambda_prog": a.lambda_prog, "seed": a.seed, "epochs": a.max_epochs, "max_steps": a.max_steps,
           "epochs_budget": FULL_EPOCHS if full else None,
           "batch_size": BATCH_SIZE, "lr": LR, "momentum": MOMENTUM, "weight_decay": WEIGHT_DECAY,
           "lr_decay_epochs": [20, 40],
           "checkpoint_rule_train": "best = lowest val BCE @1.0s lead; latest = last epoch",
           "cps_boundary": "tau<=0 excluded (PRE-ACT assigns 1); anchors k=0,k=11 excluded",
           "cps_loss": "smooth_l1(beta=1) on sigmoid(z) fp32, per-video mean then mean over videos",
           "full_run": full, "research_result": full and not pc.SYNTHETIC, "synthetic": pc.SYNTHETIC,
           "model": a.model,
           "limit_videos": a.limit_videos, "limit_val": a.limit_val, "run_name": name,
           "resume_policy": "epoch boundary only; RNG restored on CPU; no bitwise mid-epoch continuation"}
    inputs = build_inputs(a, side_meta, full)

    # model is built right after seeding -> init digest identical for B/P/S of one seed
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    model = build_model(a.model)
    init_sha = pc.model_state_digest(model)
    manifest = {"version": pc.V4_VERSION, "config": cfg, "git": git_info(a.repo_dir), "env": pc.env_identity(),
                "code": pc.source_identity(a.repo_dir), "init_state_sha256": init_sha, "inputs": inputs,
                "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    fp = run_fingerprint(cfg, manifest)

    if a.check_complete:
        ok, why = complete_ok(out_dir, fp)
        print(f"[check-complete] {name}: {'COMPLETE' if ok else 'NOT complete'} ({why})")
        sys.exit(0 if ok else 1)
    if full:
        why = full_run_gate(a, side_meta, current=dict(inputs, code_digest=manifest["code"]["digest"]))
        if why:
            print("[REFUSE full run]\n  - " + "\n  - ".join(why))
            sys.exit(3)

    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Subset
    from sklearn.metrics import average_precision_score
    sys.path.insert(0, os.path.join(a.repo_dir, "pipeline"))
    from riskprop_loss_ffr_amc import COLLISION_WEIGHT, NEG_WEIGHT
    from riskprop_dataset import RiskPropValDataset

    os.makedirs(out_dir, exist_ok=True)
    train_ds = make_train_dataset(a.cache_root, side_videos if (cps_mode != "none" or a.sidecar) else None,
                                  a.limit_videos)
    val_ds = RiskPropValDataset(a.cache_5f, lead_time=1.0)
    if a.limit_val:
        val_ds = Subset(val_ds, list(range(min(a.limit_val, len(val_ds)))))
    g = torch.Generator()
    g.manual_seed(a.seed)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=a.num_workers,
                              pin_memory=use_amp, drop_last=True, generator=g)
    val_loader = DataLoader(val_ds, batch_size=VAL_BATCH, shuffle=False, num_workers=a.num_workers,
                            pin_memory=use_amp)
    manifest.update(n_train_videos=len(train_ds), n_val_videos=len(val_ds))
    print(json.dumps({k: manifest[k] for k in ("version", "config", "init_state_sha256", "inputs")},
                     indent=2, default=str))

    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda e: 0.01 if e >= 40 else (0.1 if e >= 20 else 1.0))
    scaler = torch.amp.GradScaler(enabled=use_amp)

    start_epoch, best_val, steps = 0, float("inf"), 0
    latest = os.path.join(out_dir, "latest.pth")
    if os.path.exists(os.path.join(out_dir, "COMPLETE.json")):
        ok, why = complete_ok(out_dir, fp)
        if ok:
            print(f"[{name}] already COMPLETE with the same fingerprint -> nothing to do")
            return
        sys.exit(f"[REFUSE] {out_dir} has a COMPLETE.json that does not match ({why}); "
                 f"use a new --output-root or move the old run away")
    if os.path.exists(latest):
        ck = torch.load(latest, map_location="cpu", weights_only=False)
        old_fp = ck.get("fingerprint")
        if old_fp != fp:
            diff = sorted(k for k in set(fp) | set(old_fp or {}) if (old_fp or {}).get(k) != fp.get(k))
            sys.exit(f"[REFUSE resume] {latest} was trained with a different config/data/code: {diff}. "
                     f"Use a new --output-root/--tag or move the old run away; nothing was overwritten.")
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])     # moves state to the parameters' device
        scheduler.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, best_val, steps = ck["epoch"], ck.get("best_val_loss", best_val), ck.get("steps", 0)
        if "rng" in ck:
            pc.restore_rng(ck["rng"], use_amp)
            if ck["rng"].get("loader") is not None:
                g.set_state(ck["rng"]["loader"].cpu())
        print(f"Resumed {name} at epoch {start_epoch} (best_val_loss={best_val:.4f})")
    pc.write_json_atomic(os.path.join(out_dir, "config.json"), manifest)

    log_path = os.path.join(out_dir, "train_log.csv")
    cols = ["epoch", "steps", "train_total", "train_base", "train_bce", "train_reg", "train_mono",
            "train_cps", "cps_videos", "cps_snippets", "amp_skipped_steps", "gradnorm_base_head",
            "gradnorm_cps_head", "val_loss", "val_AP_1.0s", "lr", "time_sec", "best_val_loss",
            "peak_mem_gb"]
    if start_epoch == 0 or not os.path.exists(log_path):
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(cols)

    stop = False
    t_all = time.time()
    epoch = start_epoch - 1
    for epoch in range(start_epoch, a.max_epochs):
        model.train()
        acc = {k: 0.0 for k in ("total", "base", "bce", "reg", "mono", "cps", "cps_videos", "cps_snippets")}
        nb, skipped, gdiag = 0, 0, {"gradnorm_base_head": float("nan"), "gradnorm_cps_head": float("nan")}
        if use_amp:
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        for seq, tau, targets, dt, ctau, cmask in train_loader:
            seq = seq.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            ctau, cmask = ctau.to(device), cmask.to(device)
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss, parts, base_t, cps_t = forward_losses(model, seq, targets, float(dt[0]), ctau, cmask,
                                                            cfg, return_terms=True)
            if not torch.isfinite(loss):
                print(f"[FATAL] non-finite loss at epoch {epoch} step {steps}: {parts}")
                sys.exit(4)
            if nb == 0:
                gdiag = head_grad_norms(model, base_t, cps_t, cfg["lambda_prog"])
            scale_before = scaler.get_scale() if use_amp else 1.0
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if use_amp and scaler.get_scale() < scale_before:
                skipped += 1          # inf/nan gradients -> optimizer step skipped by GradScaler
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
        if not np.isfinite(avg_val):
            print(f"[FATAL] non-finite validation loss at epoch {epoch}")
            sys.exit(4)
        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]
        m = {k: v / max(1, nb) for k, v in acc.items()}
        peak = torch.cuda.max_memory_allocated() / 2**30 if use_amp else 0.0
        print(f"[{name}] epoch {epoch:2d} steps={steps} total={m['total']:.4f} base={m['base']:.4f} "
              f"(bce={m['bce']:.4f} reg={m['reg']:.4f} mono={m['mono']:.4f}) cps={m['cps']:.4f} "
              f"amp_skipped={skipped} g_base={gdiag['gradnorm_base_head']:.3g} g_cps={gdiag['gradnorm_cps_head']:.3g} "
              f"val_loss={avg_val:.4f} val_AP@1.0={vap:.4f} lr={lr_now:.5f} ({time.time()-t0:.0f}s)")

        ck = {"epoch": epoch + 1, "steps": steps, "model": model.state_dict(),
              "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
              "scaler": scaler.state_dict(), "best_val_loss": best_val, "seed": a.seed,
              "config": cfg, "manifest": manifest, "fingerprint": fp,
              "rng": {"torch": torch.get_rng_state(), "numpy": np.random.get_state(),
                      "python": random.getstate(), "loader": g.get_state(),
                      "cuda": torch.cuda.get_rng_state_all() if use_amp else None}}
        if avg_val < best_val:
            best_val = avg_val
            ck["best_val_loss"] = best_val
            pc.torch_save_atomic(ck, os.path.join(out_dir, "best.pth"))
            print(f"  * best.pth (val_loss={best_val:.4f})")
        pc.torch_save_atomic(ck, latest)
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, steps, f"{m['total']:.6f}", f"{m['base']:.6f}", f"{m['bce']:.6f}",
                                    f"{m['reg']:.6f}", f"{m['mono']:.6f}", f"{m['cps']:.6f}",
                                    f"{m['cps_videos']:.2f}", f"{m['cps_snippets']:.2f}", skipped,
                                    f"{gdiag['gradnorm_base_head']:.6g}", f"{gdiag['gradnorm_cps_head']:.6g}",
                                    f"{avg_val:.6f}", f"{vap:.6f}", f"{lr_now:.6f}", f"{time.time()-t0:.1f}",
                                    f"{best_val:.6f}", f"{peak:.2f}"])
        if stop:
            print(f"[{name}] --max-steps {a.max_steps} reached -> stop (dev run, not a result)")
            break
        if a.stop_after_epochs and epoch + 1 - start_epoch >= a.stop_after_epochs and epoch + 1 < a.max_epochs:
            print(f"[{name}] --stop-after-epochs {a.stop_after_epochs}: simulated interruption at epoch {epoch + 1}")
            return

    finished_budget = (epoch + 1 == a.max_epochs) and not stop
    manifest.update(finished=time.strftime("%Y-%m-%d %H:%M:%S"), train_hours=(time.time() - t_all) / 3600,
                    completed_epochs=epoch + 1, best_val_loss=best_val)
    pc.write_json_atomic(os.path.join(out_dir, "config.json"), manifest)
    if finished_budget and os.path.exists(latest) and os.path.exists(os.path.join(out_dir, "best.pth")):
        chk = {}
        for kind in ("best", "latest"):
            f = os.path.join(out_dir, f"{kind}.pth")
            torch.load(f, map_location="cpu", weights_only=False)      # must re-load
            chk[kind] = sha256_file(f)
        pc.write_json_atomic(os.path.join(out_dir, "COMPLETE.json"),
                             {"run": name, "fingerprint": fp, "checkpoints": chk, "epochs": epoch + 1,
                              "steps": steps, "full_run": full, "finished": manifest["finished"]})
        print(f"[{name}] COMPLETE -> {out_dir}")
    else:
        print(f"[{name}] stopped before the budget (epoch {epoch + 1}/{a.max_epochs}); no COMPLETE.json")


if __name__ == "__main__":
    main()
