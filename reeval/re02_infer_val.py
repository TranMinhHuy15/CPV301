"""
re02 -- Run the 24 saved checkpoints on the internal validation cache and
save the RAW per-video predictions (no metric computed here).

12 runs x {best, latest}:
    TOP             seeds 42/43/44   best_cached_seedS.pth / latest_cached_seedS.pth
    AdaLEA          seeds 42/43/44   best_adalea_seedS.pth / latest_adalea_seedS.pth
    RiskProp random seeds 42/43/44   best_riskprop_random_seedS.pth / latest_...
    RiskProp fixed  seeds 42/43/44   best_riskprop_fixed_seedS.pth  / latest_...

For every run/checkpoint one .npz is written to --out-dir containing
    vid_ids  (N,)            video ids, same order as val_index.json
    targets  (N,)            0/1 label
    taus     (3, N)          time-to-event of the cached window per lead
    leads    (3,)            [0.5, 1.0, 1.5]
    scores   (3, N)          sigmoid score           (AdaLEA / RiskProp)
             (3, N, 20)      sigmoid of all 20 heads (TOP -> every head
                             rule can be evaluated offline in re03)
plus a matching .json with checkpoint metadata.

Preprocessing is the original one: the tensors come from
cell22_adalea_dataset.AdaLEAValDataset (same _to_tensor as cell11b, which
TOP's eval used), fp32, model.eval(), batch size 8 -- identical to
cell16 / cell24 / cell35.

Usage:
    python reeval/re02_infer_val.py --ckpt-dir /workspace/ckpts
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

LEADS = [0.5, 1.0, 1.5]
SEEDS = [42, 43, 44]
KINDS = ["best", "latest"]


def run_specs():
    """(run_name, model_key, ckpt filename template)."""
    specs = []
    for s in SEEDS:
        specs.append((f"top_seed{s}", "top", "{kind}_cached_seed%d.pth" % s))
        specs.append((f"adalea_seed{s}", "adalea", "{kind}_adalea_seed%d.pth" % s))
        specs.append((f"riskprop_random_seed{s}", "riskprop",
                      "{kind}_riskprop_random_seed%d.pth" % s))
        specs.append((f"riskprop_fixed_seed{s}", "riskprop",
                      "{kind}_riskprop_fixed_seed%d.pth" % s))
    return specs


def build_model(model_key):
    if model_key == "top":
        from cell13_model import TOPModel
        return TOPModel()
    if model_key == "adalea":
        from cell20_model_adalea import AdaLEAModel
        return AdaLEAModel()
    if model_key == "riskprop":
        from cell31_model_riskprop import RiskPropModel
        return RiskPropModel()
    raise ValueError(model_key)


def load_val_tensors(cache_dir):
    """Load every cached val window once, through the ORIGINAL dataset class."""
    from cell22_adalea_dataset import AdaLEAValDataset
    data = {}
    vid_ids_ref = None
    for lead in LEADS:
        ds = AdaLEAValDataset(cache_dir, lead_time=lead)
        if vid_ids_ref is None:
            vid_ids_ref = list(ds.vid_ids)
        assert list(ds.vid_ids) == vid_ids_ref, "val_index order changed?"
        xs, taus, ys = [], [], []
        for i in range(len(ds)):
            x, tau, y = ds[i]
            xs.append(x)
            taus.append(float(tau))
            ys.append(float(y))
        data[lead] = (xs, np.array(taus), np.array(ys))
        print(f"  loaded lead {lead}s: {len(xs)} windows")
    targets = data[LEADS[0]][2]
    for lead in LEADS[1:]:
        assert np.array_equal(data[lead][2], targets)
    return vid_ids_ref, targets, data


@torch.no_grad()
def predict(model, xs, device, batch_size):
    out = []
    for i in range(0, len(xs), batch_size):
        batch = torch.stack(xs[i:i + batch_size]).to(device)
        out.append(torch.sigmoid(model(batch)).float().cpu())
    return torch.cat(out).numpy()


def sha256_head(path, nbytes=8 << 20):
    """Hash of the first 8 MB -- enough to tell checkpoints apart quickly."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(nbytes))
    return h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-dir", default="/workspace/CPV301",
                    help="folder containing cell13/cell20/cell22/cell31")
    ap.add_argument("--cache-dir", default="/workspace/CPV301/data/nexar_cache_5f")
    ap.add_argument("--ckpt-dir", required=True,
                    help="folder with the 24 .pth files (flat)")
    ap.add_argument("--out-dir", default="/workspace/CPV301/reeval_out/preds_val")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--only", default="",
                    help="comma list of run names to (re)do, e.g. top_seed42")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, args.repo_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False
    print(f"device={device}")

    print("Loading validation windows...")
    vid_ids, targets, data = load_val_tensors(args.cache_dir)
    taus = np.stack([data[l][1] for l in LEADS])
    print(f"  {len(vid_ids)} videos | positives={int(targets.sum())}")

    only = {s.strip() for s in args.only.split(",") if s.strip()}
    models = {}
    missing, done = [], 0
    t_all = time.time()
    for run_name, model_key, tmpl in run_specs():
        if only and run_name not in only:
            continue
        for kind in KINDS:
            ckpt_path = os.path.join(args.ckpt_dir, tmpl.format(kind=kind))
            out_npz = os.path.join(args.out_dir, f"{run_name}_{kind}.npz")
            if args.skip_existing and os.path.exists(out_npz):
                continue
            if not os.path.exists(ckpt_path):
                print(f"  [MISSING] {ckpt_path}")
                missing.append(ckpt_path)
                continue
            if model_key not in models:
                models[model_key] = build_model(model_key).to(device).eval()
            model = models[model_key]
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"])
            model.eval()

            t0 = time.time()
            scores = np.stack([predict(model, data[l][0], device, args.batch_size)
                               for l in LEADS])
            np.savez_compressed(out_npz, vid_ids=np.array(vid_ids),
                                targets=targets, taus=taus,
                                leads=np.array(LEADS), scores=scores)
            meta = {"run": run_name, "model": model_key, "kind": kind,
                    "ckpt_file": os.path.basename(ckpt_path),
                    "ckpt_sha256_8MB": sha256_head(ckpt_path),
                    "ckpt_epoch": ckpt.get("epoch"),
                    "ckpt_best_val_loss": ckpt.get("best_val_loss"),
                    "scores_shape": list(scores.shape)}
            with open(out_npz.replace(".npz", ".json"), "w") as f:
                json.dump(meta, f, indent=2, default=float)
            done += 1
            print(f"  [OK] {run_name:24s} {kind:6s} epoch={meta['ckpt_epoch']} "
                  f"shape={tuple(scores.shape)} ({time.time()-t0:.0f}s)")

    print(f"\nDone: {done} prediction files in {(time.time()-t_all)/60:.1f} min "
          f"-> {args.out_dir}")
    if missing:
        print(f"WARNING: {len(missing)} checkpoint(s) missing:")
        for m in missing:
            print("   ", m)


if __name__ == "__main__":
    main()
