"""
RQ1b -- score RiskProp-full checkpoints with the project's evaluation protocol.

Stage "val"  (run after training all seeds):
  * loads the project's validation windows (the re01 cache: 300 videos x 3 lead
    times, one causal 5-frame window each, ImageNet-normalised exactly like the
    authors' data_preprocessor),
  * scores EVERY epoch checkpoint of every seed,
  * computes the validation loss (plain BCE on the 1.0 s windows, pos_weight 1 --
    the BCE that RiskProp-full itself trains with) and the 3-horizon mAP,
  * applies the LOCKED project rule: candidates = {epoch with the lowest
    validation loss ("best"), final epoch ("latest")}; keep the one with the
    higher mean validation AP over 0.5/1.0/1.5 s; tie -> best,
  * also records the authors' own choice (best_mAUC@_epoch_X.pth) for the
    appendix (validation only),
  * writes predictions in the re02 format so analyze_rq1b.py can reuse re03.

Stage "test" (run ONCE, after stage val has written rq1b_chosen_checkpoints.json):
  * scores the chosen checkpoint of each seed on the official test clips (causal
    5-frame window at the end of each clip, as re04) and writes
    submission_riskprop_full_seed{S}.csv. Refuses to overwrite existing
    submissions unless --force (the test set is used once).

Usage (from the RiskProp repo root, PYTHONPATH containing riskprop_full/):
    python /workspace/CPV301/riskprop_full/infer_rq1b.py --stage val
    python /workspace/CPV301/riskprop_full/infer_rq1b.py --stage test
"""
import argparse
import glob
import hashlib
import importlib
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

LEADS = [0.5, 1.0, 1.5]
SEEDS = [42, 43, 44]


def build_model(config, device):
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmaction.registry import MODELS
    from mmaction.utils import register_all_modules
    register_all_modules()
    init_default_scope("mmaction")
    cfg = Config.fromfile(config)
    for m in cfg.custom_imports["imports"]:
        importlib.import_module(m)
    cfg.model.backbone.pretrained = None  # weights come from the checkpoint
    model = MODELS.build(cfg.model).to(device).eval()
    return model, cfg


def load_ckpt(model, path):
    ck = torch.load(path, map_location="cpu")
    sd = ck.get("state_dict", ck)
    model.load_state_dict(sd, strict=True)
    model.eval()
    return ck.get("meta", {})


@torch.no_grad()
def logits_of(model, xs, device, batch_size=32):
    """xs: list of (3, 5, 224, 224) tensors, already ImageNet-normalised."""
    out = []
    for i in range(0, len(xs), batch_size):
        x = torch.stack(xs[i:i + batch_size]).to(device)
        z = model.cls_head(model.backbone(x)).reshape(-1)
        out.append(z.float().cpu())
    return torch.cat(out).numpy()


def check_normalisation(model, device):
    """Our cached windows use (x/255 - ImageNet mean) / std; the authors'
    data_preprocessor uses (x - 255*mean) / (255*std). Confirm they agree."""
    dp = model.data_preprocessor
    raw = torch.randint(0, 256, (1, 1, 3, 5, 8, 8)).float()
    theirs = dp({"inputs": [raw[0]], "data_samples": None}, training=False)["inputs"][0, 0]
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
    ours = (raw[0, 0] / 255.0 - mean) / std
    d = float((theirs.cpu() - ours).abs().max())
    print(f"normalisation check: max |authors - ours| = {d:.2e}")
    if d > 1e-3:
        sys.exit("normalisation differs -- stop")


def epoch_ckpts(work_dir):
    out = {}
    for p in glob.glob(os.path.join(work_dir, "epoch_*.pth")):
        m = re.search(r"epoch_(\d+)\.pth$", p)
        if m:
            out[int(m.group(1))] = p
    return dict(sorted(out.items()))


def author_best(work_dir):
    ps = glob.glob(os.path.join(work_dir, "best_mAUC@_epoch_*.pth"))
    if not ps:
        return None
    return max(int(re.search(r"epoch_(\d+)\.pth$", p).group(1)) for p in ps)


def stage_val(args, device):
    sys.path.insert(0, args.repo_dir)
    sys.path.insert(0, os.path.join(args.repo_dir, "pipeline"))  # cell*.py live in pipeline/
    sys.path.insert(0, os.path.join(args.repo_dir, "reeval"))
    import re02_infer_val as r2
    import re03_analyze_rq as r3

    model, _ = build_model(args.config, device)
    check_normalisation(model, device)
    print("Loading validation windows (re01 cache)...")
    vid_ids, targets, data = r2.load_val_tensors(args.cache_dir)
    taus = np.stack([data[l][1] for l in LEADS])
    print(f"  {len(vid_ids)} videos, {int(targets.sum())} positives")

    pred_dir = os.path.join(args.out_root, "preds_val_rq1b")
    os.makedirs(pred_dir, exist_ok=True)
    chosen = {}
    for s in args.seeds:
        run = f"riskprop_full_seed{s}"
        wd = args.work_dir_tmpl.format(seed=s)
        ck = epoch_ckpts(wd)
        if not ck:
            print(f"[{run}] no epoch_*.pth in {wd} -- skipped")
            continue
        final_ep = max(ck)
        if final_ep != args.expected_epochs:
            print(f"[{run}] WARNING: last epoch is {final_ep}, expected {args.expected_epochs}")
        rows, cache = [], {}
        t0 = time.time()
        for ep, p in ck.items():
            load_ckpt(model, p)
            z = np.stack([logits_of(model, data[l][0], device, args.batch_size) for l in LEADS])
            probs = 1.0 / (1.0 + np.exp(-z))
            vl = float(F.binary_cross_entropy_with_logits(torch.from_numpy(z[1]), torch.from_numpy(targets).float()))
            mp, mauc = r3.fast_map_mauc([probs[i] for i in range(3)], targets)
            rows.append(dict(epoch=ep, val_loss=vl, val_mAP=mp, val_mAUC01=mauc,
                             **{f"AP@{l}s": float(r3.average_precision_score(targets, probs[i]))
                                for i, l in enumerate(LEADS)}))
            cache[ep] = probs
            print(f"  [{run}] epoch {ep:2d}: val_loss={vl:.4f} mAP={mp:.4f} mAUC01={mauc:.4f}")
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(args.out_root, f"rq1b_val_per_epoch_seed{s}.csv"), index=False)
        best_ep = int(df.loc[df.val_loss.idxmin(), "epoch"])
        m_best = float(df.loc[df.epoch == best_ep, "val_mAP"].iloc[0])
        m_last = float(df.loc[df.epoch == final_ep, "val_mAP"].iloc[0])
        kind = "latest" if m_last > m_best else "best"      # tie -> best (locked rule)
        a_ep = author_best(wd)
        chosen[run] = dict(kind=kind, epoch=final_ep if kind == "latest" else best_ep,
                           best_loss_epoch=best_ep, final_epoch=final_ep,
                           val_mAP_best=m_best, val_mAP_latest=m_last,
                           authorbest_epoch=a_ep,
                           file=os.path.abspath(ck[final_ep if kind == "latest" else best_ep]))
        for k, ep in (("best", best_ep), ("latest", final_ep), ("authorbest", a_ep)):
            if ep is None or ep not in cache:
                continue
            out = os.path.join(pred_dir, f"{run}_{k}.npz")
            np.savez_compressed(out, vid_ids=np.array(vid_ids), targets=targets, taus=taus,
                                leads=np.array(LEADS), scores=cache[ep])
            with open(out.replace(".npz", ".json"), "w") as f:
                json.dump({"run": run, "model": "riskprop_full", "kind": k, "ckpt_epoch": ep,
                           "ckpt_file": os.path.basename(ck[ep]),
                           "val_loss": float(df.loc[df.epoch == ep, "val_loss"].iloc[0])}, f, indent=2)
        print(f"[{run}] best-loss epoch {best_ep} (mAP {m_best:.4f}) vs final {final_ep} "
              f"(mAP {m_last:.4f}) -> {kind}; authors' best-mAUC@ epoch {a_ep} "
              f"({(time.time() - t0) / 60:.1f} min)")

    out = os.path.join(args.out_root, "rq1b_chosen_checkpoints.json")
    with open(out, "w") as f:
        json.dump({"rule": "val mAP (mean AP over 0.5/1.0/1.5 s), candidates {lowest val loss, "
                           "final epoch}, tie -> lowest val loss (same as locked_config.json)",
                   "val_loss": "BCE (pos_weight 1) on the 1.0 s validation windows",
                   "chosen": chosen,
                   "created": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)
    print(f"\nWrote {out}")


def stage_test(args, device):
    sys.path.insert(0, args.repo_dir)
    sys.path.insert(0, os.path.join(args.repo_dir, "pipeline"))  # cell*.py live in pipeline/
    sys.path.insert(0, os.path.join(args.repo_dir, "reeval"))
    import re04_infer_test as r4

    ch_path = os.path.join(args.out_root, "rq1b_chosen_checkpoints.json")
    if not os.path.exists(ch_path):
        sys.exit(f"{ch_path} not found -- run --stage val first (checkpoints must be locked on validation).")
    chosen = json.load(open(ch_path))["chosen"]
    sub_dir = os.path.join(args.out_root, "test_infer_rq1b")
    os.makedirs(sub_dir, exist_ok=True)
    existing = glob.glob(os.path.join(sub_dir, "submission_riskprop_full_seed*.csv"))
    if existing and not args.force:
        sys.exit(f"{len(existing)} RQ1b submissions already exist in {sub_dir}. The official test is "
                 "used once; pass --force only if the previous run crashed.")
    lock_hash = hashlib.sha256(open(ch_path, "rb").read()).hexdigest()[:16]

    if args.download:
        r4.download_test_assets(args.hf_local_raw)
    sample_sub = pd.read_csv(os.path.join(args.hf_local_raw, "sample_submission.csv"), dtype=str)
    id_col, score_col = sample_sub.columns[0], sample_sub.columns[1]
    sub_ids = sample_sub[id_col].astype(str).tolist()
    video_map = {r4.norm_id(k): v for k, v in r4.list_test_videos(args.hf_local_raw).items()}
    print(f"{len(sub_ids)} submission ids, {len(video_map)} test videos on disk")

    cache_p = os.path.join(args.out_root, "test_windows_u8.npz")
    if os.path.exists(cache_p):
        z = np.load(cache_p)
        ids, frames = list(z["ids"]), z["frames"]
    else:
        ids, frames, skipped = [], [], []
        t0 = time.time()
        for i, vid in enumerate(sub_ids):
            p = video_map.get(r4.norm_id(vid))
            fr = r4.extract_tail_window(p) if p else None
            if fr is None:
                skipped.append(vid)
                continue
            ids.append(vid)
            frames.append(fr)
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(sub_ids)} windows ({time.time() - t0:.0f}s)")
        frames = np.stack(frames)
        np.savez(cache_p, ids=np.array(ids), frames=frames)
        print(f"extracted {len(ids)} windows, skipped {len(skipped)}")
    xs = [r4.to_tensor(f, args.repo_dir) for f in frames]

    model, _ = build_model(args.config, device)
    check_normalisation(model, device)
    written = []
    for run, info in chosen.items():
        load_ckpt(model, info["file"])
        probs = 1.0 / (1.0 + np.exp(-logits_of(model, xs, device, args.batch_size)))
        by_id = dict(zip(ids, probs.astype(float)))
        out = os.path.join(sub_dir, f"submission_{run}.csv")
        pd.DataFrame({id_col: sub_ids, score_col: [by_id.get(i, 0.5) for i in sub_ids]}).to_csv(out, index=False)
        written.append(run)
        print(f"  [OK] {run} ({info['kind']}, epoch {info['epoch']}) -> {out}")
    with open(os.path.join(sub_dir, "test_infer_rq1b_meta.json"), "w") as f:
        json.dump({"runs": written, "chosen_json_sha256_16": lock_hash,
                   "n_submission_ids": len(sub_ids), "n_scored": len(ids),
                   "filled_with_0.5": len(sub_ids) - len(ids),
                   "window": "causal 5-frame window ending at the last frame (re04.extract_tail_window)",
                   "normalisation": "ImageNet (= authors' data_preprocessor)",
                   "created": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)
    print(f"\nWrote {len(written)} submissions to {sub_dir}. Next: analyze_rq1b.py --stage test")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["val", "test"], required=True)
    ap.add_argument("--config", default="configs/riskprop_full_nexar.py")
    ap.add_argument("--work-dir-tmpl", default="work_dirs/rq1b_seed{seed}")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--expected-epochs", type=int, default=50)
    ap.add_argument("--repo-dir", default="/workspace/CPV301")
    ap.add_argument("--cache-dir", default="/workspace/CPV301/data/nexar_cache_5f")
    ap.add_argument("--hf-local-raw", default="/workspace/CPV301/data/nexar_hf_raw")
    ap.add_argument("--download", action="store_true", help="download the test set from Hugging Face first")
    ap.add_argument("--out-root", default="/workspace/CPV301/reeval_out/rq1b")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out_root, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    (stage_val if args.stage == "val" else stage_test)(args, device)


if __name__ == "__main__":
    main()
