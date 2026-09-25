"""
re07 -- RQ2 inference on the internal validation split (GPU part).

Produces everything re08_analyze_rq2.py needs; computes no statistics
itself except the (already locked) per-run checkpoint choice.

Runs covered (15 runs x {best, latest} = 30 checkpoints):
    A  neither   riskprop_random_seed{S}_neither   (new, outputs_riskprop/)
    B  FFR-only  riskprop_random_seed{S}_ffronly   (new, outputs_riskprop/)
    C  AMC-only  riskprop_random_seed{S}_amconly   (new, outputs_riskprop/)
    D  both      riskprop_random_seed{S}           (RQ1/RQ3 "RiskProp random", ckpts/)
    F  fixed     riskprop_fixed_seed{S}            (RQ3 "RiskProp fixed", ckpts/ --
                                                    only for RQ3 temporal metrics)
    S in {42, 43, 44}

Step 1  3-lead predictions (0.5/1.0/1.5 s) for every checkpoint, on the SAME
        cached windows as re02 (nexar_cache_5f/val, AdaLEAValDataset), written
        in re02's .npz format -> reeval_out/preds_val_rq2/.
Step 2  Checkpoint choice per run with the rule LOCKED in
        results/reeval_corrected/locked_config.json ("per_run": higher val mAP
        = mean AP over 0.5/1.0/1.5 s among {best, latest}; tie -> best).
        For D and F the choice must equal the one already locked for RQ1/RQ3;
        a mismatch is reported and the locked choice is kept.
Step 3  Dense risk curves (RQ2_THEORY.md, Group 3), for the chosen checkpoint
        of every run. Sweep LOCKED here, before any temporal result exists:
          positives: 30 causal 5-frame windows ending at t_event - d,
                     d = 3.0, 2.9, ..., 0.1 s (earliest -> latest); same
                     clipping as the val cache (t_obs >= 0.4 s, <= t_event-0.1)
          negatives: 30 windows ending at mid - (d - 0.1), mid = duration/2,
                     i.e. the same 0.1 s grid, finishing exactly at the val
                     cache's negative window (mid)
        Frames/resize/normalisation identical to the val cache (re01/cell10 +
        cell22._to_tensor). Windows are cached once as uint8 in
        data/nexar_cache_dense/ (~7 GB) and reused by every run.
        -> reeval_out/dense_rq2/<run>.npz  (scores (N, 30), vid_ids, targets)

Usage (defaults match the vast.ai layout, so normally no arguments):
    python reeval/re07_infer_rq2.py
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

LEADS = [0.5, 1.0, 1.5]
SEEDS = [42, 43, 44]
KINDS = ["best", "latest"]
NUM_FRAMES, SIZE, SAMPLE_FPS = 5, 224, 10          # = re01 / cell10
DENSE_D = [round(3.0 - 0.1 * k, 1) for k in range(30)]   # 3.0 ... 0.1
CONDITIONS = {   # condition -> (run-name template, checkpoint location key)
    "A": ("riskprop_random_seed{s}_neither", "new"),
    "B": ("riskprop_random_seed{s}_ffronly", "new"),
    "C": ("riskprop_random_seed{s}_amconly", "new"),
    "D": ("riskprop_random_seed{s}", "old"),
    "F": ("riskprop_fixed_seed{s}", "old"),
}


def run_list():
    return [(c, tmpl.format(s=s), loc, s)
            for c, (tmpl, loc) in CONDITIONS.items() for s in SEEDS]


# ----------------------------------------------------------------------------
# Dense windows (no torch needed)
# ----------------------------------------------------------------------------
def dense_t_obs(row, duration):
    """30 end times (s), earliest -> latest. Positives follow re01's
    get_val_indices clipping with lead_time=d; negatives sweep up to mid."""
    min_t_obs = (NUM_FRAMES - 1) / SAMPLE_FPS
    if int(row["target"]) == 1 and not pd.isna(row["time_of_event"]):
        toe = float(row["time_of_event"])
        out = []
        for d in DENSE_D:
            t = max(min_t_obs, toe - d)
            t = min(t, toe - 0.1)
            out.append(max(min_t_obs, t))
        return out
    mid = duration * 0.5
    return [max(min_t_obs, mid - (d - 0.1)) for d in DENSE_D]


def window_indices(t_obs, fps, total_frames):
    """Verbatim index rule of re01.get_val_indices / cell10."""
    end_frame = min(int(t_obs * fps), total_frames - 1)
    frame_step = max(1, int(round(fps / SAMPLE_FPS)))
    idx = [end_frame - (NUM_FRAMES - 1 - k) * frame_step for k in range(NUM_FRAMES)]
    return np.clip(idx, 0, total_frames - 1).astype(int)


def build_dense_cache(data_dir, cache_5f_dir, dense_dir):
    """Decode each val video once and store (N, 30, 5, 224, 224, 3) uint8."""
    from decord import VideoReader, cpu
    from PIL import Image

    arr_path = os.path.join(dense_dir, "windows_u8.npy")
    meta_path = os.path.join(dense_dir, "meta.json")
    if os.path.exists(arr_path) and os.path.exists(meta_path):
        print(f"  dense cache exists -> {dense_dir} (reusing)")
        return
    os.makedirs(dense_dir, exist_ok=True)

    with open(os.path.join(cache_5f_dir, "val_index.json")) as f:
        vid_ids = list(json.load(f).keys())            # same order as re02
    df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    df["vid"] = df["id"].apply(lambda x: str(int(x)).zfill(5))
    rows = df.set_index("vid")

    n = len(vid_ids)
    arr = np.lib.format.open_memmap(arr_path, mode="w+", dtype=np.uint8,
                                    shape=(n, len(DENSE_D), NUM_FRAMES, SIZE, SIZE, 3))
    targets, t_obs_all, failed = [], [], []
    t0 = time.time()
    for i, vid in enumerate(vid_ids):
        row = rows.loc[vid]
        targets.append(int(row["target"]))
        path = os.path.join(data_dir, "train", vid + ".mp4")
        try:
            vr = VideoReader(path, ctx=cpu(0))
            total, fps = len(vr), vr.get_avg_fps()
            t_list = dense_t_obs(row, total / fps)
            idx = np.stack([window_indices(t, fps, total) for t in t_list])
            uniq = np.unique(idx)
            frames = vr.get_batch(list(uniq)).asnumpy()
            small = {int(u): np.array(Image.fromarray(fr).resize((SIZE, SIZE)))
                     for u, fr in zip(uniq, frames)}
            for k in range(len(t_list)):
                arr[i, k] = np.stack([small[int(j)] for j in idx[k]]).astype(np.uint8)
            t_obs_all.append([float(t) for t in t_list])
        except Exception as e:
            print(f"  [WARN] {vid}: {e} -> zeros")
            arr[i] = 0
            failed.append(vid)
            t_obs_all.append([float("nan")] * len(DENSE_D))
        if (i + 1) % 50 == 0:
            print(f"  [dense] {i+1}/{n} videos ({time.time()-t0:.0f}s)")
    arr.flush()
    meta = {"vid_ids": vid_ids, "targets": targets, "t_obs": t_obs_all,
            "dense_d": DENSE_D, "failed": failed,
            "rule": "pos: t_event-d, d=3.0..0.1; neg: mid-(d-0.1); re01 clipping",
            "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(meta_path, "w") as f:
        json.dump(meta, f)
    print(f"  dense cache done: {n} videos x {len(DENSE_D)} windows "
          f"(failed={len(failed)}) in {(time.time()-t0)/60:.1f} min")


# ----------------------------------------------------------------------------
# Torch parts
# ----------------------------------------------------------------------------
def ckpt_path(args, run, loc, kind):
    base = args.new_ckpt_dir if loc == "new" else args.old_ckpt_dir
    return os.path.join(base, f"{kind}_{run}.pth")


def step1_three_lead(args, device):
    import torch
    sys.path.insert(0, args.repo_dir)
    sys.path.insert(0, os.path.join(args.repo_dir, "reeval"))
    import re02_infer_val as r2
    from cell31_model_riskprop import RiskPropModel

    os.makedirs(args.preds_dir, exist_ok=True)
    print("Loading validation windows (re02 loader)...")
    vid_ids, targets, data = r2.load_val_tensors(args.cache_5f)
    taus = np.stack([data[l][1] for l in LEADS])
    model = RiskPropModel().to(device).eval()
    missing = []
    for cond, run, loc, s in run_list():
        for kind in KINDS:
            out = os.path.join(args.preds_dir, f"{run}_{kind}.npz")
            if os.path.exists(out):
                continue
            p = ckpt_path(args, run, loc, kind)
            if not os.path.exists(p):
                print(f"  [MISSING] {p}")
                missing.append(p)
                continue
            ck = torch.load(p, map_location=device, weights_only=False)
            model.load_state_dict(ck["model"])
            model.eval()
            t0 = time.time()
            scores = np.stack([r2.predict(model, data[l][0], device, 8) for l in LEADS])
            np.savez_compressed(out, vid_ids=np.array(vid_ids), targets=targets,
                                taus=taus, leads=np.array(LEADS), scores=scores)
            with open(out.replace(".npz", ".json"), "w") as f:
                json.dump({"run": run, "condition": cond, "kind": kind,
                           "ckpt_file": os.path.basename(p),
                           "ckpt_sha256_8MB": r2.sha256_head(p),
                           "ckpt_epoch": ck.get("epoch"),
                           "ckpt_best_val_loss": ck.get("best_val_loss")},
                          f, indent=2, default=float)
            print(f"  [OK] {cond} {run:34s} {kind:6s} epoch={ck.get('epoch')} "
                  f"({time.time()-t0:.0f}s)")
    return missing


def step2_choose(args):
    from sklearn.metrics import average_precision_score
    with open(args.locked_config) as f:
        locked = json.load(f)
    chosen, rows = {}, []
    for cond, run, loc, s in run_list():
        maps = {}
        for kind in KINDS:
            p = os.path.join(args.preds_dir, f"{run}_{kind}.npz")
            if os.path.exists(p):
                z = np.load(p)
                maps[kind] = float(np.mean([average_precision_score(z["targets"], z["scores"][i])
                                            for i in range(len(LEADS))]))
        if not maps:
            continue
        rule_kind = "best" if maps.get("best", -1) >= maps.get("latest", -1) else "latest"
        kind, note = rule_kind, ""
        if loc == "old":
            lk = locked["chosen_checkpoints"].get(run, {}).get("kind")
            if lk and lk != rule_kind:
                note = f"MISMATCH with locked ({lk}) -- keeping locked"
                kind = lk
            elif lk:
                note = "matches locked_config"
        chosen[run] = {"condition": cond, "seed": s, "kind": kind,
                       "val_mAP_best": maps.get("best"), "val_mAP_latest": maps.get("latest")}
        rows.append(f"  {cond} {run:34s} best={maps.get('best', float('nan')):.4f} "
                    f"latest={maps.get('latest', float('nan')):.4f} -> {kind} {note}")
    print("\n".join(rows))
    with open(os.path.join(args.out_root, "rq2_chosen_checkpoints.json"), "w") as f:
        json.dump({"rule": locked.get("checkpoint_rule_criterion"),
                   "chosen": chosen}, f, indent=2)
    return chosen


def step3_dense(args, chosen, device):
    """Videos outer, models inner: every window is normalised once and scored
    by all chosen checkpoints (15 x slow_r50 ~ 2 GB VRAM)."""
    import torch
    sys.path.insert(0, args.repo_dir)
    from cell31_model_riskprop import RiskPropModel
    from cell22_adalea_dataset import _to_tensor

    arr = np.load(os.path.join(args.dense_cache, "windows_u8.npy"), mmap_mode="r")
    with open(os.path.join(args.dense_cache, "meta.json")) as f:
        meta = json.load(f)
    os.makedirs(args.dense_out, exist_ok=True)
    todo = [(cond, run, loc) for cond, run, loc, s in run_list()
            if run in chosen and not os.path.exists(os.path.join(args.dense_out, f"{run}.npz"))]
    if not todo:
        print("  dense scores exist for every run (reusing)")
        return
    models = []
    for cond, run, loc in todo:
        m = RiskPropModel().to(device).eval()
        ck = torch.load(ckpt_path(args, run, loc, chosen[run]["kind"]),
                        map_location=device, weights_only=False)
        m.load_state_dict(ck["model"])
        m.eval()
        models.append(m)
    print(f"  {len(models)} models loaded")
    scores = np.zeros((len(todo), arr.shape[0], arr.shape[1]), dtype=np.float32)
    t0 = time.time()
    with torch.no_grad():
        for i in range(arr.shape[0]):
            x = torch.stack([_to_tensor({"frames": torch.from_numpy(np.array(arr[i, k]))})
                             for k in range(arr.shape[1])]).to(device)
            for r, m in enumerate(models):
                scores[r, i] = torch.sigmoid(m(x)).float().cpu().numpy()
            if (i + 1) % 50 == 0:
                print(f"  [dense] {i+1}/{arr.shape[0]} videos ({time.time()-t0:.0f}s)")
    for r, (cond, run, loc) in enumerate(todo):
        np.savez_compressed(os.path.join(args.dense_out, f"{run}.npz"), scores=scores[r],
                            vid_ids=np.array(meta["vid_ids"]), targets=np.array(meta["targets"]),
                            dense_d=np.array(DENSE_D), kind=np.array(chosen[run]["kind"]))
        print(f"  [dense OK] {cond} {run} ({chosen[run]['kind']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-dir", default="/workspace/CPV301")
    ap.add_argument("--data-dir", default="/workspace/CPV301/data/nexar_kaggle_style")
    ap.add_argument("--cache-5f", default="/workspace/CPV301/data/nexar_cache_5f")
    ap.add_argument("--dense-cache", default="/workspace/CPV301/data/nexar_cache_dense")
    ap.add_argument("--new-ckpt-dir", default="/workspace/CPV301/outputs_riskprop")
    ap.add_argument("--old-ckpt-dir", default="/workspace/CPV301/ckpts")
    ap.add_argument("--locked-config",
                    default="/workspace/CPV301/results/reeval_corrected/locked_config.json")
    ap.add_argument("--out-root", default="/workspace/CPV301/reeval_out")
    args = ap.parse_args()
    args.preds_dir = os.path.join(args.out_root, "preds_val_rq2")
    args.dense_out = os.path.join(args.out_root, "dense_rq2")
    os.makedirs(args.out_root, exist_ok=True)

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False
    print(f"device={device}")
    t_all = time.time()

    print("\n[1] 3-lead validation predictions (15 runs x best/latest)")
    missing = step1_three_lead(args, device)
    if missing:
        print(f"\n{len(missing)} checkpoint(s) missing -- fix and re-run (done ones are skipped).")
        sys.exit(1)

    print("\n[2] Checkpoint choice (locked per_run rule)")
    chosen = step2_choose(args)

    print("\n[3a] Dense val windows (built once)")
    build_dense_cache(args.data_dir, args.cache_5f, args.dense_cache)
    print("\n[3b] Dense risk curves for chosen checkpoints")
    step3_dense(args, chosen, device)

    print(f"\nDone in {(time.time()-t_all)/60:.1f} min. Next: python reeval/re08_analyze_rq2.py")


if __name__ == "__main__":
    main()
