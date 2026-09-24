"""
re04 -- Official Nexar test set (public + private) inference, using the
LOCKED configuration from re03 (locked_config.json): one fixed TOP head
rule, one checkpoint per run. This must run AFTER locked_config.json
exists and BEFORE looking at solution.csv scores in any way that could
feed back into rule selection (it already can't, since the rules are
already locked and this script does not read solution.csv at all).

Unlike the internal validation split (which synthesizes 3 lead-time
windows per training video), the official test videos are already
pre-clipped by Nexar to end 0.5/1.0/1.5s before the event (or, for
negatives, at some fixed point) -- so each test video gets exactly ONE
causal 5-frame window, taken from the END of the clip. No lead_time /
horizon information is given to the model at inference time (that would
be leakage): TOP uses the single locked head rule, not the oracle
LEAD_TO_HIDX mapping.

Usage (on a machine with the 24 checkpoints and locked_config.json):
    python reeval/re04_infer_test.py \
        --repo-dir /workspace/CPV301 \
        --ckpt-dir /workspace/CPV301/ckpts \
        --locked-config reeval_out/analysis/locked_config.json \
        --out-dir reeval_out/test_infer
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from decord import VideoReader, cpu
from PIL import Image

HF_REPO = "nexar-ai/nexar_collision_prediction"
NUM_FRAMES = 5
SAMPLE_FPS = 10
SIZE = 224
SPLITS = ["test-public", "test-private"]
# head_X.Y rule -> column index into TOP's 20-head output (same mapping as re03)
TOP_HEAD_IDX = {"head_0.5": 4, "head_1.0": 9, "head_1.5": 14, "head_2.0": 19}


def download_test_assets(hf_local_raw):
    os.makedirs(hf_local_raw, exist_ok=True)
    print("Downloading root CSV/py assets...")
    for fname in ["solution.csv", "time_to_accident_test_map.csv",
                  "sample_submission.csv", "evaluate_submission.py"]:
        hf_hub_download(repo_id=HF_REPO, repo_type="dataset", filename=fname,
                         local_dir=hf_local_raw)
    for split in SPLITS:
        print(f"Downloading {split}/ videos...")
        snapshot_download(repo_id=HF_REPO, repo_type="dataset",
                           local_dir=hf_local_raw,
                           allow_patterns=[f"{split}/**"])


def list_test_videos(hf_local_raw):
    """Return {video_id(str, no ext): abs_path} for every test video found."""
    out = {}
    for split in SPLITS:
        d = os.path.join(hf_local_raw, split)
        if not os.path.isdir(d):
            continue
        for root, _, files in os.walk(d):
            for f in files:
                if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
                    vid_id = os.path.splitext(f)[0]
                    out[vid_id] = os.path.join(root, f)
    return out


def extract_tail_window(vid_path, size=SIZE):
    """Causal 5-frame window ending at the LAST frame of the clip."""
    try:
        vr = VideoReader(vid_path, ctx=cpu(0))
        total, fps = len(vr), vr.get_avg_fps()
    except Exception as e:
        print(f"  [WARN] cannot open {vid_path}: {e}")
        return None
    end_frame = total - 1
    frame_step = max(1, int(round(fps / SAMPLE_FPS)))
    indices = [end_frame - (NUM_FRAMES - 1 - k) * frame_step for k in range(NUM_FRAMES)]
    indices = np.clip(indices, 0, total - 1).astype(int)
    try:
        frames = vr.get_batch(indices).asnumpy()
    except Exception as e:
        print(f"  [WARN] frame extract failed for {vid_path}: {e}")
        return None
    imgs = [np.array(Image.fromarray(f).resize((size, size))) for f in frames]
    return np.stack(imgs).astype(np.uint8)


def build_model(model_key, repo_dir):
    sys.path.insert(0, repo_dir)
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


RUN_TO_MODEL_KEY = {}
for s in [42, 43, 44]:
    RUN_TO_MODEL_KEY[f"top_seed{s}"] = "top"
    RUN_TO_MODEL_KEY[f"adalea_seed{s}"] = "adalea"
    RUN_TO_MODEL_KEY[f"riskprop_random_seed{s}"] = "riskprop"
    RUN_TO_MODEL_KEY[f"riskprop_fixed_seed{s}"] = "riskprop"


@torch.no_grad()
def predict(model, xs, device, batch_size=8):
    out = []
    for i in range(0, len(xs), batch_size):
        batch = torch.stack(xs[i:i + batch_size]).to(device)
        out.append(torch.sigmoid(model(batch)).float().cpu())
    return torch.cat(out).numpy()


def to_tensor(frames_uint8):
    # (5, H, W, 3) uint8 -> (3, 5, H, W) float32 in [0,1], matches
    # cell11b/cell22's _to_tensor convention (C, T, H, W).
    t = torch.from_numpy(frames_uint8).float() / 255.0
    t = t.permute(3, 0, 1, 2)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-dir", default="/workspace/CPV301")
    ap.add_argument("--hf-local-raw", default="/workspace/CPV301/data/nexar_hf_raw")
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--locked-config", default="reeval_out/analysis/locked_config.json")
    ap.add_argument("--out-dir", default="reeval_out/test_infer")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.locked_config) as f:
        locked = json.load(f)
    if not locked.get("legacy_check_passed"):
        sys.exit("locked_config.json says legacy_check_passed=false -- do not run the test set.")
    top_head_rule = locked["top_head_rule"]
    chosen = locked["chosen_checkpoints"]
    print(f"Locked TOP head rule: {top_head_rule}")
    print(f"Locked checkpoints: {len(chosen)} runs")

    if not args.skip_download:
        download_test_assets(args.hf_local_raw)

    sol_path = os.path.join(args.hf_local_raw, "solution.csv")
    horizon_path = os.path.join(args.hf_local_raw, "time_to_accident_test_map.csv")
    sample_sub_path = os.path.join(args.hf_local_raw, "sample_submission.csv")
    sol = pd.read_csv(sol_path)
    sample_sub = pd.read_csv(sample_sub_path)
    print("solution.csv columns:", list(sol.columns))
    print("solution.csv head:\n", sol.head())
    print("sample_submission.csv columns:", list(sample_sub.columns))
    print("sample_submission.csv head:\n", sample_sub.head())
    input("Neu cot id/target o tren hop ly (khop voi ten file video), "
          "nhan Enter de tiep tuc. Neu KHONG dung, Ctrl+C va bao lai.")

    id_col_sub = sample_sub.columns[0]
    score_col_sub = sample_sub.columns[1]
    submission_ids = sample_sub[id_col_sub].astype(str).tolist()

    video_map = list_test_videos(args.hf_local_raw)
    print(f"Found {len(video_map)} test video files on disk.")
    missing_ids = [i for i in submission_ids if i not in video_map]
    if missing_ids:
        print(f"WARNING: {len(missing_ids)} submission ids have no matching video file "
              f"(first 5: {missing_ids[:5]}). Check id<->filename convention above.")
        input("Nhan Enter de tiep tuc voi nhung video tim thay, hoac Ctrl+C de dung lai.")

    ordered_ids = [i for i in submission_ids if i in video_map]
    print(f"Extracting tail window for {len(ordered_ids)} videos...")
    t0 = time.time()
    frames_by_id = {}
    skipped = []
    for i, vid_id in enumerate(ordered_ids):
        fr = extract_tail_window(video_map[vid_id])
        if fr is None:
            skipped.append(vid_id)
            continue
        frames_by_id[vid_id] = to_tensor(fr)
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(ordered_ids)} extracted...")
    print(f"Done extracting in {(time.time()-t0)/60:.1f} min, skipped={len(skipped)}")

    final_ids = [i for i in ordered_ids if i in frames_by_id]
    xs = [frames_by_id[i] for i in final_ids]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    models = {}
    for run_name, info in chosen.items():
        model_key = RUN_TO_MODEL_KEY[run_name]
        ckpt_path = os.path.join(args.ckpt_dir, info["file"])
        if not os.path.exists(ckpt_path):
            print(f"  [MISSING] {ckpt_path} -- skipping {run_name}")
            continue
        if model_key not in models:
            models[model_key] = build_model(model_key, args.repo_dir).to(device).eval()
        model = models[model_key]
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        model.eval()

        t0 = time.time()
        raw = predict(model, xs, device, args.batch_size)
        if model_key == "top":
            hidx = TOP_HEAD_IDX[top_head_rule]
            scores = raw[:, hidx]
        else:
            scores = raw
        out_csv = os.path.join(args.out_dir, f"submission_{run_name}.csv")
        pd.DataFrame({id_col_sub: final_ids, score_col_sub: scores}).to_csv(out_csv, index=False)
        print(f"  [OK] {run_name:24s} -> {out_csv} ({time.time()-t0:.0f}s, "
              f"n={len(final_ids)}, score_range=[{scores.min():.3f},{scores.max():.3f}])")

    print(f"\nWrote {len(chosen)} submission CSVs to {args.out_dir}")
    print("Next: run reeval/re05_eval_test.py to score them with evaluate_submission.py "
          "and aggregate mean +/- SD per method.")


if __name__ == "__main__":
    main()
