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
    sys.path.insert(0, os.path.join(repo_dir, "pipeline"))  # cell*.py live in pipeline/
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


def model_key_of(run_name):
    if run_name in RUN_TO_MODEL_KEY:
        return RUN_TO_MODEL_KEY[run_name]
    if run_name.startswith("riskprop"):
        return "riskprop"
    raise KeyError(run_name)


@torch.no_grad()
def predict(model, xs, device, batch_size=8):
    out = []
    for i in range(0, len(xs), batch_size):
        batch = torch.stack(xs[i:i + batch_size]).to(device)
        out.append(torch.sigmoid(model(batch)).float().cpu())
    return torch.cat(out).numpy()


def to_tensor(frames_uint8, repo_dir):
    """(5, H, W, 3) uint8 -> (3, 5, H, W) float, IDENTICAL to the val/train
    pipeline: cell22._to_tensor = (x/255 - ImageNet mean) / ImageNet std.
    FIX (2026-09-25, before the first test run): the previous version only
    divided by 255 and skipped the ImageNet normalisation, so test inputs
    would not have matched what every model was trained / validated on."""
    sys.path.insert(0, repo_dir)
    sys.path.insert(0, os.path.join(repo_dir, "pipeline"))  # cell*.py live in pipeline/
    from cell22_adalea_dataset import _to_tensor
    return _to_tensor({"frames": torch.from_numpy(frames_uint8)})


def norm_id(x):
    """'00204' / '204' / 204 -> '204' so ids match file names either way."""
    x = str(x).strip()
    return str(int(x)) if x.isdigit() else x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-dir", default="/workspace/CPV301")
    ap.add_argument("--hf-local-raw", default="/workspace/CPV301/data/nexar_hf_raw")
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--locked-config", default="reeval_out/analysis/locked_config.json")
    ap.add_argument("--out-dir", default="reeval_out/test_infer")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--with-rq2", action="store_true",
                    help="also score RQ2 runs A/B/C (exploratory) using the checkpoint "
                         "choice in rq2_chosen_checkpoints.json")
    ap.add_argument("--rq2-chosen", default="reeval_out/rq2_chosen_checkpoints.json")
    ap.add_argument("--rq2-ckpt-dir", default="/workspace/CPV301/outputs_riskprop")
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
    run_files = {r: os.path.join(args.ckpt_dir, info["file"]) for r, info in chosen.items()}
    if args.with_rq2:
        with open(args.rq2_chosen) as f:
            rq2 = json.load(f)["chosen"]
        added = 0
        for r, info in rq2.items():
            if info.get("condition") in ("A", "B", "C") and r not in run_files:
                run_files[r] = os.path.join(args.rq2_ckpt_dir, f"{info['kind']}_{r}.pth")
                added += 1
        print(f"--with-rq2: +{added} RQ2 runs (A/B/C), exploratory")

    if not args.skip_download:
        download_test_assets(args.hf_local_raw)

    sol_path = os.path.join(args.hf_local_raw, "solution.csv")
    horizon_path = os.path.join(args.hf_local_raw, "time_to_accident_test_map.csv")
    sample_sub_path = os.path.join(args.hf_local_raw, "sample_submission.csv")
    sol = pd.read_csv(sol_path, dtype=str)
    sample_sub = pd.read_csv(sample_sub_path, dtype=str)
    print("solution.csv columns:", list(sol.columns))
    print("solution.csv head:\n", sol.head())
    print("sample_submission.csv columns:", list(sample_sub.columns))
    print("sample_submission.csv head:\n", sample_sub.head())
    input("Neu cot id/target o tren hop ly (khop voi ten file video), "
          "nhan Enter de tiep tuc. Neu KHONG dung, Ctrl+C va bao lai.")

    id_col_sub = sample_sub.columns[0]
    score_col_sub = sample_sub.columns[1]
    submission_ids = sample_sub[id_col_sub].astype(str).tolist()

    video_map = {norm_id(k): v for k, v in list_test_videos(args.hf_local_raw).items()}
    print(f"Found {len(video_map)} test video files on disk.")
    missing_ids = [i for i in submission_ids if norm_id(i) not in video_map]
    if missing_ids:
        print(f"WARNING: {len(missing_ids)} submission ids have no matching video file "
              f"(first 5: {missing_ids[:5]}). Check id<->filename convention above.")
        input("Nhan Enter de tiep tuc voi nhung video tim thay, hoac Ctrl+C de dung lai.")

    ordered_ids = [i for i in submission_ids if norm_id(i) in video_map]
    print(f"Extracting tail window for {len(ordered_ids)} videos...")
    t0 = time.time()
    frames_by_id = {}
    skipped = []
    for i, vid_id in enumerate(ordered_ids):
        fr = extract_tail_window(video_map[norm_id(vid_id)])
        if fr is None:
            skipped.append(vid_id)
            continue
        frames_by_id[vid_id] = to_tensor(fr, args.repo_dir)
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(ordered_ids)} extracted...")
    print(f"Done extracting in {(time.time()-t0)/60:.1f} min, skipped={len(skipped)}")

    final_ids = [i for i in ordered_ids if i in frames_by_id]
    xs = [frames_by_id[i] for i in final_ids]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    models = {}
    n_fill = len(submission_ids) - len(final_ids)
    if n_fill:
        print(f"NOTE: {n_fill} submission ids without a usable video get score 0.5")
    written = 0
    for run_name, ckpt_path in run_files.items():
        model_key = model_key_of(run_name)
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
        by_id = dict(zip(final_ids, scores.astype(float)))
        pd.DataFrame({id_col_sub: submission_ids,
                      score_col_sub: [by_id.get(i, 0.5) for i in submission_ids]}
                     ).to_csv(out_csv, index=False)
        written += 1
        print(f"  [OK] {run_name:24s} -> {out_csv} ({time.time()-t0:.0f}s, "
              f"n={len(final_ids)}, score_range=[{scores.min():.3f},{scores.max():.3f}])")

    with open(os.path.join(args.out_dir, "test_infer_meta.json"), "w") as f:
        json.dump({"n_submission_ids": len(submission_ids), "n_scored": len(final_ids),
                   "filled_with_0.5": n_fill, "skipped_videos": skipped,
                   "top_head_rule": top_head_rule, "runs": sorted(run_files),
                   "normalisation": "cell22._to_tensor (ImageNet)"}, f, indent=2)
    print(f"\nWrote {written} submission CSVs to {args.out_dir}")
    print("Next: run reeval/re05_eval_test.py to score them with evaluate_submission.py "
          "and aggregate mean +/- SD per method.")


if __name__ == "__main__":
    main()
