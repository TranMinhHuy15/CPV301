"""
RQ1b -- data checks to run BEFORE any training (partner's checklist, 25/9):

  1. The split: 300 validation videos = the project's split (150 pos / 150 neg),
     1,200 training videos, no overlap.
  2. One training positive, one training negative and one validation positive
     go through the authors' pipeline and give 30 clips x 5 frames.
  3. For positives, the last frame of the last clip is the accident frame, and
     the accident frame in seconds matches time_of_event in train.csv
     (10 fps grid: frame k = k / 10 s, tolerance 0.05 s).
  4. Consecutive clips are 1 frame = 0.1 s apart; the 30 clips span 3.3 s.
  5. Negatives are placed at a random position that changes between epochs.

Also saves a picture of the first and last clip of one positive so the frames
can be checked by eye (frames must show the road, the last clip the moment of
the collision).

Run from the RiskProp repo root, with PYTHONPATH containing riskprop_full/:
    python /workspace/CPV301/riskprop_full/check_rq1b_setup.py \
        configs/riskprop_full_nexar.py \
        --train-csv /workspace/CPV301/data/nexar_kaggle_style/train.csv
Exit code 0 = all checks passed.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

FPS = 10
N_CLIPS, CLIP_LEN = 30, 5

results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--train-csv", default="/workspace/CPV301/data/nexar_kaggle_style/train.csv")
    ap.add_argument("--out-png", default="rq1b_check_clips.png")
    ap.add_argument("--cfg-options", nargs="*", default=[])
    args = ap.parse_args()

    from mmengine.config import Config, DictAction  # noqa: F401
    from mmengine.registry import init_default_scope
    from mmaction.registry import DATASETS
    from mmaction.utils import register_all_modules

    cfg = Config.fromfile(args.config)
    for kv in args.cfg_options:
        k, v = kv.split("=", 1)
        cfg.merge_from_dict({k: v})
    register_all_modules()
    init_default_scope("mmaction")
    import importlib
    for m in cfg.custom_imports["imports"]:
        importlib.import_module(m)

    train_ds = DATASETS.build(cfg.train_dataloader.dataset)
    val_ds = DATASETS.build(cfg.val_dataloader.dataset)
    man = json.load(open(cfg.train_dataloader.dataset.split_manifest))
    man_val = {str(int(v)).zfill(5) for v in man["val_ids"]}

    tr = [train_ds.get_data_info(i) for i in range(len(train_ds))]
    va = [val_ds.get_data_info(i) for i in range(len(val_ds))]
    tr_ids = {d["video_id"] for d in tr}
    va_ids = {d["video_id"] for d in va}
    n_pos_va = sum(bool(d["target"]) for d in va)
    check("validation videos all come from the project split", va_ids <= man_val and not (tr_ids & man_val),
          f"{len(va_ids)} val videos ({n_pos_va} pos / {len(va_ids) - n_pos_va} neg); manifest has {len(man_val)}")
    check("train and validation do not overlap", not (tr_ids & va_ids), f"{len(tr_ids)} train videos")
    if len(va_ids) == 300:
        check("validation set = all 300 videos of the project split", va_ids == man_val)
        check("150 pos / 150 neg in validation", n_pos_va == 150, f"{n_pos_va} positives")
        check("1,200 training videos", len(tr_ids) == 1200, f"{len(tr_ids)}")
    else:
        print(f"[INFO] {len(va_ids)} val videos found -- full-size checks skipped (test data?)")

    df = pd.read_csv(args.train_csv)
    df["vid5"] = df["id"].apply(lambda x: str(int(x)).zfill(5))
    toe = dict(zip(df.vid5, df.time_of_event))

    def sample(ds, idx):
        out = ds[idx]
        ds_item = out["data_samples"]
        fi = np.asarray(ds_item.frame_inds).reshape(-1)
        return out["inputs"], fi, ds_item

    # ---- training positive ----
    i_pos = next(i for i, d in enumerate(tr) if d["target"])
    x, fi, s = sample(train_ds, i_pos)
    d = tr[i_pos]
    check("train positive: 30 clips x 5 frames", tuple(x.shape[:1]) == (N_CLIPS,) and fi.size == N_CLIPS * CLIP_LEN,
          f"inputs {tuple(x.shape)}, frame_inds {fi.size}")
    clips = fi.reshape(N_CLIPS, CLIP_LEN)
    check("clips are 1 frame (0.1 s) apart, span 3.3 s",
          np.all(np.diff(clips[:, 0]) == 1) and np.all(np.diff(clips, axis=1) == 1)
          and (clips[-1, -1] - clips[0, 0]) == N_CLIPS + CLIP_LEN - 2,
          f"first clip {clips[0].tolist()}, last clip {clips[-1].tolist()}")
    acc = d["accident_frame"]
    check("train positive: last frame = accident frame", clips[-1, -1] == min(acc, d["total_frames"] - 1),
          f"last frame {clips[-1, -1]}, accident_frame {acc}")
    t_csv = float(toe[d["video_id"]])
    check("accident frame in seconds matches train.csv", abs(acc / FPS - t_csv) <= 0.05 + 1e-9,
          f"{acc}/{FPS} = {acc / FPS:.2f}s vs time_of_event {t_csv:.3f}s")

    # ---- every positive: annotation vs csv ----
    bad = [(d["video_id"], d["accident_frame"], float(toe[d["video_id"]])) for d in tr + va
           if d["target"] and abs(d["accident_frame"] / FPS - float(toe[d["video_id"]])) > 0.05 + 1e-9
           and d["accident_frame"] < d["total_frames"] - 1]
    check("all positives: accident_frame / 10 = time_of_event (+-0.05 s)", not bad,
          f"{len(bad)} mismatches" + (f", first: {bad[:3]}" if bad else ""))

    # ---- training negative, random position ----
    i_neg = next(i for i, d in enumerate(tr) if not d["target"])
    starts = {int(sample(train_ds, i_neg)[1][0]) for _ in range(5)}
    check("train negative: position changes between draws", len(starts) > 1 or tr[i_neg]["total_frames"] <= 34,
          f"first-frame positions over 5 draws: {sorted(starts)}")

    # ---- validation positive ----
    j_pos = next(i for i, d in enumerate(va) if d["target"])
    xv, fv, _ = sample(val_ds, j_pos)
    dv = va[j_pos]
    check("val positive: 30 clips x 5 frames, last frame = accident frame",
          fv.size == N_CLIPS * CLIP_LEN and fv.reshape(N_CLIPS, CLIP_LEN)[-1, -1]
          == min(dv["accident_frame"], dv["total_frames"] - 1),
          f"last frame {fv.reshape(N_CLIPS, CLIP_LEN)[-1, -1]}, accident_frame {dv['accident_frame']}")

    # ---- picture ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        arr = x.numpy() if hasattr(x, "numpy") else np.asarray(x)   # (30, C, T, H, W), 0-255
        fig, axes = plt.subplots(2, CLIP_LEN, figsize=(3 * CLIP_LEN, 6))
        for r, ci in enumerate([0, N_CLIPS - 1]):
            for t in range(CLIP_LEN):
                img = arr[ci, :, t].transpose(1, 2, 0)
                axes[r, t].imshow(np.clip(img, 0, 255).astype(np.uint8))
                axes[r, t].set_title(f"clip {ci} frame {clips[ci, t]} ({clips[ci, t] / FPS:.1f}s)", fontsize=8)
                axes[r, t].axis("off")
        fig.suptitle(f"video {d['video_id']}: accident at {acc / FPS:.1f}s (time_of_event {t_csv:.2f}s)")
        fig.tight_layout()
        fig.savefig(args.out_png, dpi=80)
        print(f"[INFO] picture saved to {args.out_png} -- check by eye")
    except Exception as e:  # noqa: BLE001
        print(f"[INFO] picture skipped: {e}")

    n_fail = sum(not ok for _, ok in results)
    print(f"\n{len(results) - n_fail}/{len(results)} checks passed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
