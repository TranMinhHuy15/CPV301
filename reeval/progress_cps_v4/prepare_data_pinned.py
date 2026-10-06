"""
prepare_data_pinned -- Deterministic, revision-pinned replacement for data_download_from_hf's data step
(data_download_from_hf itself is kept unchanged). Gate G1 of v4.

Why: data_download_from_hf downloads the HF `main` branch and numbers videos by enumeration
order of the `datasets` loader. If the HF repo or the loader order changes,
id 00123 silently becomes another video. prepare_data_pinned instead
  1. downloads ONLY train/** at a PINNED revision (default = the audited
     revision 7535d065...);
  2. maps ids to files through results/repro/id_to_hf_source.csv (the mapping
     the RQ1-RQ3 results were produced with), never through enumeration;
  3. reads time_of_event / time_of_alert from the CSV metadata shipped in
     train/** and writes data/nexar_kaggle_style/train.csv with data_download_from_hf's exact
     columns (id,target,time_of_event,time_of_alert; LF line endings);
  4. verifies: 1500 rows, 750/750 labels, every mapped file present, labels
     agree with the folder, positives have an event and alert <= event,
     negatives have none, train.csv sha256 == split manifest, the seed-42
     stratified split recomputes to the manifest train/val ids in order.
Any failure is FATAL before any heavy cache is built. Nothing is "fixed" by
editing the manifest or dropping videos.

Usage:
  python reeval/prepare_data_pinned.py --raw-dir data/nexar_hf_raw \
      --out-dir data/nexar_kaggle_style --report-dir outputs_progress/prepare
  (add --skip-download to only verify an existing download / data_download_from_hf output)
"""
import argparse
import glob
import io
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as pc  # noqa: E402

NAME_COLS = ("file_name", "filename", "video", "video_path", "path", "file", "id")


def read_metadata(raw_dir):
    """All CSVs under raw_dir/train -> {'train/<label>/<basename>': (toe, toa, label)}."""
    out, used = {}, []
    for csv_path in sorted(glob.glob(os.path.join(raw_dir, "train", "**", "*.csv"), recursive=True)):
        df = pd.read_csv(csv_path)
        cols = {c.lower(): c for c in df.columns}
        name_col = next((cols[c] for c in NAME_COLS if c in cols), None)
        if name_col is None or "time_of_event" not in cols:
            continue
        rel_dir = os.path.relpath(os.path.dirname(csv_path), raw_dir).replace("\\", "/")
        used.append(os.path.relpath(csv_path, raw_dir))
        for _, r in df.iterrows():
            name = str(r[name_col]).replace("\\", "/")
            base = os.path.basename(name)
            if not base.lower().endswith(".mp4"):
                base = f"{base}.mp4" if base.isdigit() or "." not in base else base
            # label from the path of the file (name may already contain positive/negative)
            full = name if "/" in name else f"{rel_dir}/{base}"
            label = "positive" if "/positive/" in f"/{full}/" else "negative" if "/negative/" in f"/{full}/" else None
            if label is None:
                raise ValueError(f"cannot infer label for {name} in {csv_path}")
            key = f"train/{label}/{base}"
            toe = r[cols["time_of_event"]]
            toa = r[cols["time_of_alert"]] if "time_of_alert" in cols else np.nan
            if key in out:
                raise ValueError(f"duplicate metadata row for {key} ({csv_path})")
            out[key] = (toe, toa, label)
    return out, used


def build_train_csv(id_map, meta):
    rows = []
    for _, r in id_map.iterrows():
        vid = int(os.path.splitext(r["id_file"])[0])
        src = r["hf_source"].replace("\\", "/")
        label = "positive" if "/positive/" in src else "negative" if "/negative/" in src else None
        if label is None:
            raise ValueError(f"mapping row {r['id_file']} has no label folder: {src}")
        if src not in meta:
            raise KeyError(f"no metadata for {src}")
        toe, toa, mlabel = meta[src]
        if mlabel != label:
            raise ValueError(f"{src}: metadata label {mlabel} != folder {label}")
        rows.append({"id": vid, "target": 1 if label == "positive" else 0,
                     "time_of_event": toe if label == "positive" else np.nan,
                     "time_of_alert": toa if label == "positive" else np.nan})
    df = pd.DataFrame(rows).sort_values("id").reset_index(drop=True)
    df["time_of_event"] = pd.to_numeric(df["time_of_event"], errors="coerce")
    df["time_of_alert"] = pd.to_numeric(df["time_of_alert"], errors="coerce")
    return df


def csv_bytes(df):
    buf = io.StringIO()
    df.to_csv(buf, index=False, lineterminator="\n")
    return buf.getvalue().encode()


def check_train_df(df):
    errs = []
    if len(df) != pc.N_TRAIN_CSV:
        errs.append(f"{len(df)} rows, expected {pc.N_TRAIN_CSV}")
    if int((df.target == 1).sum()) != pc.N_POS_CSV or int((df.target == 0).sum()) != pc.N_TRAIN_CSV - pc.N_POS_CSV:
        errs.append(f"labels {int((df.target == 1).sum())}/{int((df.target == 0).sum())}, expected 750/750")
    if df["id"].duplicated().any():
        errs.append("duplicate ids")
    pos, neg = df[df.target == 1], df[df.target == 0]
    if pos["time_of_event"].isna().any() or (pos["time_of_event"] < 0).any():
        errs.append(f"{int(pos['time_of_event'].isna().sum())} positives without event / negative event")
    if pos["time_of_alert"].isna().any():
        errs.append(f"{int(pos['time_of_alert'].isna().sum())} positives without alert")
    if (pos["time_of_alert"] > pos["time_of_event"]).any():
        errs.append("alert after event for some positives")
    if neg["time_of_event"].notna().any() or neg["time_of_alert"].notna().any():
        errs.append("negatives carry event/alert times")
    return errs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True, help="where train/** is (or will be) downloaded")
    ap.add_argument("--out-dir", required=True, help="kaggle-style dir: train.csv + train/<id>.mp4")
    ap.add_argument("--report-dir", required=True)
    ap.add_argument("--id-map", default=os.path.join(pc.REPO, "results/repro/id_to_hf_source.csv"))
    ap.add_argument("--split-manifest", default=os.path.join(pc.REPO, "results/repro/split_manifest_seed42.json"))
    ap.add_argument("--revision", default=pc.HF_REVISION)
    ap.add_argument("--skip-download", action="store_true")
    a = ap.parse_args(argv)
    os.makedirs(a.report_dir, exist_ok=True)
    t0 = time.time()
    report = {"stage": "G1 metadata", "revision": a.revision, "hf_repo": pc.HF_REPO,
              "id_map": a.id_map, "id_map_sha256": pc.sha256_file(a.id_map),
              "split_manifest_sha256": pc.sha256_file(a.split_manifest), "fatal": []}

    if not a.skip_download:
        from huggingface_hub import snapshot_download
        print(f"downloading {pc.HF_REPO} train/** @ {a.revision} -> {a.raw_dir}")
        snapshot_download(repo_id=pc.HF_REPO, repo_type="dataset", revision=a.revision,
                          local_dir=a.raw_dir, allow_patterns=["train/**"])
        report["downloaded"] = True

    id_map = pd.read_csv(a.id_map, dtype=str)
    man = json.load(open(a.split_manifest))
    if len(id_map) != pc.N_TRAIN_CSV or id_map["id_file"].duplicated().any() or id_map["hf_source"].duplicated().any():
        report["fatal"].append("id map must have 1500 unique ids and 1500 unique sources")
    missing = [s for s in id_map["hf_source"] if not os.path.exists(os.path.join(a.raw_dir, s))]
    if missing:
        report["fatal"].append(f"{len(missing)} mapped videos missing in {a.raw_dir}: {missing[:5]}")
    n_mp4 = len(glob.glob(os.path.join(a.raw_dir, "train", "**", "*.mp4"), recursive=True))
    report["n_mp4_in_raw"] = n_mp4
    if n_mp4 != pc.N_TRAIN_CSV:
        report["fatal"].append(f"{n_mp4} .mp4 under {a.raw_dir}/train, expected {pc.N_TRAIN_CSV}")

    df = None
    try:
        meta, used = read_metadata(a.raw_dir)
        report["metadata_csvs"] = used
        if not used:
            raise ValueError("no metadata CSV with a file-name column and time_of_event under train/")
        df = build_train_csv(id_map, meta)
        report["fatal"] += check_train_df(df)
    except Exception as e:
        report["fatal"].append(f"metadata: {e}")

    out_csv = os.path.join(a.out_dir, "train.csv")
    if df is not None and not report["fatal"]:
        data = csv_bytes(df)
        sha = __import__("hashlib").sha256(data).hexdigest()
        report["train_csv_sha256_rebuilt"] = sha
        if sha != man["train_csv_sha256"]:
            report["fatal"].append(f"rebuilt train.csv sha256 {sha[:16]} != manifest "
                                   f"{man['train_csv_sha256'][:16]} (metadata at this revision differs "
                                   f"from the one the split was made on)")
        else:
            sys.path.insert(0, HERE)
            from audit_data import recompute_split
            tr, va = recompute_split(df)
            if tr != man["train_ids"] or va != man["val_ids"]:
                report["fatal"].append("seed-42 split recomputed from the rebuilt train.csv != manifest")
    if not report["fatal"]:
        os.makedirs(os.path.join(a.out_dir, "train"), exist_ok=True)
        if os.path.exists(out_csv) and open(out_csv, "rb").read() != data:
            report["fatal"].append(f"{out_csv} exists with different content; not overwritten "
                                   f"(move it away if it came from an unpinned data_download_from_hf run)")
        else:
            with open(out_csv, "wb") as f:
                f.write(data)
            linked, wrong = 0, []
            for _, r in id_map.iterrows():
                dst = os.path.join(a.out_dir, "train", r["id_file"])
                src = os.path.abspath(os.path.join(a.raw_dir, r["hf_source"]))
                if os.path.lexists(dst):
                    if os.path.realpath(dst) != os.path.realpath(src):
                        wrong.append(r["id_file"])
                    continue
                os.symlink(src, dst)
                linked += 1
            report["symlinks_created"] = linked
            if wrong:
                report["fatal"].append(f"{len(wrong)} existing {a.out_dir}/train files point to other "
                                       f"sources: {wrong[:5]}")
    report["pass"] = not report["fatal"]
    report["train_csv_sha256"] = pc.sha256_file(out_csv) if os.path.exists(out_csv) else None
    report["seconds"] = round(time.time() - t0, 1)
    pc.write_json_atomic(os.path.join(a.report_dir, "prepare_report.json"), report)
    print(json.dumps(report, indent=2, default=str))
    sys.exit(0 if report["pass"] else 2)


if __name__ == "__main__":
    main()
