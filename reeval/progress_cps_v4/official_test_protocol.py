"""
official_test_protocol -- Official Nexar test (1,344 clips) for the RQ3 B/P/S runs (gate G8).
infer_official_test / score_official_test are reused (imported), never edited. Three explicit stages; nothing
here runs automatically from training or validation.

  freeze  : read eval_val's chosen_checkpoints.json + rq3_decision.json; require the
            research gate to have passed (full B/P/S x 3 seeds); verify every
            chosen checkpoint's sha256; write an immutable test_freeze.json
            (run -> condition, seed, kind, path, sha256 + LOCK / decision hashes).
  infer   : download the test assets at the PINNED HF revision; take ids from
            sample_submission.csv; require EXACTLY 1,344 unique ids that match the
            test-public + test-private video files one-to-one (no missing, extra or
            duplicate ids); extract the causal 5-frame tail window of every clip
            with infer_official_test.extract_tail_window / infer_official_test.to_tensor (same transforms as
            training/validation); any unreadable clip is FATAL (no 0 / 0.5 fill);
            score every frozen checkpoint (sha re-verified), require finite scores,
            write submission_<run>.csv + infer_receipt.json (hashes).
  score   : once. Nexar's unmodified evaluate_submission.py (via score_official_test.run_official)
            for public / private mAP, plus score_official_test's group-wise re-implementation on
            public (667), private (677) and all; mean +/- SD per condition and
            paired bootstrap CIs for P-B, S-B, P-S. Writes score_receipt.json; a
            second call refuses to re-score unless the submissions are byte-identical
            (prints the stored result) -- a retry can never become a tuning loop.
The decision of RQ3 stays the validation decision of eval_val; the test is a final,
single, reported check.

Usage:
  python reeval/official_test_protocol.py freeze --eval-dir outputs_progress/eval_val \
      --lock-file outputs_progress/LOCK.json --out-dir outputs_progress/test
  python reeval/official_test_protocol.py infer  --out-dir outputs_progress/test --hf-raw data/nexar_hf_test
  python reeval/official_test_protocol.py score  --out-dir outputs_progress/test --hf-raw data/nexar_hf_test
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "pipeline"))
import common as pc  # noqa: E402

CONTRASTS = [("P - B", {"P": 1, "B": -1}), ("S - B", {"S": 1, "B": -1}), ("P - S", {"P": 1, "S": -1})]
TEST_FILES = ["solution.csv", "time_to_accident_test_map.csv", "sample_submission.csv", "evaluate_submission.py"]


def norm_id(x):
    x = str(x).strip()
    return str(int(x)) if x.isdigit() else x


def check_ids(submission_ids, video_map):
    """Exact one-to-one match between submission ids and test video files."""
    errs = []
    ids = [norm_id(i) for i in submission_ids]
    if len(ids) != pc.N_TEST:
        errs.append(f"{len(ids)} submission ids, expected {pc.N_TEST}")
    if len(set(ids)) != len(ids):
        errs.append(f"{len(ids) - len(set(ids))} duplicate submission ids")
    vids = set(video_map)
    missing = sorted(set(ids) - vids)
    extra = sorted(vids - set(ids))
    if missing:
        errs.append(f"{len(missing)} ids without a video file: {missing[:5]}")
    if extra:
        errs.append(f"{len(extra)} video files not in the submission: {extra[:5]}")
    return errs


def list_videos(hf_raw):
    """{norm_id: path}; duplicate ids across public/private are an error."""
    out, dup = {}, []
    for split in ("test-public", "test-private"):
        for root, _, files in os.walk(os.path.join(hf_raw, split)):
            for f in files:
                if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
                    k = norm_id(os.path.splitext(f)[0])
                    if k in out:
                        dup.append(k)
                    out[k] = os.path.join(root, f)
    return out, dup


def stage_freeze(a):
    if pc.SYNTHETIC:
        pc.fail("PROGRESS_SYNTHETIC_COUNTS is set: the official test is never run on synthetic runs")
    chosen_p = os.path.join(a.eval_dir, "chosen_checkpoints.json")
    dec_p = os.path.join(a.eval_dir, "rq3_decision.json")
    for p in (chosen_p, dec_p, a.lock_file):
        if not os.path.exists(p):
            pc.fail(f"missing {p} (run eval_val in research mode first)")
    dec = json.load(open(dec_p))
    if not dec.get("available") or not dec.get("research_gate_passed"):
        pc.fail("eval_val research gate did not pass -> the official test is not run "
                f"({dec.get('reason') or dec.get('research_issues')})")
    chosen = json.load(open(chosen_p))["chosen"]
    runs = {}
    for run, c in chosen.items():
        if not os.path.exists(c["ckpt"]) or pc.sha256_file(c["ckpt"]) != c["ckpt_sha256"]:
            pc.fail(f"{run}: chosen checkpoint missing or changed since validation: {c['ckpt']}")
        cfg = json.load(open(os.path.join(os.path.dirname(c["ckpt"]), "config.json")))["config"]
        runs[run] = {"condition": cfg["condition"], "seed": cfg["seed"], "kind": c["kind"],
                     "ckpt": c["ckpt"], "ckpt_sha256": c["ckpt_sha256"]}
    conds = {(v["condition"], v["seed"]) for v in runs.values()}
    need = {(c, s) for c in ("B", "P", "S") for s in pc.SEEDS_MAIN}
    if not need <= conds:
        pc.fail(f"frozen runs miss {sorted(need - conds)}")
    freeze = {"version": pc.V4_VERSION, "runs": runs, "lock_sha256": pc.sha256_file(a.lock_file),
              "decision_sha256": pc.sha256_file(dec_p), "chosen_sha256": pc.sha256_file(chosen_p),
              "hf_revision": a.revision}
    os.makedirs(a.out_dir, exist_ok=True)
    out = os.path.join(a.out_dir, "test_freeze.json")
    if os.path.exists(out):
        old = json.load(open(out))
        if {k: v for k, v in old.items() if k != "created"} != freeze:
            pc.fail(f"{out} exists with different content; the freeze is immutable")
        print(f"freeze unchanged: {out}")
        return
    freeze["created"] = time.strftime("%Y-%m-%d %H:%M:%S")
    pc.write_json_atomic(out, freeze)
    print(f"frozen {len(runs)} runs -> {out}")


def stage_infer(a):
    import torch
    import infer_official_test as r4
    freeze_p = os.path.join(a.out_dir, "test_freeze.json")
    if not os.path.exists(freeze_p):
        pc.fail("run `freeze` first")
    freeze = json.load(open(freeze_p))
    if not a.skip_download:
        from huggingface_hub import hf_hub_download, snapshot_download
        os.makedirs(a.hf_raw, exist_ok=True)
        for fn in TEST_FILES:
            hf_hub_download(repo_id=pc.HF_REPO, repo_type="dataset", filename=fn, revision=freeze["hf_revision"],
                            local_dir=a.hf_raw)
        snapshot_download(repo_id=pc.HF_REPO, repo_type="dataset", revision=freeze["hf_revision"],
                          local_dir=a.hf_raw, allow_patterns=["test-public/**", "test-private/**"])
    sample = pd.read_csv(os.path.join(a.hf_raw, "sample_submission.csv"), dtype=str)
    id_col, score_col = sample.columns[0], sample.columns[1]
    sub_ids = sample[id_col].astype(str).tolist()
    video_map, dup = list_videos(a.hf_raw)
    errs = check_ids(sub_ids, video_map) + ([f"{len(dup)} duplicate video ids across splits"] if dup else [])
    if errs:
        pc.fail("test id check failed:\n  - " + "\n  - ".join(errs))
    print(f"{len(sub_ids)} test clips, ids match the video files exactly")
    xs, failed = [], []
    t0 = time.time()
    for i, vid in enumerate(sub_ids):
        fr = r4.extract_tail_window(video_map[norm_id(vid)])
        if fr is None:
            failed.append(vid)
            continue
        xs.append(r4.to_tensor(fr, pc.REPO))
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(sub_ids)} tail windows ({time.time()-t0:.0f}s)")
    if failed:
        pc.fail(f"{len(failed)} test clips could not be decoded (no fill values allowed): {failed[:5]}")
    from train_bps import build_model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(a.model).to(device).eval()
    receipt = {"freeze_sha256": pc.sha256_file(freeze_p), "hf_revision": freeze["hf_revision"],
               "n_ids": len(sub_ids), "submissions": {}, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
               "window": "causal 5-frame tail window (infer_official_test.extract_tail_window), adalea_dataset._to_tensor"}
    for run, r in sorted(freeze["runs"].items()):
        if pc.sha256_file(r["ckpt"]) != r["ckpt_sha256"]:
            pc.fail(f"{run}: checkpoint changed after the freeze")
        ck = torch.load(r["ckpt"], map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        model.eval()
        scores = r4.predict(model, xs, device, a.batch_size)
        if len(scores) != pc.N_TEST or not np.all(np.isfinite(scores)):
            pc.fail(f"{run}: {len(scores)} scores, finite={bool(np.all(np.isfinite(scores)))}")
        out = os.path.join(a.out_dir, f"submission_{run}.csv")
        pd.DataFrame({id_col: sub_ids, score_col: scores.astype(float)}).to_csv(out, index=False)
        receipt["submissions"][run] = {"file": os.path.basename(out), "sha256": pc.sha256_file(out),
                                       "condition": r["condition"], "seed": r["seed"]}
        print(f"  [OK] {run}: range [{scores.min():.3f}, {scores.max():.3f}]")
    pc.write_json_atomic(os.path.join(a.out_dir, "infer_receipt.json"), receipt)


def stage_score(a):
    import score_official_test as r5
    rec_p = os.path.join(a.out_dir, "infer_receipt.json")
    if not os.path.exists(rec_p):
        pc.fail("run `infer` first")
    rec = json.load(open(rec_p))
    for run, s in rec["submissions"].items():
        if pc.sha256_file(os.path.join(a.out_dir, s["file"])) != s["sha256"]:
            pc.fail(f"{s['file']} changed after inference")
    score_p = os.path.join(a.out_dir, "score_receipt.json")
    if os.path.exists(score_p):
        old = json.load(open(score_p))
        if old.get("infer_receipt_sha256") == pc.sha256_file(rec_p):
            print("already scored with these exact submissions -> stored result:")
            print(open(os.path.join(a.out_dir, "test_summary.md")).read())
            return
        pc.fail("score_receipt.json exists for different submissions; the official test is scored once")
    raw = os.path.abspath(a.hf_raw)
    sol = pd.read_csv(os.path.join(raw, "solution.csv"), dtype=str)
    for c in ("id", "target", "Usage", "group"):
        if c not in sol.columns:
            pc.fail(f"solution.csv has no {c!r} column")
    sol["_id"] = sol["id"].map(norm_id)
    usage = sol["Usage"].str.lower().values
    counts = {"public": int((usage == "public").sum()), "private": int((usage == "private").sum())}
    if counts != {"public": pc.N_TEST_PUBLIC, "private": pc.N_TEST_PRIVATE}:
        pc.fail(f"solution.csv usage counts {counts} != public 667 / private 677")
    y = sol["target"].astype(float).astype(int).values
    groups = sol["group"].astype(str).values
    eval_script = os.path.join(raw, "evaluate_submission.py")
    os.makedirs(os.path.join(a.out_dir, "eval_stdout"), exist_ok=True)
    rows, scores = [], {}
    for run, s in sorted(rec["submissions"].items()):
        p = os.path.abspath(os.path.join(a.out_dir, s["file"]))
        pub, priv, _ = r5.run_official(eval_script, p, raw, os.path.join(a.out_dir, "eval_stdout", f"{run}.txt"))
        sub = pd.read_csv(p, dtype=str)
        m = dict(zip(sub.iloc[:, 0].map(norm_id), sub.iloc[:, 1].astype(float)))
        missing = [i for i in sol["_id"] if i not in m]
        if missing:
            pc.fail(f"{run}: {len(missing)} solution ids missing from the submission")
        scores[run] = np.array([m[i] for i in sol["_id"]])
        r = {"run": run, "condition": s["condition"], "seed": s["seed"],
             "mAP_public_official": pub, "mAP_private_official": priv}
        for part in ("all", "public", "private"):
            sel = np.arange(len(y)) if part == "all" else np.where(usage == part)[0]
            mm = r5.grouped_metrics(scores[run], y, groups, sel)
            r[f"mAP_{part}"], r[f"mAUC01_{part}"] = mm["mAP"], mm["mAUC01"]
        rows.append(r)
        print(f"  {run}: public={pub} private={priv} all={r['mAP_all']:.4f}")
    per = pd.DataFrame(rows)
    per.to_csv(os.path.join(a.out_dir, "test_per_run.csv"), index=False)
    members = {c: [r for r in rec["submissions"] if rec["submissions"][r]["condition"] == c] for c in ("B", "P", "S")}
    rng = np.random.default_rng(12345)
    cells = [np.where((groups == g) & (y == t))[0] for g in np.unique(groups) for t in (0, 1)]
    cells = [c for c in cells if len(c)]

    def mv(idx):
        return {c: {k: float(np.mean([r5.grouped_metrics(scores[n], y, groups, idx)[k] for n in ns]))
                    for k in ("mAP", "mAUC01")} for c, ns in members.items() if ns}
    full = mv(np.arange(len(y)))
    boot = {n: {"mAP": [], "mAUC01": []} for n, _ in CONTRASTS}
    for _ in range(a.n_boot):
        idx = np.concatenate([rng.choice(c, len(c), replace=True) for c in cells])
        v = mv(idx)
        for n, co in CONTRASTS:
            for k in ("mAP", "mAUC01"):
                boot[n][k].append(sum(w * v[c][k] for c, w in co.items()))
    ci = []
    for n, co in CONTRASTS:
        for k in ("mAP", "mAUC01"):
            est = sum(w * full[c][k] for c, w in co.items())
            lo, hi = np.percentile(boot[n][k], [2.5, 97.5])
            ci.append({"contrast": n, "metric": k, "estimate": est, "ci_low": lo, "ci_high": hi,
                       "verdict": "inconclusive (CI includes 0)" if lo <= 0 <= hi else
                       ("positive (CI > 0)" if lo > 0 else "negative (CI < 0)")})
    ci = pd.DataFrame(ci)
    ci.to_csv(os.path.join(a.out_dir, "test_bootstrap_ci.csv"), index=False)

    def msd(v):
        v = [x for x in v if x is not None and np.isfinite(x)]
        return f"{np.mean(v):.4f} +/- {np.std(v, ddof=1):.4f}" if len(v) > 1 else (f"{v[0]:.4f}" if v else "-")
    tab = pd.DataFrame([{"Condition": c, "n_seeds": int((per.condition == c).sum()),
                         "mAP public (official)": msd(per[per.condition == c].mAP_public_official.tolist()),
                         "mAP private (official)": msd(per[per.condition == c].mAP_private_official.tolist()),
                         "mAP all": msd(per[per.condition == c].mAP_all.tolist()),
                         "mAUC@0.1 all": msd(per[per.condition == c].mAUC01_all.tolist())}
                        for c in ("B", "P", "S")])
    sys.path.insert(0, os.path.join(REPO, "reeval", "pairing_sensitivity"))
    from eval_sensitivity import md_table
    ci_show = ci.copy()
    for k in ("estimate", "ci_low", "ci_high"):
        ci_show[k] = ci_show[k].map(lambda x: f"{x:+.4f}")
    rep = [f"# RQ3 official Nexar test (scored once) -- {time.strftime('%Y-%m-%d %H:%M')}\n",
           f"Public {counts['public']} / private {counts['private']} clips; checkpoints frozen from validation "
           f"(test_freeze.json). The RQ3 decision is the validation decision (eval_val); this is the final report.\n",
           md_table(tab), "", "## Paired bootstrap 95% CI (all 1,344 clips, within group x label cells, B="
           f"{a.n_boot})\n", md_table(ci_show), ""]
    with open(os.path.join(a.out_dir, "test_summary.md"), "w") as f:
        f.write("\n".join(rep))
    pc.write_json_atomic(score_p, {"infer_receipt_sha256": pc.sha256_file(rec_p), "counts": counts,
                                   "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                                   "per_run_sha256": pc.sha256_file(os.path.join(a.out_dir, "test_per_run.csv"))})
    print("\n".join(rep))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["freeze", "infer", "score"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--eval-dir", default="")
    ap.add_argument("--lock-file", default="")
    ap.add_argument("--hf-raw", default=os.path.join(pc.REPO, "data", "nexar_hf_test"))
    ap.add_argument("--revision", default=pc.HF_REVISION)
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--model", default="riskprop", choices=["riskprop", "dummy"])
    a = ap.parse_args(argv)
    {"freeze": stage_freeze, "infer": stage_infer, "score": stage_score}[a.stage](a)


if __name__ == "__main__":
    main()
