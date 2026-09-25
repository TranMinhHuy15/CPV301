"""
re05 -- Score the official Nexar test submissions written by re04.

PRIMARY (official): Nexar's own evaluate_submission.py (downloaded unmodified
from the HF dataset repo), run exactly as its README says:
    python evaluate_submission.py <submission.csv>      (cwd = folder with solution.csv)
-> mAP for the public and the private subset. Raw stdout of every call is
saved (eval_stdout/) and parsed case-insensitively.

SECONDARY (our re-implementation of the OFFICIAL definition, read from
evaluate_submission.py): within each Usage subset, AP is computed separately
inside every solution.csv "group" and averaged over groups (fillna(0) for
missing ids). We reproduce exactly that (checked against the official
numbers) and add, on the same groups, the partial AUC at FPR <= 0.1
(mAUC^0.1, mean over groups). Each group is labelled with the horizon of its
positives via time_to_accident_test_map.csv (wide format: columns 0.5/1.0/1.5,
each row = one event clipped at the three horizons).
Paired, label-stratified bootstrap 95% CIs (B=2000, seed 12345, same resampled
test videos for every run; method value = mean over its 3 seeds):
    RQ1: TOP-AdaLEA, TOP-RiskProp random, AdaLEA-RiskProp random
    RQ3: RiskProp fixed - RiskProp random
    RQ2 (only if re04 ran with --with-rq2; exploratory): B-A, C-A, D-C, D-B, interaction

Column names of solution.csv / time_to_accident_test_map.csv are detected
from their headers and printed; if detection fails the secondary part is
skipped and the official part still runs.

Usage:
    python reeval/re05_eval_test.py
"""
import argparse
import os
import re
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import auc, average_precision_score, roc_curve

METHODS = {  # label -> run-name prefix (run = prefix + _seed{S} [+ suffix])
    "TOP": ("top", ""),
    "AdaLEA": ("adalea", ""),
    "RiskProp random": ("riskprop_random", ""),
    "RiskProp fixed": ("riskprop_fixed", ""),
    "RQ2 A neither": ("riskprop_random", "_neither"),
    "RQ2 B FFR-only": ("riskprop_random", "_ffronly"),
    "RQ2 C AMC-only": ("riskprop_random", "_amconly"),
}
SEEDS = [42, 43, 44]
HORIZONS = [0.5, 1.0, 1.5]
TARGET_FAR = 0.1
RQ1_RQ3 = [("TOP", "AdaLEA"), ("TOP", "RiskProp random"),
           ("AdaLEA", "RiskProp random"), ("RiskProp fixed", "RiskProp random")]
RQ2 = [("FFR effect, AMC absent (B-A)", {"RQ2 B FFR-only": 1, "RQ2 A neither": -1}),
       ("AMC effect, FFR absent (C-A)", {"RQ2 C AMC-only": 1, "RQ2 A neither": -1}),
       ("FFR effect, AMC present (D-C)", {"RiskProp random": 1, "RQ2 C AMC-only": -1}),
       ("AMC effect, FFR present (D-B)", {"RiskProp random": 1, "RQ2 B FFR-only": -1}),
       ("Interaction (D-C)-(B-A)", {"RiskProp random": 1, "RQ2 C AMC-only": -1,
                                    "RQ2 B FFR-only": -1, "RQ2 A neither": 1})]


def norm_id(x):
    x = str(x).strip()
    return str(int(x)) if x.isdigit() else x


def pick(cols, *keys):
    for k in keys:
        for c in cols:
            if k in c.lower():
                return c
    return None


def run_official(eval_script, sub_csv, cwd, save_to):
    r = subprocess.run([sys.executable, eval_script, sub_csv], cwd=cwd,
                       capture_output=True, text=True)
    with open(save_to, "w") as f:
        f.write(r.stdout + "\n--- stderr ---\n" + r.stderr)
    if r.returncode != 0:
        print(f"  [ERROR] evaluate_submission.py failed ({os.path.basename(sub_csv)}):\n{r.stderr[-800:]}")
        return None, None, r.stdout
    num = r"([0-9]*\.[0-9]+|[01])"
    pub = re.search(r"public[^0-9\n]*" + num, r.stdout, re.I)
    priv = re.search(r"private[^0-9\n]*" + num, r.stdout, re.I)
    return (float(pub.group(1)) if pub else None,
            float(priv.group(1)) if priv else None, r.stdout)


def low_far_auc(y, s):
    fpr, tpr, _ = roc_curve(y, s)
    mask = fpr <= TARGET_FAR
    fpr_low = np.append(fpr[mask], TARGET_FAR)
    tpr_low = np.append(tpr[mask], np.interp(TARGET_FAR, fpr, tpr))
    return auc(fpr_low, tpr_low) / TARGET_FAR


def grouped_metrics(scores, y, groups, idx=None):
    """Official definition: AP inside each group, mean over groups; plus
    partial AUC (FPR<=0.1) on the same groups. idx = resample (with repeats)."""
    if idx is not None:
        scores, y, groups = scores[idx], y[idx], groups[idx]
    aps, maucs = {}, {}
    for g in np.unique(groups):
        m = groups == g
        yy, ss = y[m], scores[m]
        if yy.sum() == 0 or yy.sum() == len(yy):
            continue
        aps[g] = average_precision_score(yy, ss)
        maucs[g] = low_far_auc(yy, ss)
    return {"mAP": float(np.mean(list(aps.values()))),
            "mAUC01": float(np.mean(list(maucs.values()))),
            **{f"AP_g{g}": float(v) for g, v in aps.items()}}


def fmt(v):
    v = [x for x in v if x is not None and not np.isnan(x)]
    if not v:
        return "-"
    return f"{np.mean(v):.4f} +/- {np.std(v, ddof=1) if len(v) > 1 else 0.0:.4f}"


def md_table(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infer-dir", default="reeval_out/test_infer")
    ap.add_argument("--hf-local-raw", default="/workspace/CPV301/data/nexar_hf_raw")
    ap.add_argument("--out-dir", default="reeval_out/test_analysis")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--boot-seed", type=int, default=12345)
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out_dir, "eval_stdout"), exist_ok=True)
    raw = os.path.abspath(args.hf_local_raw)
    eval_script = os.path.join(raw, "evaluate_submission.py")
    if not os.path.exists(eval_script):
        sys.exit(f"evaluate_submission.py not found at {eval_script} (run re04 first).")

    # ---- 0. sanity: the README command on the sample submission ----
    print("[0] Sanity run: python evaluate_submission.py sample_submission.csv")
    _, _, out = run_official(eval_script, os.path.join(raw, "sample_submission.csv"), raw,
                             os.path.join(args.out_dir, "eval_stdout", "sample_submission.txt"))
    print(out.strip()[-600:] or "(no stdout)")

    # ---- runs present ----
    runs = {}
    for label, (prefix, suffix) in METHODS.items():
        for s in SEEDS:
            name = f"{prefix}_seed{s}{suffix}"
            p = os.path.join(args.infer_dir, f"submission_{name}.csv")
            if os.path.exists(p):
                runs[name] = (label, s, p)
    print(f"\n{len(runs)} submission files found")

    # ---- 1. official scorer ----
    print("\n[1] Official evaluate_submission.py")
    rows = []
    for name, (label, s, p) in runs.items():
        pub, priv, _ = run_official(eval_script, os.path.abspath(p), raw,
                                    os.path.join(args.out_dir, "eval_stdout", f"{name}.txt"))
        print(f"  {name:34s} public={pub}  private={priv}")
        rows.append({"method": label, "run": name, "seed": s,
                     "mAP_public_official": pub, "mAP_private_official": priv})
    per_run = pd.DataFrame(rows)

    # ---- 2. secondary: own metrics ----
    sec_ok = False
    group_label = {}
    try:
        sol = pd.read_csv(os.path.join(raw, "solution.csv"), dtype=str)
        tmap = pd.read_csv(os.path.join(raw, "time_to_accident_test_map.csv"), dtype=str)
        print("\n[2] solution.csv columns:", list(sol.columns),
              "| time_to_accident map columns:", list(tmap.columns))
        for c in ("id", "target", "Usage", "group"):
            if c not in sol.columns:
                raise ValueError(f"solution.csv has no {c!r} column")
        # wide map: column = horizon, cell = id of the clip cut at that horizon
        tta_of = {}
        for c in tmap.columns:
            try:
                h = float(c)
            except ValueError:
                continue
            for v in tmap[c].dropna():
                tta_of[norm_id(v)] = h
        sol["_id"] = sol["id"].map(norm_id)
        y = sol["target"].astype(float).astype(int).values
        groups = sol["group"].astype(str).values
        usage = sol["Usage"].str.lower().values
        tta = np.array([tta_of.get(i, np.nan) for i in sol["_id"]])
        for g in np.unique(groups):
            hs = pd.Series(tta[(groups == g) & (y == 1)]).dropna()
            group_label[g] = (f"{hs.mode().iloc[0]}s" if len(hs) else "?")
        print(f"    {len(sol)} test videos | positives={int(y.sum())} | groups: "
              + ", ".join(f"g{g}->{group_label[g]} (pos={int(((groups == g) & (y == 1)).sum())}, "
                          f"neg={int(((groups == g) & (y == 0)).sum())})" for g in np.unique(groups)))
        scores = {}
        for name, (label, s, p) in runs.items():
            sub = pd.read_csv(p, dtype=str)
            m = dict(zip(sub.iloc[:, 0].map(norm_id), sub.iloc[:, 1].astype(float)))
            scores[name] = np.array([m.get(i, 0.0) for i in sol["_id"]])   # fillna(0) as official
        own_rows = []
        for name, (label, s, p) in runs.items():
            r = {"run": name}
            for part in ("all", "public", "private"):
                sel = np.arange(len(y)) if part == "all" else np.where(usage == part)[0]
                mm = grouped_metrics(scores[name], y, groups, sel)
                r.update({f"{k}_{part}": v for k, v in mm.items()})
            own_rows.append(r)
        per_run = per_run.merge(pd.DataFrame(own_rows), on="run", how="left")
        sec_ok = True
    except Exception as e:
        print(f"\n[2] Secondary metrics skipped: {e}")
    per_run.to_csv(os.path.join(args.out_dir, "test_per_run.csv"), index=False)

    # ---- 3. tables ----
    rep = [f"# Official Nexar test set -- generated {time.strftime('%Y-%m-%d %H:%M')}\n",
           "Checkpoints and TOP head rule LOCKED on the internal validation split "
           "(results/reeval_corrected/locked_config.json; RQ2 runs: rq2_chosen_checkpoints.json). "
           "RQ2 rows are exploratory.\n",
           "## 1. Official scorer (Nexar evaluate_submission.py, unmodified) -- PRIMARY\n"]
    t1 = []
    for label in METHODS:
        sub = per_run[per_run.method == label]
        if sub.empty:
            continue
        t1.append({"Method": label, "n_seeds": len(sub),
                   "mAP public": fmt(sub.mAP_public_official.tolist()),
                   "mAP private": fmt(sub.mAP_private_official.tolist())})
    rep += [md_table(pd.DataFrame(t1)), ""]

    if sec_ok:
        gcols = sorted(c for c in per_run.columns if c.startswith("AP_g") and c.endswith("_all"))
        cols = ["mAP_all"] + gcols + ["mAUC01_all"]
        t2 = []
        for label in METHODS:
            sub = per_run[per_run.method == label]
            if sub.empty:
                continue
            row = {"Method": label}
            for c in cols:
                key = c[:-4]
                if key.startswith("AP_g"):
                    key = f"AP {key[3:]} ({group_label.get(key[4:], '?')})"
                row[key] = fmt(sub[c].tolist())
            t2.append(row)
        rep += ["## 2. Re-implemented official metric on the whole test set (public + private)\n",
                "AP inside each solution.csv group, averaged over groups (official definition); "
                "mAUC0.1 = partial AUC at FPR<=0.1 on the same groups. Group labels = horizon of "
                "their positives.\n",
                md_table(pd.DataFrame(t2)), ""]
        chk = per_run.dropna(subset=["mAP_public_official", "mAP_private_official"])
        if "mAP_public" in chk.columns and len(chk):
            d = max((chk["mAP_public"] - chk["mAP_public_official"]).abs().max(),
                    (chk["mAP_private"] - chk["mAP_private_official"]).abs().max())
            rep.append(f"Check: max |our mAP - official mAP| over runs and both subsets = {d:.6f} "
                       f"({'identical to the official scorer' if d < 1e-4 else 'DIFFERS -- trust the official column'}).\n")

        # ---- 4. bootstrap ----
        print(f"\n[3] Paired stratified bootstrap on the test set, B={args.n_boot}")
        rng = np.random.default_rng(args.boot_seed)
        cells = [np.where((groups == g) & (y == t))[0]
                 for g in np.unique(groups) for t in (0, 1)]
        cells = [c for c in cells if len(c)]
        members = {lab: [n for n, (l, s, p) in runs.items() if l == lab] for lab in METHODS}
        members = {k: v for k, v in members.items() if v}

        def mvals(idx):
            out = {}
            for lab, names in members.items():
                ms = [grouped_metrics(scores[n], y, groups, idx) for n in names]
                out[lab] = {k: float(np.mean([m[k] for m in ms])) for k in ("mAP", "mAUC01")}
            return out

        full = mvals(np.arange(len(y)))
        comps = [(f"{a} - {b}", {a: 1, b: -1}) for a, b in RQ1_RQ3] + RQ2
        comps = [(n, c) for n, c in comps if all(k in members for k in c)]
        boot = {n: {"mAP": [], "mAUC01": []} for n, _ in comps}
        t0 = time.time()
        for b in range(args.n_boot):
            idx = np.concatenate([rng.choice(c, len(c), replace=True) for c in cells])
            mv = mvals(idx)
            for n, c in comps:
                for k in ("mAP", "mAUC01"):
                    boot[n][k].append(sum(w * mv[m][k] for m, w in c.items()))
            if (b + 1) % 250 == 0:
                print(f"  {b+1}/{args.n_boot} ({time.time()-t0:.0f}s)")
        ci = []
        for n, c in comps:
            for k in ("mAP", "mAUC01"):
                est = sum(w * full[m][k] for m, w in c.items())
                lo, hi = np.percentile(boot[n][k], [2.5, 97.5])
                verdict = "inconclusive (CI includes 0)" if lo <= 0 <= hi else (
                    "positive (CI > 0)" if lo > 0 else "negative (CI < 0)")
                ci.append({"comparison": n, "metric": k, "estimate": round(est, 4),
                           "ci_low": round(lo, 4), "ci_high": round(hi, 4), "verdict": verdict})
        ci_df = pd.DataFrame(ci)
        ci_df.to_csv(os.path.join(args.out_dir, "test_bootstrap_ci.csv"), index=False)
        rep += [f"## 3. Paired bootstrap 95% CI on the whole test set (B={args.n_boot}, "
                "resampling within every group x label cell)\n",
                "Estimate = first minus second; value per method = mean over its 3 seeds. "
                "RQ2 rows exploratory.\n", md_table(ci_df), ""]

    report = "\n".join(rep)
    with open(os.path.join(args.out_dir, "test_summary.md"), "w") as f:
        f.write(report + "\n")
    print("\n" + report)
    print(f"\nWritten to {args.out_dir}")


if __name__ == "__main__":
    main()
