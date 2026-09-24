"""
re05 -- Score the submission CSVs written by re04 with Nexar's own
evaluate_submission.py (downloaded from the HF dataset repo, so the
public/private mAP numbers are computed exactly the way the official
leaderboard computes them), then aggregate mean +/- SD over the 3 seeds
of each method (TOP / AdaLEA / RiskProp random / RiskProp fixed).

Usage:
    python reeval/re05_eval_test.py \
        --infer-dir reeval_out/test_infer \
        --hf-local-raw /workspace/CPV301/data/nexar_hf_raw \
        --out-dir reeval_out/test_analysis
"""
import argparse
import glob
import os
import re
import subprocess
import sys

import numpy as np
import pandas as pd

METHODS = {
    "TOP": "top",
    "AdaLEA": "adalea",
    "RiskProp random": "riskprop_random",
    "RiskProp fixed": "riskprop_fixed",
}
SEEDS = [42, 43, 44]


def run_eval(eval_script, sub_csv, cwd):
    r = subprocess.run([sys.executable, eval_script, sub_csv],
                        cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  [ERROR] evaluate_submission.py failed for {sub_csv}:\n{r.stderr}")
        return None, None
    out = r.stdout
    m_pub = re.search(r"mAP \(Public\):\s*([0-9.]+)", out)
    m_priv = re.search(r"mAP \(Private\):\s*([0-9.]+)", out)
    pub = float(m_pub.group(1)) if m_pub else None
    priv = float(m_priv.group(1)) if m_priv else None
    return pub, priv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infer-dir", default="reeval_out/test_infer")
    ap.add_argument("--hf-local-raw", default="/workspace/CPV301/data/nexar_hf_raw")
    ap.add_argument("--out-dir", default="reeval_out/test_analysis")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    eval_script = os.path.join(args.hf_local_raw, "evaluate_submission.py")
    if not os.path.exists(eval_script):
        sys.exit(f"evaluate_submission.py not found at {eval_script} "
                  f"(run re04 first, or check --hf-local-raw).")

    rows = []
    for method, prefix in METHODS.items():
        for seed in SEEDS:
            run_name = f"{prefix}_seed{seed}"
            sub_csv = os.path.join(args.infer_dir, f"submission_{run_name}.csv")
            if not os.path.exists(sub_csv):
                print(f"  [MISSING] {sub_csv}")
                continue
            pub, priv = run_eval(eval_script, os.path.abspath(sub_csv), args.hf_local_raw)
            print(f"  {run_name:26s} public={pub}  private={priv}")
            rows.append({"method": method, "run": run_name, "seed": seed,
                         "mAP_public": pub, "mAP_private": priv})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out_dir, "test_per_run.csv"), index=False)

    agg = (df.groupby("method")[["mAP_public", "mAP_private"]]
           .agg(["mean", "std", "count"]))
    agg.to_csv(os.path.join(args.out_dir, "test_mean_sd.csv"))

    lines = ["# Official Nexar test set results (public + private)\n",
             "| Method | n_seeds | mAP (Public) | mAP (Private) |",
             "|---|---|---|---|"]
    for method in METHODS:
        sub = df[df.method == method]
        if sub.empty:
            continue
        pub_m, pub_s = sub.mAP_public.mean(), sub.mAP_public.std(ddof=0)
        priv_m, priv_s = sub.mAP_private.mean(), sub.mAP_private.std(ddof=0)
        lines.append(f"| {method} | {len(sub)} | {pub_m:.4f} +/- {pub_s:.4f} "
                      f"| {priv_m:.4f} +/- {priv_s:.4f} |")
    lines.append("\nPer-run detail: `test_per_run.csv`. Scored with Nexar's own "
                  "`evaluate_submission.py` against `solution.csv` "
                  "(downloaded from the HF dataset repo, not modified).")
    report = "\n".join(lines)
    with open(os.path.join(args.out_dir, "test_summary.md"), "w") as f:
        f.write(report + "\n")
    print("\n" + report)
    print(f"\nWrote {args.out_dir}/test_summary.md, test_mean_sd.csv, test_per_run.csv")


if __name__ == "__main__":
    main()
