"""
re20 -- Create the immutable LOCK.json for RQ3 (gate G5) from REAL inputs.

LOCK = (a) the pre-registered design and decision rule the team agreed on, and
(b) fingerprints of everything a full run depends on: code (source digest),
HF revision + train.csv/split hashes, audit, sidecar (incl. cache content
digest), validation-cache content digest, environment/weights identity from
preflight. re14 --full-run refuses when any of these differ at train time;
re15/re19 read the decision fields.

It refuses to write unless the evidence of G1-G4 exists and passed:
prepare_report.json (G1), audit_summary.json + sidecar_summary.json (G2),
preflight.json with --require-cuda (G3), COMPLETE.json of the B/P/S mini runs (G4).
A LOCK is never overwritten: re-running with identical content is a no-op,
anything else is refused.

Usage:
  LOCK_CONFIRMED=1 python reeval/re20_lock_progress.py --out outputs_progress/LOCK.json \
     --prepare outputs_progress/prepare/prepare_report.json --audit outputs_progress/audit/audit_summary.json \
     --sidecar outputs_progress/sidecar/sidecar.json --preflight outputs_progress/preflight/preflight.json \
     --mini-root outputs_progress/dev --cache-5f data/nexar_cache_5f
"""
import argparse
import glob
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import progress_common as pc  # noqa: E402

RQ3 = ("Can event-distance-aware progress supervision improve RiskProp's early collision anticipation "
       "and temporal risk progression while maintaining predictive accuracy under the temporal-pairing "
       "configuration selected in RQ2?")
DESIGN_KEYS = ("pairing_mode", "horizon_sec", "alpha", "lambda_prog", "seeds", "conditions", "epochs")


def design(a):
    return {
        "research_question": RQ3,
        "pairing_mode": a.pairing, "horizon_sec": a.horizon, "alpha": a.alpha, "lambda_prog": a.lam,
        "seeds": [int(s) for s in a.seeds.split(",")], "conditions": a.conditions.split(","),
        "epochs": 50, "batch_size": 2, "lr": 0.01, "momentum": 0.9, "weight_decay": 1e-4,
        "lr_decay_epochs": [20, 40],
        "sensitivity_horizons": [1.5], "sensitivity_conditions": ["P"], "sensitivity_seeds": [42],
        "primary_contrast": "P - B", "primary_metric": "AP@1.5s",
        "ni_metric": "mAP", "ni_margin": 0.02, "ps_metric": "AP@1.5s",
        "success_rule": "RQ3 supported iff P - B on AP@1.5s has 95% CI lower bound > 0 AND P - B on mAP has "
                        "CI lower bound > -0.02 (raw floats, 3 seeds x B/P/S complete, research gate passed). "
                        "P - S on AP@1.5s reported regardless of CI.",
        "supporting_evidence": ["PVR", "ADS", "RCJ", "rise_pos", "score_neg_mean", "seed_stability",
                                "mAUC01", "Coverage"],
        "checkpoint_rule": "trainer keeps best (lowest val BCE @1.0 s) and latest (epoch 50); evaluator picks "
                           "per run the one with higher mean val AP over 0.5/1.0/1.5 s; tie (equal float) -> best",
        "ci": "paired label-stratified video bootstrap, B=2000, seed 12345, accuracy on all 300 val videos, "
              "temporal on all 150 val positives (only if the dense timing gate passes)",
        "test_policy": "official test (1344 clips) scored once by re19 after freezing chosen checkpoints",
        "pairing_rationale": "fixed lag 1.0 s = RiskProp v2 default; RQ2 found no accuracy difference fixed vs "
                             "random (mAP -0.0026 [-0.031, 0.025])",
        "lambda_rationale": "SmoothL1(beta=1) on probabilities = 0.5*(a-r)^2, same algebraic scale as FFR "
                            "(0.5*MSE); fixed before training, not tuned on mini; not PRE-ACT's 10. Gradient "
                            "norms of both terms are logged as a diagnostic only.",
        "adaptation": "PRE-ACT-inspired loss-level adaptation on RiskProp (one sigmoid head, positive-only CPS, "
                      "BCE anchors excluded, tau<=0 excluded); not a reproduction of PRE-ACT",
    }


def evidence(a):
    errs, ev = [], {}

    def need(path, label, cond=lambda j: j.get("pass")):
        if not path or not os.path.exists(path):
            errs.append(f"{label}: missing {path}")
            return None
        j = json.load(open(path))
        if not cond(j):
            errs.append(f"{label}: did not pass ({path})")
        ev[label] = {"path": path, "sha256": pc.sha256_file(path)}
        return j
    prep = need(a.prepare, "G1 prepare")
    audit = need(a.audit, "G2 audit", lambda j: j.get("pass") and j.get("complete"))
    side_sum = need(os.path.join(os.path.dirname(a.sidecar), "sidecar_summary.json"), "G2 sidecar",
                    lambda j: j.get("usable_for_full_run"))
    pre = need(a.preflight, "G3 preflight", lambda j: j.get("pass") and j.get("require_cuda"))
    for c in a.conditions.split(","):
        hits = glob.glob(os.path.join(a.mini_root, f"{c}_*_mini", "COMPLETE.json"))
        if not hits:
            errs.append(f"G4 mini: no COMPLETE.json for condition {c} under {a.mini_root}")
    side = json.load(open(a.sidecar)) if os.path.exists(a.sidecar) else {"meta": {}}
    fp = {
        "code": pc.source_identity(),
        "hf_revision": (prep or {}).get("revision"),
        "train_csv_sha256": (prep or {}).get("train_csv_sha256"),
        "id_map_sha256": (prep or {}).get("id_map_sha256"),
        "split_manifest_sha256": side["meta"].get("split_manifest_sha256"),
        "train_index_sha256": side["meta"].get("train_index_sha256"),
        "train_cache_digest": side["meta"].get("cache_digest"),
        "sidecar_sha256": pc.sha256_file(a.sidecar) if os.path.exists(a.sidecar) else None,
        "audit_sha256": pc.sha256_file(a.audit) if os.path.exists(a.audit) else None,
        "val_cache_digest": pc.val_cache_digest(a.cache_5f) if os.path.exists(a.cache_5f) else None,
        "env": (pre or {}).get("env"), "init_state_sha256_seed0": (pre or {}).get("init_state_sha256_seed0"),
        "hub_weight_files": (pre or {}).get("hub_weight_files"),
    }
    if audit and audit.get("inputs", {}).get("train_csv_sha256") != fp["train_csv_sha256"]:
        errs.append("audit was run on a different train.csv than prepare_report")
    if side_sum and side_sum.get("audit_sha256") and side_sum.get("audit_sha256") != fp["audit_sha256"]:
        errs.append("sidecar was built from a different audit file")
    return errs, ev, fp


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--prepare", required=True)
    ap.add_argument("--audit", required=True)
    ap.add_argument("--sidecar", required=True)
    ap.add_argument("--preflight", required=True)
    ap.add_argument("--mini-root", required=True)
    ap.add_argument("--cache-5f", required=True)
    ap.add_argument("--pairing", default="fixed")
    ap.add_argument("--horizon", type=float, default=2.0)
    ap.add_argument("--alpha", type=float, default=3.0)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--conditions", default="B,P,S")
    a = ap.parse_args(argv)
    if os.environ.get("LOCK_CONFIRMED") != "1":
        pc.fail("set LOCK_CONFIRMED=1 only after the team agreed on the design printed in the README")
    if pc.SYNTHETIC:
        pc.fail("PROGRESS_SYNTHETIC_COUNTS is set: a LOCK is never created on synthetic data")
    errs, ev, fp = evidence(a)
    if errs:
        pc.fail("LOCK refused:\n  - " + "\n  - ".join(errs))
    lock = {"confirmed": True, "version": pc.V4_VERSION, **design(a), "fingerprints": fp, "evidence": ev,
            "git": pc.git_info()}
    content_key = pc.sha256_json({k: v for k, v in lock.items() if k not in ("git",)})
    lock["content_sha256"] = content_key
    if os.path.exists(a.out):
        old = json.load(open(a.out))
        if old.get("content_sha256") == content_key:
            print(f"LOCK unchanged: {a.out}")
            return
        pc.fail(f"{a.out} already exists with different content; LOCK is immutable. Differences in: "
                f"{sorted(k for k in set(old) | set(lock) if old.get(k) != lock.get(k) and k not in ('created', 'git'))}")
    lock["created"] = time.strftime("%Y-%m-%d %H:%M:%S")
    pc.write_json_atomic(a.out, lock)
    with open(a.out + ".sha256", "w") as f:
        f.write(pc.sha256_file(a.out) + "\n")
    print(json.dumps({k: lock[k] for k in DESIGN_KEYS}, indent=2))
    print(f"LOCK written: {a.out} (sha256 in {a.out}.sha256)")


if __name__ == "__main__":
    main()
