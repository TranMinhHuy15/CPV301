"""
preflight -- Environment preflight for Progressive RiskProp v4 (run BEFORE paid
training; part of G3). Never installs or upgrades anything.

Checks (each reported PASS / FAIL / WARN / NOT RUN):
  * python / torch / torchvision / numpy / pandas / sklearn / decord / PIL /
    huggingface_hub importable; versions vs the recorded RTX-4080 environment
    (results/repro/environment_vast_rtx4080.txt: torch 2.11.0+cu128,
    torchvision 0.26.0+cu128) -- a different version is WARN, not silently ok;
  * CUDA: available, device name, memory, a matmul + autocast step;
  * backbone: torch.hub 'facebookresearch/pytorchvideo' slow_r50 pretrained
    loads; RiskPropModel builds; sha256 of the initial state (weights identity)
    is recorded so every run can be tied to the same pretrained weights;
  * forward + backward of RiskPropModel on a (2, 12, 3, 5, 224, 224) batch
    under AMP, peak memory recorded;
  * disk free under --out-dir, HF token presence (never printed).
`--require-cuda` (used before smoke/mini/full) turns missing CUDA into FAIL.

Usage: python reeval/preflight.py --out-dir outputs_progress/preflight --require-cuda
"""
import argparse
import importlib
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as pc  # noqa: E402

RECORDED = {"python": "3.12", "torch": "2.11.0+cu128", "torchvision": "0.26.0+cu128",
            "decord": "0.6.0", "numpy": "2.5.3", "pandas": "3.0.6", "sklearn": "1.9.1"}


def check(results, name, fn):
    t0 = time.time()
    try:
        status, detail = fn()
    except Exception as e:
        status, detail = "FAIL", f"{type(e).__name__}: {str(e)[:300]}"
    results.append({"check": name, "status": status, "detail": detail, "sec": round(time.time() - t0, 1)})
    print(f"[{status:7s}] {name}: {detail}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--require-cuda", action="store_true")
    ap.add_argument("--min-free-gb", type=float, default=60.0)
    ap.add_argument("--skip-backbone", action="store_true", help="CPU-only environments / tests")
    a = ap.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)
    res, info = [], {"env": pc.env_identity(), "recorded_env": RECORDED}

    def versions():
        bad, got = [], {}
        for mod, key in (("torch", "torch"), ("torchvision", "torchvision"), ("numpy", "numpy"),
                         ("pandas", "pandas"), ("sklearn", "sklearn"), ("decord", "decord"),
                         ("PIL", "PIL"), ("huggingface_hub", "huggingface_hub")):
            m = importlib.import_module(mod)
            got[key] = getattr(m, "__version__", "?")
            if key in RECORDED and got[key] != RECORDED[key]:
                bad.append(f"{key} {got[key]} (recorded {RECORDED[key]})")
        info["versions"] = got
        py = info["env"]["python"]
        if not py.startswith(RECORDED["python"]):
            bad.append(f"python {py} (recorded {RECORDED['python']})")
        return ("WARN" if bad else "PASS"), ("differs from recorded env: " + "; ".join(bad)) if bad else "matches recorded env"
    check(res, "imports+versions", versions)

    def cuda():
        import torch
        if not torch.cuda.is_available():
            return ("FAIL" if a.require_cuda else "NOT RUN"), "torch.cuda.is_available() is False"
        x = torch.randn(512, 512, device="cuda")
        with torch.autocast("cuda"):
            y = (x @ x).float().sum().item()
        p = torch.cuda.get_device_properties(0)
        return "PASS", f"{p.name}, {p.total_memory / 2**30:.1f} GB, matmul ok ({y:.1f})"
    check(res, "cuda", cuda)

    def disk():
        free = shutil.disk_usage(a.out_dir).free / 2**30
        info["disk_free_gb"] = round(free, 1)
        return ("PASS" if free >= a.min_free_gb else "FAIL"), f"{free:.1f} GB free under {a.out_dir} (need >= {a.min_free_gb})"
    check(res, "disk", disk)

    def token():
        has = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
        return ("PASS" if has else "WARN"), ("HF token set (not printed)" if has else "no HF_TOKEN in env (needed to download)")
    check(res, "hf_token", token)

    def backbone():
        if a.skip_backbone:
            return "NOT RUN", "--skip-backbone"
        import torch
        sys.path.insert(0, os.path.join(pc.REPO, "pipeline"))
        from riskprop_model import RiskPropModel
        torch.manual_seed(0)
        m = RiskPropModel()
        info["init_state_sha256_seed0"] = pc.model_state_digest(m)
        hub_dir = torch.hub.get_dir()
        ck = [os.path.join(r, f) for r, _, fs in os.walk(hub_dir) for f in fs if f.endswith((".pyth", ".pth"))]
        info["hub_weight_files"] = {os.path.basename(p): pc.sha256_file(p) for p in ck}
        if not torch.cuda.is_available():
            return ("FAIL" if a.require_cuda else "PASS"), "model built on CPU (no CUDA step)"
        m = m.cuda().train()
        torch.cuda.reset_peak_memory_stats()
        x = torch.randn(2, 12, 3, 5, 224, 224, device="cuda")
        with torch.autocast("cuda"):
            z = m(x)
        z.float().sum().backward()
        peak = torch.cuda.max_memory_allocated() / 2**30
        info["peak_mem_gb_fwd_bwd_batch2"] = round(peak, 2)
        return "PASS", f"slow_r50 pretrained loaded, fwd+bwd (2x12 snippets) peak {peak:.2f} GB"
    check(res, "backbone", backbone)

    hard_fail = [r for r in res if r["status"] == "FAIL"]
    out = {"version": pc.V4_VERSION, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
           "require_cuda": a.require_cuda, "pass": not hard_fail, "checks": res, **info}
    pc.write_json_atomic(os.path.join(a.out_dir, "preflight.json"), out)
    print(f"PREFLIGHT {'PASS' if out['pass'] else 'FAIL'} -> {os.path.join(a.out_dir, 'preflight.json')}")
    sys.exit(0 if out["pass"] else 2)


if __name__ == "__main__":
    main()
