"""
progress_common -- shared helpers for the Progressive RiskProp (RQ3) v4 scripts.
Pure python / numpy at import time (torch is imported lazily), so CPU tests and
CLI --help work without a GPU.

Contents
  * constants pinned for v4 (HF revision, split counts, PRE-ACT reference)
  * hashing: sha256_file, sha256_json, digest_files (content digest of many files)
  * atomic writes: write_json_atomic, torch_save_atomic (tmp -> fsync -> verify -> replace)
  * source_identity: sha256 of every source file a run depends on (code fingerprint)
  * env_identity: python / torch / CUDA / GPU description
  * restore_rng: put RNG state back on the right device after torch.load
"""
import hashlib
import json
import os
import platform
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

V4_VERSION = "v4.0.0"
HF_REPO = "nexar-ai/nexar_collision_prediction"
HF_REVISION = "7535d0656dac31d7da2846913bb69c6331d2a70a"      # audited 2026-10-05
PREACT_REF = "giddyyupp/PRE-ACT@5e0d38fb3eac80b145615b782494bac9be861fe3"
BASE_COMMIT = "cf7de850ae1159dbe6b88cd223244d8656810dbc"      # GitHub base the v4 overlay applies to

N_TRAIN_CSV, N_POS_CSV = 1500, 750
N_TRAIN, N_VAL, N_VAL_POS = 1200, 300, 150
N_TEST, N_TEST_PUBLIC, N_TEST_PRIVATE = 1344, 667, 677
TRAIN_CSV_SHA256 = "86cf32b23ad359b92b02f7779f786f37a8917d119d6b67010c753d6fa8bd146d"
# Synthetic plumbing tests only (CPU dry runs on tiny fake data): override the expected
# counts. Everything produced while this is set is marked "synthetic": true and can never
# pass the research gate, the LOCK or the official-test freeze.
_SYN = os.environ.get("PROGRESS_SYNTHETIC_COUNTS", "").strip()
SYNTHETIC = bool(_SYN)
if SYNTHETIC:
    N_TRAIN_CSV, N_POS_CSV, N_TRAIN, N_VAL, N_VAL_POS = (int(x) for x in _SYN.split(","))
SEEDS_MAIN = (42, 43, 44)

# Files whose content defines a training run (code fingerprint).
SOURCE_FILES = [
    "pipeline/cell22_adalea_dataset.py", "pipeline/cell31_model_riskprop.py",
    "pipeline/cell32_riskprop_loss.py", "pipeline/cell33_riskprop_dataset.py",
    "reeval/progress_common.py", "reeval/re12_build_progress_sidecar.py",
    "reeval/re13_progress_supervision.py", "reeval/re14_train_progress_riskprop.py",
]
# Baseline files that v4 must not modify (checked by re16 / release manifest).
BASELINE_FILES = [
    "pipeline/cell09_prepare_hf_data.py", "pipeline/cell10_precache_5f.py",
    "pipeline/cell22_adalea_dataset.py", "pipeline/cell30_precache_riskprop.py",
    "pipeline/cell31_model_riskprop.py", "pipeline/cell32_riskprop_loss.py",
    "pipeline/cell33_riskprop_dataset.py", "pipeline/cell34_train_riskprop_cached.py",
    "reeval/re01_precache_val.py", "reeval/re02_infer_val.py", "reeval/re03_analyze_rq.py",
    "reeval/re04_infer_test.py", "reeval/re05_eval_test.py", "reeval/re07_infer_rq2.py",
    "reeval/re10_rq3_sensitivity_eval.py",
]


# ----------------------------------------------------------------------------
# Hashing
# ----------------------------------------------------------------------------
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def digest_files(paths, keys=None):
    """Order-independent content digest of many files: sha256 over sorted
    'key:sha256(file)' lines. Returns (digest, {key: sha})."""
    keys = list(keys) if keys is not None else list(paths)
    per = {k: sha256_file(p) for k, p in zip(keys, paths)}
    h = hashlib.sha256()
    for k in sorted(per):
        h.update(f"{k}:{per[k]}\n".encode())
    return h.hexdigest(), per


def val_cache_digest(cache_5f):
    """Content digest of the validation cache: val_index.json + every .pt it lists
    (all leads). A one-pixel change in any val window changes the digest."""
    idx_path = os.path.join(cache_5f, "val_index.json")
    with open(idx_path) as f:
        idx = json.load(f)
    paths, keys = [idx_path], ["val_index.json"]
    for vid, leads in idx.items():
        for lead, fn in sorted(leads.items()):
            paths.append(os.path.join(cache_5f, "val", fn))
            keys.append(f"{vid}@{lead}")
    return digest_files(paths, keys)[0]


def dense_cache_digest(dense_cache):
    """Content digest of the re07 dense cache (windows_u8.npy + meta.json)."""
    return digest_files([os.path.join(dense_cache, "windows_u8.npy"),
                         os.path.join(dense_cache, "meta.json")],
                        ["windows_u8.npy", "meta.json"])[0]


def source_identity(repo=REPO, files=SOURCE_FILES):
    per = {}
    for rel in files:
        p = os.path.join(repo, rel)
        per[rel] = sha256_file(p) if os.path.exists(p) else None
    return {"digest": sha256_json(per), "files": per}


def git_info(repo=REPO):
    def run(*a):
        try:
            return subprocess.check_output(["git", "-C", repo, *a], stderr=subprocess.DEVNULL,
                                           text=True).strip()
        except Exception:
            return None
    status = run("status", "--porcelain") or ""
    return {"sha": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "uncommitted_files": [ln[3:] for ln in status.splitlines() if ln.strip()]}


def env_identity():
    out = {"python": platform.python_version(), "platform": platform.platform()}
    try:
        import torch
        out.update(torch=torch.__version__, cuda_build=torch.version.cuda,
                   cudnn=torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
                   cuda_available=torch.cuda.is_available())
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            out.update(gpu=p.name, gpu_mem_gb=round(p.total_memory / 2**30, 1),
                       gpu_count=torch.cuda.device_count())
    except Exception as e:  # torch missing / broken
        out["torch_error"] = str(e)[:200]
    try:
        import torchvision
        out["torchvision"] = torchvision.__version__
    except Exception:
        pass
    return out


# ----------------------------------------------------------------------------
# Atomic writes
# ----------------------------------------------------------------------------
def write_json_atomic(path, obj):
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def torch_save_atomic(obj, path, verify_keys=("model",)):
    """Save to a temp file, fsync, reload it on CPU to prove it is readable,
    then atomically replace `path`. The previous file stays intact on any
    failure (disk full, kill, corrupt write)."""
    import torch
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp, "wb") as f:
            torch.save(obj, f)
            f.flush()
            os.fsync(f.fileno())
        chk = torch.load(tmp, map_location="cpu", weights_only=False)
        for k in verify_keys:
            if k not in chk:
                raise RuntimeError(f"checkpoint verification failed: key {k!r} missing")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return sha256_file(path)


# ----------------------------------------------------------------------------
# RNG
# ----------------------------------------------------------------------------
def restore_rng(rng, use_cuda):
    """Restore RNG states saved by re14. torch.set_rng_state needs a CPU
    ByteTensor; a checkpoint loaded with map_location='cuda' would have moved
    it to the GPU (v3 bug class), so always force .cpu()."""
    import random
    import numpy as np
    import torch
    torch.set_rng_state(rng["torch"].cpu().to(torch.uint8))
    np.random.set_state(rng["numpy"])
    random.setstate(rng["python"])
    if use_cuda and rng.get("cuda") is not None:
        torch.cuda.set_rng_state_all([t.cpu().to(torch.uint8) for t in rng["cuda"]])


def model_state_digest(model):
    """sha256 of a model's parameters/buffers (init identity)."""
    import torch
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        t = v.detach().cpu().contiguous()
        if t.dtype == torch.bfloat16:
            t = t.float()
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def fail(msg, code=2):
    print(f"[FATAL] {msg}", file=sys.stderr)
    sys.exit(code)
