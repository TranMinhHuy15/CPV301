"""
regression_tests_v4 -- v4 regression tests (CPU, synthetic data only; never touches the real
dataset). Complements unit_tests_v3 (the 33 v3 protocol tests, kept). Each test names
the V4 item it covers. CUDA-only checks are SKIPPED without a GPU and must be
reported as NOT RUN, never as PASS.

    python reeval/regression_tests_v4.py
"""
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.pop("PROGRESS_SYNTHETIC_COUNTS", None)   # tests always use the real expected counts
REPO = os.path.dirname(os.path.dirname(HERE))  # repo root (two levels up: reeval/progress_cps_v4/)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "pipeline"))
sys.path.insert(0, os.path.join(REPO, "reeval"))               # precache_val.py and other reeval-root siblings

import common as pc  # noqa: E402
import audit_data as au  # noqa: E402
import train_bps as tr  # noqa: E402
import eval_val as ev  # noqa: E402
import prepare_data_pinned as prep  # noqa: E402
import official_test_protocol as tst  # noqa: E402
import lock_config as lk  # noqa: E402


def make_fixture(root, n_train=8, n_val=4):
    """Tiny riskprop_precache_snippet_sequence/precache_val-format caches + manifest + matching sidecar (8x8 frames)."""
    g = torch.Generator().manual_seed(0)
    cache = os.path.join(root, "cache_rp")
    c5 = os.path.join(root, "c5")
    os.makedirs(os.path.join(cache, "train"))
    os.makedirs(os.path.join(c5, "val"))
    tr_ids = [f"{i:05d}" for i in range(n_train)]
    va_ids = [f"{i:05d}" for i in range(100, 100 + n_val)]
    idx, side = {}, {}
    for i, v in enumerate(tr_ids):
        pos = i % 2 == 0
        torch.save({"frames": torch.randint(0, 255, (12, 5, 8, 8, 3), dtype=torch.uint8, generator=g),
                    "tau": torch.linspace(5.6, 0.1, 12) if pos else torch.full((12,), float("inf")),
                    "target": float(pos), "dt": 0.5}, os.path.join(cache, "train", f"{v}.pt"))
        idx[v] = f"{v}.pt"
        side[v] = {"target": int(pos), "tau_actual": [5.6 - 0.5 * k for k in range(12)] if pos else [None] * 12,
                   "cps_mask": [0 < k < 11 for k in range(12)] if pos else [False] * 12}
    json.dump(idx, open(os.path.join(cache, "train_index.json"), "w"))
    vidx = {}
    for i, v in enumerate(va_ids):
        pos = i % 2 == 0
        vidx[v] = {}
        for lead in ("0.5", "1.0", "1.5"):
            fn = f"{v}_lead{lead}.pt"
            torch.save({"frames": torch.randint(0, 255, (5, 8, 8, 3), dtype=torch.uint8, generator=g),
                        "tau": float(lead) if pos else None, "target": float(pos)}, os.path.join(c5, "val", fn))
            vidx[v][lead] = fn
    json.dump(vidx, open(os.path.join(c5, "val_index.json"), "w"))
    man = os.path.join(root, "manifest.json")
    json.dump({"train_ids": tr_ids, "val_ids": va_ids, "train_csv_sha256": "x"}, open(man, "w"))
    from build_sidecar import cache_digest
    dig, _ = cache_digest(cache, idx, tr_ids)
    meta = {"usable_for_full_run": True, "pixel_verified": 24, "pixel_identical": 24,
            "split_manifest_sha256": pc.sha256_file(man),
            "train_index_sha256": pc.sha256_file(os.path.join(cache, "train_index.json")),
            "cache_digest": dig, "cache_digest_ids": len(tr_ids)}
    sc = os.path.join(root, "sidecar.json")
    json.dump({"meta": meta, "videos": side}, open(sc, "w"))
    return {"cache": cache, "c5": c5, "man": man, "sidecar": sc, "train_ids": tr_ids, "val_ids": va_ids}


def train_argv(fx, out, cond="P", extra=()):
    argv = ["--condition", cond, "--seed", "42", "--pairing-mode", "fixed", "--cache-root", fx["cache"],
            "--cache-5f", fx["c5"], "--sidecar", fx["sidecar"], "--split-manifest", fx["man"],
            "--output-root", out, "--model", "dummy", "--num-workers", "0", "--device", "cpu"]
    if cond != "B":
        argv += ["--lambda-prog", "1.0"]
    return argv + list(extra)


class TestPackageAndBaseline(unittest.TestCase):
    def test_v4_01_all_modules_import_and_cli_help(self):
        for f in ("audit_data", "build_sidecar", "train_bps",
                  "eval_val", "prepare_data_pinned", "preflight",
                  "official_test_protocol", "lock_config"):
            r = subprocess.run([sys.executable, os.path.join(HERE, f + ".py"), "--help"], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, f"{f} --help failed: {r.stderr[-300:]}")

    def test_v4_22_baseline_files_unchanged_vs_base_commit(self):
        if not os.path.isdir(os.path.join(REPO, ".git")):
            self.skipTest("not a git checkout (fresh-extract test covers hashes)")
        r = subprocess.run(["git", "-C", REPO, "cat-file", "-e", pc.BASE_COMMIT], capture_output=True)
        if r.returncode != 0:
            self.skipTest("base commit not available in this clone")
        d = subprocess.run(["git", "-C", REPO, "diff", "--name-only", pc.BASE_COMMIT, "--"] + pc.BASELINE_FILES,
                           capture_output=True, text=True)
        self.assertEqual(d.stdout.strip(), "", f"baseline files changed: {d.stdout}")


class TestPrepare(unittest.TestCase):
    """V4-04: deterministic mapping + metadata -> train.csv; failures are fatal."""

    def _raw(self, d, conflict=False):
        for lab in ("positive", "negative"):
            os.makedirs(os.path.join(d, "train", lab))
        pd.DataFrame({"file_name": ["01.mp4", "02.mp4"], "time_of_event": [5.5, 7.25],
                      "time_of_alert": [4.0, 6.0]}).to_csv(os.path.join(d, "train", "positive", "metadata.csv"), index=False)
        pd.DataFrame({"file_name": ["03.mp4", "04.mp4"], "time_of_event": [np.nan, 3.0 if conflict else np.nan],
                      "time_of_alert": [np.nan, np.nan]}).to_csv(os.path.join(d, "train", "negative", "metadata.csv"),
                                                                 index=False)
        return pd.DataFrame({"id_file": ["00000.mp4", "00001.mp4", "00002.mp4", "00003.mp4"],
                             "hf_source": ["train/negative/03.mp4", "train/positive/01.mp4",
                                           "train/negative/04.mp4", "train/positive/02.mp4"]})

    def test_build_and_serialize(self):
        with tempfile.TemporaryDirectory() as d:
            idm = self._raw(d)
            meta, used = prep.read_metadata(d)
            self.assertEqual(len(used), 2)
            df = prep.build_train_csv(idm, meta)
            self.assertEqual(df["target"].tolist(), [0, 1, 0, 1])
            self.assertEqual(df["time_of_event"].tolist()[1], 5.5)
            b = prep.csv_bytes(df)
            self.assertTrue(b.startswith(b"id,target,time_of_event,time_of_alert\n0,0,,\n1,1,5.5,4.0\n"))
            self.assertNotIn(b"\r", b)

    def test_metadata_problems_are_fatal(self):
        with tempfile.TemporaryDirectory() as d:
            idm = self._raw(d, conflict=True)
            meta, _ = prep.read_metadata(d)
            df = prep.build_train_csv(idm, meta)
            errs = prep.check_train_df(df)
            self.assertTrue(any("rows" in e for e in errs))           # 4 != 1500
            idm.loc[0, "hf_source"] = "train/positive/03.mp4"         # mapping vs metadata label conflict
            with self.assertRaises((KeyError, ValueError)):
                prep.build_train_csv(idm, meta)


class TestTimingAndSplit(unittest.TestCase):
    def test_v4_06_pts_policy(self):
        fps = 25.0
        rel, info, err = au.pts_policy(np.arange(100) / fps + 0.2, fps)
        self.assertIsNone(err)
        self.assertAlmostEqual(info["pts_offset_s"], 0.2)
        self.assertAlmostEqual(info["frame_period_s"], 0.04)        # from the video's fps, not 1/30
        self.assertFalse(info["vfr_suspect"])
        ts = np.arange(100) / fps
        ts[50:] += 0.05                                              # > 1 frame jump -> VFR suspect
        self.assertTrue(au.pts_policy(ts, fps)[1]["vfr_suspect"])
        bad = np.arange(100) / fps
        bad[10] = bad[9]
        self.assertIsNotNone(au.pts_policy(bad, fps)[2])
        self.assertIsNotNone(au.pts_policy(np.arange(10) / fps - 0.5, fps)[2])
        self.assertIsNotNone(au.pts_policy(np.array([]), fps)[2])

    def test_v4_13_val_timing_rows(self):
        fps, total, toe = 25.0, 500, 12.0
        rel = np.arange(total) / fps
        rows = au.val_timing_rows("00007", 1, toe, rel, fps, total)
        self.assertEqual(len(rows), 33)                               # 3 leads + 30 dense
        self.assertFalse(any(r["future_frame"] for r in rows))
        self.assertTrue(all(abs(r["dev_frames"]) <= 1.0 for r in rows))
        self.assertTrue(all(abs(r["frame_period_s"] - 0.04) < 1e-12 for r in rows))
        late = au.val_timing_rows("00007", 1, toe, rel + 0.5, fps, total)   # shifted video clock
        self.assertTrue(any(r["future_frame"] for r in late))

    def test_lead_and_dense_rules_match_precache_val_and_infer_val(self):
        sys.path.insert(0, os.path.join(REPO, "reeval", "ablation_ffr_amc")); import infer_val as r7
        for toe in (3.2, 9.87, 30.0):
            row = {"target": 1, "time_of_event": toe}
            self.assertEqual(au.dense_t_obs(1, toe, 40.0), r7.dense_t_obs(pd.Series(row), 40.0))
        try:
            import precache_val as r1
        except Exception:
            self.skipTest("precache_val needs decord")
        for toe in (3.2, 9.87):
            for lead in (0.5, 1.0, 1.5):
                idx, _ = r1.get_val_indices(pd.Series({"target": 1, "time_of_event": toe}), 30.0, 1200, lead)
                self.assertEqual(int(idx[-1]), au.end_frame_of(au.val_lead_t_obs(1, toe, 40.0, lead), 30.0, 1200))

    def test_split_order_change_detected(self):
        rng = np.random.default_rng(0)
        n = 60
        df = pd.DataFrame({"id": np.arange(n), "target": [1, 0] * (n // 2),
                           "time_of_event": [rng.uniform(5, 20) if i % 2 == 0 else np.nan for i in range(n)]})
        df["time_of_alert"] = df["time_of_event"] - 1.2
        tr_ids, va_ids = au.recompute_split(df)
        shuffled = df.sample(frac=1.0, random_state=1).reset_index(drop=True)
        tr2, va2 = au.recompute_split(shuffled)
        self.assertNotEqual((tr_ids, va_ids), (tr2, va2))


class TestProvenance(unittest.TestCase):
    def test_v4_08_pixel_mutation_changes_val_and_dense_digest(self):
        with tempfile.TemporaryDirectory() as d:
            fx = make_fixture(d)
            v0 = pc.val_cache_digest(fx["c5"])
            p = os.path.join(fx["c5"], "val", f"{fx['val_ids'][1]}_lead1.0.pt")
            x = torch.load(p, weights_only=False)
            x["frames"][0, 0, 0, 0] ^= 1
            torch.save(x, p)
            self.assertNotEqual(v0, pc.val_cache_digest(fx["c5"]))           # index/meta unchanged
            dense = os.path.join(d, "dense")
            os.makedirs(dense)
            arr = np.zeros((2, 30, 5, 4, 4, 3), np.uint8)
            np.save(os.path.join(dense, "windows_u8.npy"), arr)
            json.dump({"vid_ids": ["a", "b"]}, open(os.path.join(dense, "meta.json"), "w"))
            d0 = pc.dense_cache_digest(dense)
            arr[1, 3, 0, 0, 0, 0] = 1
            np.save(os.path.join(dense, "windows_u8.npy"), arr)
            self.assertNotEqual(d0, pc.dense_cache_digest(dense))
            ck = os.path.join(d, "best.pth")
            open(ck, "wb").write(b"w")
            out = os.path.join(d, "r_best.npz")
            np.savez(out, scores=np.zeros(3))
            json.dump({"ckpt_sha256": pc.sha256_file(ck), "val_cache_digest": v0}, open(out[:-4] + ".json", "w"))
            self.assertFalse(ev.preds_fresh(out, ck, pc.val_cache_digest(fx["c5"])))

    def test_v4_09_fingerprint_code_init_lock(self):
        cfg = {"condition": "P", "seed": 42, "lambda_prog": 1.0, "run_name": "x"}
        man = {"inputs": {"lock_sha256": "L"}, "code": {"digest": "C"}, "init_state_sha256": "I"}
        fp = tr.run_fingerprint(cfg, man)
        for mut in ({"code": {"digest": "C2"}}, {"init_state_sha256": "I2"}, {"inputs": {"lock_sha256": "L2"}}):
            self.assertNotEqual(fp, tr.run_fingerprint(cfg, dict(man, **mut)), mut)

    def test_v4_03_complete_marker(self):
        with tempfile.TemporaryDirectory() as d:
            for k in ("best", "latest"):
                open(os.path.join(d, f"{k}.pth"), "wb").write(k.encode())
            fp = {"a": 1}
            self.assertFalse(tr.complete_ok(d, fp)[0])                     # no marker
            json.dump({"fingerprint": fp, "checkpoints": {k: pc.sha256_file(os.path.join(d, f"{k}.pth"))
                                                         for k in ("best", "latest")}},
                      open(os.path.join(d, "COMPLETE.json"), "w"))
            self.assertTrue(tr.complete_ok(d, fp)[0])
            self.assertFalse(tr.complete_ok(d, {"a": 2})[0])               # lambda/alpha/lock/code changed
            open(os.path.join(d, "best.pth"), "wb").write(b"other")
            self.assertFalse(tr.complete_ok(d, fp)[0])                     # checkpoint replaced

    def test_v4_11_atomic_save_keeps_last_valid(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "latest.pth")
            pc.torch_save_atomic({"model": {"w": torch.ones(2)}}, p)
            before = pc.sha256_file(p)
            with mock.patch("torch.load", side_effect=RuntimeError("corrupt")):
                with self.assertRaises(RuntimeError):
                    pc.torch_save_atomic({"model": {"w": torch.zeros(2)}}, p)
            self.assertEqual(before, pc.sha256_file(p))
            self.assertEqual([f for f in os.listdir(d) if ".tmp." in f], [])
            with self.assertRaises(RuntimeError):
                pc.torch_save_atomic({"no_model": 1}, p)                  # verification of required keys
            self.assertEqual(before, pc.sha256_file(p))


class TestTrainerIntegration(unittest.TestCase):
    """V4-10/11/12 on CPU with the dummy model (real CUDA resume: see test_cuda_resume)."""

    def test_resume_equals_uninterrupted_and_marker(self):
        with tempfile.TemporaryDirectory() as d:
            fx = make_fixture(d)
            tr.main(train_argv(fx, os.path.join(d, "o1"), extra=["--max-epochs", "2"]))
            tr.main(train_argv(fx, os.path.join(d, "o2"), extra=["--max-epochs", "2", "--stop-after-epochs", "1"]))
            r2 = os.path.join(d, "o2", "dev", "P_fixed_H2_a3_lam1_seed42")
            self.assertFalse(os.path.exists(os.path.join(r2, "COMPLETE.json")))     # interrupted -> no marker
            tr.main(train_argv(fx, os.path.join(d, "o2"), extra=["--max-epochs", "2"]))
            r1 = os.path.join(d, "o1", "dev", "P_fixed_H2_a3_lam1_seed42")
            a = torch.load(os.path.join(r1, "latest.pth"), weights_only=False)["model"]
            b = torch.load(os.path.join(r2, "latest.pth"), weights_only=False)["model"]
            for k in a:
                self.assertTrue(torch.equal(a[k], b[k]), k)
            self.assertTrue(os.path.exists(os.path.join(r2, "COMPLETE.json")))
            l1 = pd.read_csv(os.path.join(r1, "train_log.csv"))
            l2 = pd.read_csv(os.path.join(r2, "train_log.csv"))
            self.assertTrue(np.allclose(l1["train_total"], l2["train_total"]))
            with self.assertRaises(SystemExit) as cm:                              # check-complete: same fp
                tr.main(train_argv(fx, os.path.join(d, "o2"), extra=["--max-epochs", "2", "--check-complete"]))
            self.assertEqual(cm.exception.code, 0)

    def test_resume_refused_when_config_or_data_changes(self):
        with tempfile.TemporaryDirectory() as d:
            fx = make_fixture(d)
            out = os.path.join(d, "o")
            tr.main(train_argv(fx, out, extra=["--max-epochs", "2", "--stop-after-epochs", "1"]))
            side = json.load(open(fx["sidecar"]))
            side["videos"]["00000"]["cps_mask"][3] = False                 # rebuilt sidecar
            json.dump(side, open(fx["sidecar"], "w"))
            with self.assertRaises(SystemExit) as cm:
                tr.main(train_argv(fx, out, extra=["--max-epochs", "2"]))
            self.assertIn("sidecar_sha256", str(cm.exception.code))

    def test_v4_12_same_init_for_conditions_and_lambda0_parity(self):
        with tempfile.TemporaryDirectory() as d:
            fx = make_fixture(d)
            inits = []
            for c in ("B", "P", "S"):
                tr.main(train_argv(fx, os.path.join(d, "o"), cond=c, extra=["--max-steps", "1", "--limit-val", "2"]))
                rd = os.path.join(d, "o", "dev", tr.run_name(c, "fixed", 2.0, 3.0, 0.0 if c == "B" else 1.0, 42))
                inits.append(json.load(open(os.path.join(rd, "config.json")))["init_state_sha256"])
            self.assertEqual(len(set(inits)), 1)
        torch.manual_seed(0)
        logits = torch.randn(4, 12)
        tgt = torch.tensor([1.0, 0.0, 1.0, 0.0])
        tau = torch.full((4, 12), 1.0)
        mask = torch.ones(4, 12, dtype=torch.bool)

        class M(torch.nn.Module):
            def forward(self, x):
                return logits.clone()
        cfgB = {"cps_mode": "none", "use_ffr": True, "use_amc": True, "pairing_mode": "fixed",
                "horizon_sec": 2.0, "alpha": 3.0, "lambda_prog": 0.0}
        torch.manual_seed(5)
        b, _ = tr.forward_losses(M(), None, tgt, 0.5, tau, mask, cfgB)
        torch.manual_seed(5)
        p, _ = tr.forward_losses(M(), None, tgt, 0.5, tau, mask, dict(cfgB, cps_mode="continuous", lambda_prog=0.0))
        self.assertEqual(b.item(), p.item())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available -> NOT RUN")
    def test_cuda_resume(self):
        with tempfile.TemporaryDirectory() as d:
            fx = make_fixture(d)
            argv = train_argv(fx, os.path.join(d, "o"), extra=["--max-epochs", "2"])
            argv[argv.index("--device") + 1] = "cuda"          # replace the value, keep every other flag
            tr.main(argv + ["--stop-after-epochs", "1"])
            ck = torch.load(os.path.join(d, "o", "dev", "P_fixed_H2_a3_lam1_seed42", "latest.pth"),
                            map_location="cuda", weights_only=False)
            pc.restore_rng(ck["rng"], True)                                # cuda-mapped state must restore
            tr.main(argv)


class TestEvaluation(unittest.TestCase):
    def test_v4_14_accuracy_cohort_not_shrunk_by_dense(self):
        targets = np.array([1, 1, 0, 0])
        has_toe = np.array([True, True, False, False])
        d1 = np.ones((4, 30))
        d1[1, 5] = np.nan
        acc, tpos, _ = ev.make_cohorts(targets, has_toe, [d1, np.ones((4, 30))])
        self.assertEqual(len(acc), 4)
        self.assertEqual(tpos.tolist(), [True, False, False, False])

    def _ci(self, lo, ni_lo=-0.01):
        return pd.DataFrame([dict(contrast="P - B", metric="AP@1.5s", estimate=0.01, ci_low=lo, ci_high=0.05, verdict=""),
                             dict(contrast="P - B", metric="mAP", estimate=0.0, ci_low=ni_lo, ci_high=0.01, verdict=""),
                             dict(contrast="P - S", metric="AP@1.5s", estimate=0.0, ci_low=-0.02, ci_high=0.02,
                                  verdict="inconclusive (CI includes 0)")])

    LOCK = {"primary_contrast": "P - B", "primary_metric": "AP@1.5s", "ni_metric": "mAP", "ni_margin": 0.02,
            "ps_metric": "AP@1.5s"}

    def test_v4_15_raw_float_precision(self):
        self.assertTrue(ev.decision(self._ci(1e-6), self.LOCK)["rq3_supported"])
        self.assertFalse(ev.decision(self._ci(0.0), self.LOCK)["rq3_supported"])
        self.assertFalse(ev.decision(self._ci(-1e-6), self.LOCK)["rq3_supported"])
        self.assertFalse(ev.decision(self._ci(float("nan")), self.LOCK)["available"])
        self.assertTrue(ev.decision(self._ci(0.01, ni_lo=-0.0199999), self.LOCK)["non_inferiority_met"])
        self.assertFalse(ev.decision(self._ci(0.01, ni_lo=-0.02), self.LOCK)["non_inferiority_met"])
        d = ev.decision(self._ci(0.01), self.LOCK, ["missing run B seed 44"])
        self.assertFalse(d["rq3_supported"])
        self.assertIsNotNone(d["p_minus_s"])

    def test_v4_16_research_gate(self):
        with tempfile.TemporaryDirectory() as d:
            lockp = os.path.join(d, "LOCK.json")
            json.dump({"x": 1}, open(lockp, "w"))
            runs, rc = {}, {}
            for c in ("B", "P", "S"):
                for s in (42, 43, 44):
                    rd = os.path.join(d, f"{c}{s}")
                    os.makedirs(rd)
                    for k in ("best", "latest"):
                        open(os.path.join(rd, f"{k}.pth"), "wb").write(f"{c}{s}{k}".encode())
                    json.dump({"config": {"research_result": True, "full_run": True},
                               "inputs": {"lock_sha256": pc.sha256_file(lockp)},
                               "init_state_sha256": f"init{s}", "code": {"digest": "c"}},
                              open(os.path.join(rd, "config.json"), "w"))
                    json.dump({"checkpoints": {k: pc.sha256_file(os.path.join(rd, f"{k}.pth")) for k in ("best", "latest")}},
                              open(os.path.join(rd, "COMPLETE.json"), "w"))
                    runs[f"{c}{s}"], rc[f"{c}{s}"] = rd, (c, s)
            self.assertEqual(ev.research_gate(runs, rc, lockp, ["B", "P", "S"], [42, 43, 44]), [])
            r2 = {k: v for k, v in runs.items() if k != "S44"}
            self.assertTrue(any("missing run S seed 44" in i for i in
                                ev.research_gate(r2, {k: rc[k] for k in r2}, lockp, ["B", "P", "S"], [42, 43, 44])))
            cfg = json.load(open(os.path.join(runs["P43"], "config.json")))
            cfg["init_state_sha256"] = "other"
            json.dump(cfg, open(os.path.join(runs["P43"], "config.json"), "w"))
            self.assertTrue(any("initial weights" in i for i in ev.research_gate(runs, rc, lockp, [], [])))
            open(os.path.join(runs["B42"], "best.pth"), "wb").write(b"changed")
            self.assertTrue(any("changed after COMPLETE" in i for i in ev.research_gate(runs, rc, lockp, [], [])))


class TestDenseTolerance(unittest.TestCase):
    def test_exactly_one_frame_plus_rounding_passes_but_two_ms_more_fails(self):
        D = [round(3.0 - 0.1 * k, 1) for k in range(30)]
        meta = {"vid_ids": ["00001", "00002"], "targets": [1, 0], "failed": [], "dense_d": D,
                "t_obs": [[10.0 - x for x in D], [5.0] * 30]}
        with tempfile.TemporaryDirectory() as d:
            def vt(dev_s):
                rows = [{"vid": "00001", "target": 1, "kind": "dense", "k": k, "requested_end_s": 10.0 - x,
                         "end_frame": int((10.0 - x) * 30), "actual_end_s": 10.0 - x - dev_s, "time_of_event": 10.0,
                         "future_frame": False, "dev_s": -dev_s, "dev_frames": -dev_s * 30,
                         "frame_period_s": 1 / 30, "vfr_suspect": False} for k, x in enumerate(D)]
                p = os.path.join(d, "vt.csv")
                pd.DataFrame(rows).to_csv(p, index=False)
                return p
            ok = ev.dense_gate(meta, ["00001", "00002"], {"00001": 10.0}, vt(1 / 30 + 1e-9), expected=(2, 1))
            self.assertTrue(ok["pass"], ok["issues"])                   # observed on real data: +0.00003 frame
            bad = ev.dense_gate(meta, ["00001", "00002"], {"00001": 10.0}, vt(1 / 30 + 0.002), expected=(2, 1))
            self.assertEqual(bad["counts"]["dev_gt_1_frame"], 1)


class TestOfficialTest(unittest.TestCase):
    def test_v4_17_exact_ids(self):
        ids = [str(i) for i in range(pc.N_TEST)]
        vm = {i: f"/x/{i}.mp4" for i in ids}
        self.assertEqual(tst.check_ids(ids, vm), [])
        self.assertTrue(tst.check_ids(ids[:-1], vm))                     # count + extra video
        self.assertTrue(tst.check_ids(ids[:-1] + ids[:1], vm))           # duplicate id
        vm2 = dict(vm)
        vm2.pop("7")
        self.assertTrue(any("without a video" in e for e in tst.check_ids(ids, vm2)))
        self.assertEqual(tst.check_ids(["00012"] + ids[1:12] + ids[13:] + ["0"], vm)[:0], [])
        with tempfile.TemporaryDirectory() as d:
            for sp in ("test-public", "test-private"):
                os.makedirs(os.path.join(d, sp, "positive"))
                open(os.path.join(d, sp, "positive", "00012.mp4"), "w").close()
            _, dup = tst.list_videos(d)
            self.assertEqual(dup, ["12"])

    def test_v4_18_freeze_requires_research_gate(self):
        with tempfile.TemporaryDirectory() as d:
            json.dump({"chosen": {}}, open(os.path.join(d, "chosen_checkpoints.json"), "w"))
            json.dump({"available": True, "research_gate_passed": False}, open(os.path.join(d, "rq3_decision.json"), "w"))
            lockp = os.path.join(d, "LOCK.json")
            json.dump({}, open(lockp, "w"))
            with self.assertRaises(SystemExit):
                tst.main(["freeze", "--eval-dir", d, "--lock-file", lockp, "--out-dir", os.path.join(d, "t")])


class TestLock(unittest.TestCase):
    def test_g5_lock_requires_evidence_and_is_immutable(self):
        with tempfile.TemporaryDirectory() as d:
            fx = make_fixture(d)
            ev_ = {}
            for name, body in (("prepare.json", {"pass": True, "revision": "r", "train_csv_sha256": "t"}),
                               ("audit.json", {"pass": True, "complete": True, "inputs": {"train_csv_sha256": "t"}}),
                               ("preflight.json", {"pass": True, "require_cuda": True, "env": {}})):
                ev_[name] = os.path.join(d, name)
                json.dump(body, open(ev_[name], "w"))
            json.dump({"usable_for_full_run": True}, open(os.path.join(d, "sidecar_summary.json"), "w"))
            mini = os.path.join(d, "dev")
            argv = ["--out", os.path.join(d, "LOCK.json"), "--prepare", ev_["prepare.json"], "--audit", ev_["audit.json"],
                    "--sidecar", fx["sidecar"], "--preflight", ev_["preflight.json"], "--mini-root", mini,
                    "--cache-5f", fx["c5"]]
            with mock.patch.dict(os.environ, {"LOCK_CONFIRMED": "1"}):
                with self.assertRaises(SystemExit):                       # no mini COMPLETE.json yet
                    lk.main(argv)
                for c in ("B", "P", "S"):
                    os.makedirs(os.path.join(mini, f"{c}_x_mini"))
                    open(os.path.join(mini, f"{c}_x_mini", "COMPLETE.json"), "w").write("{}")
                lk.main(argv)
                lk.main(argv)                                             # identical -> no-op
                with self.assertRaises(SystemExit):
                    lk.main(argv + ["--lam", "0.5"])                      # different design -> refused
            lock = json.load(open(os.path.join(d, "LOCK.json")))
            self.assertEqual(lock["lambda_prog"], 1.0)
            self.assertEqual(lock["fingerprints"]["val_cache_digest"], pc.val_cache_digest(fx["c5"]))
            with mock.patch.dict(os.environ, {"LOCK_CONFIRMED": ""}):
                with self.assertRaises(SystemExit):
                    lk.main(argv)


class TestShell(unittest.TestCase):
    """V4-02 / V4-03: the run script stops on failure and never skips unverified runs."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.tmp, "reeval"))
        os.makedirs(os.path.join(self.tmp, "results", "repro"))
        shutil.copy(os.path.join(HERE, "run_all.sh"), os.path.join(self.tmp, "reeval"))
        self.calls = os.path.join(self.tmp, "calls.txt")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def stub(self, name, body):
        p = os.path.join(self.tmp, "reeval", name)
        with open(p, "w") as f:
            f.write("import sys, os\nopen(%r, 'a').write(' '.join(sys.argv) + '\\n')\n" % self.calls + body + "\n")
        os.chmod(p, os.stat(p).st_mode | stat.S_IEXEC)

    def run_stage(self, stage, env=None):
        e = dict(os.environ, REPO=self.tmp, OUT=os.path.join(self.tmp, "out"), DATA=os.path.join(self.tmp, "data"),
                 MODEL="dummy", PATH=os.path.dirname(sys.executable) + os.pathsep + os.environ["PATH"])
        e.update(env or {})
        shim = os.path.join(self.tmp, "bin")
        os.makedirs(shim, exist_ok=True)
        with open(os.path.join(shim, "python"), "w") as f:
            f.write(f"#!/bin/sh\nexec {sys.executable} \"$@\"\n")
        os.chmod(os.path.join(shim, "python"), 0o755)
        e["PATH"] = shim + os.pathsep + e["PATH"]
        return subprocess.run(["bash", os.path.join(self.tmp, "reeval", "run_all.sh"), stage],
                              capture_output=True, text=True, env=e, cwd=self.tmp)

    def calls_list(self):
        return open(self.calls).read().splitlines() if os.path.exists(self.calls) else []

    def test_failing_step_stops_stage_and_keeps_log(self):
        self.stub("unit_tests_v3.py", "print('BOOM-ORIGINAL-ERROR'); sys.exit(3)")
        self.stub("audit_data.py", "pass")
        r = self.run_stage("check")
        self.assertNotEqual(r.returncode, 0)
        self.assertFalse(any("audit_data" in c for c in self.calls_list()), "stage continued after a failure")
        log = open(os.path.join(self.tmp, "out", "logs", "unit_tests.log")).read()
        self.assertIn("BOOM-ORIGINAL-ERROR", log)
        self.assertIn("rc=3", r.stderr)

    def test_filter_without_matches_is_not_a_failure_and_child_failure_is(self):
        self.stub("train_bps.py", "pass")             # prints nothing at all
        self.assertEqual(self.run_stage("smoke").returncode, 0)
        self.stub("train_bps.py", "sys.exit(7)")
        self.assertNotEqual(self.run_stage("smoke").returncode, 0)

    def test_full_skips_only_verified_and_fails_unverified(self):
        os.makedirs(os.path.join(self.tmp, "out"), exist_ok=True)
        open(os.path.join(self.tmp, "out", "LOCK.json"), "w").write("{}")
        self.stub("preflight.py", "pass")
        self.stub("train_bps.py", "sys.exit(0)")      # every run verifies complete
        r = self.run_stage("full")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.count("[skip]"), 9)
        self.assertTrue(all("--check-complete" in c for c in self.calls_list() if "train_bps" in c))
        os.remove(self.calls)
        # training "succeeds" but the completion check never verifies -> stage must fail, not mark done
        self.stub("train_bps.py", "sys.exit(1 if '--check-complete' in sys.argv else 0)")
        r = self.run_stage("full")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("does not verify", r.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
