"""
unit_tests_v3 -- CPU unit tests for the Progressive RiskProp protocol (Gate 1).
No dataset, no GPU, no internet. Run from the repository root:
    python reeval/unit_tests_v3.py
"""
import json
import math
import os
import sys
import tempfile
import unittest

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.pop("PROGRESS_SYNTHETIC_COUNTS", None)   # tests always use the real expected counts
REPO = os.path.dirname(os.path.dirname(HERE))  # repo root (two levels up: reeval/progress_cps_v4/)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "pipeline"))

import supervision_targets as ps  # noqa: E402
import train_bps as tr  # noqa: E402
from audit_data import nominal_t_obs  # noqa: E402
from build_sidecar import precache_indices_for_t_obs, check_pts, choose_subset, cache_digest  # noqa: E402
import eval_val as ev  # noqa: E402
from riskprop_loss_ffr_amc import riskprop_loss  # noqa: E402


def cfg_for(cond, lam=1.0, pairing="fixed", H=2.0, alpha=3.0):
    mode, ffr, amc = tr.CONDITIONS[cond]
    return {"cps_mode": mode, "use_ffr": ffr, "use_amc": amc, "pairing_mode": pairing,
            "horizon_sec": H, "alpha": alpha, "lambda_prog": 0.0 if mode == "none" else lam}


class TestTargets(unittest.TestCase):
    def test_continuous_values(self):
        tau = torch.tensor([2.0, 1.5, 1.0, 2.5, 0.0, -0.1, float("inf"), float("nan")])
        r, v = ps.continuous_target(tau, 2.0, 3.0)
        self.assertAlmostEqual(r[0].item(), 0.0, places=6)
        self.assertAlmostEqual(r[1].item(), 0.5553, places=4)
        self.assertAlmostEqual(r[2].item(), 0.8176, places=4)
        self.assertAlmostEqual(r[3].item(), 0.0, places=6)
        self.assertEqual(v.tolist(), [True, True, True, True, False, False, False, False])
        self.assertFalse(torch.isnan(r).any() or torch.isinf(r).any())

    def test_scalar_matches_tensor_and_preact_formula(self):
        for tau in [0.05, 0.1, 0.6, 1.1, 1.6, 1.99, 2.0, 3.0]:
            s = ps.continuous_target_scalar(tau, 2.0, 3.0)
            t = ps.continuous_target(torch.tensor([tau]), 2.0, 3.0)[0].item()
            self.assertAlmostEqual(s, t, places=5)
            # PRE-ACT exp_above_linear with ttc in 10-fps frames
            f = tau * 10
            pre = 0.0 if f >= 20 else (1 - math.exp(-3 * (1 - f / 20))) / (1 - math.exp(-3))
            self.assertAlmostEqual(s, pre, places=6)
        self.assertIsNone(ps.continuous_target_scalar(0.0))   # PRE-ACT would give 1.0 (excluded here)

    def test_binary_and_h15_boundaries(self):
        b, v = ps.binary_target(torch.tensor([1.999, 2.0, 0.0, -1.0, 0.01]), 2.0)
        self.assertEqual(b.tolist(), [1.0, 0.0, 0.0, 0.0, 1.0])
        self.assertEqual(v.tolist(), [True, True, False, False, True])
        b, _ = ps.binary_target(torch.tensor([1.4999, 1.5]), 1.5)
        self.assertEqual(b.tolist(), [1.0, 0.0])
        r, _ = ps.continuous_target(torch.tensor([1.5, 1.4999]), 1.5, 3.0)
        self.assertEqual(r[0].item(), 0.0)
        self.assertGreater(r[1].item(), 0.0)

    def test_monotone_and_bounded(self):
        tau = torch.linspace(2.5, 0.001, 500)       # decreasing tau
        r, _ = ps.continuous_target(tau, 2.0, 3.0)
        self.assertTrue(bool((r[1:] >= r[:-1] - 1e-7).all()))
        self.assertTrue(bool(((r >= 0) & (r <= 1)).all()))

    def test_grid_values(self):
        g = {k: c for k, tau, c, b in ps.grid_table()}
        self.assertAlmostEqual(g[8], 0.4749, places=3)   # tau 1.6
        self.assertAlmostEqual(g[9], 0.7796, places=3)   # tau 1.1
        self.assertEqual(g[0], 0.0)


class TestMasks(unittest.TestCase):
    def _precache_like(self, toe, fps=30.0, total=1200, positive=True):
        t_nom = nominal_t_obs(1 if positive else 0, toe, total / fps)
        idx = [precache_indices_for_t_obs(t, fps, total) for t in t_nom]
        ends = np.array([int(x[-1]) for x in idx])
        return ends, ends / fps

    def test_normal_positive(self):
        ends, times = self._precache_like(20.0)
        rec = ps.snippet_records(True, 20.0, ends, times, np.ones(12, bool), np.zeros(12, bool))
        self.assertEqual(rec["reason"][0], "anchor_first")
        self.assertEqual(rec["reason"][11], "anchor_last")
        self.assertEqual(int(rec["cps_mask"].sum()), 10)
        self.assertTrue(np.all(rec["tau_actual"][rec["cps_mask"]] > 0))

    def test_early_event_duplicates(self):
        # handover case: k=0..5 clamp to t_obs=0.4 s; at 30 fps k=6 (t_obs 0.432 s) also
        # lands on frame 12 -> 7 snippets share one endpoint, only k=6 is kept
        ends, times = self._precache_like(3.032)
        self.assertEqual(len(set(ends[:7])), 1)
        rec = ps.snippet_records(True, 3.032, ends, times, np.ones(12, bool), np.zeros(12, bool))
        self.assertEqual(rec["reason"][1:6], ["duplicate_endpoint"] * 5)
        self.assertEqual(rec["reason"][6], "ok")
        self.assertEqual(len(set(ends[rec["cps_mask"]])), int(rec["cps_mask"].sum()))

    def test_decode_zero_post_event_negative(self):
        ends = np.arange(12) * 15 + 100
        times = ends / 30.0
        dec = np.ones(12, bool); dec[3] = False
        zero = np.zeros(12, bool); zero[4] = True
        toe = times[9] - 0.0                          # snippet 9 at the event -> tau 0
        rec = ps.snippet_records(True, toe, ends, times, dec, zero)
        self.assertEqual(rec["reason"][3], "decode_fail")
        self.assertEqual(rec["reason"][4], "zero_unchecked")    # zero = suspect, not decode failure
        self.assertEqual(rec["reason"][9], "post_event")
        self.assertEqual(rec["reason"][10], "post_event")
        neg = ps.snippet_records(False, None, ends, times, dec, zero)
        self.assertFalse(neg["cps_mask"].any())
        self.assertTrue(np.all(np.isinf(neg["tau_actual"])))

    def test_nonmonotonic(self):
        ends = np.arange(12) * 15 + 100
        times = ends / 30.0
        times[6] = times[5] - 0.01
        rec = ps.snippet_records(True, 50.0, ends, times, np.ones(12, bool), np.zeros(12, bool))
        self.assertEqual(rec["reason"][6], "nonmonotonic_time")

    def test_p_and_s_same_mask(self):
        tau = torch.tensor([[3.0, 2.0, 1.9, 1.0, 0.5, 0.0, -1, float("inf")]])
        _, vc = ps.continuous_target(tau)
        _, vb = ps.binary_target(tau)
        self.assertTrue(torch.equal(vc, vb))


class TestLoss(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.logits = torch.randn(4, 12, requires_grad=True)
        self.targets = torch.tensor([1.0, 0.0, 1.0, 0.0])
        tau = torch.full((4, 12), float("inf"))
        tau[0] = torch.linspace(5.6, 0.1, 12)
        tau[2] = torch.linspace(5.6, 0.1, 12)
        self.tau = tau
        mask = torch.zeros(4, 12, dtype=torch.bool)
        mask[0, 1:11] = True
        mask[2, 1:11] = True
        self.mask = mask

    def test_empty_mask_zero_and_finite(self):
        for mode in ("continuous", "binary", "none"):
            l, st = ps.cps_loss(self.logits, self.tau, torch.zeros_like(self.mask), mode)
            self.assertEqual(l.item(), 0.0)
            l.backward()
            self.assertTrue(torch.isfinite(self.logits.grad).all())
            self.logits.grad = None
        base, _ = riskprop_loss(self.logits, torch.zeros(4), dt=0.5)
        self.assertTrue(torch.isfinite(base))

    def test_gradient_only_on_valid(self):
        l, st = ps.cps_loss(self.logits, self.tau, self.mask, "continuous")
        l.backward()
        g = self.logits.grad
        self.assertTrue(torch.isfinite(g).all())
        self.assertGreater(g[self.mask].abs().sum().item(), 0)
        self.assertEqual(g[~self.mask].abs().sum().item(), 0.0)
        self.assertEqual(st["cps_videos"], 2)
        self.assertEqual(st["cps_snippets"], 20)
        r, _ = ps.continuous_target(self.tau)
        self.assertFalse(r.requires_grad)

    def test_per_video_then_mean_reduction(self):
        mask = self.mask.clone()
        mask[2, 2:11] = False                      # video 2 keeps one snippet only
        l, _ = ps.cps_loss(self.logits, self.tau, mask, "binary")
        a = torch.sigmoid(self.logits.detach())
        b, _ = ps.binary_target(self.tau)
        e = torch.nn.functional.smooth_l1_loss(a, b, reduction="none")
        exp = (e[0, 1:11].mean() + e[2, 1:2].mean()) / 2
        self.assertAlmostEqual(l.item(), exp.item(), places=6)

    def test_parity_with_riskprop_loss(self):
        """B total == riskprop_loss_ffr_amc.riskprop_loss; P total - lambda*cps == same base (same RNG)."""
        model = RecordingModel(self.logits.detach())
        seq = torch.zeros(4, 12, 3, 5, 8, 8)
        for pairing in ("fixed", "random"):
            torch.manual_seed(123)
            ref, _ = riskprop_loss(self.logits.detach(), self.targets, dt=0.5, pairing_mode=pairing)
            torch.manual_seed(123)
            totB, pB = tr.forward_losses(model, seq, self.targets, 0.5, self.tau, self.mask,
                                         cfg_for("B", pairing=pairing))
            self.assertAlmostEqual(totB.item(), ref.item(), places=6)
            torch.manual_seed(123)
            totP, pP = tr.forward_losses(model, seq, self.targets, 0.5, self.tau, self.mask,
                                         cfg_for("P", lam=0.7, pairing=pairing))
            self.assertAlmostEqual(totP.item() - 0.7 * pP["cps"], ref.item(), places=5)
            self.assertAlmostEqual(pP["base"], ref.item(), places=6)
        torch.manual_seed(1)
        refF, _ = riskprop_loss(self.logits.detach(), self.targets, dt=0.5, use_amc=False)
        torch.manual_seed(1)
        totF, _ = tr.forward_losses(model, seq, self.targets, 0.5, self.tau, self.mask, cfg_for("F"))
        self.assertAlmostEqual(totF.item(), refF.item(), places=6)


class TestPtsAndZero(unittest.TestCase):
    def test_pts_offset_subtracted(self):
        ts = np.arange(300) / 30.0 + 0.5           # stream starts at 0.5 s
        rel, info, err = check_pts(ts, [30, 60], 30.0)
        self.assertIsNone(err)
        self.assertAlmostEqual(info["pts_offset_s"], 0.5)
        np.testing.assert_allclose(rel, [1.0, 2.0])

    def test_pts_zero_start_no_offset(self):
        rel, info, err = check_pts(np.arange(90) / 30.0, [45], 30.0)
        self.assertIsNone(err)
        self.assertEqual(info["pts_offset_s"], 0.0)

    def test_pts_nonincreasing_or_negative_rejected(self):
        ts = np.arange(90) / 30.0
        ts[40] = ts[39]
        self.assertIsNotNone(check_pts(ts, [45], 30.0)[2])
        self.assertIsNotNone(check_pts(np.arange(90) / 30.0 - 0.2, [45], 30.0)[2])
        bad = np.arange(90) / 30.0
        bad[3] = np.nan
        self.assertIsNotNone(check_pts(bad, [45], 30.0)[2])

    def test_nonfinite_time_never_supervised(self):
        rec = ps.snippet_records(True, 20.0, np.arange(12) * 15, np.full(12, np.nan),
                                 np.ones(12, bool), np.zeros(12, bool))
        self.assertFalse(rec["cps_mask"].any())

    def test_zero_kinds(self):
        zero = np.zeros(12, bool)
        zero[[2, 3, 4]] = True
        kinds = [None] * 12
        kinds[2], kinds[3], kinds[4] = "decode_fail", "black", "cache_only"
        rec = ps.snippet_records(True, 50.0, np.arange(12) * 15 + 100, (np.arange(12) * 15 + 100) / 30.0,
                                 np.ones(12, bool), zero, zero_kind=kinds)
        self.assertEqual(rec["reason"][2:5], ["zero_decode_fail", "zero_black", "zero_cache_only"])
        self.assertFalse(rec["cps_mask"][2:5].any())
        with self.assertRaises(ValueError):
            ps.snippet_records(True, 50.0, np.arange(12), np.arange(12) / 30.0, np.ones(12, bool), zero,
                               zero_kind=["weird"] * 12)


class TestEvalProvenance(unittest.TestCase):
    def test_stale_predictions_detected(self):
        with tempfile.TemporaryDirectory() as d:
            ck = os.path.join(d, "best.pth")
            open(ck, "wb").write(b"weights-v1")
            out = os.path.join(d, "run_best.npz")
            np.savez(out, scores=np.zeros(3))
            json.dump({"ckpt_sha256": ev.file_sha(ck), "val_cache_digest": "v"}, open(out.replace(".npz", ".json"), "w"))
            self.assertTrue(ev.preds_fresh(out, ck, "v"))
            self.assertFalse(ev.preds_fresh(out, ck, "other-val-cache"))
            ck2 = os.path.join(d, "best2.pth")
            open(ck2, "wb").write(b"weights-v2 (retrained)")
            self.assertFalse(ev.preds_fresh(out, ck2, "v"))
            dense = os.path.join(d, "run.npz")
            np.savez(dense, scores=np.zeros((2, 3)), ckpt_sha256=np.array(ev.file_sha(ck)),
                     dense_cache_digest=np.array("m"))
            self.assertTrue(ev.dense_fresh(dense, ck, "m"))
            self.assertFalse(ev.dense_fresh(dense, ck2, "m"))
            np.savez(dense, scores=np.zeros((2, 3)))       # old file without provenance
            self.assertFalse(ev.dense_fresh(dense, ck, "m"))


class TestDenseGateAndDecision(unittest.TestCase):
    D = [round(3.0 - 0.1 * k, 1) for k in range(30)]

    def _meta(self, toe=10.0):
        return {"vid_ids": ["00001", "00002"], "targets": [1, 0], "failed": [], "dense_d": self.D,
                "t_obs": [[toe - d for d in self.D], [5.0] * 30]}

    def _vt(self, d, toe=10.0, dev=0.01, future=False, vfr=False, req=None):
        """audit_data-style val_timing.csv for the positive 00001 (30 dense rows)."""
        import pandas as pd
        req = req if req is not None else [toe - x for x in self.D]
        rows = [{"vid": "00001", "target": 1, "kind": "dense", "k": k, "requested_end_s": r,
                 "end_frame": int(r * 30), "actual_end_s": r + dev, "time_of_event": toe,
                 "future_frame": future and k == 29, "dev_s": dev, "dev_frames": dev * 30,
                 "frame_period_s": 1 / 30, "vfr_suspect": vfr} for k, r in enumerate(req)]
        p = os.path.join(d, "val_timing.csv")
        pd.DataFrame(rows).to_csv(p, index=False)
        return p

    def gate(self, m, ids, toe, vt):
        return ev.dense_gate(m, ids, toe, vt, expected=(2, 1))

    def test_clean_grid_passes(self):
        with tempfile.TemporaryDirectory() as d:
            g = self.gate(self._meta(), ["00001", "00002"], {"00001": 10.0}, self._vt(d))
            self.assertTrue(g["pass"], g["issues"])

    def test_clamped_duplicate_post_event_and_pts(self):
        m = self._meta(toe=2.0)                    # 2.0 - 3.0 < 0.4 -> clamped + duplicates in practice
        m["t_obs"][0] = [max(0.4, 2.0 - d) for d in self.D]
        with tempfile.TemporaryDirectory() as d:
            g = self.gate(m, ["00001", "00002"], {"00001": 2.0}, self._vt(d, toe=2.0, req=m["t_obs"][0]))
            self.assertFalse(g["pass"])
            self.assertGreater(g["counts"]["clamped"], 0)
            self.assertGreater(g["counts"]["duplicate"], 0)
            m2 = self._meta()
            m2["t_obs"][0][-1] = 10.0                  # requested endpoint at the event
            self.assertGreater(self.gate(m2, ["00001", "00002"], {"00001": 10.0},
                                         self._vt(d))["counts"]["post_event"], 0)
            # actual endpoint more than one frame away from the requested one
            g3 = self.gate(self._meta(), ["00001", "00002"], {"00001": 10.0}, self._vt(d, dev=0.05))
            self.assertEqual(g3["counts"]["dev_gt_1_frame"], 1)
            # actual endpoint at/after the event, VFR video, audit made for another dense cache
            self.assertFalse(self.gate(self._meta(), ["00001", "00002"], {"00001": 10.0}, self._vt(d, future=True))["pass"])
            self.assertEqual(self.gate(self._meta(), ["00001", "00002"], {"00001": 10.0},
                                       self._vt(d, vfr=True))["counts"]["vfr"], 1)
            self.assertEqual(self.gate(self._meta(), ["00001", "00002"], {"00001": 10.0},
                                       self._vt(d, req=[9.0 - x for x in self.D]))["counts"]["audit_mismatch"], 1)
            # coverage must be exactly the expected 300 / 150 by default
            self.assertFalse(ev.dense_gate(self._meta(), ["00001", "00002"], {"00001": 10.0}, self._vt(d))["pass"])
        self.assertFalse(self.gate(self._meta(), ["00001", "00002"], {"00001": 10.0}, "")["pass"])
        self.assertFalse(self.gate(self._meta(), ["00002", "00001"], {"00001": 10.0}, "")["pass"])

    def test_decision_rule(self):
        import pandas as pd
        lock = {"primary_contrast": "P - B", "primary_metric": "AP@1.5s", "ni_metric": "mAP",
                "ni_margin": 0.02, "ps_metric": "AP@1.5s"}

        def ci(p_lo, ni_lo):
            return pd.DataFrame([
                dict(contrast="P - B", metric="AP@1.5s", estimate=0.03, ci_low=p_lo, ci_high=0.06, verdict=""),
                dict(contrast="P - B", metric="mAP", estimate=0.0, ci_low=ni_lo, ci_high=0.02, verdict=""),
                dict(contrast="P - S", metric="AP@1.5s", estimate=0.01, ci_low=-0.01, ci_high=0.03,
                     verdict="inconclusive (CI includes 0)")])
        d = ev.decision(ci(0.005, -0.015), lock)
        self.assertTrue(d["primary_met"] and d["non_inferiority_met"] and d["rq3_supported"])
        self.assertIsNotNone(d["p_minus_s"])                     # reported even though CI includes 0
        self.assertFalse(ev.decision(ci(-0.001, -0.015), lock)["rq3_supported"])   # primary CI includes 0
        self.assertFalse(ev.decision(ci(0.005, -0.025), lock)["rq3_supported"])    # NI fails
        self.assertFalse(ev.decision(ci(0.005, -0.015), {})["available"])


class RecordingModel(torch.nn.Module):
    """Returns fixed logits and records exactly what forward() received."""

    def __init__(self, logits):
        super().__init__()
        self.logits = logits
        self.calls = []

    def forward(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.logits.clone().requires_grad_(True)


class TestModelContract(unittest.TestCase):
    def test_model_sees_only_video(self):
        logits = torch.zeros(2, 12)
        m = RecordingModel(logits)
        seq = torch.zeros(2, 12, 3, 5, 8, 8)
        tau = torch.full((2, 12), 1.0)
        tr.forward_losses(m, seq, torch.tensor([1.0, 0.0]), 0.5, tau, torch.ones(2, 12, dtype=torch.bool),
                          cfg_for("P"))
        args, kwargs = m.calls[0]
        self.assertEqual(len(args), 1)
        self.assertEqual(kwargs, {})
        self.assertIs(args[0], seq)

    def test_dummy_model_shapes(self):
        m = tr.build_model("dummy")
        self.assertEqual(tuple(m(torch.zeros(2, 12, 3, 5, 8, 8)).shape), (2, 12))
        self.assertEqual(tuple(m(torch.zeros(3, 3, 5, 8, 8)).shape), (3,))


class TestDataAndGates(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = self.tmp.name
        self.cache = os.path.join(d, "cache_rp")
        os.makedirs(os.path.join(self.cache, "train"))
        self.ids = ["00001", "00002", "00003", "00004"]
        idx = {}
        for i, v in enumerate(self.ids):
            tgt = float(i % 2 == 0)
            torch.save({"frames": torch.randint(0, 255, (12, 5, 8, 8, 3), dtype=torch.uint8),
                        "tau": torch.linspace(5.6, 0.1, 12) if tgt else torch.full((12,), float("inf")),
                        "target": tgt, "dt": 0.5}, os.path.join(self.cache, "train", f"{v}.pt"))
            idx[v] = f"{v}.pt"
        json.dump(idx, open(os.path.join(self.cache, "train_index.json"), "w"))
        self.side = {}
        for i, v in enumerate(self.ids):
            pos = i % 2 == 0
            self.side[v] = {"target": int(pos),
                            "tau_actual": [5.6 - 0.5 * k for k in range(12)] if pos else [None] * 12,
                            "cps_mask": [0 < k < 11 for k in range(12)] if pos else [False] * 12}
        self.c5 = os.path.join(d, "c5")
        os.makedirs(self.c5)
        json.dump({"00009": {}}, open(os.path.join(self.c5, "val_index.json"), "w"))
        self.man = os.path.join(d, "man.json")
        json.dump({"train_ids": self.ids, "val_ids": ["00009"]}, open(self.man, "w"))

    def tearDown(self):
        self.tmp.cleanup()

    def test_dataset_items(self):
        ds = tr.make_train_dataset(self.cache, self.side, 0)
        seq, tau, tgt, dt, ctau, cmask = ds[ds.vid_ids.index("00001")]
        self.assertEqual(tuple(seq.shape), (12, 3, 5, 8, 8))
        self.assertEqual(int(cmask.sum()), 10)
        seq, tau, tgt, dt, ctau, cmask = ds[ds.vid_ids.index("00002")]
        self.assertFalse(cmask.any())
        self.assertTrue(torch.isinf(ctau).all())

    def test_sidecar_target_mismatch_raises(self):
        bad = dict(self.side)
        bad["00002"] = dict(self.side["00001"])
        ds = tr.make_train_dataset(self.cache, bad, 0)
        with self.assertRaises(ValueError):
            ds[ds.vid_ids.index("00002")]

    def test_limit_videos_balanced(self):
        ds = tr.make_train_dataset(self.cache, self.side, 2)
        self.assertEqual(sorted(self.side[v]["target"] for v in ds.vid_ids), [0, 1])
        self.assertEqual(choose_subset(self.ids, {v: self.side[v]["target"] for v in self.ids}, 3),
                         ["00001", "00002", "00003"])

    def _args(self, extra):
        return tr.parse_args(["--condition", "P", "--seed", "42", "--pairing-mode", "fixed",
                              "--lambda-prog", "1.0", "--cache-root", self.cache, "--cache-5f", self.c5,
                              "--sidecar", "x.json", "--split-manifest", self.man,
                              "--output-root", self.tmp.name] + extra)

    def test_full_run_refused_without_lock_and_audit(self):
        why = tr.full_run_gate(self._args(["--full-run"]), {"usable_for_full_run": True})
        self.assertTrue(any("audit" in w for w in why))
        self.assertTrue(any("lock" in w for w in why))

    def test_full_run_refused_with_limits_or_wrong_lambda(self):
        lock = os.path.join(self.tmp.name, "LOCK.json")
        lk = {"confirmed": True, "pairing_mode": "fixed", "conditions": ["B", "P", "S"],
              "seeds": [42, 43, 44], "horizon_sec": 2.0, "alpha": 3.0, "lambda_prog": 0.5,
              "primary_contrast": "P - B", "primary_metric": "AP@1.5s", "ni_metric": "mAP",
              "ni_margin": 0.02, "ps_metric": "AP@1.5s", "success_rule": "primary CI>0 and NI"}
        json.dump(lk, open(lock, "w"))
        audit = os.path.join(self.tmp.name, "audit.json")
        json.dump({"pass": True, "limited": False, "complete": True, "probed_videos": 4, "scanned_cache": 4},
                  open(audit, "w"))
        good0 = self._good_meta()
        current = {"code_digest": "code1", "sidecar_sha256": "side1", "audit_sha256": tr.sha256_file(audit),
                   "val_cache_digest": "val1", "split_manifest_sha256": good0["split_manifest_sha256"],
                   "train_index_sha256": good0["train_index_sha256"], "train_cache_digest": good0["cache_digest"]}
        lk["fingerprints"] = {"code": {"digest": "code1"}, "sidecar_sha256": "side1",
                              "audit_sha256": current["audit_sha256"], "val_cache_digest": "val1",
                              "split_manifest_sha256": current["split_manifest_sha256"],
                              "train_index_sha256": current["train_index_sha256"],
                              "train_cache_digest": current["train_cache_digest"]}
        json.dump(lk, open(lock, "w"))
        why = tr.full_run_gate(self._args(["--full-run", "--lock-file", lock, "--audit", audit,
                                           "--max-steps", "1"]), {"usable_for_full_run": True})
        self.assertTrue(any("limits" in w for w in why))
        self.assertTrue(any("lambda" in w for w in why))
        a = self._args(["--full-run", "--lock-file", lock, "--audit", audit, "--lambda-prog", "0.5"])
        good = self._good_meta()
        self.assertEqual(tr.full_run_gate(a, good, current), [])
        lk2 = dict(lk); lk2.pop("ps_metric")                       # decision fields are mandatory
        json.dump(lk2, open(lock, "w"))
        self.assertTrue(any("decision fields" in w for w in tr.full_run_gate(a, good, current)))
        json.dump(lk, open(lock, "w"))
        self.assertTrue(tr.full_run_gate(a, dict(good, usable_for_full_run=False), current))
        # too few / non-identical pixel checks
        self.assertTrue(any("pixel" in w for w in tr.full_run_gate(a, dict(good, pixel_verified=8, pixel_identical=8), current)))
        self.assertTrue(any("pixel" in w for w in tr.full_run_gate(a, dict(good, pixel_identical=23), current)))
        # sidecar built for another split / index / cache content
        self.assertTrue(any("split manifest" in w for w in
                            tr.full_run_gate(a, dict(good, split_manifest_sha256="x"), current)))
        self.assertTrue(any("train_index" in w for w in
                            tr.full_run_gate(a, dict(good, train_index_sha256="x"), current)))
        # V4-09: code / sidecar / val cache changed after LOCK
        for key, label in (("code_digest", "code"), ("sidecar_sha256", "sidecar"), ("val_cache_digest", "val cache")):
            why = tr.full_run_gate(a, good, dict(current, **{key: "changed"}))
            self.assertTrue(any(w.startswith(label) for w in why), (key, why))
        # V4-05: an audit that is not complete is refused
        json.dump({"pass": True, "limited": False, "complete": False}, open(audit, "w"))
        self.assertTrue(any("complete" in w for w in tr.full_run_gate(a, good, current)))
        pt = os.path.join(self.cache, "train", "00003.pt")
        d = torch.load(pt, weights_only=False)
        d["frames"][0] = 0
        torch.save(d, pt)                                  # cache changed after the sidecar
        self.assertTrue(any("cache_digest" in w for w in tr.full_run_gate(a, good, current)))

    def _good_meta(self):
        tr_index = json.load(open(os.path.join(self.cache, "train_index.json")))
        dig, _ = cache_digest(self.cache, tr_index, self.ids)
        return {"usable_for_full_run": True, "pixel_verified": 24, "pixel_identical": 24,
                "split_manifest_sha256": tr.sha256_file(self.man),
                "train_index_sha256": tr.sha256_file(os.path.join(self.cache, "train_index.json")),
                "cache_digest": dig, "cache_digest_ids": len(self.ids)}

    def test_dev_run_refuses_mismatched_sidecar(self):
        a = self._args([])
        self.assertEqual(tr.sidecar_binding_errors(a, self._good_meta()), [])
        self.assertTrue(tr.sidecar_binding_errors(a, dict(self._good_meta(), train_index_sha256="old")))

    def test_resume_fingerprint(self):
        cfg = {"condition": "P", "cps_mode": "continuous", "use_ffr": True, "use_amc": True,
               "pairing_mode": "fixed", "horizon_sec": 2.0, "alpha": 3.0, "lambda_prog": 1.0, "seed": 42,
               "batch_size": 2, "lr": 0.01, "momentum": 0.9, "weight_decay": 1e-4, "model": "riskprop",
               "limit_videos": 0, "run_name": "P", "epochs": 50}
        man = {"inputs": {"split_manifest_sha256": "a", "train_index_sha256": "b", "val_index_sha256": "c",
                          "sidecar_sha256": "d", "lock_sha256": "e"}}
        fp = tr.run_fingerprint(cfg, man)
        self.assertEqual(fp, tr.run_fingerprint(dict(cfg, epochs=60), man))      # longer run: same run
        self.assertNotEqual(fp, tr.run_fingerprint(dict(cfg, lambda_prog=0.5), man))
        man2 = {"inputs": dict(man["inputs"], sidecar_sha256="NEW")}
        self.assertNotEqual(fp, tr.run_fingerprint(cfg, man2))

    def test_lambda_rules(self):
        with self.assertRaises(SystemExit):
            tr.parse_args(["--condition", "P", "--seed", "42", "--pairing-mode", "fixed", "--sidecar", "x",
                           "--cache-root", "c", "--cache-5f", "c", "--split-manifest", "m", "--output-root", "o"])
        with self.assertRaises(SystemExit):
            tr.parse_args(["--condition", "B", "--seed", "42", "--pairing-mode", "fixed", "--lambda-prog", "1",
                           "--cache-root", "c", "--cache-5f", "c", "--split-manifest", "m", "--output-root", "o"])
        self.assertEqual(tr.run_name("B", "fixed", 2.0, 3.0, 1.0, 42), "B_fixed_seed42")
        self.assertEqual(tr.run_name("B", "fixed", 1.5, 3.0, 1.0, 42), "B_fixed_seed42")
        self.assertEqual(tr.run_name("P", "fixed", 1.5, 3.0, 1.0, 42, "smoke"), "P_fixed_H1.5_a3_lam1_seed42_smoke")


if __name__ == "__main__":
    unittest.main(verbosity=2)
