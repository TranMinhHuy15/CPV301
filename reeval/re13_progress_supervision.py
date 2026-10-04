"""
re13 -- Continuous Progress Supervision (CPS) for Progressive RiskProp.
Pure functions only: target generators, snippet-validity rules and the CPS
loss. No model, no dataset, no video decoding -> unit-testable on CPU
(reeval/re16_test_progress_protocol.py).

PRE-ACT reference (github.com/giddyyupp/PRE-ACT @ 5e0d38fb,
engine/risk_targets.py::exp_above_linear, losses/preact.py):
    ttc in 10-fps-equivalent frames, horizon 20 frames (= 2.0 s)
    ttc <= 0   -> 1.0            (PRE-ACT)
    ttc >= 20  -> 0.0
    else u = 1 - ttc/20,  r = (1 - exp(-alpha*u)) / (1 - exp(-alpha))
    progress loss = F.smooth_l1_loss(progress_logit, risk_target) on a
    SEPARATE progress head (raw output, no sigmoid).

CPV301 adaptation (documented differences, NOT a reproduction of PRE-ACT):
  * Same curve, expressed in seconds: H = horizon (main 2.0 s, sensitivity
    1.5 s), tau = time_of_event - actual endpoint time of the snippet.
  * tau <= 0 (endpoint at/after the event) is EXCLUDED from CPS instead of
    being given target 1 (PRE-ACT). RiskProp's cache never intentionally
    samples after the event; any such snippet is a timing error.
  * No progress head: the loss is applied to RiskProp's single scalar score
    a = sigmoid(z). No preference loss. lambda_prog is NOT PRE-ACT's 10.
  * Smooth L1 with beta = 1 (torch default, same as PRE-ACT). Because
    |a - r| < 1 always, this equals 0.5 * (a - r)^2 here.
  * Reduction: mean over valid snippets inside each video, then mean over
    positive videos that have >= 1 valid snippet. No valid snippet in the
    batch -> loss = 0 (finite, still attached to the graph).

Condition S (dense binary control) uses the same mask and loss with
b = 1[0 < tau < H] (tau = H -> 0).
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

H_MAIN = 2.0
H_SENS = 1.5
ALPHA_MAIN = 3.0

# reason codes (first matching rule wins, in this order)
REASONS = ["ok", "negative", "anchor_first", "anchor_last", "decode_fail",
           "zero_decode_fail", "zero_black", "zero_cache_only", "zero_unchecked",
           "nonfinite_time", "post_event", "duplicate_endpoint", "nonmonotonic_time"]
# An all-zero cached snippet is only SUSPECT (cell30 writes zeros when decoding
# fails, but a genuinely black scene also gives zeros). re12 re-decodes every
# zero snippet and classifies it:
#   zero_decode_fail : re-decoding raises            -> decode failure confirmed
#   zero_black       : re-decoded pixels are zero too -> real black content
#   zero_cache_only  : re-decoded pixels are NOT zero -> cache corrupted / stale
#   zero_unchecked   : not re-decoded
# All four are excluded from CPS; B/P/S still see these pixels in BCE/FFR/AMC
# (unchanged RiskProp behaviour), so their counts must be reported.
ZERO_KINDS = ("decode_fail", "black", "cache_only", "unchecked")


# ----------------------------------------------------------------------------
# Targets
# ----------------------------------------------------------------------------
def continuous_target_scalar(tau, horizon=H_MAIN, alpha=ALPHA_MAIN):
    """Scalar version (seconds). Returns None when tau is not supervisable."""
    if tau is None or not math.isfinite(tau) or tau <= 0:
        return None
    if tau >= horizon:
        return 0.0
    u = 1.0 - tau / horizon
    return (1.0 - math.exp(-alpha * u)) / (1.0 - math.exp(-alpha))


def continuous_target(tau, horizon=H_MAIN, alpha=ALPHA_MAIN):
    """tau: tensor (any shape, seconds). Returns (target, valid) tensors.
    valid = finite & tau > 0. target is 0 where invalid (ignored by mask)."""
    tau = tau.float()
    valid = torch.isfinite(tau) & (tau > 0)
    t = torch.where(valid, tau, torch.full_like(tau, horizon))
    u = (1.0 - t / horizon).clamp(min=0.0)
    r = (1.0 - torch.exp(-alpha * u)) / (1.0 - math.exp(-alpha))
    r = torch.where(t >= horizon, torch.zeros_like(r), r)
    r = torch.where(valid, r, torch.zeros_like(r))
    return r.detach(), valid


def binary_target(tau, horizon=H_MAIN):
    """b = 1[0 < tau < H]; tau = H -> 0; tau <= 0 / non-finite invalid."""
    tau = tau.float()
    valid = torch.isfinite(tau) & (tau > 0)
    b = (valid & (tau < horizon)).float()
    return b.detach(), valid


# ----------------------------------------------------------------------------
# Snippet validity (used by re12 to build the sidecar)
# ----------------------------------------------------------------------------
def snippet_records(is_positive, toe, end_frames, end_times, decode_ok,
                    zero_frames, exclude_anchors=True, zero_kind=None):
    """
    One video's N snippets, ordered earliest -> latest (cell30 order).
      is_positive : bool
      toe         : float time_of_event (s) or nan for negatives
      end_frames  : (N,) int, index of the LAST frame actually decoded
      end_times   : (N,) float, its timestamp in seconds (PTS if available)
      decode_ok   : (N,) bool, frame extraction did not raise
      zero_frames : (N,) bool, cached snippet is all zeros (suspect)
      zero_kind   : optional (N,) str in ZERO_KINDS for the zero snippets
    Returns dict with tau_actual (N,), cps_mask (N,) bool, reason (N,) str.
    Rules (first match wins): negative, anchor_first (k=0), anchor_last
    (k=N-1), decode_fail, zero_<kind>, nonfinite_time, post_event (tau<=0),
    duplicate_endpoint (same end frame as the NEXT snippet -> only the latest
    snippet of a run of identical endpoints is kept), nonmonotonic_time.
    """
    end_frames = np.asarray(end_frames, dtype=np.int64)
    end_times = np.asarray(end_times, dtype=np.float64)
    decode_ok = np.asarray(decode_ok, dtype=bool)
    zero_frames = np.asarray(zero_frames, dtype=bool)
    n = len(end_frames)
    tau = np.full(n, np.inf)
    if is_positive and toe is not None and np.isfinite(toe):
        tau = float(toe) - end_times
    reason = []
    last_kept_time = -np.inf
    for k in range(n):
        if not is_positive:
            r = "negative"
        elif exclude_anchors and k == 0:
            r = "anchor_first"
        elif exclude_anchors and k == n - 1:
            r = "anchor_last"
        elif not decode_ok[k]:
            r = "decode_fail"
        elif zero_frames[k]:
            kind = zero_kind[k] if zero_kind is not None and zero_kind[k] else "unchecked"
            if kind not in ZERO_KINDS:
                raise ValueError(f"unknown zero kind {kind!r}")
            r = f"zero_{kind}"
        elif not (np.isfinite(end_times[k]) and np.isfinite(tau[k])):
            r = "nonfinite_time"
        elif tau[k] <= 0:
            r = "post_event"
        elif k + 1 < n and end_frames[k] == end_frames[k + 1]:
            r = "duplicate_endpoint"
        elif end_times[k] <= last_kept_time:
            r = "nonmonotonic_time"
        else:
            r = "ok"
        if r == "ok":
            last_kept_time = end_times[k]
        reason.append(r)
    mask = np.array([r == "ok" for r in reason], dtype=bool)
    if not is_positive:
        tau = np.full(n, np.inf)
    return {"tau_actual": tau, "cps_mask": mask, "reason": reason}


# ----------------------------------------------------------------------------
# Loss
# ----------------------------------------------------------------------------
def cps_loss(logits, cps_tau, cps_mask, mode, horizon=H_MAIN, alpha=ALPHA_MAIN,
             beta=1.0):
    """
    logits   : (B, N) raw RiskProp z (any dtype; computed in fp32 here)
    cps_tau  : (B, N) actual time-to-event per snippet (inf for negatives)
    cps_mask : (B, N) bool/0-1, sidecar validity (False for negatives)
    mode     : "none" (B) | "continuous" (P) | "binary" (S)
    Returns (loss scalar tensor, stats dict).
    """
    zero = logits.float().sum() * 0.0
    if mode == "none":
        return zero, {"cps": 0.0, "cps_videos": 0, "cps_snippets": 0}
    a = torch.sigmoid(logits.float())
    tau = cps_tau.to(a.device).float()
    if mode == "continuous":
        tgt, valid = continuous_target(tau, horizon, alpha)
    elif mode == "binary":
        tgt, valid = binary_target(tau, horizon)
    else:
        raise ValueError(f"unknown CPS mode {mode!r}")
    m = valid & cps_mask.to(a.device).bool()
    if not bool(m.any()):
        return zero, {"cps": 0.0, "cps_videos": 0, "cps_snippets": 0}
    elem = F.smooth_l1_loss(a, tgt, reduction="none", beta=beta)
    mf = m.float()
    n_per_video = mf.sum(dim=1)
    has = n_per_video > 0
    per_video = (elem * mf).sum(dim=1)[has] / n_per_video[has]
    loss = per_video.mean()
    return loss, {"cps": float(loss.detach()), "cps_videos": int(has.sum()),
                  "cps_snippets": int(mf.sum())}


def grid_table(horizon=H_MAIN, alpha=ALPHA_MAIN, n=12, stride=0.5, last_tau=0.1):
    """Nominal 12-snippet grid -> (tau, continuous, binary). For logs/docs."""
    rows = []
    for k in range(n):
        tau = last_tau + (n - 1 - k) * stride
        c = continuous_target_scalar(tau, horizon, alpha)
        b = 1.0 if 0 < tau < horizon else 0.0
        rows.append((k, round(tau, 3), None if c is None else round(c, 4), b))
    return rows


if __name__ == "__main__":
    print("k  tau    P-target  S-target   (H=2.0, alpha=3; k=0 and k=11 are BCE anchors)")
    for k, tau, c, b in grid_table():
        print(f"{k:2d} {tau:5.2f}  {c:8.4f}  {b:4.1f}")
