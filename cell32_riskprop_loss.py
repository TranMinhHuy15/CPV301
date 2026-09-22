"""
RiskProp losses (Zou et al., CVPR 2026), adapted to a per-video SEQUENCE
of N=12 causal 5-frame snippets (stride 0.5s) instead of the paper's
full-video frame-by-frame sequence -- team decision (see cell30 docstring)
to keep precache/train time feasible for a 1-GPU Kaggle simulation while
still giving FFR/AMC a real ordered chain to operate on.

L = L_bce + lambda1 * L_reg + lambda2 * L_mono          (paper Eq. 7)

L_bce   (Eq. 6): weighted BCE. Positive video: labeled ONLY at the first
                 snippet (k=0, y=0) and last/collision snippet (k=N-1,
                 y=1), collision frame gets extra weight (exact weight
                 value NOT given in the paper -- "higher weight... " is
                 stated qualitatively only; ADAPTED: reuse AdaLEA's
                 documented pos_weight=5.0 for consistency across the
                 project's baselines). Negative video: ALL N snippets
                 labeled y=0, normal weight -- monotonicity/collision
                 anchor don't apply (paper Sec 3.4), so FFR/AMC are
                 disabled for negative videos.
L_reg   (Eq. 1): Future-Frame Regularization -- z_k learns toward
                 detach(z_{k+1}), chained backward from the collision
                 snippet. Positive videos only.
L_mono  (Eq. 3/4/5): Adaptive Monotonic Constraint -- randomly sampled
                 snippet pairs (i,j), j>i, penalized if a_i > a_j - delta.
                 delta adapts to temporal gap AND prediction confidence.
                 Positive videos only.

Sampling mode (paper's own default = "random", Fig. 3): offset
d ~ Uniform(d_min, d_max) of the sequence length, mapped to our discrete
N=12 grid as gap = round(d * (N-1)). This IS the paper's default
"RiskProp-full" config used for today's RQ1 run. A "fixed" mode (constant
gap, no d~Uniform) is included for RQ3 (fixed-lag vs random-offset
temporal pairing -- the project's actual research question) but NOT used
today.

USE_FFR / USE_AMC flags let cell34's training script be reused unchanged
for RQ2's 4-way ablation (neither / FFR-only / AMC-only / both) later --
today's RQ1 run uses the full model (both True).

---
v2 UPDATE (post-RQ1 optimization pass, per partner feedback
"KE HOACH KHAC PHUC VA TOI UU RISKPROP"):
- lambda_reg / lambda_mono: 1.5/1.1 -> 0.5/0.5 (was suppressing mAP by
  over-weighting the temporal-consistency terms vs. the primary BCE task).
- COLLISION_WEIGHT: 5.0 -> 8.0 (heavier weight on the true collision
  anchor frame, aimed at improving recall/coverage).
- NUM_PAIRS_PER_VIDEO: 4 -> 8 (denser AMC pair sampling per step, for a
  more stable/representative monotonicity estimate per batch).
- Default pairing_mode: "random" -> "fixed" with FIXED_LAG_SEC=1.0s
  (partner's plan prioritizes fixed 1.0s lag to reduce AMC noise vs.
  paper's default random-offset sampling; "random" is kept available via
  the pairing_mode argument for the RQ3 comparison later).
- L_reg (FFR) now computed on sigmoid probabilities a_t=sigma(z_t)
  instead of raw logits z_t, for better-conditioned targets in [0,1]
  (partner-proposed alternative formula, Eq. L_FFR on probabilities).
"""
import torch
import torch.nn.functional as F

# ============ CONFIG (v2, post-optimization -- see docstring above) ============
LAMBDA1 = 0.5     # L_reg (FFR) weight -- was 1.5
LAMBDA2 = 0.5     # L_mono (AMC) weight -- was 1.1
D_MIN, D_MAX = 0.1, 0.9   # AMC pair-offset range (used only when pairing_mode="random")
DELTA0 = 0.01              # AMC margin scaling coefficient (paper's stated value)
COLLISION_WEIGHT = 8.0     # was 5.0
NEG_WEIGHT = 1.0
NUM_PAIRS_PER_VIDEO = 8     # was 4
FIXED_LAG_SEC = 1.0        # default fixed AMC pairing gap, in seconds (v2 default)
EPS = 1e-7


def riskprop_loss(logits, target, dt=0.5, use_ffr=True, use_amc=True,
                   pairing_mode="fixed", fixed_gap=None,
                   lambda1=LAMBDA1, lambda2=LAMBDA2):
    """
    logits : (B, N) raw z_t (pre-sigmoid) for N ordered snippets/video
    target : (B,) 0/1 video-level label
    dt     : seconds between consecutive snippets (stride, e.g. 0.5)
    pairing_mode : "fixed" (v2 default -- constant gap corresponding to
                   FIXED_LAG_SEC, or fixed_gap in snippet-steps if given
                   explicitly) or "random" (paper's original default,
                   d~Uniform(D_MIN,D_MAX) -- kept for the RQ3 comparison)
    Returns: scalar total loss, dict of component losses (for logging)
    """
    B, N = logits.shape
    device = logits.device
    pos_mask = (target == 1)
    a = torch.sigmoid(logits).clamp(EPS, 1 - EPS)

    # ---------------- L_bce (Eq. 6) ----------------
    y = torch.zeros_like(logits)
    w = torch.full_like(logits, NEG_WEIGHT)
    valid = torch.zeros_like(logits)

    if pos_mask.any():
        y[pos_mask, 0] = 0.0
        y[pos_mask, N - 1] = 1.0
        w[pos_mask, 0] = 1.0
        w[pos_mask, N - 1] = COLLISION_WEIGHT
        valid[pos_mask, 0] = 1.0
        valid[pos_mask, N - 1] = 1.0
    if (~pos_mask).any():
        valid[~pos_mask, :] = 1.0
        w[~pos_mask, :] = NEG_WEIGHT

    bce_elem = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    denom = (valid * w).sum().clamp_min(EPS)
    L_bce = (bce_elem * valid * w).sum() / denom

    # ---------------- L_reg (Eq. 1) -- FFR, positive videos only ----------------
    # v2: computed on sigmoid probabilities a_t=sigma(z_t) instead of raw
    # logits z_t (partner-proposed alternative formula) -- keeps the
    # regularization target bounded in [0,1], better-conditioned than
    # unbounded raw logits.
    L_reg = torch.tensor(0.0, device=device)
    if use_ffr and pos_mask.any():
        a_pos_ffr = a[pos_mask]                         # (P, N), sigmoid probs
        target_next = a_pos_ffr[:, 1:].detach()
        L_reg = ((target_next - a_pos_ffr[:, :-1]) ** 2).mean()

    # ---------------- L_mono (Eq. 3/4/5) -- AMC, positive videos only ----------------
    L_mono = torch.tensor(0.0, device=device)
    if use_amc and pos_mask.any():
        a_pos = a[pos_mask]                             # (P, N)
        P = a_pos.shape[0]
        a_bar = a_pos.mean()                            # batch-wise mean over positive videos
        terms = []
        for _ in range(NUM_PAIRS_PER_VIDEO):
            if pairing_mode == "fixed":
                # v2 default: gap = FIXED_LAG_SEC worth of snippet-steps
                # (round(1.0s / dt), e.g. dt=0.5s -> gap=2 snippet-steps),
                # unless an explicit fixed_gap (in snippet-steps) is given.
                default_gap = max(1, int(round(FIXED_LAG_SEC / dt)))
                gap = torch.full((P,), fixed_gap if fixed_gap is not None
                                  else default_gap,
                                  device=device, dtype=torch.long)
            else:
                d = torch.empty(P, device=device).uniform_(D_MIN, D_MAX)
                gap = (d * (N - 1)).round().long().clamp(1, N - 1)
            i = (torch.rand(P, device=device) * (N - gap).clamp_min(1).float()).long()
            i = i.clamp(0, N - 1)
            j = (i + gap).clamp(0, N - 1)
            ai = a_pos.gather(1, i.unsqueeze(1)).squeeze(1)
            aj = a_pos.gather(1, j.unsqueeze(1)).squeeze(1)
            delta_t = (j - i).float() * dt
            ci = 2 * (ai - a_bar).abs()
            cj = 2 * (aj - a_bar).abs()
            c_bar = (ci + cj) / 2
            delta = DELTA0 * delta_t * c_bar
            terms.append(F.relu(ai - aj + delta))
        L_mono = torch.cat(terms).mean()

    total = L_bce + lambda1 * L_reg + lambda2 * L_mono
    return total, {"bce": L_bce.item(), "reg": float(L_reg), "mono": float(L_mono)}
