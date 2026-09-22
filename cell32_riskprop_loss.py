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
"""
import torch
import torch.nn.functional as F

# ============ CONFIG (paper's stated values where given) ============
LAMBDA1 = 1.5     # L_reg weight (Sec 4.1, paper)
LAMBDA2 = 1.1     # L_mono weight (Sec 4.1, paper)
D_MIN, D_MAX = 0.1, 0.9   # AMC pair-offset range (paper's stated values)
DELTA0 = 0.01              # AMC margin scaling coefficient (paper's stated value)
COLLISION_WEIGHT = 5.0     # ADAPTED (paper underspecified w_i) -- matches
                            # AdaLEA's documented pos_weight=5.0
NEG_WEIGHT = 1.0
NUM_PAIRS_PER_VIDEO = 4     # AMC pairs sampled per positive video per step
EPS = 1e-7


def riskprop_loss(logits, target, dt=0.5, use_ffr=True, use_amc=True,
                   pairing_mode="random", fixed_gap=None,
                   lambda1=LAMBDA1, lambda2=LAMBDA2):
    """
    logits : (B, N) raw z_t (pre-sigmoid) for N ordered snippets/video
    target : (B,) 0/1 video-level label
    dt     : seconds between consecutive snippets (stride, e.g. 0.5)
    pairing_mode : "random" (paper default, d~Uniform(D_MIN,D_MAX)) or
                   "fixed" (constant gap = fixed_gap, for RQ3)
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
    L_reg = torch.tensor(0.0, device=device)
    if use_ffr and pos_mask.any():
        z_pos = logits[pos_mask]                       # (P, N)
        target_next = z_pos[:, 1:].detach()
        L_reg = ((target_next - z_pos[:, :-1]) ** 2).mean()

    # ---------------- L_mono (Eq. 3/4/5) -- AMC, positive videos only ----------------
    L_mono = torch.tensor(0.0, device=device)
    if use_amc and pos_mask.any():
        a_pos = a[pos_mask]                             # (P, N)
        P = a_pos.shape[0]
        a_bar = a_pos.mean()                            # batch-wise mean over positive videos
        terms = []
        for _ in range(NUM_PAIRS_PER_VIDEO):
            if pairing_mode == "fixed":
                gap = torch.full((P,), fixed_gap if fixed_gap is not None
                                  else max(1, int(round(0.5 * (N - 1)))),
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
