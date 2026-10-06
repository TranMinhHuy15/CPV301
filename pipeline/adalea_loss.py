"""
AdaLEA loss (Suzuki et al., CVPR 2018), continuous-time (seconds)
formulation, adapted to single-causal-window supervision (one
(frames, tau, target) sample per item -- matches TOP/RiskProp's shared
RQ1 protocol) instead of the paper's per-frame QRNN roll-out.

    alpha(tau, Phi) = 1.0                                    if tau <= Phi + gamma_sec
                    = exp(-(tau - (Phi + gamma_sec)) / sigma) otherwise
    L_pos = -pos_weight * alpha * log(r)
    L_neg = -log(1 - r)

gamma_sec = 0.5s  (paper's gamma=5 frames @ F=10fps -> 5/10 = 0.5s)
sigma     = 0.1s  (exact math translation of the paper's implicit decay
                    rate: alpha=exp(-max(0,d-F*Phi-gamma)) with d,gamma
                    in frames == exp(-F*max(0,tau-Phi-gamma_sec)) in
                    seconds, i.e. sigma = 1/F = 0.1s -- kept faithful to
                    the original paper's math, per team decision)
pos_weight = 5.0   (extra class-imbalance weighting on top of alpha,
                    documented adaptation -- not in the original paper)
Phi bounds = [0.3s, 2.0s]  (keeps the curriculum from stalling: never
                    lets gamma_sec+Phi collapse the margin to ~0, and
                    never lets it exceed the eval horizon range)
Phi(0)     = 0.5s  (safe warm start; ATTC is noisy/unreliable in the
                    first ~10-15 epochs per the paper's own Fig. 5)
"""
import torch
import numpy as np
from sklearn.metrics import roc_curve

GAMMA_SEC = 0.5
SIGMA = 0.1
POS_WEIGHT = 5.0
PHI_MIN, PHI_MAX = 0.3, 2.0
PHI_INIT = 0.5
EMA_BETA = 0.9   # Phi_e = EMA_BETA * Phi_(e-1) + (1-EMA_BETA) * ATTC_measured
THRESH_FAR = 0.1  # FAR<=0.1 threshold used to pick theta per lead time
EPS = 1e-7


def adalea_loss(logits, tau, target, phi):
    """
    logits : (B,) raw model output (pre-sigmoid)
    tau    : (B,) time-to-accident in seconds (float('inf') for negatives)
    target : (B,) 0/1
    phi    : float, current Phi (already EMA-updated, clamped)
    Returns: scalar loss (mean over batch)
    """
    logits = logits.float()  # force fp32: under AMP autocast, torch.log()
                              # promotes its output to fp32 regardless of
                              # input dtype, so keeping logits/r in fp16
                              # here causes a Half<-Float scatter mismatch
    r = torch.sigmoid(logits).clamp(EPS, 1 - EPS)
    pos = (target == 1)
    loss = torch.zeros_like(r)

    if (~pos).any():
        loss[~pos] = -torch.log(1 - r[~pos])

    if pos.any():
        margin = phi + GAMMA_SEC
        excess = (tau[pos] - margin).clamp(min=0.0)
        alpha = torch.exp(-excess / SIGMA)
        loss[pos] = -POS_WEIGHT * alpha * torch.log(r[pos])

    return loss.mean()


def _theta_at_far(targets, scores, target_far=THRESH_FAR):
    """Same low-FAR threshold logic as the TOP/AdaLEA eval scripts."""
    fpr, tpr, thr = roc_curve(targets, scores)
    mask = fpr <= target_far
    if np.any(mask):
        idx = np.where(mask)[0][-1]
        return thr[idx]
    return 1.0  # nothing qualifies -> nothing passes


def measure_attc(lead_results):
    """
    lead_results: dict {lead_time: (scores(np.array), targets(np.array))}
                  for the 3 fixed val lead-times (0.5/1.0/1.5s), SAME
                  video order across leads (guaranteed by val_index.json
                  iteration order).
    For each positive video, find the FARTHEST lead-time at which the
    model's score clears that lead's own FAR<=0.1 threshold -- this is
    the model's "earliest confident detection distance". ATTC_measured
    is the mean of that quantity over positives with >=1 detected lead.
    Returns (attc_measured, n_detected_videos) -- attc_measured is None
    if no positive was detected at any lead this epoch.
    """
    leads = sorted(lead_results.keys())
    thetas = {l: _theta_at_far(lead_results[l][1], lead_results[l][0]) for l in leads}

    targets_ref = lead_results[leads[0]][1]
    pos_idx = np.where(targets_ref == 1)[0]

    detected_leads = []
    for i in pos_idx:
        hits = [l for l in leads if lead_results[l][0][i] >= thetas[l]]
        if hits:
            detected_leads.append(max(hits))

    if not detected_leads:
        return None, 0
    return float(np.mean(detected_leads)), len(detected_leads)


def update_phi(phi_prev, attc_measured):
    """EMA update + clamp. If attc_measured is None (no detections this
    epoch), Phi is left unchanged."""
    if attc_measured is None:
        return phi_prev
    phi_new = EMA_BETA * phi_prev + (1 - EMA_BETA) * attc_measured
    return float(np.clip(phi_new, PHI_MIN, PHI_MAX))
