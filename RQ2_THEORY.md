# RQ2 — Theoretical Background and Metric Definitions

*Written for the team before running the RQ2 ablation. Covers: what RQ2 asks, the exact algorithm/losses used, every metric with its formula, theoretical predictions per condition, and how this connects to RQ1/RQ3.*

---

## 1. The question (from the proposal)

> **RQ2**: Within RiskProp, how much do future-frame regularization (FFR) and random-pair adaptive monotonic constraint (AMC) contribute — **individually and jointly** — to (1) accuracy, (2) low-FAR anticipation, (3) temporal monotonicity, and (4) run-to-run stability?

This is a **2×2 ablation**: train RiskProp 4 ways, toggling FFR and AMC on/off, and measure the 4 outcome groups above. All 4 conditions share the *same* backbone, split, batch construction, optimizer schedule, pair count, and stopping rule (proposal §5.4, "Controls") — only the loss terms differ.

| Condition | FFR | AMC | Status |
|---|---|---|---|
| A — neither | off | off | to train (3 seeds) |
| B — FFR-only | on | off | to train (3 seeds) |
| C — AMC-only (random) | off | on | to train (3 seeds) |
| D — both | on | on | **already trained** — this is "RiskProp random" from RQ1/RQ3 |

Condition D is reused unchanged (same 3 checkpoints already on GitHub). Only A, B, C are new: **9 training runs**.

---

## 2. The algorithm: RiskProp (sequence formulation)

Per video, we extract a sequence of **N = 12 causal 5-frame snippets**, spaced **dt = 0.5 s** apart (stride), ending near the collision for positive videos, or centered on the clip midpoint for negative videos (proposal §5.3, project's own adaptation §5.2 — the published paper uses per-frame supervision on the full video; this project uses a snippet-sequence approximation to keep training feasible on one GPU, and this deviation is disclosed).

For each snippet `k`, a SlowOnly-R50 encoder (same backbone as TOP/AdaLEA, 224×224 input, 10 fps sampling) produces a raw logit `z_k`, converted to a risk score `a_k = sigmoid(z_k) ∈ (0,1)`.

**Total loss** (proposal §5.3, Eq. form): 

```
L = L_BCE + λ1 · L_FFR + λ2 · L_AMC
```

with **λ1 = λ2 = 0.5** (project's tuned value; the RiskProp paper uses 1.5/1.1 — this is a documented deviation, see §7).

### 2.1 L_BCE — supervision anchor

Only 2 of the 12 snippets carry a direct label for a **positive** video:
- snippet `k=0` (earliest) → `y=0`
- snippet `k=N-1` (collision-adjacent) → `y=1`, with extra weight `COLLISION_WEIGHT = 8.0`

All 10 middle snippets of a positive video get **no direct BCE gradient** — this is intentional (the paper's design): the middle of the risk curve is shaped entirely by FFR/AMC, not by hand-picked labels. For a **negative** video, all 12 snippets are labeled `y=0` with weight 1.0 (no FFR/AMC applied to negatives — there is no collision anchor to propagate from or monotonic trend to enforce).

```
L_BCE = Σ (w_k · BCE(z_k, y_k)) / Σ w_k     (sum over labeled positions only)
```

**Why this matters for RQ2**: condition A (BCE-only) has almost no supervision for the 3 evaluation horizons (0.5/1.0/1.5s before the event), since those windows fall inside the 10 unlabeled middle snippets. Condition A is expected to generalize poorly there — this is the reference/floor condition, not a competitive baseline.

### 2.2 L_FFR — Future-Frame Regularization

```
L_FFR = mean over positive videos of  (a_{k+1}.detach() − a_k)²   for k = 0..N-2
```

Each snippet's score is pulled toward the **next** snippet's score, with the target `detach()`-ed (no gradient flows back through it). Because of the detach, information propagates in one direction only: **gradient flows into the earlier snippet**, pulled by the value at the later one. Chained across the whole sequence, this "leaks" the collision-adjacent signal (`a_{N-1}≈1`) backward toward earlier snippets — this is the literal mechanism behind the name "RiskProp" (risk *propagation*).

*Implementation note*: computed on **sigmoid probabilities** `a_k`, not raw logits `z_k` (project's v2 change, kept in the current/v3 code) — keeps the regression target bounded in [0,1].

**Theoretical prediction**: FFR should help most at the **earlier** evaluation horizons (AP@1.0s, AP@1.5s, mTTA) — since it is the only mechanism that pushes risk information into the middle, unlabeled snippets, ahead of the collision.

### 2.3 L_AMC — Adaptive Monotonic Constraint

For each positive video, sample `NUM_PAIRS_PER_VIDEO = 8` pairs `(i, j)` with `j > i`, and penalize a **downward** step:

```
L_AMC = mean over sampled pairs of  ReLU(a_i − a_j + δ)
```

where the margin `δ` **adapts** to both the temporal gap and the model's own confidence:

```
δ = DELTA0 · Δt_ij · c̄          (DELTA0 = 0.01)
Δt_ij = (j − i) · dt              (elapsed time between the pair)
c̄     = mean( 2|a_i − ā|, 2|a_j − ā| )     (confidence relative to the batch mean ā)
```

**Pair sampling (`pairing_mode`)** — this is the variable RQ3 changes, held fixed to `random` for RQ2:
- `random` (RQ2's setting, and RQ1's "RiskProp"): gap `~ Uniform(D_MIN, D_MAX)` of the sequence length, i.e. a different temporal separation on every sampled pair, every step.
- `fixed` (RQ3's variant only, not used in RQ2): gap corresponds to a constant `τ = 1.0 s` (2 snippet-steps at dt=0.5s).

**Theoretical prediction**: AMC enforces *non-decreasing* order but does not, by itself, say *when* the risk should start rising — a step function (flat-low, then a single jump at the last snippet) satisfies AMC perfectly but is useless for early warning. So AMC-only (C) is expected to reduce **violations** (metric group 3) more than it improves **early accuracy** (metric group 1). Because the pair offset is re-sampled randomly every training step, AMC-only is also expected to have the **highest seed-to-seed variance** (metric group 4) among B/C/D, since each training run is exposed to a different realized distribution of temporal relationships.

### 2.4 Interaction hypothesis (FFR × AMC)

FFR already pulls every score toward its neighbor, which by construction tends to *produce* a smooth, non-decreasing curve as a side effect (each step is small because it is regressed toward the very next point). AMC, layered on top, may therefore have **diminishing marginal effect once FFR is already active** — i.e., we expect the interaction term `(D−C) − (B−A)` to be **negative** (sub-additive): AMC's added benefit is smaller when FFR is already doing similar work.

This is a hypothesis to test, not an assumption to build into the code — the ablation framework treats all 4 conditions symmetrically.

---

## 3. Metrics (the 4 outcome groups RQ2 asks about)

All computed on the **same internal validation split** already re-validated for RQ1/RQ3 (300 videos, 150/150), using the **same locked TOP-independent scoring** (RiskProp/AdaLEA always emit one scalar score per window — no head-selection issue applies here, that was TOP-specific).

### Group 1 — Accuracy

| Metric | Formula | Meaning |
|---|---|---|
| AP@0.5s / 1.0s / 1.5s | `average_precision_score(target, score)` per lead-time window | Standard average precision at each fixed horizon |
| mAP | mean of the 3 AP@ values | Primary metric for RQ2's comparisons |

### Group 2 — Low-FAR anticipation

| Metric | Formula | Meaning |
|---|---|---|
| mAUC@0.1 | partial AUC of the ROC curve restricted to FPR ≤ 0.1, normalized by 0.1 | Ranking quality specifically in the low-false-alarm regime |
| Recall@FAR≤0.1 | TPR at the operating point where FPR ≤ 0.1 | Detection rate at an acceptable false-alarm budget |
| ActualFAR | the FPR actually achieved at the chosen threshold | Sanity check that the FAR constraint is respected |
| mTTA@0.1 | mean, over detected positives, of the earliest horizon (0.5/1.0/1.5s) whose score clears the FAR≤0.1 threshold | Average warning lead time |
| Coverage | fraction of positive videos detected at any of the 3 horizons | What share of accidents get *any* warning under this FAR budget |

*(Identical formulas to the ones already implemented and validated in `reeval/re03_analyze_rq.py` for RQ1/RQ3 — reused as-is, no new code needed for this group.)*

### Group 3 — Temporal monotonicity (new for RQ2/RQ3; not yet implemented)

Requires a **denser** sequence of scores per validation video than the 3 fixed lead-times used above — e.g. windows every 0.1s from `t_event − 3.0s` to `t_event − 0.1s` for positives (and an equivalent sweep for negatives, to also check false-alarm jitter). Proposed formulas (proposal §5.5 names these metrics but does not give formulas — these definitions must be locked *before* looking at results):

| Metric | Formula | Meaning |
|---|---|---|
| Pairwise violation rate | fraction of ordered pairs `(t_i < t_j)` in the dense sequence with `a_{t_i} > a_{t_j} + ε` | How often the risk curve goes "backward" |
| Average downward step | `mean( max(0, a_t − a_{t+1}) )` over consecutive dense steps | Average size of a downward move, when one occurs |
| Risk-curve jitter | `mean( |a_{t+1} − 2·a_t + a_{t-1}| )` | Discrete second-derivative — how jagged/noisy the curve is |

### Group 4 — Run-to-run stability

| Metric | Formula | Meaning |
|---|---|---|
| SD across 3 seeds | standard deviation of any Group 1–3 metric, computed over the 3 seeds of a condition | How consistent training is, independent of any one lucky/unlucky seed |

---

## 4. Comparison framework (how the 4 conditions are read together)

For any metric `M` (mean over 3 seeds per condition):

| Quantity | Formula | Interpretation |
|---|---|---|
| Main effect of FFR (AMC absent) | `M(B) − M(A)` | Value of adding FFR alone |
| Main effect of AMC (FFR absent) | `M(C) − M(A)` | Value of adding AMC alone |
| Main effect of FFR (AMC present) | `M(D) − M(C)` | Value of adding FFR once AMC is already there |
| Main effect of AMC (FFR present) | `M(D) − M(B)` | Value of adding AMC once FFR is already there |
| Interaction | `(M(D) − M(C)) − (M(B) − M(A))` | Whether the two losses combine additively (≈0), synergistically (>0), or redundantly (<0) |

Every difference above gets a **paired, label-stratified bootstrap 95% CI** (same method as RQ1/RQ3: B=2000, same resampled validation videos applied to every condition/seed). A CI that includes 0 is reported as **inconclusive**, never as "no effect" or "improvement" — per the proposal's decision rule (§5.5), also already applied consistently in the RQ1/RQ3 report.

`mAP` is designated the **primary** metric for headline claims; all other metrics are secondary/exploratory, reported but not used to cherry-pick a "winning" condition — this avoids multiple-comparisons fishing across 4 conditions × 3 metric groups.

---

## 5. How RQ2 connects to RQ1 and RQ3

| | Compares | Held fixed | New training? | Primary outcome |
|---|---|---|---|---|
| RQ1 | 3 architectures (TOP / AdaLEA / RiskProp) | protocol, split, budget | No (done) | Official test mAP |
| **RQ2** | **4 loss combinations inside RiskProp** | architecture, split, budget, sampler=random | **Yes — 9 runs** | Internal validation, 4 metric groups |
| RQ3 | RiskProp's AMC sampler: random vs. fixed-1.0s-lag | architecture, split, budget, **both losses on (=D)** | No (done) | Internal validation, mAP + temporal |

RQ2's most useful output for the team is **explaining** the RQ3 finding already in hand: RQ3 showed fixed-lag ≈ random-offset in mean mAP, but fixed-lag has much lower seed-to-seed SD (0.033 → 0.005). RQ2 will show whether that instability actually comes from AMC's random sampler specifically:

- If `SD(B)` (FFR-only, no AMC at all) is already as low as fixed-lag's SD, while `SD(D)` (AMC-random) is high → **random-offset AMC is the source of the seed variance**, and RQ3's fix (constraining the offset) is a direct, mechanistically-explained solution — not a coincidence.
- If `M(D) ≈ M(B)` on mAP → AMC contributes little to accuracy on its own, which is consistent with RQ3 showing the sampler change (random→fixed) doesn't move mAP either.

Caveat: with only 3 seeds, SD estimates are themselves noisy — this comparison is suggestive, not a formal statistical test of "AMC causes variance."

---

## 6. Deviations from the published RiskProp paper (disclose in the report)

- **Snippet-sequence approximation**: 12 causal 5-frame snippets at 0.5s stride, vs. the paper's per-frame supervision on the full video.
- **λ1 = λ2 = 0.5** vs. the paper's 1.5/1.1 (project's own re-tuned value, chosen because the paper's weights over-suppressed the primary BCE task in earlier project experiments).
- **COLLISION_WEIGHT = 8.0**, **NUM_PAIRS_PER_VIDEO = 8** — project-chosen values, not from the paper (paper does not state an exact collision weight).
- **L_FFR computed on sigmoid probabilities**, not raw logits — a project-proposed reformulation for a better-conditioned regression target.
- **Matched-budget training** (LR=0.01, 50 epochs, no gradient clipping, checkpoint by validation loss, no early stopping) — same recipe as TOP/AdaLEA, not the paper's original 8×A800/50-epoch large-scale setup.

All 4 RQ2 conditions use this **same** recipe — only `use_ffr`/`use_amc` toggle. This is required by the proposal's Controls rule and is also what makes RQ2's numbers directly comparable to the already-trained RQ1/RQ3 condition D.

---

## 7. What must be locked before training (and before touching results)

1. This document's metric definitions (Section 3, temporal metrics especially — no formula is given in the proposal, so ours must be fixed and committed *before* any run is scored).
2. The comparison list in Section 4 — decided in advance, not chosen after seeing which comparisons look good.
3. Checkpoint-selection rule per condition/seed: reuse the same `{best, latest} → higher validation mAP` rule already locked in `results/reeval_corrected/locked_config.json` for RQ1/RQ3.
4. `mAP` as the single primary metric for headline conclusions.

Only after this file is committed do the 9 new training runs start.
