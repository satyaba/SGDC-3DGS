# SGDC — Experiment Report

**Branch:** `feature/sgdc-error-budget`
**Scope:** segmentation-guided density control for 3D Gaussian Splatting
**Data:** Replica `office_0/Sequence_1` (COLMAP scene == `DATASET_DIR`; GT class
labels under `semantic_class/`)
**Status:** exploratory findings; not yet a paper-ready result

This document is the single, self-contained report of *our* experiments and
findings. The method/math spec is in `docs/sgdc_spec.md`; the harness docs are in
`experiments/README.md`; raw numbers also live in the "SGDC" section of
`results.md`. Where the three disagree, **this file is the current source of
truth** for conclusions.

---

## 1. Objective

We set out to test **SGDC (Segmentation-Guided Density Control)**: use per-view
ground-truth semantic labels (transferred to Gaussians by projection) to place,
grow, and prune Gaussians adaptively near semantic boundaries instead of
uniformly, aiming for better reconstruction quality — especially on boundaries —
at a comparable Gaussian budget.

Two ideas were implemented for this round:

1. **Segment-level error budgeting** (Component B) — a hard per-refine capacity
   budget allocated across semantic segments by *fixable error*.
2. **Mask-texture → clone/split signal** (Component A.1) — convert the label
   "texture" into a densification signal that decides clone vs split.

Alongside the pre-existing boundary-aware densification (A) and cross-boundary
scale loss (C).

---

## 2. What is implemented

| Component | Mechanism | Code | Status |
|---|---|---|---|
| **A** | per-Gaussian clone/split threshold scaled by boundary distance `d_i` | `scene/gaussian_model.py` (`compute_sgdc_thresholds`) | implemented, measured |
| **B** | hard per-refine budget `B·N`, per-segment allocation by `q=|grad|·residual` | `sgdc/budget.py` | implemented, measured |
| **A.1** | split candidates whose footprint straddles a boundary (`|sd| ≤ k·σ_perp`) | `gaussian_model.boundary_straddle` | implemented, measured |
| **C** | `erfc` cross-boundary scale regularizer | `gaussian_model.boundary_loss` | implemented, measured |
| **D** | semantic-consistency pruning | — | spec only |
| **E** | semantic-aware opacity reset | — | spec only |

Key CLI switches: `--sgdc_enable`, `--f_b/--f_int/--eta/--sgdc_dt_norm` (A);
`--sgdc_budget_enable --sgdc_budget_B --sgdc_budget_grad_only --sgdc_budget_global`
(B); `--sgdc_signal_enable --sgdc_split_k` (A.1); `--lambda_bnd` (C).

> **Segmentation unit.** This Replica copy has no `semantic_instance/`, so a
> "segment" is a semantic **class** (16 classes). Per-view connected components
> were considered as pseudo-instances but rejected: component ids are not
> consistent across views, so the per-Gaussian top-1 vote would scatter. True
> 3D-fused instances are a separate sub-project.

---

## 3. Protocol

- **Dataset:** Replica `office_0/Sequence_1` (90 images, 640×480).
- **Split:** `--eval` (llffhold=8) for every test-split row.
- **Iterations:** 30 000, default densification schedule.
- **Rasterizer:** this fork's `dr_aa`; `base_30k` (same code, `--sgdc_enable`
  omitted) is the directly comparable baseline. `vanilla_30k` (upstream
  graphdeco, stock rasterizer) is **not** row-comparable.
- **Metrics:** `experiments/metrics.py` → `eval_sgdc.py`, same code path for all
  rows. **Band** = test PSNR on a 4 px dilation of the class-boundary mask;
  **Interior** = test PSNR off the band.
- **Determinism:** `safe_state` seeds `random`/`numpy` but **not `torch`**, and
  the rasterizer backwards use atomics. Runs are therefore stochastic; measured
  spread is ~±0.13 dB (base, n=3) and ±0.03–0.08 dB (budget arms, n=3). All
  "±" below are sample std with n=3 unless stated.

---

## 4. Results

### 4.1 Component B — budget sweep (single runs)

All rows keep Component A neutral (`--f_b 1.0 --f_int 1.0 --eta 0.0`).

| Run | B | Gaussians | Train PSNR | Test PSNR | SSIM | LPIPS | Band | Interior | min |
|---|---|---|---|---|---|---|---|---|---|
| `base_30k`            | —               | 355,256 | 43.50 | 36.50 | 0.9769 | 0.0487 | 31.47 | 39.45 | 16.99 |
| `sgdc_b_30k_B0.02`    | 0.02            | 100,217 | 39.67 | 35.28 | 0.9631 | 0.1180 | 30.66 | 37.42 |  9.95 |
| `sgdc_b_30k`          | 0.05            | 267,931 | 42.81 | 36.86 | 0.9761 | 0.0537 | 31.60 | 39.85 | 15.26 |
| `sgdc_b_30k_B0.05`    | 0.05            | 261,519 | 42.86 | 36.66 | 0.9758 | 0.0539 | 31.41 | 39.66 | 14.43 |
| `sgdc_b_30k_gradonly` | 0.05 (no error) | 260,867 | 42.87 | 36.91 | 0.9762 | 0.0537 | 31.73 | 39.89 | 14.39 |
| `sgdc_b_30k_B0.1`     | 0.10            | 310,566 | 43.23 | 36.79 | 0.9765 | 0.0513 | 31.75 | 39.66 | 16.18 |
| `sgdc_b_30k_B0.15`    | 0.15            | 338,209 | 43.43 | 36.90 | 0.9767 | 0.0497 | 31.68 | 39.91 | 16.98 |
| `sgdc_bglobal_30k`    | 0.10 (global)   | 316,686 | 43.36 | 36.66 | 0.9765 | 0.0504 | 31.51 | 39.68 | 16.05 |

- Gaussian count tracks `B` monotonically (100k → 261k → 310k → 338k): a clean
  capacity knob.
- The **error term does not help**: `grad_only` (Test 36.91, band 31.73) ties the
  error-weighted version (36.86, band 31.60).
- **`base_30k` here is one sample** and turned out low; see 4.2.

### 4.2 Component B — repeated comparison (n=3, B = 0.10) — **headline**

3 stochastic repeats per arm.

| Arm | n | Gaussians | Train PSNR | **Test PSNR** | Test SSIM | Test LPIPS | **Band** | **Interior** | Train−Test |
|---|---|---|---|---|---|---|---|---|---|
| base | 3 | 353,126 ± 1,849 | 43.58 ± 0.10 | 36.64 ± 0.13 | 0.97693 ± 0.00006 | **0.04867 ± 0.00035** | 31.53 ± 0.07 | 39.68 ± 0.20 | 6.94 |
| per-segment | 3 | **313,684 ± 2,918** | 43.30 ± 0.09 | 36.75 ± 0.03 | 0.97657 ± 0.00012 | 0.05120 ± 0.00010 | **31.66 ± 0.09** | 39.67 ± 0.03 | 6.55 |
| global | 3 | 316,312 ± 348 | 43.34 ± 0.09 | 36.74 ± 0.08 | 0.97660 ± 0.00010 | 0.05070 ± 0.00025 | 31.58 ± 0.08 | 39.76 ± 0.07 | 6.60 |

Run ids — base: `base_30k`, `base_30k_r1`, `base_30k_r2`; per-segment:
`sgdc_b_30k_B0.1`, `sgdc_b_30k_seg_r1`, `sgdc_b_30k_seg_r2`; global:
`sgdc_bglobal_30k`, `sgdc_bglobal_30k_glob_r1`, `sgdc_bglobal_30k_glob_r2`.

Deltas:

| Comparison | ΔGaussians | ΔTest PSNR | ΔSSIM | ΔLPIPS | ΔBand | ΔInterior |
|---|---|---|---|---|---|---|
| per-segment − global | −2,628 (−0.8%) | +0.01 | 0.0000 | +0.0005 | +0.08 | −0.09 |
| per-segment − base | **−39,442 (−11.2%)** | +0.11 | −0.0004 | +0.0025 | +0.13 | −0.01 |
| global − base | **−36,814 (−10.4%)** | +0.10 | −0.0003 | +0.0020 | +0.05 | +0.08 |

**Reading.**
- **Quality is statistically tied with base.** Test PSNR +0.10/+0.11 (SE ≈ 0.08,
  ~1.2–1.4σ), band/interior within ~2σ. The earlier "+0.25 dB" was base's single
  low sample.
- **The robust effect is count:** ~**11% fewer Gaussians** (313.7k/316.3k vs
  353.1k) at statistically equal test PSNR.
- **Small but significant perceptual cost:** SSIM −0.0004, LPIPS +0.0020–0.0025
  (well outside the noise floor).
- **Regularization:** lower train PSNR and smaller train−test gap (6.55/6.60 vs
  6.94).
- **Segmentation is inert:** per-segment vs global band gap +0.08 (SE ≈ 0.07,
  n.s.), test PSNR/SSIM/LPIPS tied. The per-segment allocation adds nothing.

### 4.3 Component A.1 — mask-boundary straddle clone/split (single runs)

| Run | Config | Gaussians | Train PSNR | Test PSNR | SSIM | LPIPS | Band | Interior |
|---|---|---|---|---|---|---|---|---|
| `base_30k`      | —                    | 355,256 | 43.50 | 36.50 | 0.9769 | 0.0487 | 31.47 | 39.45 |
| `sgdc_sig_30k`  | A.1 only (k=1.0)     | 316,501 | 43.64 | 36.73 | 0.9769 | 0.0489 | 31.57 | 39.77 |
| `sgdc_bsig_30k` | B 0.05 + A.1 (k=1.0) | 247,067 | 43.03 | 36.79 | 0.9757 | 0.0544 | 31.61 | 39.73 |

- **A.1 alone ≈ base quality at ~11% fewer Gaussians.** Against the n=3 base mean
  (36.64), A.1's 36.73 is +0.09 on a single run (within noise); SSIM/LPIPS tied,
  band +0.04, interior +0.09. The earlier "+0.23" was against base's low run.
- **Not boundary-specific:** band +0.04 over the base mean, below B's band —
  despite A.1 targeting boundaries.
- **Not additive with B:** `bsig` 36.79 vs B-alone 36.86/36.66, band flat, LPIPS
  worse.

### 4.4 Components A and C — earlier session (**train split**, not comparable)

> Measured in a separate earlier session on the **train split**
> (`sgdc.eval_boundary`, band 4 px) — a different protocol from §4.1–4.3.

**A sweep (30k, train-view PSNR):**

| Run | Config | Train PSNR | Gaussians |
|---|---|---|---|
| `base` | SGDC off | 41.745 | 530,009 |
| `sgdc_a` | `f_b 0.5, f_int 2.0, dt 100` | 42.273 | 866,915 |
| `a_fb1_fi3_dt60` | `f_b 1.0, f_int 3.0, dt 60` | 40.279 | 316,518 |
| `a_fb05_fi4_dt50` | `f_b 0.5, f_int 4.0, dt 50` | 40.217 | 522,161 |
| `a_fb1_fi6_dt40` | `f_b 1.0, f_int 6.0, dt 40` | 37.835 | 201,792 |

Only `sgdc_a` beat base, and only by adding **+63% Gaussians**; at matched count
(`a_fb05_fi4_dt50`, 522k vs 530k) it is **−1.5 dB**.

**Boundary-restricted eval (train split, band 4 px):**

| Run | Gaussians | Global | Band | Interior |
|---|---|---|---|---|
| `base` | 534.0k | 43.70 | 38.92 | 45.64 |
| `a_fb05_fi4_dt50` | 522.2k | 41.68 | **39.11** | 42.50 |
| `c_bnd02` | 498.1k | 42.89 | 37.92 | 44.93 |
| `c_bnd10` | 414.3k | 41.94 | 36.54 | 44.29 |
| `c_bnd50` | 349.1k | 40.32 | 34.53 | 42.99 |

Deltas vs base: **A** band **+0.20** but interior **−3.13** → net **−2.02**;
**C** harmful everywhere (−0.81 … −3.38 global; band −1.00 … −4.39).

---

## 5. Findings

1. **The budget (B) is a genuine capacity/quality knob** — Gaussian count tracks
   `B` monotonically.
2. **B does not improve quality on this scene.** With n=3, both budget arms are
   statistically tied with base on test PSNR at **~11% fewer Gaussians**, paying
   a small (significant) SSIM/LPIPS regression. It works by *regularization*
   (smaller train−test gap), not better placement.
3. **B's segmentation is inert.** Per-segment allocation ≈ scene-global budget
   (band +0.08, n.s.; PSNR tied). The **error weighting is inert** too
   (`grad_only` ties).
4. **A is net-negative.** It reallocates to boundaries (band +0.20) but collapses
   interiors (−3.13) → net −2.0 global.
5. **C is harmful** on every metric, including the band it targets.
6. **A.1 is not boundary-specific** — it acts as a mild capacity regularizer, and
   stacking it on B is not additive.
7. **Net: no segmentation-attributable improvement** was found on this scene.
   Every semantic mechanism (A, A.1, C, and B's per-segment split) failed its
   control. The one robust effect — fewer Gaussians at equal PSNR — comes from
   the **capacity cap**, which is not segmentation.

**Principle (hypothesis for the negative result).** Density control's target
(PSNR) and its signal (view-space photometric gradient) are the *same modality*.
On a densely-viewed, well-textured scene the gradient already encodes the
information a semantic mask would supply, so semantics are close to orthogonal.
Semantics should only matter where the gradient is a poor proxy: **sparse views,
unbounded backgrounds, low-contrast boundaries, thin structures, transient
artifacts.**

---

## 6. What we can and cannot claim

| Claim | Verdict |
|---|---|
| A code-tied SGDC operator family (A/A.1/B/C; D/E spec'd) exists | **Supported** |
| A hard density budget removes ~11% of Gaussians at statistically equal test PSNR | **Supported** (n=3, single scene) |
| The budget improves PSNR / boundary quality | **Not supported** |
| Segmentation guidance improves reconstruction | **Not supported** |
| The budget's effect is segmentation-driven | **Not supported** (per-segment ≈ global) |
| The budget's effect is error-driven | **Not supported** (`grad_only` ties) |
| Boundary-semantic operators (A, C) help | **Refuted** (net-negative / harmful) |
| A.1 improves boundaries | **Not supported** |

**Framing.** The defensible framing is **"budgeted density control for 3DGS"**
(a capacity/quality knob), *not* "segmentation-guided density control." The
convergent failure of four semantic mechanisms is itself a reportable
**negative result**.

---

## 7. Limitations

- **Single scene** (`office_0/Sequence_1`). Nothing is shown to generalize.
- **Small n.** Only the `B=0.10` arm vs base was repeated (n=3). The sweep, A.1,
  and all A/C rows are single runs.
- **Mixed protocols.** A/C are train-split; B/A.1 are test-split.
- **No formal significance testing** beyond ~σ comparisons.
- **Segments = classes (16).** Instance-level segmentation was not available, so
  fine-grained "per-object" allocation was never actually tested.

---

## 8. Recommendations / next steps

**If the goal is a positive segmentation-guided result:**
1. **Diagnostic first:** measure the fraction of mask-boundary pixels whose
   photometric `|∇I|` is low. If it is tiny on this dataset, *no* mask mechanism
   can help here — go to a regime where it isn't (sparse views, unbounded bg,
   low-texture, thin structures).
2. Redesign the mask signal to be **orthogonal** to the gradient and to affect
   **candidate selection**, not just clone-vs-split (A.1's failure is the proof).
   e.g. inject a boundary term where the gradient is weak.
3. Test with metrics matched to the target (low-contrast-boundary PSNR, thin-
   structure regions) and controls (mask-off, shuffled-mask).

**If the goal is a solid efficiency result:**
4. Reframe around the **budget cap** Pareto; add a second scene and repeats of
   `base`/`B`/`B-global`.

**Either way:**
5. Keep D/E out of scope until a positive signal exists to build on.

---

## 9. Reproduce

```bash
python -m experiments.runs --run base_30k
python -m experiments.runs --run sgdc_b_30k
python -m experiments.metrics --run-id sgdc_b_30k

# budget sweep / controls (tagged):
# run_by_id("sgdc_b_30k", tag=f"B{B}", extra=f"--sgdc_budget_B {B}")
# run_by_id("sgdc_b_30k", tag="gradonly",  extra="--sgdc_budget_grad_only")
# run_by_id("sgdc_bglobal_30k")                     # scene-global control
# run_by_id("sgdc_sig_30k")                          # A.1 alone
```

Manifest run ids in `experiments/runs.py`; harness docs in `experiments/README.md`.
