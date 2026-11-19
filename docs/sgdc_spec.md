# SGDC — Segmentation-Guided Density Control for 3DGS
### Method & Math Spec (code-tied to original INRIA Gaussian Splatting)

---

## 0. Provenance & repository layout

| Item | Value |
|---|---|
| Branch | `sgdc` |
| Base | pristine upstream `graphdeco-inria/gaussian-splatting` @ `54c035f` |
| Isolation | prior project work remains on `main` / `feature/*` / `snapshot/*`; **not read, not reused** |
| Semantic source | 2D semantic / instance labels (semantic seg or SAM2), transferred to Gaussians by projection |
| Placement | our code only in `src/sgdc/` (or top-level `sgdc/`), spec in `docs/`, run configs in `experiments/` |

**Files this spec edits (all pristine upstream at the commit above):**

- `train.py` — densification block, **lines 163–174**
- `scene/gaussian_model.py` — `GaussianModel` density-control methods (**lines 258–473**)
- `gaussian_renderer/__init__.py` — `render()` outputs (**lines 118–128**)
- `arguments/__init__.py` — `OptimizationParams` (**lines 74–100**)
- `scene/cameras.py` — `Camera` fields (**lines 19–89**)

### Build prerequisites (Colab / modern toolchains)

On CUDA 12.9 / GCC 14 / Python 3.13 the CUDA submodules fail with `uint32_t`/`uint64_t`/`std::uintptr_t` undefined (missing `<cstdint>`). Fix: `bash scripts/patch_cuda_submodules.sh`, then `pip install .` each of `submodules/diff-gaussian-rasterization`, `submodules/simple-knn`, `submodules/fused-ssim` with `TORCH_CUDA_ARCH_LIST` set to the GPU arch (T4=7.5, V100=7.0, A100=8.0, L4=8.9). Requires `ninja` for a reasonable build time.

---

## 1. Baseline density control (exact hooks we modify)

### 1.1 Training-loop block — `train.py:163–174`

```python
# Densification
if iteration < opt.densify_until_iter:
    gaussians.max_radii2D[visibility_filter] = torch.max(
        gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005,
                                    scene.cameras_extent, size_threshold, radii)

    if iteration % opt.opacity_reset_interval == 0 or \
       (dataset.white_background and iteration == opt.densify_from_iter):
        gaussians.reset_opacity()
```

### 1.2 Model methods — `scene/gaussian_model.py`

| Method | Signature | Line | Role |
|---|---|---|---|
| `add_densification_stats` | `(viewspace_point_tensor, update_filter)` | 471 | accumulate `‖∂L/∂μ²ᴰ‖` into `xyz_gradient_accum`, `denom` |
| `densify_and_prune` | `(max_grad, min_opacity, extent, max_screen_size, radii)` | 452 | orchestrator: computes `grads`, calls clone/split/prune |
| `densify_and_clone` | `(grads, grad_threshold, scene_extent)` | 435 | clone where grad high ∧ small |
| `densify_and_split` | `(grads, grad_threshold, scene_extent, N=2)` | 409 | split where grad high ∧ large |
| `densification_postfix` | `(new_xyz, …, new_tmp_radii)` | 388 | appends tensors, resets accumulators |
| `prune_points` | `(mask)` | 349 | removes Gaussians + buffers |
| `reset_opacity` | `()` | 258 | min(opacity, 0.01) reset |

**Buffers carried per-Gaussian (must be extended by SGDC):** `xyz_gradient_accum [N,1]`, `denom [N,1]`, `max_radii2D [N]`, `tmp_radii [N]`. They are appended in `densification_postfix` (`cat`) and sliced in `prune_points` (lines 360–364).

**Renderer outputs used (`gaussian_renderer/__init__.py:120–126`):** `viewspace_points` (=2D means, carries grad), `visibility_filter = (radii>0).nonzero()`, `radii`, `depth`.

**Camera fields (`scene/cameras.py`):** `image_name`, `image_width`, `image_height`, `FoVx/FoVy`, `world_view_transform`, `full_proj_transform` (= world→clip, used by the rasterizer as `projmatrix`), `camera_center`, `original_image`.

### 1.3 Baseline decision rule (to be replaced)

$$
\text{clone if } \|g_i\|>\tau_g \wedge \max(s_i)\le \theta_0,\qquad
\text{split if } \|g_i\|>\tau_g \wedge \max(s_i)>\theta_0,
$$
$$
\text{prune if } \alpha_i<\epsilon_\alpha \;\vee\; \text{max\_radii2D}_i>S \;\vee\; \max(s_i)>0.1\,\mathcal S,
\qquad \theta_0=\texttt{percent\_dense}\cdot\mathcal S .
$$

SGDC conditions every one of these on the **semantic signature** of Gaussian $i$.

---

## 2. Semantic input pipeline

### 2.0 Segment definition & label sources

A **segment** is the composite id `seg = (class_id, instance_id)`. Components pick their view:

- **A** (thresholds) keys on **class** boundaries;
- **B** (budgets) uses the composite unit — instance for objects, class for structures;
- **D** (consistency) keys on **instance** (strongest cross-view signal).

**Scope now: Replica GT + SAM2 instance pseudo-labels.** The Replica adapter
reads GT class labels; `InstanceDirLabelAdapter` (`sam2` / `instance_dir`) reads
offline, class-agnostic instance maps produced by `sgdc/segment_sam2.py` for
non-segmented datasets (Mip-NeRF 360, ...). See `docs/datasets.md`. A
class-associated `SemsegAdapter` remains deferred.

Canonical contract consumed by SGDC (all aligned to `camera.image_height × camera.image_width`; **`0` = void**):

| Object | Type | Meaning |
|---|---|---|
| `L_v` | `int32 [H,W]` | composite segment id per pixel |
| `dt_v` | `float [H,W]` | distance transform to nearest label boundary (px) |
| `n_v` | `float [H,W,2]` | unit boundary normal (`∇dt`) |
| `area_c`,`perim_c` | per-segment | for Component B |

Adapter layer `src/sgdc/datasets/`:

```python
class LabelAdapter:                       # base
    name: str
    def resolve(self, image_name) -> Path | None    # label file(s) for this frame
    def load_raw(self, path) -> np.ndarray          # native HxW ints
    def to_canonical(self, raw) -> np.ndarray       # remap ids, set void, class/instance
    def meta(self) -> dict                          # {"K","ignore_ids","source"}

class ReplicaLabelAdapter(LabelAdapter): ...           # Replica GT PNGs
class InstanceDirLabelAdapter(LabelAdapter): ...       # SAM2/instance maps (in scope)
# SemsegAdapter (class-associated)                      -> deferred
```

`SemanticStore` loads `L_v`, **nearest-neighbor** resizes to `[H_v,W_v]` (ids must never be bilinearly interpolated), computes `dt_v`/`n_v`, and caches.

### 2.1 Replica labels (in scope) — CONFIRMED

Confirmed layout (the COLMAP scene *is* `DATASET_DIR`, so `-s DATASET_DIR` works):

- `images/rgb_<N>.png`, `sparse/` (COLMAP), `depth/`, `traj_w_c.txt`
- `semantic_class/rgb_<N>.png` — GT **class** labels, native `480×640`, uint8
- `semantic_instance/` **absent** in this copy → composite **degenerates to class-only**

Sparse Replica class ids (`0 12 17 20 … 98`; 16 present per frame) are remapped to a contiguous range `1..K`; **`void_ids` defaults to empty** (class `0` is a real class in this copy, so nothing is treated as void — and Component D's void-floater branch is simply inactive, leaving purity-based pruning). `ReplicaLabelAdapter` resolves labels by filename **stem** (`rgb_0.png` ↔ `rgb_0.png`), so it is robust to extension differences. Verified: **90 images ↔ 90 labels, 0 mismatches, both `640×480`**. Implementation: `sgdc/datasets/replica.py`.

### 2.2 Precompute — `src/sgdc/precompute.py`

Per-view, cached `.npz` at training resolution:

- `dt_v`  — Euclidean **distance transform** to nearest label boundary (pixels)
- `gdt_v` — gradient/normal field of `dt_v` (per-pixel unit normal `n_v` pointing away from boundary)
- `perim_c`, `area_c` — per-segment perimeter and area (for Component B)

### 2.3 Loading — `src/sgdc/semantic_store.py`

A standalone `SemanticStore(source_path, device)` keyed by `image_name`, returning `L_v`, `dt_v`, `gdt_v` on GPU on demand. **No change to the `Camera` constructor** — `image_name` (cameras.py:33) is the key.

### 2.4 Alignment verification

One-time check (`src/sgdc/verify_alignment.py`): overlay `dt_v` boundaries on the RGB/render frame and assert pixel alignment; confirm the NDC $y$-convention from §3.1. Catches the most common failure (label/image resolution mismatch).

---

## 3. Per-Gaussian semantic signature (new buffers)

Between refine steps, accumulate per-Gaussian statistics from *visible* projections — exactly parallel to `add_densification_stats`. **O(N), constant memory, no learned params.**

### 3.1 Projection of a Gaussian center into view $v$

Using `full_proj_transform` `P` (cameras.py:88; column-major, as read by CUDA), for mean $\mu$:

$$
\tilde p = [\mu;1]\,P,\quad p=\tilde p_{1:3}/\tilde p_4,\quad
u=\tfrac{(p_x+1)\,W-1}{2},\quad v=\tfrac{(p_y+1)\,H-1}{2}.
$$

> Convention confirmed against `diff-gaussian-rasterization` (`auxiliary.h`): `ndc2Pix(v,S)=((v+1)S-1)/2` for both axes (no $y$-flip), and CUDA reads `full_proj_transform` column-major, so the row-vector product $[\mu;1]P$ is required (equivalently $P^\top[\mu;1]^\top$). Clamp indices to `[0,W-1]×[0,H-1]`; discard if $\tilde p_4\le0$.

### 3.2 Buffers

| Buffer | Shape | Meaning |
|---|---|---|
| `seg_top1` | `[N]` long | running argmax segment id $s_i$ |
| `seg_top1_cnt` | `[N,1]` | count for top label |
| `seg_tot_cnt` | `[N,1]` | total labeled projections |
| `seg_void_cnt` | `[N,1]` | projections landing in `void` |
| `seg_dt_sum` | `[N,1]` | Σ `dt_v(u)` (boundary distance) |
| `seg_norm_sum` | `[N,1]` | Σ visibility weight (for means) |

Derived (computed at refine time):

$$
\rho_i=\frac{\texttt{seg\_top1\_cnt}_i}{\texttt{seg\_tot\_cnt}_i},\quad
\nu_i=\frac{\texttt{seg\_void\_cnt}_i}{\texttt{seg\_tot\_cnt}_i},\quad
d_i=\mathrm{norm}\!\Big(\tfrac{\texttt{seg\_dt\_sum}_i}{\texttt{seg\_tot\_cnt}_i}\Big)\in[0,1].
$$

`d_i` is normalized by the per-view max distance (or a fixed `dt_norm`), so 0 = on a boundary, 1 = deep interior.

> Optional dense form `seg_hist [N,K]` for exact purity when $K$ is small (semantic classes). Default is the compact O(N) form above; `ρ_i` = top-count ratio is the standard estimator.

### 3.3 Update (extends `add_densification_stats`)

```python
# scene/gaussian_model.py  (SGDC variant)
def add_semantic_stats(self, view, camera):
    vis = view["visibility_filter"].squeeze(-1)          # from render()
    if vis.numel() == 0: return
    with torch.no_grad():
        mu  = self.get_xyz[vis]                            # [M,3]
        P   = camera.full_proj_transform                   # [4,4]
        p   = (P @ torch.cat([mu, torch.ones_like(mu[:, :1])], -1).T).T
        ndc = p[:, :3] / p[:, 3:].clamp_min(1e-6)
        u = ((ndc[:, 0] + 1) * camera.image_width  - 1) / 2
        v = ((ndc[:, 1] + 1) * camera.image_height - 1) / 2
        ui = u.round().long().clamp(0, camera.image_width - 1)
        vi = v.round().long().clamp(0, camera.image_height - 1)
        Lv = self.semantic_store.label(camera.image_name)  # [H,W]
        lab = Lv[vi, ui]                                    # [M]
        dt  = self.semantic_store.dt(camera.image_name)    # [H,W]
        d   = dt[vi, ui]

        void = lab == 0
        self.seg_void_cnt[vis]  += void[:, None].float()
        self.seg_dt_sum[vis]    += d[:, None]
        self.seg_tot_cnt[vis]   += (~void)[:, None].float()
        self.seg_norm_sum[vis]  += 1.0

        # top-1 update
        prev = self.seg_top1[vis]
        hit  = (~void) & (lab == prev)
        self.seg_top1_cnt[vis] += hit[:, None].float()
        miss = (~void) & (~hit)
        new  = miss & (self.seg_top1_cnt[vis, 0] == 0)     # first label seen
        self.seg_top1[vis]  = torch.where(new, lab, prev)
        self.seg_top1_cnt[vis] = torch.where(new, torch.ones_like(self.seg_top1_cnt[vis]), self.seg_top1_cnt[vis])
```

### 3.4 Per-segment accumulators (global, for Component B)

Updated each iteration from the rendered vs. GT residual, restricted to each label region:

```python
# src/sgdc/budget.py, called once per iteration before densify
R_c += |Î - I| masked by (Lv == c)          # [K]
seg_residual[c] += R_c.sum();  seg_area[c] += (Lv==c).sum()
# perimeter/area precomputed offline (perim_c, area_c)
```

---

## 4. Components (exact edits)

### Component A — Boundary-aware densification

**Mechanism.** Replace the scalar threshold $\tau_g$ with a per-Gaussian threshold driven by boundary distance $d_i$:

$$
\tau_i=\tau_g\big[f_b(1-d_i)+f_{\text{int}}\,d_i\big],\qquad f_b<1<f_{\text{int}}.
$$

Boundary-adjacent Gaussians densify aggressively; interior ones conservatively. Split-vs-clone gate also becomes boundary-modulated:

$$
\theta_i=\mathcal S\cdot\texttt{percent\_dense}\cdot\big(1-\eta(1-d_i)\big)
\quad\Rightarrow\quad
\text{split if } \max(s_i)>\theta_i \text{ else clone.}
$$

**Edit `scene/gaussian_model.py` — `densify_and_clone` / `densify_and_split`.** Instead of receiving a scalar `grad_threshold`, receive the per-point tensor `tau_i` (length $N$) and use `grads >= tau_i`:

```python
# densify_and_clone (conceptually)
g = grads.squeeze(-1)                                  # [N]
sel = (g >= tau) & (self.get_scaling.max(dim=1).values <= theta)   # tau, theta per-point

# densify_and_split: same, with  sel = (g >= tau) & (max_scale > theta)
```

where `tau`, `theta` are computed in a new `compute_sgdc_thresholds()` from `seg_dt_sum/seg_tot_cnt`.

### Component A.1 — Mask-boundary straddle decides clone vs split (implemented)

**Mechanism.** The baseline size gate decides clone (small) vs split (large).
A.1 adds a footprint-relative boundary signal: a Gaussian whose *projected mass
reaches across* the semantic boundary of its own segment is **split** even when
small, so capacity is instantiated as finer Gaussians exactly where the mask says
a geometric edge must be resolved:

$$
\text{straddle}_i=\big(|sd_i|\le k\,\sigma_{\perp,i}\big)\wedge(dt_i<\delta_{\text{sig}})\wedge(s_i>0),
\qquad
\text{split if } (\max(s_i)>\theta_i)\ \vee\ \text{straddle}_i .
$$

$sd_i,\sigma_{\perp,i}$ are the signed boundary distance and projected std along
the boundary normal (shared with Component C via `_boundary_geometry`). Unlike
Component A's centre-distance $d_i$, this is footprint-relative: a large Gaussian
near an edge is flagged even when its centre is a few pixels away.

**Implementation — `boundary_straddle()`**, applied in `densify_and_prune` after
the B budget (chain: budget picks candidates → straddle picks clone vs split).
CLI: `--sgdc_signal_enable`, `--sgdc_split_k`, `--sgdc_signal_band`.

### Component B — Per-segment error budgeting (implemented)

**Mechanism.** Cap additions per refine step at $B$ (fraction of current $N$) and
distribute across segments by **fixable error**. The priority folds the
view-space gradient (a *benefit* proxy) into the photometric residual (a *need*
proxy), so the budget is not wasted on unrecoverable regions (sky, specular,
blur) where adding capacity cannot help:

$$
q_i=\|g_i\|\cdot e_i,\qquad e_i=\frac{\texttt{seg\_err\_sum}_i}{\texttt{seg\_norm\_sum}_i},
\qquad
p_c=\frac{\sum_{i:s_i=c}q_i}{\sqrt{n_c}},\qquad
\Delta N_c=B\,\frac{p_c}{\sum_{c'}p_{c'}},
$$

where $n_c$ is the number of Gaussians currently assigned to $c$ (a
capacity-normalized error: "error per unit capacity"). Candidates within a
segment are ranked by the same $q_i$ and the top $\Delta N_c$ are kept; leftover
slack (segments short of candidates) is topped up from the global pool, so the
cap $B$ is a hard ceiling but is never left unspent while candidates remain.

**Implementation — `sgdc/budget.py`** (`per_gaussian_priority`, `segment_priority`,
`allocate` [largest-remainder], `budget_mask` [per-segment top-k + top-up]). The
mask is applied in `densify_and_prune` *before* clone/split run; `seg_err_sum` is
accumulated in `add_semantic_stats` from the current view's `|render − GT|` map.
CLI: `--sgdc_budget_enable`, `--sgdc_budget_B`, `--sgdc_budget_grad_only`.

**Non-segmented control — `--sgdc_budget_global`.** Applies the *same* priority
`q_i` and the *same* hard cap, but keeps the top-``B·N`` candidates scene-globally
(`sgdc.budget.global_mask`) instead of allocating per segment. Comparing it to the
per-segment path isolates how much of B comes from the segmentation/allocation
versus a plain capacity cap — the control the segmentation-attribution claim
needs.

### Component C — Cross-boundary scale regularizer

**Mechanism.** For a projection with boundary normal $n$, $\sigma_\perp=\sqrt{n^\top\Sigma'_{iv}n}$, signed boundary distance $z=\mathrm{sd}/\sigma_\perp$:

$$
\psi_{iv}\approx\tfrac12\mathrm{erfc}\!\big(z/\sqrt2\big),\qquad
\mathcal L_{\text{bnd}}=\frac{\sum_i\sum_v \alpha_i\,\psi_{iv}}{\sum_i\sum_v \mathbb 1[\text{visible}]}.
$$

Minimizing elongates $\Sigma_i$ tangentially, shrinks it normally → anisotropic, contour-hugging Gaussians.

**Edit `train.py`** — inside the autograd region (before `loss.backward()`, line 142), for visible Gaussians:

```python
# src/sgdc/boundary_loss.py
def boundary_loss(gaussians, camera, vis, semantic_store, pipe, eps=1e-6):
    mu   = gaussians.get_xyz[vis]
    Rs   = build_rotation(gaussians._rotation[vis])
    Ss   = gaussians.get_scaling[vis]
    Sig  = Rs @ torch.diag_embed(Ss**2) @ Rs.transpose(1, 2)      # world cov [M,3,3]
    W    = camera.world_view_transform[:3, :3]                    # rot part
    # project cov to image plane (J from perspective at mu)
    J    = jacobian_projective(mu, camera)                        # [M,2,3]
    Sig2 = J @ (W @ Sig @ W.T) @ J.transpose(1, 2)                # [M,2,2] (+eps I)
    # pixel coords + fields
    ui, vi = project_to_pixels(mu, camera)
    sd  = semantic_store.signed_dist(camera.image_name, ui, vi, seg=gaussians.seg_top1[vis])
    nrm = semantic_store.normal(camera.image_name, ui, vi)        # [M,2] unit
    sig_perp = torch.sqrt((nrm[:,None,:] @ Sig2 @ nrm[:,:,None]).squeeze() + eps)
    z   = sd / sig_perp
    psi = 0.5 * torch.erfc(z / math.sqrt(2))
    alpha = gaussians.get_opacity[vis].squeeze(-1)
    return (alpha * psi).sum() / vis.sum().clamp_min(1)
# train.py:  loss += opt.lambda_bnd * boundary_loss(...)
```

`signed_dist` and `normal` are read from the cached `dt_v`/`gdt_v`, with sign from the indicator `Lv == seg_i`.

### Component D — Semantic-consistency pruning

**Edit `densify_and_prune` (`scene/gaussian_model.py:452`)** — append to the prune mask (excluding a boundary band $d_i<\delta_{\text{band}}$ to respect SAM2 edge uncertainty):

```python
rho = (self.seg_top1_cnt / self.seg_tot_cnt.clamp_min(1)).squeeze(-1)
nu  = (self.seg_void_cnt / self.seg_tot_cnt.clamp_min(1)).squeeze(-1)
d   = self.normalized_dt().squeeze(-1)

band  = d < self.sgdc_cfg.delta_band
low_p = (~band) & (rho < self.sgdc_cfg.rho_min) & (self.avg_alpha < self.sgdc_cfg.eps_imp)
floater = (~band) & (nu > self.sgdc_cfg.nu_max) & (self.avg_residual > self.sgdc_cfg.r_thr)
prune_mask = prune_mask | low_p | floater
```

`avg_alpha` = mean effective blended alpha over its visible projections; `avg_residual` = mean photometric residual at its projections (accumulated alongside the semantic stats).

### Component E — Semantic-aware opacity reset

**Edit `reset_opacity` (`scene/gaussian_model.py:258`).** Baseline is `min(opacity, 0.01)` for all. Make the floor per-Gaussian:

$$
\alpha_i \leftarrow \sigma\!\big(\beta_0-\beta_1(1-\rho_i)-\beta_2\nu_i\big)\cdot s_{\text{stab}}.
$$

Stable pure interiors reset **higher** (geometry preserved); inconsistent/void Gaussians reset **lower** (ready for Component D).

```python
def reset_opacity(self):
    rho = (self.seg_top1_cnt / self.seg_tot_cnt.clamp_min(1))
    nu  = (self.seg_void_cnt / self.seg_tot_cnt.clamp_min(1))
    logit = self.sgdc_cfg.beta0 - self.sgdc_cfg.beta1*(1-rho) - self.sgdc_cfg.beta2*nu
    opacities_new = torch.sigmoid(logit) * self.sgdc_cfg.s_stab
    self._opacity = self.replace_tensor_to_optimizer(opacities_new, "opacity")["opacity"]
```

---

## 5. Buffer plumbing (must-follow)

New buffers are **not** nn.Parameters, so they are handled exactly like `xyz_gradient_accum`/`denom`/`max_radii2D`:

- **Allocate** in `create_from_pcd` / `training_setup` alongside `xyz_gradient_accum` (`gaussian_model.py:172,180`).
- **`cat`** in `densification_postfix` (line 388–407): append zeros/slices for cloned/split points (replicate `seg_top1*` from the parent for split/clone children — children inherit parent label).
- **`prune`** in `prune_points` (line 349–364): slice every new buffer by `valid_points_mask`.
- **`reset`** accumulators in `densification_postfix` (line 405–407): zero per-Gaussian counters after each refine, matching `xyz_gradient_accum`.

---

## 6. New CLI args — `arguments/__init__.py` (`OptimizationParams`)

```python
# SGDC (as implemented; A, B, A.1 and C are live, D/E are spec-only)
self.sgdc_enable        = False   # master switch (code-tied ablation)
self.sgdc_adapter       = "replica"
self.f_b                = 0.5     # A: boundary threshold factor
self.f_int              = 2.0     # A: interior threshold factor
self.eta                = 0.3     # A: split-threshold shrink
self.sgdc_dt_norm       = 100.0   # A: px normalizer for boundary distance d_i
self.sgdc_budget_enable = False   # B: hard per-refine error budget
self.sgdc_budget_B      = 0.05    # B: budget / refine step (frac of N)
self.sgdc_budget_grad_only=False # B: drop residual (q = |grad| only)
self.sgdc_budget_global =False   # B: scene-global top-k control (no per-segment split)
self.sgdc_signal_enable = False   # A.1: straddle-driven clone/split
self.sgdc_split_k       = 1.0     # A.1: straddle if |sd| <= k * sigma_perp
self.sgdc_signal_band   = 32.0    # A.1: boundary band (px)
self.lambda_bnd         = 0.0     # C: cross-boundary scale loss weight
self.sgdc_bnd_start     = 500     # C: apply from this iteration
self.sgdc_bnd_band      = 32.0    # C: boundary band (px)
```

---

## 7. Unified refine algorithm → function map

```
every opt.densification_interval iters, densify_from_iter < it < densify_until_iter:
  # stats (train.py:166-167, extended)
  max_radii2D          <- update                                 (train.py:166)
  add_densification_stats(viewspace_point_tensor, vis)           (gaussian_model.py:471)
  add_semantic_stats(render_pkg, viewpoint_cam)                  [SGDC §3.3]

  grads <- xyz_gradient_accum / denom                            (gaussian_model.py:453)
  rho, nu, d <- derive(seg_*)                                    [SGDC §3.2]

  # A: per-Gaussian clone / split gates
  tau   <- tau_g [ f_b(1-d) + f_int d ]                          [SGDC §4A]
  theta <- S * percent_dense * (1 - eta(1-d))                    [SGDC §4A]
  cand  <- grads >= tau

  # B: hard error budget over candidates (chain step 1)            [SGDC §4B]
  q     <- |grads| * (seg_err_sum/seg_norm_sum)                  [SGDC §4B]
  cand  <- budget_mask(q, seg_top1, cand, B*N)                   (sgdc/budget.py)

  # A.1: mask-boundary straddle picks clone vs split (chain step 2) [SGDC §4A.1]
  straddle  <- (|sd| <= k*sig_perp) & (dt<band) & (seg>0)        (camera = current view)
  split_sel <- cand & (max_scale > theta | straddle)
  clone_sel <- cand & ~split_sel
  densify_and_clone(grads, tau, theta, sel_mask=clone_sel)       (gaussian_model.py)
  densify_and_split(grads, tau, theta, extent, sel_mask=split_sel)

  # D: prune
  prune_mask <- opacity<eps | radii>S*| scale>0.1 S
                 | (not band & rho<rho_min & avg_alpha<eps_imp)
                 | (not band & nu>nu_max & avg_residual>r_thr)   [SGDC §4D]
  prune_points(prune_mask)                                       (gaussian_model.py:349)

# every opt.opacity_reset_interval iters:
  reset_opacity()  # semantic floor                                [SGDC §4E]

# every training step (before backward):
  loss += lambda_bnd * boundary_loss(...)                          [SGDC §4C]
```

---

## 8. Hyperparameters (initial)

| Param | Role | Init |
|---|---|---|
| $f_b$ | boundary threshold factor | 0.5 |
| $f_{\text{int}}$ | interior threshold factor | 2.0 |
| $\eta$ | split-threshold shrink | 0.3 |
| $B$ | slot budget / refine step | 5% of $N$ |
| $\gamma_{\text{edge}}$ | perimeter bonus | 1.0 |
| $\lambda_{\text{bnd}}$ | boundary loss weight | 0.02 |
| $\delta_{\text{band}}$ | boundary exclusion band | ~3 px |
| $\rho_{\min}$ | purity prune bound | 0.4 |
| $\epsilon_{\text{imp}}$ | importance prune bound | 0.02 |
| $\nu_{\max}$ | void-frac bound | 0.6 |
| $r_{\text{thr}}$ | residual prune bound | 0.05 |
| $\beta_{0,1,2}$ | reset biases | −4.6, 2.0, 2.0 |

## 9. Cost

- Semantic stats: one label lookup per visible projection per refine step → **O(N · avg_views)**, negligible vs. rasterization.
- Memory: +O(N) buffers (+O(NK) only if dense histograms enabled).
- Component C: one extra projected-covariance computation per visible Gaussian — no second rasterization pass (unlike feature-splatting closed-loop designs).
- Net: **same asymptotic cost as baseline 3DGS.**

## 10. Evaluation & ablations

**Benchmarks:** Mip-NeRF 360, Tanks & Temples, Deep Blending.
**Metrics:** PSNR / SSIM / LPIPS; **Boundary-PSNR** (PSNR on dilated semantic edges); **floater count** (high-opacity Gaussians projecting into `void`); **#Gaussians** / peak GPU mem / train time → **quality-vs-count Pareto**.
**Ablations:** components A/B/C/D/E individually + cumulative (via `use_component`); semantic source (SAM2 vs semseg vs GT); sensitivity to $f_b, f_{\text{int}}, \lambda_{\text{bnd}}, B$.
**Baselines:** vanilla 3DGS; uniform-budget 3DGS (same $B$, no semantics); segmentation-as-prune-only.

## 11. Risks

1. **Center-projection label read is footprint-blind** → mitigated by boundary band + `dt` weighting.
2. **SAM2 is class-agnostic** → use instance IDs directly (ideal for B/D) or associate with classes.
3. **Void handling** → never prune on void alone; require persistent photometric error.
4. **Label noise** → multi-view top-count voting denoises; add confidence gate before trusting $s_i$.
5. **NDC y-convention** → verify projection sign once vs a rendered view.

## 12. Build order

1. **A** alone (lowest risk; clearest signal: Boundary-PSNR).
2. **E + D** (floater/opacity wins, cheap).
3. **C** (shape quality, anisotropic contours).
4. **B** (capacity/Pareto claim — needs calibration).
5. Full ablation grid + Pareto plots.
