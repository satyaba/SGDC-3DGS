"""Segment-level error budgeting + budget-conditioned clone/split selection.

This implements SGDC's *Component B* (``docs/sgdc_spec.md`` §4B) in its
error-driven form. Instead of letting every segment densify until the (per-Gaussian,
boundary-modulated) gradient threshold is exhausted, a refine step is given a
**hard budget** of ``B`` new slots and those slots are allocated across segments
in proportion to each segment's *fixable error*:

    q_i = |g_i| * mean_residual_i                  (per-Gaussian, error x gradient)
    p_c = sum_{i in c} q_i / n_c^gamma              (per-segment; gamma = size discount)
    dN_c = B * p_c / sum_c' p_c'                    (integer shares)

``gamma`` (``--sgdc_budget_gamma``) tunes the size discount: 0 = raw sum,
0.5 = capacity-normalized (default), 1 = per-capita mean. ``floor``
(``--sgdc_budget_alloc_floor``) blends each share toward an equal slice per
class so low-priority classes are not starved.

``q_i`` folds the view-space positional gradient ``g_i`` (a *benefit* proxy) into
the photometric residual ``e_i`` (a *need* proxy). Pure residual would spend the
budget on unrecoverable regions (sky, specular, motion blur) where adding
capacity cannot help; pure gradient ignores where the reconstruction is actually
wrong. Multiplying the two asks for *fixable* error.

Candidates inside a segment are ranked by the same ``q_i`` and the top ``dN_c``
are kept. Leftover budget (segments with fewer candidates than their share) is
topped up from the global candidate pool by ``q_i``, so the cap is never exceeded
but is also never left unspent while candidates remain.

Everything here is pure ``torch`` and device-agnostic so it can be unit-tested on
CPU.
"""
from __future__ import annotations

import torch


def per_gaussian_priority(grads, err_sum, norm_sum, use_error: bool = True):
    """``q_i``: error x gradient, or gradient-only when ``use_error`` is False.

    ``grads`` is ``[N, 1]`` (``xyz_gradient_accum / denom``), ``err_sum`` and
    ``norm_sum`` are ``[N, 1]`` accumulators. Returns ``[N]`` with no gradient.
    """
    g = grads.reshape(-1).abs()
    if not use_error:
        return g
    e = (err_sum.reshape(-1) / norm_sum.reshape(-1).clamp_min(1.0))
    return g * e


def segment_priority(q, seg, n_seg: int, gamma: float = 0.5):
    """``p_c`` for segment ids ``0..n_seg`` (0 == unlabeled/unseen).

    ``p_c = sum_{i in c} q_i / n_c**gamma`` where ``n_c`` is the number of
    Gaussians currently assigned to ``c``. ``gamma`` is the size-discount
    exponent: ``0`` = raw sum (large segments win), ``0.5`` = capacity-normalized
    (default), ``1`` = per-capita mean (small, high-value segments win).
    Returns ``[n_seg + 1]``.
    """
    size = int(n_seg) + 1
    q = q.reshape(-1)
    seg = seg.reshape(-1).clamp(0, size - 1)
    num = torch.zeros(size, device=q.device, dtype=q.dtype).index_add_(0, seg, q)
    cnt = torch.zeros(size, device=q.device, dtype=q.dtype).index_add_(
        0, seg, torch.ones_like(q))
    denom = cnt.clamp_min(1.0).pow(float(gamma))
    return num / denom


def allocate(delta_total: int, p):
    """Split ``delta_total`` integer slots across ids by weight ``p``.

    Largest-remainder allocation: each id gets its floor share, the remaining
    slots go to the ids with the largest fractional parts. Returns ``long`` with
    the same shape as ``p`` and ``sum() == min(delta_total, ...)`` (the tail is
    simply dropped when the total exceeds available mass only at rounding).
    """
    p = p.reshape(-1)
    total = p.sum()
    long = torch.zeros_like(p, dtype=torch.long)
    delta_total = int(delta_total)
    if delta_total <= 0 or total <= 0:
        return long
    exact = p / total * delta_total
    long = torch.floor(exact).to(torch.long)
    rem = delta_total - int(long.sum())
    if rem > 0:
        # give the leftover to the largest fractional parts
        frac = exact - long.to(exact.dtype)
        k = min(rem, frac.numel())
        top = torch.topk(frac, k).indices
        long[top] += 1
    return long


def topk_mask(q, seg, delta_n, candidate_mask, n_seg: int):
    """Per-segment top-``delta_n[c]`` selection inside ``candidate_mask``.

    Returns a boolean mask over the same ``[N]`` support as ``q``.
    """
    q = q.reshape(-1)
    seg = seg.reshape(-1)
    candidate_mask = candidate_mask.reshape(-1)
    keep = torch.zeros_like(candidate_mask)
    for c in torch.unique(seg[candidate_mask]).tolist():
        budget = int(delta_n[int(c)]) if 0 <= int(c) < delta_n.numel() else 0
        if budget <= 0:
            continue
        idx = (seg == c) & candidate_mask
        vals = q[idx]
        if vals.numel() <= budget:
            keep[idx] = True
        else:
            sub = torch.nonzero(idx, as_tuple=False).squeeze(-1)
            keep[sub[torch.topk(vals, budget).indices]] = True
    return keep


def budget_mask(q, seg, candidate_mask, delta_total: int, n_seg: int, topup: bool = True,
                floor: float = 0.0, gamma: float = 0.5):
    """Full Component-B selection: allocate ``delta_total`` slots and pick candidates.

    ``q``           [N]   per-Gaussian priority (error x gradient)
    ``seg``         [N]   per-Gaussian segment id
    ``candidate_mask`` [N] bool, the pre-budget gradient candidates
    ``delta_total`` int   hard budget of new slots for this refine step
    ``n_seg``       int   number of real segments (max id; 0 is unlabeled)
    ``floor``       float in [0,1]; reserve this fraction of the budget as an
                    equal slice per class and allocate the rest by priority
                    (0 = pure priority, 1 = equal per class)
    ``gamma``       float; size-discount exponent passed to :func:`segment_priority`

    Returns ``(keep, delta_n)``.
    """
    candidate_mask = candidate_mask.reshape(-1)
    if delta_total <= 0 or not bool(candidate_mask.any()):
        return torch.zeros_like(candidate_mask), torch.zeros(int(n_seg) + 1,
                                                              dtype=torch.long,
                                                              device=q.device)
    p = segment_priority(q, seg, n_seg, gamma=gamma).clone()
    p[0] = 0.0  # never budget the unlabeled pool
    if floor > 0.0 and n_seg > 0:
        # Blend the per-class weights toward an equal slice:
        #   w_c = floor / n_seg + (1 - floor) * p_c / sum(p).
        real = p[1:]
        s = real.sum()
        if s > 0:
            r = real / s
            p[1:] = float(floor) / n_seg + (1.0 - float(floor)) * r
        else:
            p[1:] = 1.0 / n_seg
    delta_n = allocate(delta_total, p)
    keep = topk_mask(q, seg, delta_n, candidate_mask, n_seg)

    if topup:
        remaining = int(delta_total) - int(keep.sum())
        pool = candidate_mask & ~keep
        if remaining > 0 and bool(pool.any()):
            pool_idx = torch.nonzero(pool, as_tuple=False).squeeze(-1)
            take = min(remaining, int(pool_idx.numel()))
            keep = keep.clone()
            keep[pool_idx[torch.topk(q[pool_idx], take).indices]] = True
    return keep, delta_n


def global_mask(q, candidate_mask, delta_total: int):
    """Scene-global budget: keep the top-``delta_total`` candidates by ``q``.

    This is the *non-segmented* control for Component B. It applies the same
    priority ``q`` and the same hard cap, but ranks every candidate together
    instead of allocating per segment. Comparing it against
    :func:`budget_mask` isolates how much of B's effect comes from the
    segmentation/allocation rather than a plain capacity cap.
    """
    q = q.reshape(-1)
    candidate_mask = candidate_mask.reshape(-1)
    keep = torch.zeros_like(candidate_mask)
    if delta_total <= 0 or not bool(candidate_mask.any()):
        return keep
    pool = torch.nonzero(candidate_mask, as_tuple=False).squeeze(-1)
    take = min(int(delta_total), int(pool.numel()))
    keep[pool[torch.topk(q[pool], take).indices]] = True
    return keep

