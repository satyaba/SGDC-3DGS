"""Synthetic CPU probe for the Component-B "why is per-segment ~ global" theories.

Drives the *real* ``sgdc.budget`` functions on synthetic distributions that
isolate each mechanism. No rasterizer, no dataset, CPU torch only.

Reported per regime:
  * ``IoU(s,g)``  -- overlap of the per-segment mask (``budget_mask``) and the
    global mask (``global_mask``) at the same ``delta``. ~1 means the two training
    arms pick nearly the same Gaussians.
  * ``IoU(g,q0)`` -- overlap of the global priority mask with the *grad-only*
    mask (``q = |grad|``): how much the residual changes selection at all.
  * ``spear``     -- Spearman between ``q`` and ``|grad|`` over candidates: ~1
    means the residual is rank-preserving and ``q`` carries no new ordering.
  * ``unlab``     -- fraction of candidates that are unlabeled (id 0). The
    unlabeled pool is excluded from the per-segment allocation (``p[0]=0``) and
    only ever enters through the global top-up path, so it biases the segment arm
    toward labeled classes.

    python scripts/debug_budget.py
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sgdc import budget as sgb  # noqa: E402

N = 20_000


def _spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    """Rank correlation without scipy (average ranks on ties)."""
    def rank(x):
        order = x.argsort()
        r = torch.empty(x.numel(), dtype=torch.float)
        r[order] = torch.arange(x.numel(), dtype=torch.float)
        return r
    ra, rb = rank(a), rank(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    return float((ra @ rb) / (ra.norm() * rb.norm()).clamp_min(1e-12))


def _classes(n, sizes):
    seg = torch.zeros(n, dtype=torch.long)
    off = 0
    for c, s in enumerate(sizes, start=1):
        seg[off:off + s] = c
        off += s
    return seg


def _iou(a, b):
    return int((a & b).sum()) / max(int((a | b).sum()), 1)


def _sim(N, seg, grads, q, cand, bs=(0.02, 0.05, 0.10, 0.15)):
    n_seg = int(seg.max().item())
    rows = []
    for B in bs:
        delta = int(B * N)
        keep_seg, delta_n = sgb.budget_mask(q, seg, cand, delta, n_seg)
        keep_glob = sgb.global_mask(q, cand, delta)
        keep_gradonly = sgb.global_mask(grads, cand, delta)

        quota_used = sum(
            min(int(delta_n[c].item()), int(((seg == c) & cand).sum()))
            for c in range(1, n_seg + 1))
        topup = int(keep_seg.sum()) - quota_used
        unlab = (int((cand & (seg == 0)).sum())
                 / max(int(cand.sum()), 1))
        rows.append(dict(
            B=B, n_cand=int(cand.sum()), delta=delta,
            binding=int(cand.sum()) > delta,
            iou_sg=_iou(keep_seg, keep_glob),
            iou_gq0=_iou(keep_glob, keep_gradonly),
            spear=_spearman(q[cand], grads[cand]),
            unlab=unlab, topup=topup))
    return rows


def _show(name, rows):
    print(f"\n== {name} ==")
    print(f"{'B':>5} {'nCand':>6} {'delta':>6} {'bind':>5} "
          f"{'IoU(s,g)':>9} {'IoU(g,q0)':>9} {'spear':>6} {'unlab':>6} {'topup':>6}")
    for r in rows:
        print(f"{r['B']:>5.2f} {r['n_cand']:>6} {r['delta']:>6} "
              f"{str(r['binding']):>5} {r['iou_sg']:>9.3f} {r['iou_gq0']:>9.3f} "
              f"{r['spear']:>6.3f} {r['unlab']:>6.3f} {r['topup']:>6}")


def check_T1_exact():
    """T1: when candidates <= delta the two arms are byte-identical."""
    n = 500
    q = torch.rand(n)
    seg = torch.randint(1, 4, (n,))
    cand = torch.rand(n) < 0.05           # ~25 candidates
    delta = int(0.10 * n)                 # 50 > 25 -> not tight
    keep_seg, _ = sgb.budget_mask(q, seg, cand, delta, n_seg=3)
    keep_glob = sgb.global_mask(q, cand, delta)
    ok = (bool((keep_seg == cand).all()) and bool((keep_glob == cand).all())
          and bool((keep_seg == keep_glob).all()))
    print("\n== T1: non-binding cap is an exact no-op ==")
    print(f"candidates={int(cand.sum())} delta={delta} "
          f"seg==cand==global: {ok}")
    assert ok


def check_T4_normalization():
    """T4: 1/sqrt(n_c) reallocation needs BOTH unequal size and unequal q."""
    n = 300
    eq = _classes(n, [100, 100, 100])
    zipf = _classes(n, [210, 60, 30])
    q_flat = torch.ones(n)
    q_tilt = torch.ones(n)
    q_tilt[zipf == 1] = 4.0                     # big class also much higher q
    print("\n== T4: capacity normalization  p_c = sum(q)/sqrt(n_c) ==")
    for label, seg, q in (
        ("equal size, equal q     ", eq, q_flat),
        ("zipf size,  equal q     ", zipf, q_flat),
        ("zipf size,  big class hi", zipf, q_tilt),
    ):
        p = sgb.segment_priority(q, seg, int(seg.max()))
        share = (p[1:] / p[1:].sum()).tolist()
        print(f"{label} -> p={[round(float(x), 1) for x in p[1:]]}  "
              f"share={[round(float(s), 2) for s in share]}")


def rank_collapse_report():
    """T2: IoU(global priority mask, grad-only mask) vs how much residual reorders."""
    g = torch.rand(N) ** 3 * 1e-3 + 1e-5
    cand = g >= torch.quantile(g, 0.85)
    seg = _classes(N, [int(N * 0.7), int(N * 0.2), int(N * 0.1)])
    delta = int(0.10 * N)
    print("\n== T2: does the residual change *which* points are picked? ==")
    print(f"{'resid weight':>13} {'spear(q,|g|)':>13} {'IoU(g,q0)':>10}")
    for w in (0.0, 0.05, 0.25, 1.0, 4.0):
        q = g.clone() if w == 0 else g * (1.0 + w * torch.rand(N))
        kg = sgb.global_mask(q, cand, delta)
        kg0 = sgb.global_mask(g, cand, delta)
        print(f"{w:>13.2f} {_spearman(q[cand], g[cand]):>13.3f} {_iou(kg, kg0):>10.3f}")


def allocation_sweep():
    """Options 1/2: how floor and gamma move IoU(per-segment, global)."""
    g = torch.rand(N) ** 3 * 1e-3 + 1e-5
    cand = g >= torch.quantile(g, 0.85)
    seg = _classes(N, [int(N * 0.7), int(N * 0.2), int(N * 0.1)])
    q = g * ((g / g.max()) + 0.02 * torch.rand(N))
    delta = int(0.10 * N)
    kg = sgb.global_mask(q, cand, delta)

    def share_max(floor, gamma):
        p = sgb.segment_priority(q, seg, 3, gamma=gamma).clone()
        p[0] = 0.0
        if floor > 0:
            r = p[1:] / p[1:].sum()
            p[1:] = floor / 3 + (1 - floor) * r
        return float((p[1:].max() / p[1:].sum()).item())

    print("\n== Option 1: IoU(per-segment, global) vs floor (gamma=0.5) ==")
    print(f"{'floor':>6} {'IoU':>6} {'maxshare':>9}")
    for f in (0.0, 0.25, 0.5, 0.75, 1.0):
        ks, _ = sgb.budget_mask(q, seg, cand, delta, 3, floor=f)
        print(f"{f:>6.2f} {_iou(ks, kg):>6.3f} {share_max(f, 0.5):>9.3f}")

    print("\n== Option 2: IoU(per-segment, global) vs gamma (floor=0) ==")
    print(f"{'gamma':>6} {'IoU':>6} {'maxshare':>9}")
    for gm in (0.0, 0.5, 1.0):
        ks, _ = sgb.budget_mask(q, seg, cand, delta, 3, gamma=gm)
        print(f"{gm:>6.2f} {_iou(ks, kg):>6.3f} {share_max(0.0, gm):>9.3f}")


if __name__ == "__main__":
    g = torch.rand(N) ** 3 * 1e-3 + 1e-5
    tau = torch.quantile(g, 0.85)
    cand = g >= tau

    print("=" * 72)
    print("Component-B synthetic probe  (CPU, no dataset)")
    print(f"N={N}; the gradient gate keeps ~15% of points as candidates")
    print("=" * 72)
    check_T1_exact()
    check_T4_normalization()
    rank_collapse_report()
    allocation_sweep()

    res_rank = (g / g.max()) + 0.02 * torch.rand(N)      # rank-preserving residual
    eq = _classes(N, [N // 3] * 3)
    zipf = _classes(N, [int(N * 0.7), int(N * 0.2), int(N * 0.1)])
    _show("eq-classes, residual ranks with grad",
          _sim(N, eq, g, g * res_rank, cand))
    _show("zipf-classes, residual ranks with grad",
          _sim(N, zipf, g, g * res_rank, cand))

    informative = torch.zeros(N)
    small = zipf == 3
    informative[small] = torch.rand(int(small.sum()))
    _show("zipf-classes, residual informative (small class)",
          _sim(N, zipf, g, g * (0.02 + informative), cand))

    # first half stays id 0 (unlabeled); two labeled classes fill the rest
    _show("zipf-classes, 50% unlabeled",
          _sim(N, _classes(N, [N // 4, N // 4]), g, g * res_rank, cand))
    print()
