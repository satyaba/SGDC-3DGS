"""Unit tests for ``sgdc.budget`` (CPU torch only).

These cover the pure-tensor Component-B logic: the error x gradient priority,
the capacity-normalized per-segment priority, largest-remainder allocation, and
the hard budget cap (including the top-up path). No GPU / rasterizer required.

    pytest tests/test_sgdc_budget.py
    python tests/test_sgdc_budget.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
except ImportError:  # pragma: no cover - torch is expected in this project
    torch = None

from sgdc import budget as sgb


def _skip_if_no_torch():
    if torch is None:
        print("SKIP (torch not installed)")
        return True
    return False


def test_per_gaussian_priority_error_x_gradient():
    if _skip_if_no_torch():
        return
    grads = torch.tensor([[0.5], [0.25], [1.0]])
    err = torch.tensor([[0.2], [0.4], [0.0]])
    norm = torch.tensor([[2.0], [1.0], [1.0]])
    q = sgb.per_gaussian_priority(grads, err, norm, use_error=True)
    expected = grads.reshape(-1) * (err.reshape(-1) / norm.reshape(-1))
    assert torch.allclose(q, expected)
    # grad-only ignores the residual entirely
    qg = sgb.per_gaussian_priority(grads, err, norm, use_error=False)
    assert torch.allclose(qg, grads.reshape(-1))
    # unseen Gaussians (norm 0) contribute zero error, not NaN
    z = sgb.per_gaussian_priority(torch.ones(1, 1), torch.ones(1, 1), torch.zeros(1, 1))
    assert torch.isfinite(z).all()


def test_segment_priority_capacity_normalized():
    if _skip_if_no_torch():
        return
    q = torch.tensor([1.0, 1.0, 1.0, 4.0])          # seg 1 has three, seg 2 one
    seg = torch.tensor([1, 1, 1, 2])
    p = sgb.segment_priority(q, seg, n_seg=2)
    assert p.shape == (3,)
    assert p[0] == 0.0                               # unlabeled never prioritized
    assert abs(float(p[1]) - 3.0 / (3 ** 0.5)) < 1e-6
    assert abs(float(p[2]) - 4.0) < 1e-6


def test_allocate_sums_exactly_and_tracks_weights():
    if _skip_if_no_torch():
        return
    p = torch.tensor([0.0, 1.0, 3.0])
    d = sgb.allocate(100, p)
    assert int(d.sum()) == 100
    assert d[0] == 0
    assert d[2] > d[1] > 0
    # degenerate inputs
    assert int(sgb.allocate(0, p).sum()) == 0
    assert int(sgb.allocate(10, torch.zeros(3)).sum()) == 0


def test_budget_mask_is_a_hard_cap():
    if _skip_if_no_torch():
        return
    n = 1000
    q = torch.rand(n)
    seg = torch.randint(1, 5, (n,))          # four segments
    cand = torch.ones(n, dtype=torch.bool)
    delta_total = 50
    keep, delta_n = sgb.budget_mask(q, seg, cand, delta_total, n_seg=4)
    assert int(keep.sum()) <= delta_total                 # never exceeds the cap
    assert int(keep.sum()) > 0
    assert keep.sum() <= cand.sum()
    # every kept point was a candidate
    assert bool((~cand[keep]).sum() == 0)


def test_budget_mask_topup_fills_sparse_segments():
    if _skip_if_no_torch():
        return
    # Only 3 candidates exist; a budget of 10 must not invent slots.
    q = torch.tensor([0.9, 0.5, 0.1, 0.0, 0.0])
    seg = torch.tensor([1, 1, 2, 3, 3])
    cand = torch.tensor([True, True, True, False, False])
    keep, _ = sgb.budget_mask(q, seg, cand, delta_total=10, n_seg=3)
    assert int(keep.sum()) == 3


def test_budget_mask_all_spendable_goes_to_high_priority_segment():
    if _skip_if_no_torch():
        return
    q = torch.tensor([10.0, 9.0, 1.0, 0.5])
    seg = torch.tensor([1, 1, 2, 2])
    cand = torch.ones(4, dtype=torch.bool)
    keep, _ = sgb.budget_mask(q, seg, cand, delta_total=2, n_seg=2, topup=False)
    # segment 1 dominates: its two candidates are kept
    assert bool(keep[0]) and bool(keep[1]) and not bool(keep[2]) and not bool(keep[3])


def test_segment_priority_gamma_extremes():
    if _skip_if_no_torch():
        return
    q = torch.tensor([1.0, 1.0, 1.0, 4.0])
    seg = torch.tensor([1, 1, 1, 2])
    # gamma=0 -> raw sum
    p0 = sgb.segment_priority(q, seg, 2, gamma=0.0)
    assert abs(float(p0[1]) - 3.0) < 1e-6
    assert abs(float(p0[2]) - 4.0) < 1e-6
    # gamma=1 -> per-capita mean
    p1 = sgb.segment_priority(q, seg, 2, gamma=1.0)
    assert abs(float(p1[1]) - 1.0) < 1e-6
    assert abs(float(p1[2]) - 4.0) < 1e-6
    # default gamma=0.5 still capacity-normalized
    p5 = sgb.segment_priority(q, seg, 2)
    assert abs(float(p5[1]) - 3.0 / (3 ** 0.5)) < 1e-6


def test_budget_mask_floor_equalizes_shares():
    if _skip_if_no_torch():
        return
    n = 4000
    q = torch.ones(n)
    seg = torch.cat([torch.ones(3000, dtype=torch.long),
                     2 * torch.ones(1000, dtype=torch.long)])
    cand = torch.ones(n, dtype=torch.bool)
    _, dn0 = sgb.budget_mask(q, seg, cand, 400, n_seg=2, topup=False, floor=0.0)
    _, dn1 = sgb.budget_mask(q, seg, cand, 400, n_seg=2, topup=False, floor=1.0)
    # floor=0: proportional -> the larger class gets more
    assert int(dn0[1]) > int(dn0[2])
    # floor=1: equal slices (rounding aside) summing exactly to the budget
    assert abs(int(dn1[1]) - int(dn1[2])) <= 1
    assert int(dn1[1]) + int(dn1[2]) == 400


def test_budget_mask_floor_zero_matches_default():
    if _skip_if_no_torch():
        return
    torch.manual_seed(0)
    n = 500
    q = torch.rand(n)
    seg = torch.randint(1, 5, (n,))
    cand = torch.rand(n) < 0.3
    a, _ = sgb.budget_mask(q, seg, cand, 50, n_seg=4)
    b, _ = sgb.budget_mask(q, seg, cand, 50, n_seg=4, floor=0.0, gamma=0.5)
    assert bool((a == b).all())


def test_global_mask_is_a_cap_independent_of_segments():
    if _skip_if_no_torch():
        return
    q = torch.tensor([0.1, 5.0, 0.2, 4.0, 3.0])
    cand = torch.ones(5, dtype=torch.bool)
    keep = sgb.global_mask(q, cand, delta_total=2)
    assert int(keep.sum()) == 2
    assert bool(keep[1]) and bool(keep[3])            # two highest q, across all
    assert not bool(keep[0]) and not bool(keep[2]) and not bool(keep[4])
    # respects the candidate mask and is bounded by available candidates
    keep2 = sgb.global_mask(q, torch.tensor([True, False, False, False, False]), 3)
    assert int(keep2.sum()) == 1
    assert int(sgb.global_mask(q, cand, 0).sum()) == 0


if __name__ == "__main__":
    import traceback

    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception:
                failures += 1
                print(f"FAIL {name}")
                traceback.print_exc()
    sys.exit(1 if failures else 0)
