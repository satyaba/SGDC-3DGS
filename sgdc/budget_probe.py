"""Runtime probe for Component B: records the per-refine-step evidence needed to
explain why the per-segment budget tracks the global control.

Enabled with ``--sgdc_budget_debug``. Each refine step appends one JSON object to
``$SGDC_BUDGET_DEBUG_PATH`` (default ``sgdc_budget_probe.jsonl`` in the cwd):

    iter, N, n_seg, delta, n_cand, binding,
    iou_sg    IoU(per-segment mask, global mask)      -> ~1 means identical arms
    iou_gq0   IoU(global mask, grad-only mask)        -> how much residual matters
    iou_sel   IoU(selected mask, global mask)         -> sanity for the live arm
    spear     Spearman(q, |grad|) over candidates     -> ~1 means residual inert
    unlab     fraction of candidates with segment id 0
    topup     slots filled by the global top-up path
    class_cv  coeff. of variation of per-class candidate counts
    share_max largest single-class budget share

Summarise a file with:

    python -m sgdc.budget_probe sgdc_budget_probe.jsonl
"""
from __future__ import annotations

import json
import os
import sys

import torch

from sgdc import budget as sgb

_DEFAULT_PATH = "sgdc_budget_probe.jsonl"


def _path() -> str:
    return os.environ.get("SGDC_BUDGET_DEBUG_PATH", _DEFAULT_PATH)


def _spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() < 2:
        return float("nan")

    def rank(x):
        order = x.argsort()
        r = torch.empty(x.numel(), dtype=torch.float, device=x.device)
        r[order] = torch.arange(x.numel(), dtype=torch.float, device=x.device)
        return r

    ra, rb = rank(a), rank(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    return float((ra @ rb) / (ra.norm() * rb.norm()).clamp_min(1e-12))


def _iou(a: torch.Tensor, b: torch.Tensor) -> float:
    return int((a & b).sum()) / max(int((a | b).sum()), 1)


def record(q, grads, seg, cand_pre, selected, delta_total, n_seg, mode, iteration=None):
    """Compute and append one probe record. All tensors are ``[N]`` / ``[N,1]``."""
    q = q.reshape(-1).detach()
    g = grads.reshape(-1).abs().detach()
    seg = seg.reshape(-1).detach()
    cand_pre = cand_pre.reshape(-1).bool()
    selected = selected.reshape(-1).bool()

    keep_glob = sgb.global_mask(q, cand_pre, delta_total)
    keep_gq0 = sgb.global_mask(g, cand_pre, delta_total)
    keep_seg = (sgb.budget_mask(q, seg, cand_pre, delta_total, n_seg)[0]
                if n_seg > 0 else keep_glob)

    n_cand = int(cand_pre.sum())
    counts = []
    for c in range(1, n_seg + 1):
        counts.append(int(((seg == c) & cand_pre).sum()))
    if counts:
        import statistics
        mean = statistics.fmean(counts)
        class_cv = (statistics.pstdev(counts) / mean) if mean > 0 else 0.0
    else:
        class_cv = 0.0

    p = sgb.segment_priority(q, seg, n_seg)
    alloc = sgb.allocate(delta_total, p)
    share_max = float((p[1:].max() / p[1:].sum()).item()) if n_seg > 0 and p[1:].sum() > 0 else 0.0

    quota_used = sum(min(counts[c - 1], int(alloc[c].item()))
                     for c in range(1, n_seg + 1))
    rec = dict(
        iter=int(iteration) if iteration is not None else -1,
        N=int(q.numel()),
        n_seg=int(n_seg),
        delta=int(delta_total),
        n_cand=n_cand,
        binding=int(n_cand > delta_total),
        mode=mode,
        iou_sg=_iou(keep_seg, keep_glob),
        iou_gq0=_iou(keep_glob, keep_gq0),
        iou_sel=_iou(selected, keep_glob),
        spear=_spearman(q[cand_pre], g[cand_pre]) if n_cand >= 2 else float("nan"),
        unlab=float((cand_pre & (seg == 0)).sum().item() / max(n_cand, 1)),
        topup=int(keep_seg.sum().item()) - quota_used,
        class_cv=float(class_cv),
        share_max=share_max,
    )
    with open(_path(), "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")
    return rec


def summarize(path=None):
    path = path or _path()
    if not os.path.exists(path):
        print(f"no probe file at {path}")
        return None
    recs = [json.loads(ln) for ln in open(path, encoding="utf-8") if ln.strip()]
    if not recs:
        print(f"empty probe file at {path}")
        return None
    n = len(recs)
    import statistics as st

    def mean(rows, key):
        vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))
                and r[key] == r[key]]
        return st.fmean(vals) if vals else float("nan")

    binding = [r for r in recs if r["binding"]]
    slack = [r for r in recs if not r["binding"]]
    print(f"\n=== Component-B runtime probe: {path}  ({n} refine steps) ===")
    print(f"binding fraction (n_cand > delta) : {len(binding)}/{n} "
          f"= {len(binding) / n:.2f}   (else the cap is an exact no-op)")

    def line(label, key):
        print(f"  {label:<34} all={mean(recs, key):.3f} | "
              f"binding={mean(binding, key):.3f} | slack={mean(slack, key):.3f}")

    line("IoU(per-segment, global)", "iou_sg")
    line("IoU(global, grad-only)", "iou_gq0")
    line("Spearman(q, |grad|)", "spear")
    line("unlabeled candidate fraction", "unlab")
    line("max single-class share", "share_max")
    line("class-size CV", "class_cv")
    topup = mean(recs, "topup") / max(mean(recs, "delta"), 1)
    print(f"  {'top-up slots / delta':<34} {topup:.3f}")

    # Adaptive read-out: the binding fraction is the dominant knob.
    frac = len(binding) / n
    iou_s = mean(recs, "iou_sg")
    print("\nReading:")
    if frac < 0.25:
        print(f"  The cap binds in only {frac:.0%} of steps, so per-segment and global")
        print("  are byte-identical the other " + f"{(1 - frac):.0%}" + " -> arms must tie.")
    elif iou_s > 0.9:
        print(f"  Cap binds often ({frac:.0%}) but IoU(per-segment, global)={iou_s:.2f}")
        print("  is high -> the two allocations pick nearly the same points.")
    else:
        print(f"  Cap binds often ({frac:.0%}) and IoU(per-segment, global)={iou_s:.2f}")
        print("  is low -> segmentation is actually reallocating; check if quality moves.")
    return recs


if __name__ == "__main__":
    summarize(sys.argv[1] if len(sys.argv) > 1 else None)
