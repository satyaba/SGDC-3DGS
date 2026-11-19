"""Budget headroom test for Component B (pre-training acceptance check).

Motivation
----------
The office_0 study found that B's *segmentation* (per-segment vs scene-global
allocation) is inert. Before spending a 30k-iteration run on a new dataset, we
want to know whether the label partition even *can* move B's allocation away from
the non-segmented control. If per-segment and global allocate the budget to the
same places, no labeling -- however good -- can produce a segmentation effect.

What this measures
------------------
B allocates a refine budget ``B * N`` across segments by
``p_c = sum_{i in c} q_i / n_c^gamma`` and ranks candidates inside each segment by
``q_i = |grad_i| * residual_i``; the non-segmented control ranks every candidate
globally by ``q_i``. Pre-training we have no Gaussians, so we stand in for them
with the **COLMAP points** (one per surface sample) and approximate ``q`` with an
image-**gradient** proxy (B's gradient term; the ``--sgdc_budget_grad_only`` arm
that was measured to tie the error-weighted one). We then call the *real*
:mod:`sgdc.budget` functions and report how much the per-segment selection differs
from the global one:

* ``TV``      -- total-variation distance between the per-segment and global
                 *count vectors* (fraction of the budget allocated differently).
                 0 = identical to the control; 1 = disjoint.
* ``changed`` -- fraction of the budget placed on different points than global.
* ``jaccard`` -- overlap of the two kept sets.

Calibration
-----------
The number alone is not the answer; we compare against controls that hold the
segment **sizes** fixed but destroy the priority<->segment structure:

* ``single``  -- one segment  -> exactly the global control (TV == 0).
* ``random``  -- the segment ids are permuted -> chance-level TV for this size
                 distribution. Fused TV must beat this for the *structure* to be
                 doing anything.

Usage
-----
    python -m sgdc.headroom --dataset_dir "$DATASET_ROOT/mipnerf360/garden" \
        --fused_dir semantic_instance_fused --raw_dir semantic_instance \
        --B 0.10 --gamma 0.5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

from sgdc import budget as sgb
from sgdc.prep import scene_io as sio
from sgdc.segment_sam2 import IMAGE_EXTS


# --------------------------------------------------------------------------- #
# structure statistics                                                          #
# --------------------------------------------------------------------------- #
def _gini(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = x.size
    if n == 0 or x.sum() <= 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * (idx * x).sum() / (n * x.sum())) - (n + 1) / n)


def segment_stats(seg: np.ndarray, support: Optional[np.ndarray] = None) -> dict:
    """Size/shape summary of a per-point segment partition."""
    seg = np.asarray(seg)
    labeled = seg > 0
    n_seg = int(seg.max()) if seg.size else 0
    if n_seg == 0 or not labeled.any():
        return {"n_seg": 0, "labeled": int(labeled.sum()), "mean_size": 0.0,
                "p90_size": 0.0, "max_size": 0, "gini": 0.0}
    sizes = np.bincount(seg[labeled], minlength=n_seg + 1)[1:]
    out = {
        "n_seg": n_seg,
        "labeled": int(labeled.sum()),
        "mean_size": float(sizes.mean()),
        "p90_size": float(np.percentile(sizes, 90)),
        "max_size": int(sizes.max()),
        "gini": _gini(sizes),
    }
    if support is not None:
        sup = np.asarray(support)
        if sup.size > n_seg:
            out["mean_views_per_seg"] = float(sup[1:n_seg + 1].mean())
            out["min_views_per_seg"] = int(sup[1:n_seg + 1].min())
    return out


# --------------------------------------------------------------------------- #
# allocation divergence (uses the real sgdc.budget functions)                   #
# --------------------------------------------------------------------------- #
def _entropy(counts: np.ndarray) -> float:
    p = np.asarray(counts, dtype=np.float64)
    p = p[p > 0]
    if p.size <= 1:
        return 0.0
    p = p / p.sum()
    return float(-(p * np.log(p)).sum())


def allocation_divergence(q: np.ndarray, seg: np.ndarray, *, B: float = 0.10,
                          gamma: float = 0.5, topup: bool = True,
                          candidate: Optional[np.ndarray] = None,
                          labeled_only: bool = False) -> dict:
    """Fraction of B's per-segment budget that lands differently than global.

    ``q``/``seg`` are per-sample (per-3D-point) arrays. ``candidate`` defaults to
    ``q > 0`` (samples seen by at least one view). ``labeled_only`` restricts the
    candidate pool to labeled points (``seg > 0``) for *both* B and the global
    control, which removes the unlabeled-pool asymmetry and makes the
    single-segment control read exactly 0 (a clean diagnostic floor).
    """
    q = torch.as_tensor(np.asarray(q), dtype=torch.float32)
    seg = torch.as_tensor(np.asarray(seg), dtype=torch.long).clamp_min(0)
    n = q.numel()
    if candidate is None:
        candidate = q > 0
    candidate = torch.as_tensor(np.asarray(candidate), dtype=torch.bool)
    if labeled_only:
        candidate = candidate & (seg > 0)
    n_seg = int(seg.max().item()) if n else 0
    delta_total = int(B * n)
    empty = {"delta_total": delta_total, "kept_seg": 0, "kept_glob": 0,
             "tv": 0.0, "changed": 0.0, "jaccard": 1.0, "share_entropy": 0.0}
    if n_seg == 0 or delta_total <= 0 or not bool(candidate.any()):
        return empty

    keep_seg, delta_n = sgb.budget_mask(q, seg, candidate, delta_total, n_seg,
                                        topup=topup, gamma=gamma)
    keep_glob = sgb.global_mask(q, candidate, delta_total)

    c_seg = torch.bincount(seg[keep_seg], minlength=n_seg + 1)
    c_glob = torch.bincount(seg[keep_glob], minlength=n_seg + 1)
    denom = max(1, int(keep_glob.sum()))
    tv = 0.5 * float((c_seg - c_glob).abs().sum()) / denom
    inter = int((keep_seg & keep_glob).sum())
    union = int((keep_seg | keep_glob).sum())
    changed = int((keep_seg ^ keep_glob).sum()) / max(1, delta_total)
    return {
        "delta_total": delta_total,
        "kept_seg": int(keep_seg.sum()),
        "kept_glob": int(keep_glob.sum()),
        "tv": tv,
        "changed": changed,
        "jaccard": (inter / union) if union else 1.0,
        "share_entropy": _entropy(delta_n.numpy()),
    }


# --------------------------------------------------------------------------- #
# labelings + priority proxy                                                    #
# --------------------------------------------------------------------------- #
def first_view_point_labels(scene: sio.ColmapScene, labels_dir: Path) -> np.ndarray:
    """Simulate SGDC's ``seg_top1``: each point takes the id from the **first**
    labeled view that sees it (never updated). This is the *raw* (unfused) case."""
    return sio.point_labels_from_views(scene, labels_dir, mode="first")


def _read_gray(images_dir: Path, name: str) -> Optional[np.ndarray]:
    from PIL import Image
    p = images_dir / name
    if not p.exists():
        cand = next((c for c in images_dir.rglob(f"{Path(name).stem}.*")
                     if c.suffix.lower() in IMAGE_EXTS), None)
        if cand is None:
            return None
        p = cand
    return np.array(Image.open(p).convert("L"))


def point_gradient_priority(scene: sio.ColmapScene, images_dir: Path,
                            max_dim: int = 512) -> np.ndarray:
    """Per-point mean image-gradient magnitude over the views that see it.

    A pre-training stand-in for B's per-Gaussian ``|grad|`` signal. Images are
    (optionally) downscaled and the intrinsics scaled to match, so this stays
    cheap on 1-2K frames. Points seen by no view get priority 0.
    """
    import cv2

    p = scene.xyz.shape[0]
    acc = np.zeros(p, dtype=np.float64)
    cnt = np.zeros(p, dtype=np.int64)
    for cam in scene.cameras:
        gray = _read_gray(images_dir, cam.name)
        if gray is None:
            continue
        H, W = gray.shape
        s = 1.0
        if max_dim and max(H, W) > max_dim:
            s = max_dim / max(H, W)
            gray = cv2.resize(gray, (int(round(W * s)), int(round(H * s))))
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        Hh, Ww = mag.shape
        u, v, z, valid = sio.project_points(scene.xyz, cam)
        u = np.round(u * s).astype(np.int64)
        v = np.round(v * s).astype(np.int64)
        inb = valid & (u >= 0) & (u < Ww) & (v >= 0) & (v < Hh)
        if inb.any():
            idx = np.nonzero(inb)[0]
            acc[idx] += mag[v[idx], u[idx]]
            cnt[idx] += 1
    q = np.zeros(p, dtype=np.float64)
    seen = cnt > 0
    q[seen] = acc[seen] / cnt[seen]
    return q


# --------------------------------------------------------------------------- #
# driver                                                                        #
# --------------------------------------------------------------------------- #
def _fmt_row(name, st, dv):
    return (f"{name:<22} {st['n_seg']:>6d} {st['labeled']:>8d} "
            f"{dv['tv']:>7.3f} {dv['changed']:>8.3f} {dv['jaccard']:>8.3f} "
            f"{dv['share_entropy']:>8.3f}")


def _drop_large(seg: np.ndarray, frac: float) -> np.ndarray:
    """Void segments covering more than ``frac`` of the labeled points."""
    seg = np.asarray(seg)
    if frac >= 1.0 or seg.size == 0:
        return seg
    out = seg.copy()
    lab = seg > 0
    if not lab.any():
        return out
    u, c = np.unique(seg[lab], return_counts=True)
    big = u[c > frac * lab.sum()]
    if big.size:
        out[np.isin(seg, big)] = 0
    return out


def run_headroom(dataset_dir: str | Path, *, fused_dir: Optional[str] = None,
                 raw_dir: Optional[str] = None, sparse_subdir: str = "sparse/0",
                 images_subdir: str = "images", B: float = 0.10, gamma: float = 0.5,
                 gammas: Optional[Sequence[float]] = None,
                 drop_above_frac: float = 1.0, labeled_only: bool = True,
                 priority: str = "gradient", seed: int = 0, max_dim: int = 512) -> dict:
    dataset_dir = Path(dataset_dir)
    scene = sio.load_colmap_scene(dataset_dir / sparse_subdir)
    images_dir = dataset_dir / images_subdir
    gamma_list = list(gammas) if gammas else [gamma]
    if gamma not in gamma_list:
        gamma_list = [gamma] + gamma_list

    if priority == "gradient":
        q = point_gradient_priority(scene, images_dir, max_dim=max_dim)
        prio_txt = "image |grad| (mean over views)"
    elif priority == "random":
        q = np.random.default_rng(seed).random(scene.xyz.shape[0])
        prio_txt = "random (i.i.d.)"
    else:  # uniform
        q = np.ones(scene.xyz.shape[0])
        prio_txt = "uniform"
    print(f"[headroom] {len(scene.cameras)} cams, {scene.xyz.shape[0]} points | "
          f"priority = {prio_txt} | B={B} | labeled_only={labeled_only} | "
          f"drop_above_frac={drop_above_frac}", flush=True)

    labelings: dict = {}
    if fused_dir:
        pl = Path(fused_dir)
        pl = pl if pl.is_absolute() else dataset_dir / pl
        npy = pl / "point_labels.npy"
        if npy.exists():
            labelings["fused"] = np.load(npy).astype(np.int32)
        else:
            print(f"[headroom] no {npy}; skipping fused labeling", flush=True)
    if raw_dir:
        rd = Path(raw_dir)
        rd = rd if rd.is_absolute() else dataset_dir / rd
        if rd.exists():
            labelings["raw (first-view)"] = first_view_point_labels(scene, rd)
    if not labelings:
        raise SystemExit("nothing to evaluate: pass --fused_dir and/or --raw_dir")
    if drop_above_frac < 1.0:
        labelings = {k: _drop_large(v, drop_above_frac) for k, v in labelings.items()}

    fused = labelings.get("fused")
    raw = labelings.get("raw (first-view)")

    # Controls: single segment (== global) and a size-preserving permutation.
    alllabel = dict(labelings)
    ref = fused if fused is not None else raw
    if ref is not None:
        alllabel["single-segment (ctrl)"] = np.where(ref > 0, 1, 0)
    ref_rand = fused if _n_seg(fused) > 1 else (raw if _n_seg(raw) > 1 else None)
    if ref_rand is not None:
        rng = np.random.default_rng(seed)
        seg_rand = np.zeros_like(ref_rand)
        lab = np.nonzero(ref_rand > 0)[0]
        seg_rand[lab] = rng.permutation(ref_rand[lab])      # keep sizes, shuffle structure
        src = "fused" if _n_seg(fused) > 1 else "raw"
        alllabel[f"random (ctrl,{src})"] = seg_rand
    else:
        print("[headroom] random control skipped: no non-degenerate partition.", flush=True)

    # --- main table at the primary gamma ----------------------------------- #
    pg = gamma
    header = (f"{'labeling':<22} {'n_seg':>6} {'labeled':>8} {'TV':>7} "
              f"{'changed':>8} {'jaccard':>8} {'H(share)':>8}")
    print("\n" + header)
    print("-" * len(header))
    report = {"points": int(scene.xyz.shape[0]), "B": B, "gamma": pg,
              "gammas": gamma_list, "labeled_only": labeled_only,
              "drop_above_frac": drop_above_frac, "priority": prio_txt, "labelings": {}}
    stats_cache = {}
    for name, seg in alllabel.items():
        st = segment_stats(seg, _support_per_segment(seg, scene))
        stats_cache[name] = st
        dv = allocation_divergence(q, seg, B=B, gamma=pg, labeled_only=labeled_only)
        print(_fmt_row(name, st, dv))
        report["labelings"][name] = {"stats": st, "divergence": dv}

    # --- gamma sweep -------------------------------------------------------- #
    if len(gamma_list) > 1:
        print(f"\n[headroom] gamma sweep (TV vs global; floor=0 when labeled_only)")
        cols = "".join(f"{('g=%.2f' % g):>9}" for g in gamma_list)
        print(f"{'labeling':<22}{cols}")
        print("-" * (22 + 9 * len(gamma_list)))
        for name, seg in alllabel.items():
            tvs = [allocation_divergence(q, seg, B=B, gamma=g,
                                         labeled_only=labeled_only)["tv"] for g in gamma_list]
            report["labelings"][name]["tv_by_gamma"] = dict(zip(gamma_list, tvs))
            print(f"{name:<22}" + "".join(f"{t:>9.3f}" for t in tvs))

    _print_verdict(report)
    return report


def _n_seg(seg: Optional[np.ndarray]) -> int:
    return int(seg.max()) if seg is not None and seg.size else 0


def _support_per_segment(seg: np.ndarray, scene: sio.ColmapScene) -> np.ndarray:
    """#distinct frames in which each point-segment has at least one visible point."""
    n_seg = int(seg.max()) if seg.size else 0
    out = np.zeros(n_seg + 1, dtype=int)
    if n_seg == 0:
        return out
    for cam in scene.cameras:
        u, v, z, valid = sio.project_points(scene.xyz, cam)
        ui = np.round(u).astype(np.int64)
        vi = np.round(v).astype(np.int64)
        inb = valid & (ui >= 0) & (ui < cam.width) & (vi >= 0) & (vi < cam.height)
        for g in np.unique(seg[inb]).tolist():
            if g > 0:
                out[g] += 1
    return out


def _print_verdict(report: dict) -> None:
    labels = report["labelings"]
    fused_lab = labels.get("fused", {})
    fused = fused_lab.get("divergence")
    rand = next((v.get("divergence") for k, v in labels.items() if k.startswith("random")), None)
    raw = labels.get("raw (first-view)", {}).get("divergence")
    if not fused:
        return
    n_seg_fused = fused_lab.get("stats", {}).get("n_seg", 0)
    print(f"\n[headroom] reading (B={report['B']}, gamma={report['gamma']}, "
          f"labeled_only={report.get('labeled_only')}):")
    if n_seg_fused <= 1:
        print(f"  fused has n_seg={n_seg_fused}: the labeling is DEGENERATE "
              "(fusion collapsed to one segment), so TV=0 is an artifact, not a "
              "verdict. Fix the fusion and re-run.")
        return
    print(f"  fused TV={fused['tv']:.3f} vs random-control TV="
          f"{(rand or {}).get('tv', float('nan')):.3f}"
          + (f", raw TV={raw['tv']:.3f}" if raw else ""))
    tv, rtv = fused["tv"], (rand or {}).get("tv")
    if tv < 0.05:
        print("  -> LITTLE headroom: per-segment allocation is ~the global control.")
    elif rtv is not None and tv <= rtv + 0.02:
        print("  -> NO semantic advantage: fused reallocates no more than a size-matched "
              "random partition -- the segmentation is not steering the budget. "
              "Check the gamma sweep; if fused <= random at every gamma, B's "
              "segmentation is inert (as on office_0).")
    else:
        print("  -> HEADROOM: fused reallocates the budget beyond the size-matched "
              "control; a segmentation effect is at least possible.")


def _main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Component-B budget headroom test (pre-training)")
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--fused_dir", default="semantic_instance_fused")
    ap.add_argument("--raw_dir", default="semantic_instance")
    ap.add_argument("--sparse_subdir", default="sparse/0")
    ap.add_argument("--images_subdir", default="images")
    ap.add_argument("--B", type=float, default=0.10)
    ap.add_argument("--gamma", type=float, default=0.5, help="size-discount exponent")
    ap.add_argument("--gammas", nargs="+", type=float, default=None,
                    help="sweep several gammas and print a TV table")
    ap.add_argument("--drop_above_frac", type=float, default=1.0,
                    help="void segments covering more than this fraction of points "
                         "(e.g. 0.2 to drop a dominant background blob)")
    ap.add_argument("--include_unlabeled", action="store_true",
                    help="let the global control also use unlabeled points (training-"
                         "faithful); default restricts both to labeled points")
    ap.add_argument("--priority", default="gradient", choices=["gradient", "random", "uniform"])
    ap.add_argument("--max_dim", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json_out", default="")
    args = ap.parse_args(argv)
    rep = run_headroom(
        args.dataset_dir, fused_dir=args.fused_dir, raw_dir=args.raw_dir,
        sparse_subdir=args.sparse_subdir, images_subdir=args.images_subdir,
        B=args.B, gamma=args.gamma, gammas=args.gammas,
        drop_above_frac=args.drop_above_frac,
        labeled_only=not args.include_unlabeled,
        priority=args.priority, seed=args.seed, max_dim=args.max_dim)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(rep, indent=2))
        print(f"[headroom] wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
