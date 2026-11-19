"""Tests for the Component-B budget headroom test (sgdc.headroom).

Verifies the structure stats and the allocation-divergence metric (which calls
the real :mod:`sgdc.budget`), plus the first-view labeling simulation. CPU only.

    pytest tests/test_headroom.py
    python tests/test_headroom.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

from sgdc import headroom as hr
from sgdc.prep import scene_io as sio


def test_segment_stats_basic():
    seg = np.array([1, 1, 2, 0, 3])
    st = hr.segment_stats(seg)
    assert st["n_seg"] == 3
    assert st["labeled"] == 4
    assert st["max_size"] == 2
    assert 0.0 <= st["gini"] <= 1.0


def test_segment_stats_empty():
    st = hr.segment_stats(np.zeros(5, dtype=int))
    assert st["n_seg"] == 0 and st["labeled"] == 0


def _skip_if_no_torch():
    if not HAVE_TORCH:
        print("SKIP (torch not installed)")
        return True
    return False


def test_divergence_single_segment_is_global():
    # one segment -> per-segment selection is exactly the global one
    if _skip_if_no_torch():
        return
    q = np.array([5.0, 4.0, 3.0, 2.0, 1.0, 0.9, 0.8, 0.7])
    seg = np.ones(8, dtype=int)
    dv = hr.allocation_divergence(q, seg, B=0.5)
    assert abs(dv["tv"]) < 1e-6
    assert abs(dv["jaccard"] - 1.0) < 1e-6


def test_divergence_structural_reallocation():
    # seg 1 = 3 high-priority points; seg 2 = 100 low-priority points.
    # Global keeps the 3 highs + 1 low; per-segment shares give seg2 an extra slot
    # (sqrt-capacity normalization) -> a measurable, deterministic divergence.
    if _skip_if_no_torch():
        return
    q = np.concatenate([np.full(3, 9.0), np.full(100, 1.0)])
    seg = np.concatenate([np.ones(3, dtype=int), np.full(100, 2, dtype=int)])
    dv = hr.allocation_divergence(q, seg, B=0.04, gamma=0.5)
    assert dv["delta_total"] == 4
    assert abs(dv["tv"] - 0.25) < 1e-6
    assert abs(dv["jaccard"] - 0.6) < 1e-6
    assert abs(dv["changed"] - 0.5) < 1e-6


def test_divergence_empty_and_guards():
    if _skip_if_no_torch():
        return
    q = np.array([1.0, 2.0])
    seg = np.zeros(2, dtype=int)                      # all unlabeled
    dv = hr.allocation_divergence(q, seg, B=0.5)
    assert dv["delta_total"] == 1 and dv["tv"] == 0.0


def test_first_view_labels_are_first_seen():
    if _skip_if_no_torch():
        return
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        sp = root / "sparse" / "0"
        sp.mkdir(parents=True)
        (sp / "cameras.txt").write_text("1 SIMPLE_PINHOLE 4 4 4.0 2.0 2.0\n")
        (sp / "images.txt").write_text(
            "1 1.0 0.0 0.0 0.0 0.0 0.0 0.0 1 rgb_0.png\n0.0 0.0 -1\n"
            "2 1.0 0.0 0.0 0.0 0.0 0.0 0.0 1 rgb_1.png\n0.0 0.0 -1\n")
        (sp / "points3D.txt").write_text(
            "1 0.0 0.0 1.0 255 0 0 0.1\n2 0.0 0.0 1.0 0 255 0 0.1\n")
        scene = sio.load_colmap_scene(sp)

        lab_dir = root / "semantic_instance"
        lab_dir.mkdir()
        m0 = np.zeros((4, 4), dtype=np.int32); m0[2, 2] = 5
        m1 = np.zeros((4, 4), dtype=np.int32); m1[2, 2] = 9   # later view, higher id
        sio.write_id_map(lab_dir / "rgb_0.png", m0)
        sio.write_id_map(lab_dir / "rgb_1.png", m1)

        labels = hr.first_view_point_labels(scene, lab_dir)
        assert set(labels.tolist()) == {5}               # first view wins, never 9


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
