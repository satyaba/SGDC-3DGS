"""Tests for the non-Replica label path: SAM2-style instance labels.

Covers the pure void-fill helper, :class:`InstanceDirLabelAdapter` (including
`min_area` and `fill_void`), and the cross-dataset run manifest. No GPU / torch
required.

    pytest tests/test_instance_labels.py
    python tests/test_instance_labels.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sgdc import semantic_utils as su
from sgdc.datasets import get_adapter
from experiments import runs


# --------------------------------------------------------------------------- #
# fill_void_nearest                                                             #
# --------------------------------------------------------------------------- #
def test_fill_void_nearest_propagates_ids():
    ids = np.array([[0, 0, 5],
                    [0, 5, 5]], dtype=np.int32)
    out = su.fill_void_nearest(ids)
    assert set(np.unique(out).tolist()) == {5}
    assert out.dtype == np.int32


def test_fill_void_nearest_edge_cases():
    all_void = np.zeros((3, 3), dtype=np.int32)
    assert np.array_equal(su.fill_void_nearest(all_void), all_void)
    empty = np.zeros((0, 0), dtype=np.int32)
    assert su.fill_void_nearest(empty).shape == (0, 0)
    solid = np.full((2, 2), 7, dtype=np.int32)  # nothing to fill
    assert np.array_equal(su.fill_void_nearest(solid), solid)


# --------------------------------------------------------------------------- #
# InstanceDirLabelAdapter                                                       #
# --------------------------------------------------------------------------- #
def _write_ids(path: Path, ids: np.ndarray, sixteen: bool = True):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = ids.astype(np.uint16) if sixteen else ids.astype(np.uint8)
    Image.fromarray(arr).save(path)


def test_instance_dir_adapter_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        ds = Path(d)
        (ds / "images").mkdir()
        ids = np.zeros((4, 4), dtype=np.int32)
        ids[:, :2] = 1
        ids[:, 2:] = 2
        _write_ids(ds / "semantic_instance" / "rgb_0.png", ids)

        adapter = get_adapter("sam2", ds)  # alias for instance_dir
        out = adapter.load("rgb_0.png")
        assert out is not None and out.shape == (4, 4)
        assert set(np.unique(out).tolist()) == {1, 2}
        # resolved by stem even when the RGB extension differs
        assert adapter.resolve("rgb_0.jpg") is not None
        assert adapter.resolve("missing.png") is None
        assert adapter.meta()["num_frames"] == 1


def test_instance_dir_adapter_min_area_and_fill():
    with tempfile.TemporaryDirectory() as d:
        ds = Path(d)
        ids = np.zeros((3, 3), dtype=np.int32)
        ids[:2, :] = 4          # 6-px instance
        ids[2, 2] = 9           # 1-px speckle
        _write_ids(ds / "semantic_instance" / "a.png", ids)

        # min_area drops the speckle to void, leaving a hole
        a = get_adapter("instance_dir", ds, {"min_area": 3})
        kept = a.load("a.png")
        assert set(np.unique(kept).tolist()) == {0, 4}

        # ... which fill_void then repairs with the nearest instance
        b = get_adapter("instance_dir", ds, {"min_area": 3, "fill_void": True})
        filled = b.load("a.png")
        assert set(np.unique(filled).tolist()) == {4}


def test_instance_dir_relative_label_dir_resolves_against_dataset():
    with tempfile.TemporaryDirectory() as d:
        ds = Path(d)
        _write_ids(ds / "semantic_instance_fused" / "a.png",
                   np.array([[0, 1], [2, 2]], dtype=np.int32))
        # relative label_dir must resolve against the dataset, not the cwd
        a = get_adapter("instance_dir", ds, {"label_dir": "semantic_instance_fused"})
        assert Path(a.label_dir).resolve() == (ds / "semantic_instance_fused").resolve()
        assert a.load("a.png") is not None


def test_instance_dir_adapter_missing_dir_raises():
    with tempfile.TemporaryDirectory() as d:
        try:
            get_adapter("sam2", Path(d))
        except FileNotFoundError:
            return
        raise AssertionError("expected FileNotFoundError for a dataset without labels")


# --------------------------------------------------------------------------- #
# cross-dataset run manifest                                                    #
# --------------------------------------------------------------------------- #
def test_cross_dataset_runs_registered():
    for rid in ("garden_base_30k", "garden_b10_30k", "garden_sig_30k",
                "bicycle_base_30k", "bicycle_b10_30k", "bicycle_sig_30k"):
        assert rid in runs.RUNS, f"missing run {rid}"


def test_sam2_runs_carry_the_adapter_flag():
    # SGDC arms on the Mip-NeRF 360 scenes must select the sam2 adapter ...
    for rid in ("garden_b10_30k", "garden_sig_30k",
                "bicycle_b10_30k", "bicycle_sig_30k"):
        extra = runs.RUNS[rid].extra
        assert "--sgdc_adapter" in extra
        assert extra[extra.index("--sgdc_adapter") + 1] == "sam2"
    # ... the plain baselines do not (SGDC is off)
    assert "--sgdc_adapter" not in runs.RUNS["garden_base_30k"].extra
    assert runs.RUNS["garden_b10_30k"].dataset == "mipnerf360/garden"


def test_run_dataset_override():
    captured = {}

    def fake_run_method(method_id, **kw):
        captured["source"] = str(kw["source"])
        return kw["model_dir"]

    original = runs.run_method
    runs.run_method = fake_run_method
    try:
        runs.run_by_id("base_30k", dataset="room_0/Sequence_1",
                       home_dir="/h", dataset_root="/ds")
    finally:
        runs.run_method = original
    assert captured["source"].replace("\\", "/") == "/ds/room_0/Sequence_1"


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
