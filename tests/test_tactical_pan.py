from argparse import Namespace

import numpy as np
import pytest

from tactical_pan import apply_deep_coverage, apply_manual_corrections, smooth_centers, weighted_median


def test_weighted_median_uses_weights_not_input_order():
    values = np.array([30.0, 10.0, 20.0])
    weights = np.array([1.0, 5.0, 1.0])

    assert weighted_median(values, weights) == 10.0


@pytest.mark.parametrize(
    ("values", "weights"),
    [
        (np.array([]), np.array([])),
        (np.array([1.0]), np.array([0.0])),
        (np.array([1.0]), np.array([-1.0])),
        (np.array([1.0, 2.0]), np.array([1.0])),
    ],
)
def test_weighted_median_rejects_invalid_inputs(values, weights):
    with pytest.raises(ValueError):
        weighted_median(values, weights)


def test_smooth_centers_stays_inside_crop_bounds_and_speed_limit():
    raw = np.array([100.0, 100.0, 100.0, 3800.0, 3800.0, 3800.0])
    result = smooth_centers(raw, crop_width=2560, frame_width=3840, sample_rate=2.0)

    assert np.all(result >= 1280.0)
    assert np.all(result <= 2560.0)
    assert np.all(np.abs(np.diff(result)) <= 260.0 + 1e-9)


def test_deep_coverage_keeps_pair_inside_safe_frame():
    args = Namespace(no_deep_tracking=False, crop_width=2560, sample_rate=2.0)
    records = [
        {"deep_active": True, "deep_center_x": 3500.0},
        {"deep_active": True, "deep_center_x": 3500.0},
        {"deep_active": True, "deep_center_x": 3500.0},
    ]
    centers = np.array([1920.0, 1920.0, 1920.0])

    result = apply_deep_coverage(args, records, centers)

    assert np.all(result > centers)
    assert np.all(np.abs(np.diff(result)) <= 310.0 + 1e-9)


def test_deep_coverage_can_be_disabled():
    args = Namespace(no_deep_tracking=True, crop_width=2560, sample_rate=2.0)
    centers = np.array([1800.0, 1900.0])

    result = apply_deep_coverage(args, [{}, {}], centers)

    assert result is centers


def test_manual_correction_reaches_operator_target(tmp_path):
    correction_file = tmp_path / "corrections.csv"
    correction_file.write_text("source_time_seconds,center_x\n12,3000\n")
    args = Namespace(corrections=correction_file, start=10.0, correction_radius=2.0)
    times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    centers = np.full(5, 1900.0)

    result = apply_manual_corrections(args, times, centers)

    assert result[2] == 3000.0
    assert result[0] == 1900.0
    assert result[4] == 1900.0
