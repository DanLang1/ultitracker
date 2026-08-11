from argparse import Namespace

import numpy as np
import pytest

from tactical_pan import (
    apply_deep_coverage,
    apply_manual_corrections,
    identify_deep_pairs,
    select_field_players,
    smooth_centers,
    suppress_stationary_tracks,
    weighted_median,
)


def player(track_id, x, velocity=180.0, center_y=1200.0):
    return {
        "track_id": track_id,
        "center_x": float(x),
        "center_y": center_y,
        "height": 100.0,
        "weight": 1.0,
        "velocity_x": velocity,
        "velocity_y": 0.0,
        "speed_body": abs(velocity) / 100.0,
        "stationary_factor": 1.0,
    }


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


def test_tracker_keeps_identity_through_a_missed_sample():
    records = [
        {"players": [player(0, 500.0)]},
        {"players": [player(0, 700.0)]},
        {"players": []},
        {"players": [player(0, 1100.0)]},
    ]
    for record in records:
        for tracked_player in record["players"]:
            tracked_player.pop("track_id")

    suppress_stationary_tracks(records, sample_rate=2.0)

    assert records[0]["players"][0]["track_id"] == records[3]["players"][0]["track_id"]


def test_tracker_prefers_a_recent_track_over_a_stale_nearby_track():
    records = [
        {"players": [player(0, 500.0), player(0, 600.0)]},
        {"players": [player(0, 500.0)]},
        {"players": [player(0, 545.0)]},
    ]
    for record in records:
        for tracked_player in record["players"]:
            tracked_player.pop("track_id")

    suppress_stationary_tracks(records, sample_rate=2.0)

    assert records[2]["players"][0]["track_id"] == records[1]["players"][0]["track_id"]


def test_deep_confirmation_requires_the_same_pair():
    records = [
        {"raw_center_x": 1000.0, "players": [player(1, 1400), player(2, 1500)]},
        {"raw_center_x": 1000.0, "players": [player(3, 1450), player(4, 1550)]},
        {"raw_center_x": 1000.0, "players": [player(3, 1540), player(4, 1640)]},
    ]

    identify_deep_pairs(records, sample_rate=2.0)

    assert records[-1]["deep_track_ids"] == [3, 4]
    assert all(record.get("deep_track_ids") != [1, 2] for record in records)


def test_deep_pair_survives_dropout_with_predicted_leading_edge():
    records = [
        {"raw_center_x": 1000.0, "players": [player(1, 1400), player(2, 1500)]},
        {"raw_center_x": 1000.0, "players": [player(1, 1500), player(2, 1600)]},
        {"raw_center_x": 1000.0, "players": []},
        {"raw_center_x": 1000.0, "players": []},
        {"raw_center_x": 1000.0, "players": []},
        {"raw_center_x": 1000.0, "players": []},
    ]

    identify_deep_pairs(records, sample_rate=2.0)

    assert records[2]["deep_active"]
    assert records[2]["deep_leading_x"] > records[1]["deep_leading_x"]
    assert records[0]["deep_preroll"]
    assert not records[-1]["deep_active"]


def test_deep_coverage_favors_projected_leading_edge():
    args = Namespace(no_deep_tracking=False, crop_width=2560, sample_rate=2.0, deep_lookahead=0.85)
    record = {
        "deep_active": True,
        "deep_center_x": 3250.0,
        "deep_pair_min_x": 3000.0,
        "deep_pair_max_x": 3500.0,
        "deep_leading_x": 3500.0,
        "deep_velocity_x": 400.0,
        "deep_direction": 1,
    }
    centers = np.full(8, 1920.0)

    result = apply_deep_coverage(args, [record.copy() for _ in centers], centers)

    assert result[-1] + (args.crop_width / 2 - 310) == pytest.approx(3840.0)


def test_operator_mask_allows_small_far_player_in_4k_source():
    boxes = np.array([[800.0, 550.0, 830.0, 580.0]])
    confidences = np.array([0.8])
    mask = np.full((2160, 3840), 255, dtype=np.uint8)

    selected = select_field_players(boxes, confidences, 3840, mask)

    assert len(selected) == 1


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
