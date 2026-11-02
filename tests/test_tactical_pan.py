import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import cv2

from panstudio.calibration import Calibration
from panstudio.media import analysis_decoder_command, letterbox_crop, render, sampled_analysis_frames
from panstudio.planning import (
    apply_crop_geometry, apply_deep_coverage, apply_manual_corrections, plan_crop_widths, smooth_centers,
)
from panstudio.models import ProcessingConfig
from panstudio.tracking import identify_deep_pairs, select_field_players, suppress_stationary_tracks, weighted_median

SOURCE_WIDTH = 4608


def settings(**overrides):
    """Processing settings with test overrides; the source and time range are placeholders."""
    values = {'input': Path('source.mp4'), 'start': 0.0} | overrides
    return ProcessingConfig(**({'end': values['start'] + 60.0} | values))


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


def test_analysis_decoder_requests_one_sequential_fixed_rate_pass(tmp_path):
    source = tmp_path / "source.mp4"
    args = settings(input=source, start=12.5, sample_rate=2.0)

    command = analysis_decoder_command(args, sample_count=21)

    assert command.count("-ss") == 1
    assert command[command.index("-ss") + 1] == "12.5"
    assert command[command.index("-vf") + 1] == "fps=2.0:start_time=0"
    assert command[command.index("-frames:v") + 1] == "21"


def test_sampled_analysis_frames_returns_expected_count_and_shape(tmp_path):
    source = tmp_path / "source.mp4"
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "testsrc2=size=64x48:rate=10:duration=2",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(source),
        ],
        check=True,
    )
    args = settings(input=source, start=0.25, sample_rate=2.0)
    sample_times = np.array([0.25, 0.75, 1.25])

    samples = list(sampled_analysis_frames(args, 64, 48, sample_times))

    assert [sample[0] for sample in samples] == [0, 1, 2]
    assert [sample[1] for sample in samples] == pytest.approx(sample_times)
    assert all(sample[2].shape == (48, 64, 3) for sample in samples)


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
    result = smooth_centers(raw, crop_width=2560, frame_width=3840, sample_rate=2.0, calibration=Calibration())

    assert np.all(result >= 1280.0)
    assert np.all(result <= 2560.0)
    assert np.all(np.abs(np.diff(result)) <= 260.0 + 1e-9)


def test_deep_coverage_keeps_pair_inside_safe_frame():
    args = settings(no_deep_tracking=False, crop_width=2560, sample_rate=2.0)
    records = [
        {"deep_active": True, "deep_center_x": 3500.0},
        {"deep_active": True, "deep_center_x": 3500.0},
        {"deep_active": True, "deep_center_x": 3500.0},
    ]
    centers = np.array([1920.0, 1920.0, 1920.0])

    result = apply_deep_coverage(args, records, centers, SOURCE_WIDTH)

    assert np.all(result > centers)
    assert np.all(np.abs(np.diff(result)) <= 285.0 + 1e-9)


def test_deep_coverage_can_be_disabled():
    args = settings(no_deep_tracking=True, crop_width=2560, sample_rate=2.0)
    centers = np.array([1800.0, 1900.0])

    result = apply_deep_coverage(args, [{}, {}], centers, SOURCE_WIDTH)

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

    suppress_stationary_tracks(records, sample_rate=2.0, calibration=Calibration())

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

    suppress_stationary_tracks(records, sample_rate=2.0, calibration=Calibration())

    assert records[2]["players"][0]["track_id"] == records[1]["players"][0]["track_id"]


def test_deep_confirmation_requires_the_same_pair():
    records = [
        {"raw_center_x": 1000.0, "players": [player(1, 1400), player(2, 1500)]},
        {"raw_center_x": 1000.0, "players": [player(3, 1450), player(4, 1550)]},
        {"raw_center_x": 1000.0, "players": [player(3, 1540), player(4, 1640)]},
    ]

    identify_deep_pairs(records, sample_rate=2.0, calibration=Calibration())

    assert records[-1]["deep_track_ids"] == [3, 4]
    assert all(record.get("deep_track_ids") != [1, 2] for record in records)


def test_deep_preroll_is_limited_to_half_a_second():
    records = [
        {"raw_center_x": 1000.0, "players": []},
        {"raw_center_x": 1000.0, "players": []},
        {"raw_center_x": 1000.0, "players": [player(1, 1400), player(2, 1500)]},
        {"raw_center_x": 1000.0, "players": [player(1, 1490), player(2, 1590)]},
    ]

    identify_deep_pairs(records, sample_rate=2.0, calibration=Calibration())

    assert not records[1].get("deep_active")
    assert records[2]["deep_preroll"]
    assert records[3]["deep_active"]


def test_deep_cutter_who_beats_their_defender_stays_tracked():
    # Cutter 1 and defender 2 run deep together, then the cutter pulls clear (beyond the pair gap)
    # and the defender pairs up with a trailing player 3 instead.
    records = [
        {"raw_center_x": 1000.0, "players": [player(1, 1500), player(2, 1400)]},
        {"raw_center_x": 1000.0, "players": [player(1, 1700), player(2, 1550)]},
        {"raw_center_x": 1000.0, "players": [player(1, 2200), player(2, 1650), player(3, 1600)]},
        {"raw_center_x": 1000.0, "players": [player(1, 2500), player(2, 1750), player(3, 1700)]},
        {"raw_center_x": 1000.0, "players": [player(1, 2520, velocity=0.0), player(2, 1850), player(3, 1800)]},
        {"raw_center_x": 1000.0, "players": [player(1, 2520, velocity=0.0), player(2, 1950), player(3, 1900)]},
    ]

    identify_deep_pairs(records, sample_rate=2.0, calibration=Calibration())

    assert records[1]["deep_track_ids"] == [1, 2] and not records[1]["deep_breakaway"]
    assert sorted(records[3]["deep_track_ids"]) == [1, 2, 3] and records[3]["deep_breakaway"]
    assert records[3]["deep_leading_x"] == pytest.approx(2500.0)
    # Once the cutter stops running, the pair behind is tracked on its own again.
    assert sorted(records[5]["deep_track_ids"]) == [2, 3] and not records[5]["deep_breakaway"]


def test_breakaway_is_not_inherited_by_an_unrelated_pair():
    # The cutter's pair ends (the defender stops); a separate pair 4 and 5 starts a cut elsewhere.
    records = [
        {"raw_center_x": 1000.0, "players": [player(1, 1500), player(2, 1400)]},
        {"raw_center_x": 1000.0, "players": [player(1, 1700), player(2, 1550)]},
        {"raw_center_x": 1000.0, "players": [player(1, 2200), player(2, 1600, velocity=0.0), player(4, 1350), player(5, 1300)]},
        {"raw_center_x": 1000.0, "players": [player(1, 2500), player(2, 1600, velocity=0.0), player(4, 1450), player(5, 1400)]},
    ]

    identify_deep_pairs(records, sample_rate=2.0, calibration=Calibration())

    assert sorted(records[3]["deep_track_ids"]) == [4, 5] and not records[3]["deep_breakaway"]


def test_deep_coverage_keeps_a_breakaway_cutter_and_the_pair_behind_in_frame():
    records = [
        {"raw_center_x": 1300.0, "players": [player(1, 1800), player(2, 1700), player(9, 1100, velocity=0.0)]},
        {"raw_center_x": 1300.0, "players": [player(1, 2000), player(2, 1850), player(9, 1100, velocity=0.0)]},
        {"raw_center_x": 1300.0, "players": [player(1, 2500), player(2, 1950), player(3, 1900), player(9, 1100, velocity=0.0)]},
        {"raw_center_x": 1300.0, "players": [player(1, 2800), player(2, 2050), player(3, 2000), player(9, 1100, velocity=0.0)]},
        {"raw_center_x": 1300.0, "players": [player(1, 3000), player(2, 2150), player(3, 2100), player(9, 1100, velocity=0.0)]},
    ]
    identify_deep_pairs(records, sample_rate=2.0, calibration=Calibration())
    assert records[-1]["deep_breakaway"] and len(records[-1]["deep_track_ids"]) == 3
    args = settings(no_deep_tracking=False, crop_width=2560, sample_rate=2.0, deep_lookahead=0.6)

    result = apply_deep_coverage(args, records, np.full(len(records), 1300.0), SOURCE_WIDTH)

    left, right = result[-1] - 1280, result[-1] + 1280
    assert right >= 3000                                   # the cutter who got away
    assert left <= 1100                                    # the formation player behind is not cut


def test_breakaway_widens_with_room_ahead_of_the_cutter():
    args = settings(crop_width=2560, crop_height=1440, sample_rate=5.0, no_auto_widen=False, no_deep_tracking=False)
    deep = {"time": 0.0, "deep_active": True, "deep_direction": 1, "deep_pair_min_x": 3000.0, "deep_pair_max_x": 3600.0}
    centers = np.full(30, 2300.0)
    pair = plan_crop_widths(args, [dict(deep, time=i / 5.0) for i in range(30)], centers, 4608, 1728)
    breakaway = plan_crop_widths(args, [dict(deep, time=i / 5.0, deep_breakaway=True) for i in range(30)], centers, 4608, 1728)
    assert pair[15] == pytest.approx(2 * (3600 + 40 - 2300))
    assert breakaway[15] == pytest.approx(2560 * 1.15)  # room ahead of the cutter, up to the cap
    assert breakaway[15] > pair[15]


def test_deep_pair_survives_dropout_with_predicted_leading_edge():
    records = [
        {"raw_center_x": 1000.0, "players": [player(1, 1400), player(2, 1500)]},
        {"raw_center_x": 1000.0, "players": [player(1, 1500), player(2, 1600)]},
        {"raw_center_x": 1000.0, "players": []},
        {"raw_center_x": 1000.0, "players": []},
        {"raw_center_x": 1000.0, "players": []},
        {"raw_center_x": 1000.0, "players": []},
    ]

    identify_deep_pairs(records, sample_rate=2.0, calibration=Calibration())

    assert records[2]["deep_active"]
    assert records[2]["deep_leading_x"] > records[1]["deep_leading_x"]
    assert records[0]["deep_preroll"]
    assert not records[-1]["deep_active"]


def test_deep_coverage_favors_projected_leading_edge():
    args = settings(no_deep_tracking=False, crop_width=2560, sample_rate=2.0, deep_lookahead=0.85)
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

    result = apply_deep_coverage(args, [record.copy() for _ in centers], centers, SOURCE_WIDTH)

    assert result[-1] + (args.crop_width / 2 - 310) == pytest.approx(3840.0)


def test_deep_coverage_protects_a_non_deep_player_from_being_cut():
    args = settings(
        no_deep_tracking=False, crop_width=2560, sample_rate=2.0,
        deep_lookahead=0.6,
    )
    record = {
        "deep_active": True,
        "deep_track_ids": [1, 2],
        "deep_direction": 1,
        "deep_pair_min_x": 3400.0,
        "deep_pair_max_x": 3500.0,
        "deep_center_x": 3450.0,
        "deep_leading_x": 3500.0,
        "deep_velocity_x": 0.0,
        "players": [
            {"track_id": 1, "center_x": 3400.0, "stationary_factor": 1.0},
            {"track_id": 2, "center_x": 3500.0, "stationary_factor": 1.0},
            {"track_id": 9, "center_x": 800.0, "box": [780, 1000, 820, 1100], "stationary_factor": 1.0},
        ],
    }
    result = apply_deep_coverage(args, [record.copy() for _ in range(4)], np.full(4, 1920.0), SOURCE_WIDTH)

    assert np.all(result <= 2060.0 + 1e-9)


def test_deep_coverage_keeps_a_feasible_pair_and_formation_visible():
    args = settings(
        no_deep_tracking=False, crop_width=2560, sample_rate=2.0,
        deep_lookahead=0.6,
    )
    record = {
        "deep_active": True, "deep_track_ids": [1, 2], "deep_direction": 1,
        "deep_pair_min_x": 2800.0, "deep_pair_max_x": 3200.0,
        "deep_center_x": 3000.0, "deep_leading_x": 3200.0,
        "deep_velocity_x": 0.0,
        "players": [
            {"track_id": 1, "center_x": 2800.0, "stationary_factor": 1.0},
            {"track_id": 2, "center_x": 3200.0, "stationary_factor": 1.0},
            {"track_id": 9, "center_x": 1800.0, "box": [1780, 1000, 1820, 1100], "stationary_factor": 1.0},
        ],
    }
    records = [dict(record) for _ in range(3)]
    result = apply_deep_coverage(args, records, np.full(3, 1920.0), SOURCE_WIDTH)

    assert records[0]["deep_pair_feasible"]
    assert result[-1] == pytest.approx(2230.0)


def test_deep_coverage_has_no_motion_without_a_deep_signal():
    args = settings(no_deep_tracking=False, crop_width=2560, sample_rate=2.0)
    centers = np.array([1800.0, 1900.0, 1850.0])

    result = apply_deep_coverage(args, [{}, {}, {}], centers, SOURCE_WIDTH)

    assert result is centers


@pytest.mark.parametrize("direction", [-1, 1])
def test_deep_coverage_is_symmetric(direction):
    args = settings(
        no_deep_tracking=False, crop_width=2560, sample_rate=2.0,
        deep_lookahead=0.6,
    )
    pair_min, pair_max = (1108.0, 1508.0) if direction < 0 else (3012.0, 3412.0)
    record = {
        "deep_active": True, "deep_track_ids": [1, 2], "deep_direction": direction,
        "deep_pair_min_x": pair_min, "deep_pair_max_x": pair_max,
        "deep_center_x": (pair_min + pair_max) / 2, "deep_leading_x": pair_min if direction < 0 else pair_max,
        "deep_velocity_x": 0.0,
        "players": [
            {"track_id": 1, "center_x": pair_min, "stationary_factor": 1.0},
            {"track_id": 2, "center_x": pair_max, "stationary_factor": 1.0},
            {"track_id": 9, "center_x": 2300.0, "box": [2280, 1000, 2320, 1100], "stationary_factor": 1.0},
        ],
    }
    centers = np.full(3, 2300.0)
    result = apply_deep_coverage(args, [dict(record) for _ in centers], centers, SOURCE_WIDTH)

    if direction < 0:
        assert result[0] < 2300.0
    else:
        assert result[0] > 2300.0
    assert np.all(np.abs(np.diff(result)) <= 285.0 + 1e-9)


def test_offscreen_formation_detection_does_not_disable_deep_pair():
    args = settings(
        no_deep_tracking=False, crop_width=2560, sample_rate=2.0,
        deep_lookahead=0.6,
    )
    record = {
        "deep_active": True, "deep_track_ids": [1, 2], "deep_direction": 1,
        "deep_pair_min_x": 3000.0, "deep_pair_max_x": 3400.0,
        "deep_center_x": 3200.0, "deep_leading_x": 3400.0, "deep_velocity_x": 0.0,
        "players": [{"track_id": 1, "center_x": 3000.0}, {"track_id": 2, "center_x": 3400.0},
                    {"track_id": 9, "center_x": 100.0, "box": [50, 1000, 150, 1100], "stationary_factor": 1.0}],
    }
    result = apply_deep_coverage(args, [record], np.array([1920.0]), SOURCE_WIDTH)

    assert result[0] > 1920.0


def test_formation_protection_can_be_disabled_for_ab_comparison():
    args = settings(
        no_deep_tracking=False, no_formation_protection=True,
        crop_width=2560, sample_rate=2.0, deep_lookahead=0.6,
    )
    record = {
        "deep_active": True, "deep_track_ids": [1, 2], "deep_direction": 1,
        "deep_pair_min_x": 3000.0, "deep_pair_max_x": 3400.0,
        "deep_center_x": 3200.0, "deep_leading_x": 3400.0, "deep_velocity_x": 0.0,
        "players": [{"track_id": 1, "center_x": 3000.0}, {"track_id": 2, "center_x": 3400.0},
                    {"track_id": 9, "center_x": 800.0, "box": [780, 1000, 820, 1100], "stationary_factor": 1.0}],
    }
    result = apply_deep_coverage(args, [record], np.array([1920.0]), SOURCE_WIDTH)

    assert result[0] > 1920.0


def test_protection_plans_for_a_late_opposite_edge_player():
    args = settings(
        no_deep_tracking=False, crop_width=2560, sample_rate=2.0,
        deep_lookahead=0.6,
    )
    records = []
    for index in range(6):
        players = [
            {"track_id": 1, "center_x": 3300.0, "stationary_factor": 1.0},
            {"track_id": 2, "center_x": 3500.0, "stationary_factor": 1.0},
        ]
        if index >= 3:
            players.append({"track_id": 9, "center_x": 800.0, "box": [780, 1000, 820, 1100], "stationary_factor": 1.0})
        records.append({
            "deep_active": True, "deep_track_ids": [1, 2], "deep_direction": 1,
            "deep_pair_min_x": 3300.0, "deep_pair_max_x": 3500.0,
            "deep_center_x": 3400.0, "deep_leading_x": 3500.0, "deep_velocity_x": 0.0,
            "players": players,
        })
    result = apply_deep_coverage(args, records, np.full(6, 1920.0), SOURCE_WIDTH)

    assert np.all(np.abs(np.diff(result)) <= 285.0 + 1e-9)
    assert not any(record.get("deep_protection_speed_limited") for record in records)
    assert np.all(result[3:] <= 2060.0 + 1e-9)


def test_deep_margin_is_sacrificed_when_actual_pair_and_formation_fit():
    args = settings(
        no_deep_tracking=False, crop_width=2560, sample_rate=2.0,
        deep_lookahead=0.6,
    )
    record = {
        "deep_active": True, "deep_track_ids": [1, 2], "deep_direction": 1,
        "deep_pair_min_x": 2700.0, "deep_pair_max_x": 3200.0,
        "deep_center_x": 2950.0, "deep_leading_x": 3200.0, "deep_velocity_x": 0.0,
        "players": [
            {"track_id": 1, "center_x": 2700.0, "stationary_factor": 1.0},
            {"track_id": 2, "center_x": 3200.0, "stationary_factor": 1.0},
            {"track_id": 9, "center_x": 800.0, "box": [780, 1000, 820, 1100], "stationary_factor": 1.0},
        ],
    }
    records = [dict(record) for _ in range(3)]
    result = apply_deep_coverage(args, records, np.full(3, 1920.0), SOURCE_WIDTH)

    assert all(not record["deep_pair_feasible"] for record in records)
    assert all(record["deep_actual_pair_feasible"] for record in records)
    assert result[-1] == pytest.approx(2060.0)


def test_deep_coverage_mirrors_left_and_right_outputs():
    args = settings(
        no_deep_tracking=False, crop_width=2560, sample_rate=2.0,
        deep_lookahead=0.6,
    )

    def make_record(left_side):
        pair_min, pair_max = (1500.0, 1900.0) if left_side else (2708.0, 3108.0)
        direction = -1 if left_side else 1
        return {
            "deep_active": True, "deep_track_ids": [1, 2], "deep_direction": direction,
            "deep_pair_min_x": pair_min, "deep_pair_max_x": pair_max,
            "deep_center_x": (pair_min + pair_max) / 2,
            "deep_leading_x": pair_min if left_side else pair_max, "deep_velocity_x": 0.0,
            "players": [
                {"track_id": 1, "center_x": pair_min, "stationary_factor": 1.0},
                {"track_id": 2, "center_x": pair_max, "stationary_factor": 1.0},
                {"track_id": 9, "center_x": 2304.0, "box": [2284, 1000, 2324, 1100], "stationary_factor": 1.0},
            ],
        }

    centers = np.full(4, 2304.0)
    left = apply_deep_coverage(args, [make_record(True) for _ in centers], centers, SOURCE_WIDTH)
    right = apply_deep_coverage(args, [make_record(False) for _ in centers], centers, SOURCE_WIDTH)

    assert right == pytest.approx(4608.0 - left)


def test_auto_widen_is_capped_and_rate_limited():
    args = settings(crop_width=2560, crop_height=1440, sample_rate=2.0, no_auto_widen=False, no_deep_tracking=False)
    records = [{"deep_active": True, "deep_pair_min_x": 1000.0, "deep_pair_max_x": 4500.0}] * 4
    widths = plan_crop_widths(args, records, np.full(4, 2304.0), 4608, 1728)

    assert widths[-1] <= 2944.0 + 1e-9
    assert np.all(np.diff(widths) <= 140.0 + 1e-9)


def test_no_deep_keeps_fixed_crop_geometry_and_aspect_ratio():
    args = settings(crop_width=2560, crop_height=1440, sample_rate=2.0, no_auto_widen=False, no_deep_tracking=True)
    records = [{}, {}]
    centers = np.array([1800.0, 1900.0])
    apply_crop_geometry(args, records, centers, 4608, 1728)

    assert [record["crop_width"] for record in records] == [2560.0, 2560.0]
    assert [record["crop_height"] for record in records] == [1440.0, 1440.0]


def widening_args(**overrides):
    values = dict(crop_width=2560, crop_height=1440, sample_rate=2.0,
                  no_auto_widen=False, no_deep_tracking=False)
    return settings(**(values | overrides))


def deep_width_event(right=4000.0):
    return {"deep_active": True, "deep_pair_min_x": right - 100,
            "deep_pair_max_x": right}


def test_widening_prepares_for_tight_offscreen_pair_then_holds_and_releases():
    records = [{} for _ in range(24)]
    records[6] = deep_width_event()
    widths = plan_crop_widths(widening_args(), records, np.full(24, 2304.0), 4608, 1728)

    assert widths[0] == 2560
    assert widths[5] > 2560  # Offline ramp begins before the event.
    assert widths[6] == 2944  # No late zoom when the pair arrives.
    assert np.all(widths[6:10] == 2944)  # 1.5 seconds after the event.
    assert widths[10] < widths[9]
    assert widths[-1] == 2560
    assert np.all(np.diff(widths) <= 140 + 1e-9)
    assert np.all(np.diff(widths) >= -110 - 1e-9)


def test_later_small_widening_does_not_inherit_previous_peak():
    records = [{} for _ in range(45)]
    records[4] = deep_width_event()
    records[30] = deep_width_event(3600)
    widths = plan_crop_widths(widening_args(), records, np.full(45, 2304.0), 4608, 1728)

    assert widths[20] == 2560
    assert widths[30] == 2672
    assert widths[33] == 2672
    assert widths[34] < 2672
    assert widths[-1] == 2560


@pytest.mark.parametrize("flag", ["no_auto_widen", "no_deep_tracking"])
def test_widening_opt_out_ignores_existing_deep_records(flag):
    records = [deep_width_event() for _ in range(3)]
    widths = plan_crop_widths(widening_args(**{flag: True}), records, np.full(3, 2304.0), 4608, 1728)
    assert widths.tolist() == [2560] * 3


def test_widening_is_unchanged_without_signals_and_accepts_empty_input():
    args = widening_args()
    assert plan_crop_widths(args, [], np.array([]), 4608, 1728).size == 0
    assert plan_crop_widths(args, [{}] * 3, np.full(3, 2304.0), 4608, 1728).tolist() == [2560] * 3


@pytest.mark.parametrize("left_side", [True, False])
def test_widening_at_source_edge_preserves_existing_crop_and_avoids_wasted_margin(left_side):
    center = 1280.0 if left_side else 3328.0
    pair_min, pair_max = (2800.0, 2900.0) if left_side else (1708.0, 1808.0)
    record = {"deep_active": True, "deep_pair_min_x": pair_min, "deep_pair_max_x": pair_max}
    apply_crop_geometry(widening_args(), [record], np.array([center]), 4608, 1728)

    assert record["crop_width"] == 2940  # Includes the 40px margin, after clamping the center.
    assert 0 <= record["crop_x"] <= 4608 - record["crop_width"]
    assert record["crop_x"] <= center - 1280
    assert record["crop_x"] + record["crop_width"] >= center + 1280
    assert record["crop_y"] + record["crop_height"] == 1728


@pytest.mark.parametrize("sample_rate", [1., 2., 5.])
def test_widening_rates_hold_at_different_sample_rates(sample_rate):
    records = [{} for _ in range(int(20 * sample_rate))]
    records[int(5 * sample_rate)] = deep_width_event()
    widths = plan_crop_widths(widening_args(sample_rate=sample_rate), records,
                              np.full(len(records), 2304.), 4608, 1728)
    assert widths.max() == pytest.approx(2944.)
    assert np.all(np.diff(widths) <= 280 / sample_rate + 1e-9)
    assert np.all(np.diff(widths) >= -220 / sample_rate - 1e-9)


def test_widening_with_no_spare_height_keeps_normal_width():
    records = [deep_width_event() for _ in range(8)]
    widths = plan_crop_widths(widening_args(), records, np.full(8, 2304.), 4608, 1440)
    assert widths.tolist() == [2560.] * 8


def test_widening_never_exceeds_source_dimensions():
    records = [deep_width_event(4000)]
    apply_crop_geometry(widening_args(), records, np.array([1400.0]), 3000, 1600)
    assert records[0]["crop_width"] == pytest.approx(1600 * 16 / 9, abs=.01)
    assert records[0]["crop_height"] == 1600
    assert records[0]["crop_x"] == records[0]["crop_y"] == 0


def test_letterbox_preserves_square_and_adds_black_bars():
    frame = np.full((160, 320, 3), 100, dtype=np.uint8)
    frame[60:100, 140:180] = 255
    output = letterbox_crop(frame, 0, 0, 320, 160, 160, 120)
    ys, xs = np.where(output[:, :, 0] > 240)
    assert output.shape == (120, 160, 3)
    assert xs.max() - xs.min() + 1 == ys.max() - ys.min() + 1 == 20
    assert np.all(output[:20] == 0)
    assert np.all(output[-20:] == 0)
    assert np.all(output[20, :, :] == 100)


def test_variable_crop_renders_decodable_video_with_fixed_output_dimensions(tmp_path):
    source = tmp_path / "gray.mp4"
    output = tmp_path / "rendered.mp4"
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "color=gray:size=192x108:rate=10:duration=1",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(source),
    ], check=True)
    args = widening_args(input=source, start=0., end=1., crop_width=128,
                         crop_height=72, output_width=128, output_height=72)
    records = [{"time": 0.}, {"time": .5, **deep_width_event(192)}]
    apply_crop_geometry(args, records, np.full(2, 96.), 192, 108)
    render(args, records, 192, 108, 10., output, debug=False)
    cap = cv2.VideoCapture(str(output))
    frames = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()
    assert len(frames) == 10
    assert all(frame.shape == (72, 128, 3) for frame in frames)
    assert frames[0][0].mean() > 80
    assert records[-1]["crop_width"] > args.crop_width
    assert all(frame[0].mean() > 80 and frame[-1].mean() > 80 for frame in frames)
    assert frames[-1][36].mean() > 80


def test_deep_diagnostics_are_json_serializable_and_actual_feasibility_ignores_projection():
    args = widening_args(deep_lookahead=.6)
    records = [{
        "deep_active": True, "deep_track_ids": [1, 2], "deep_direction": 1,
        "deep_pair_min_x": 2900., "deep_pair_max_x": 3200.,
        "deep_center_x": 3050., "deep_leading_x": 3200., "deep_velocity_x": 600.,
        "players": [{"track_id": 9, "center_x": 800., "box": [780, 1000, 820, 1100]}],
    }]
    centers = apply_deep_coverage(args, records, np.array([1920.]), SOURCE_WIDTH)
    apply_crop_geometry(args, records, centers, 4608, 1728)
    assert records[0]["deep_actual_pair_feasible"] is True
    assert records[0]["deep_pair_feasible"] is False
    json.dumps(records)


def test_deep_processing_leaves_distant_formation_only_motion_unchanged():
    args = widening_args()
    centers = np.array([1800., 1850., 1800., 1850.] * 5)
    records = [{} for _ in centers]
    records[-1] = {"deep_active": True, "deep_center_x": 3500.}
    result = apply_deep_coverage(args, records, centers, SOURCE_WIDTH)
    assert result[:10] == pytest.approx(centers[:10])


def test_operator_mask_allows_small_far_player_in_4k_source():
    boxes = np.array([[800.0, 550.0, 830.0, 580.0]])
    confidences = np.array([0.8])
    mask = np.full((2160, 3840), 255, dtype=np.uint8)

    selected = select_field_players(boxes, confidences, 3840, mask, Calibration())

    assert len(selected) == 1


def test_manual_correction_reaches_operator_target(tmp_path):
    correction_file = tmp_path / "corrections.csv"
    correction_file.write_text("source_time_seconds,center_x\n12,3000\n")
    args = settings(corrections=correction_file, start=10.0, correction_radius=2.0)
    times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    centers = np.full(5, 1900.0)

    result = apply_manual_corrections(args, times, centers)

    assert result[2] == 3000.0
    assert result[0] == 1900.0
    assert result[4] == 1900.0


def test_time_based_smoothing_matches_original_kernels_at_two_hertz():
    from panstudio.planning import triangular_kernel
    assert np.allclose(triangular_kernel(1.5, 2.0) * 16, [1, 2, 3, 4, 3, 2, 1])
    assert np.allclose(triangular_kernel(1.0, 2.0) * 9, [1, 2, 3, 2, 1])
    assert len(triangular_kernel(1.5, 5.0)) == 17


def test_smoothing_response_is_independent_of_sample_rate():
    """A 2 Hz and a 5 Hz analysis of the same step should settle on the same timeline."""
    results = {}
    for rate in (2.0, 5.0):
        raw = np.array([1500.0] * int(10 * rate) + [2500.0] * int(10 * rate))
        results[rate] = smooth_centers(raw, crop_width=2560, frame_width=4608, sample_rate=rate, calibration=Calibration())
    times = {rate: np.arange(len(r)) / rate for rate, r in results.items()}
    resampled = np.interp(times[2.0], times[5.0], results[5.0])
    assert np.max(np.abs(resampled - results[2.0])) < 100.0


def test_frame_rate_path_is_smooth_and_never_overshoots():
    from panstudio.media import frame_rate_path
    records = [{"time": t / 5.0, "crop_x": 1000.0 if t < 10 else 1600.0, "crop_width": 2560.0, "crop_height": 1440.0} for t in range(20)]
    sample_times = np.array([r["time"] for r in records])
    frame_times = np.arange(80) / 20.0
    xs, widths, heights = frame_rate_path(records, sample_times, frame_times, 2560, 1440)
    assert xs.min() >= 1000.0 - 1e-9 and xs.max() <= 1600.0 + 1e-9
    assert np.allclose(widths, 2560.0) and np.allclose(heights, 1440.0)
    linear = np.interp(frame_times, sample_times, [r["crop_x"] for r in records])
    # Acceleration (second difference) should be gentler than plain interpolation.
    assert np.max(np.abs(np.diff(xs, 2))) < np.max(np.abs(np.diff(linear, 2)))


def test_masked_weights_do_not_favor_near_sideline_players():
    boxes = np.array([[100, 1500, 150, 1728], [2000, 950, 2015, 995]], dtype=float)  # 228 px and 45 px tall
    confidences = np.array([0.8, 0.8])
    mask = np.full((1728, 4608), 255, dtype=np.uint8)
    near, far = select_field_players(boxes, confidences, 4608, mask, Calibration())
    assert near["weight"] / far["weight"] < 2.0
    unmasked = select_field_players(np.array([[1000, 1300, 1050, 1600], [2000, 950, 2015, 1000]], dtype=float), confidences, 4608, None, Calibration())
    assert unmasked[0]["weight"] / unmasked[1]["weight"] > 5.0


def test_track_gate_keeps_a_near_sideline_sprinter_at_five_hertz():
    """A 250 px tall player starting a sprint moves ~230 px in 0.2 s with no velocity history."""
    def tall(x):
        p = player(0, x, velocity=0.0, center_y=1500.0)
        p["height"] = 250.0
        p.pop("track_id")
        return p
    records = [{"players": [tall(x)]} for x in (1000.0, 1230.0, 1460.0)]
    suppress_stationary_tracks(records, sample_rate=5.0, calibration=Calibration())
    assert len({r["players"][0]["track_id"] for r in records}) == 1
    # At 2 Hz the same tolerance is not extended to a 500 px jump of a small far player.
    far = [{"players": [dict(player(0, x), height=40.0)]} for x in (1000.0, 1900.0)]
    for r in far:
        r["players"][0].pop("track_id")
    suppress_stationary_tracks(far, sample_rate=2.0, calibration=Calibration())
    assert far[0]["players"][0]["track_id"] != far[1]["players"][0]["track_id"]


def test_bystander_suppression_keeps_runners_and_demotes_loiterers_and_benches():
    from panstudio.tracking import suppress_bystanders
    rate = 5.0
    records = []
    for i in range(int(400 * rate)):  # 400 s
        t = i / rate
        runner = player(1, 1500 + 300 * np.sin(t / 3.0), center_y=1200.0)     # sprints back and forth, 100 px tall
        runner["box"] = [runner["center_x"] - 20, 1100.0, runner["center_x"] + 20, 1200.0]
        loiterer = player(2, 2500 + 8 * np.sin(t), center_y=1000.0)           # shuffles within a few px
        loiterer["box"] = [loiterer["center_x"] - 20, 900.0, loiterer["center_x"] + 20, 1000.0]
        seated = player(3, 3500 + 300 * np.sin(t / 3.0), center_y=1000.0)      # mobile but squat box
        seated["box"] = [seated["center_x"] - 60, 940.0, seated["center_x"] + 60, 1000.0]
        records.append({"time": t, "players": [runner, loiterer, seated]})
    suppress_bystanders(records, rate, 4608, 1728)
    mid = records[len(records) // 2]["players"]
    assert mid[0]["weight"] == 1.0 and mid[0]["stationary_factor"] == 1.0
    assert mid[0]["mobility"] > 2.0
    assert mid[1]["weight"] <= 0.2 and mid[1]["occupancy"] > 0.5
    assert mid[2]["weight"] == 0.5
    # Early samples lack a full occupancy window but mobility still applies.
    assert records[2]["players"][1]["weight"] <= 0.2


def test_bystander_suppression_tolerates_broken_tracks_in_a_crowd():
    """A crowd where the tracker swaps identities every second is still suppressed by occupancy."""
    from panstudio.tracking import suppress_bystanders
    rate = 5.0
    records = []
    for i in range(int(300 * rate)):
        t = i / rate
        crowd = []
        for k in range(5):
            p = player(100 * int(t) + k, 2400 + 60 * k + 15 * np.sin(t + k), center_y=1000.0)
            p["box"] = [p["center_x"] - 15, 940.0, p["center_x"] + 15, 1000.0]
            crowd.append(p)
        records.append({"time": t, "players": crowd})
    suppress_bystanders(records, rate, 4608, 1728)
    assert all(p["weight"] <= 0.2 for p in records[len(records) // 2]["players"])


def test_flight_target_pans_to_landing_from_the_release_and_returns():
    from panstudio.planning import annotate_flights, apply_flight_targets
    rate = 5.0
    records = [{"time": i / rate, "players": []} for i in range(int(30 * rate))]
    centers = np.full(len(records), 2000.0)
    flights = [{"start": 10.0, "end": 12.0, "x0": 2100.0, "y0": 1000.0, "x1": 3600.0, "y1": 1000.0, "direction": 1, "chord": 1500.0}]
    annotate_flights(records, flights, 2560, 4608, 0.15)
    assert records[int(10.5 * rate)]["flight_active"] and not records[int(5 * rate)]["flight_active"]
    assert records[int(10.5 * rate)]["flight_target_x"] == pytest.approx(min(3600 + 0.15 * 2560, 4608 - 1280))  # look room, clipped to the frame
    assert records[int(13.5 * rate)]["flight_active"] and not records[int(15 * rate)]["flight_active"]
    args = settings(crop_width=2560, sample_rate=rate)
    result = apply_flight_targets(args, records, centers, SOURCE_WIDTH)
    assert result[int(13.0 * rate)] == pytest.approx(3328.0)          # clipped to the right edge, on the landing
    assert np.all(result[:int(10.0 * rate)] == pytest.approx(2000.0))  # nothing moves before the release
    assert result[int(10.0 * rate)] > 2000.0                                   # and the pan starts right after it
    assert np.all(np.abs(np.diff(result)) <= 570 / rate + 1e-6)        # within the deep pan speed
    assert result[-1] == pytest.approx(2000.0)                         # back on the formation afterwards
    # A flight first found mid-air (a high pull) had its release out of view: the camera may start early.
    pull = [{"time": i / rate, "players": []} for i in range(int(30 * rate))]
    annotate_flights(pull, [{**flights[0], "launch_seen": False}], 2560, 4608, 0.15)
    assert apply_flight_targets(args, pull, centers, SOURCE_WIDTH)[int(9.0 * rate)] > 2000.0
    # Receivers still running the throw's way after the hold keep the camera from falling back behind them.
    for record in records[int(14.2 * rate):int(17 * rate)]:
        record.update({"deep_active": True, "deep_direction": 1, "deep_leading_x": 4000.0})
    held = apply_flight_targets(args, records, centers, SOURCE_WIDTH)
    assert np.all(held[int(14.2 * rate):int(17 * rate)] >= 4000 + 0.1 * 2560 - 1280 - 1e-6)
    assert held[-1] == pytest.approx(2000.0)
    # A short throw never moves the camera, whether measured in pixels or in body heights.
    short = [{"time": i / rate, "players": []} for i in range(int(30 * rate))]
    annotate_flights(short, [{**flights[0], "chord": 300.0}], 2560, 4608, 0.15)
    assert not any(r["flight_active"] for r in short)
    annotate_flights(short, [{**flights[0], "chord_bodies": 5.0}], 2560, 4608, 0.15)
    assert not any(r["flight_active"] for r in short)
    # A far-sideline huck covers few pixels but many body heights.
    annotate_flights(short, [{**flights[0], "chord": 700.0, "chord_bodies": 16.0}], 2560, 4608, 0.15)
    assert any(r["flight_active"] for r in short)


def test_flight_with_a_runner_alongside_is_ignored():
    """A sprinter's bobbing head can look like a long flight; a real disc outruns everyone."""
    from panstudio.planning import annotate_flights
    rate = 5.0
    flight = {"start": 10.0, "end": 12.0, "x0": 800.0, "y0": 1080.0, "x1": 1500.0, "y1": 1180.0, "direction": 1,
              "chord": 700.0, "chord_bodies": 20.0,
              "points": [[10.0 + i / 20, 800.0 + 17.5 * i, 1080.0 + 2.5 * i] for i in range(41)]}
    def records(runner_offset):
        return [{"time": i / rate, "players": [{"track_id": 7, "center_x": 800.0 + 350.0 * (i / rate - 10.0) + runner_offset,
                                               "center_y": 1180.0, "height": 200.0, "velocity_x": 350.0}]}
                for i in range(int(20 * rate))]
    escorted = records(80.0)        # the runner a step ahead of the "disc", at its speed
    assert annotate_flights(escorted, [flight], 2560, 4608, 0.15) == [flight]
    assert not any(r["flight_active"] for r in escorted)
    alone = records(-900.0)         # the nearest runner far behind: a real throw
    assert annotate_flights(alone, [flight], 2560, 4608, 0.15) == []
    assert any(r["flight_active"] for r in alone)
    # A throw moving mostly toward the camera, past a standing stack: barely any sideways speed,
    # so standing players "match" it, but that says nothing and the throw is kept.
    toward = {**flight, "x1": 900.0, "points": [[10.0 + i / 20, 800.0 + 2.5 * i, 1000.0 + 8.0 * i] for i in range(41)]}
    stack = [{"time": i / rate, "players": [{"track_id": k, "center_x": 790.0 + 40 * k, "center_y": 1000.0 + 80 * k,
                                             "height": 200.0, "velocity_x": 0.0} for k in range(5)]}
             for i in range(int(20 * rate))]
    assert annotate_flights(stack, [toward], 2560, 4608, 0.15) == []
    assert any(r["flight_active"] for r in stack)


def test_flight_widening_holds_both_ends_when_possible():
    records = [{"time": i / 5.0, "flight_active": i == 25, "flight_min_x": 800.0, "flight_max_x": 3400.0} for i in range(50)]
    widths = plan_crop_widths(settings(crop_width=2560, crop_height=1440, sample_rate=5.0, no_auto_widen=False, no_deep_tracking=False),
                              records, np.full(50, 2100.0), 4608, 1728)
    assert widths[25] > 2560 and widths[25] <= 2560 * 1.15 + 1e-6


def test_flight_qualification_accepts_a_smooth_arc_and_rejects_chained_clutter():
    from panstudio.disc import _qualify
    from panstudio.models import Detection, DetectionSample, DetectionSet, VideoInfo
    samples = tuple(DetectionSample(t / 5, t / 5, (Detection((2000.0, 1100.0, 2060.0, 1250.0), .9), Detection((3500.0, 1050.0, 3550.0, 1180.0), .9))) for t in range(60))
    detections = DetectionSet(VideoInfo(4608, 1728, 20.0, 12.0), {'path': 'x', 'size': 1, 'mtime_ns': 1}, 0.0, 12.0, 5.0,
                              {'model': 'm', 'inference_size': 1920, 'confidence': .2, 'weights': None}, samples)
    # A smooth parabola from the first player to the second, 40 frames, blobs of ~20 px.
    arc = [(f, 2030 + (3525 - 2030) * f / 40, 1100 - 300 * np.sin(np.pi * f / 40), 300) for f in range(41)]
    # Chained clutter: fast, straight, but tiny blobs jumping 150 px a frame along the far edge.
    chain = [(f, 500 + 150 * f, 950 + 2 * f, 8) for f in range(20)]
    mask = np.zeros((1728, 4608), np.uint8); mask[1000:1700, 200:4400] = 255
    tops = np.full(4608, 1000, dtype=np.int32)
    flights = _qualify([{'points': arc}, {'points': chain}], detections, 20.0, mask, tops)
    assert len(flights) == 1 and flights[0].direction == 1 and flights[0].x1 == pytest.approx(3525.0)


def test_flight_qualification_accepts_a_low_far_sideline_arc_but_not_traffic():
    from panstudio.disc import _qualify
    from panstudio.models import Detection, DetectionSample, DetectionSet, VideoInfo
    samples = tuple(DetectionSample(t / 5, t / 5, (Detection((2620.0, 900.0, 2650.0, 970.0), .9), Detection((3830.0, 905.0, 3860.0, 975.0), .9))) for t in range(60))
    detections = DetectionSet(VideoInfo(4608, 1728, 20.0, 12.0), {'path': 'x', 'size': 1, 'mtime_ns': 1}, 0.0, 12.0, 5.0,
                              {'model': 'm', 'inference_size': 1920, 'confidence': .2, 'weights': None}, samples)
    mask = np.zeros((1728, 4608), np.uint8); mask[970:1700, 200:4400] = 255
    tops = np.full(4608, 970, dtype=np.int32)
    # A huck seen along the far sideline: 1200 px long but arching only ~90 px, as at 5:29 in the field footage.
    arc = [(f, 2650 + 1200 * f / 48, 968 - 90 * np.sin(np.pi * f / 48), 80) for f in range(49)]
    # A car on the road behind the far sideline: the same speed and length, but level.
    car = [(f, 2650 + 1200 * f / 48, 960 - 0.2 * f, 80) for f in range(49)]
    assert len(_qualify([{'points': arc}], detections, 20.0, mask, tops)) == 1
    assert _qualify([{'points': car}], detections, 20.0, mask, tops) == []


def test_disc_candidates_keep_small_unblocked_movers_in_source_units():
    from panstudio.disc import _candidates
    gray = np.zeros((400, 600), np.uint8)
    median = np.zeros((400, 600), np.int16)
    gray[50:54, 100:104] = 200       # a disc-sized blob
    gray[200:300, 300:400] = 200     # a person-sized blob
    gray[20:24, 500:504] = 200       # a disc-sized blob under a person box
    gray[350:352, 10:30] = 200       # a streak too elongated to be a disc
    blocked = np.full((400, 600), 255, np.uint8)
    blocked[10:40, 490:520] = 0
    assert _candidates(gray, median, blocked, scale=2) == [(203.0, 103.0, 64, 200)]


def test_disc_linker_follows_a_steady_mover_and_ignores_clutter_seeds():
    from panstudio.disc import ClutterMap, Linker
    linker, clutter = Linker(), ClutterMap(1728, 4608)
    for frame in range(12):
        disc = (500.0 + 40 * frame, 800.0 - 10 * frame, 60, 180)
        noise = (3000.0 + 30 * frame, 1500.0, 60, 180)  # moves like a disc but only ever offered as clutter
        clutter.update([disc])
        linker.step(frame, [disc], [noise])
    tracks = linker.finish()
    assert len(tracks) == 1
    assert [p[1] for p in tracks[0]['points']] == [500.0 + 40 * f for f in range(12)]


def test_single_sample_protection_flicker_does_not_yank_the_camera():
    """A deep pair runs right. One far-left detection appears for a single sample; the camera must not dip."""
    rate = 5.0
    records = []
    for i in range(int(8 * rate)):
        players = [player(1, 3300 + 40 * i), player(2, 3350 + 40 * i)]
        for k in range(3, 9):
            players.append(player(k, 2600 + 60 * k, velocity=0.0))
        if i == 20:
            players.append(player(99, 1700.0, velocity=0.0))   # the flicker
        records.append({"players": players, "deep_active": True, "deep_track_ids": [1, 2], "deep_direction": 1,
                        "deep_pair_min_x": 3300 + 40 * i, "deep_pair_max_x": 3350 + 40 * i,
                        "deep_center_x": 3325 + 40 * i, "deep_leading_x": 3350 + 40 * i, "deep_velocity_x": 200.0})
    centers = np.full(len(records), 2900.0)
    args = settings(no_deep_tracking=False, crop_width=2560, sample_rate=rate, deep_lookahead=0.6)
    result = apply_deep_coverage(args, records, centers, SOURCE_WIDTH)
    steps = np.diff(result)
    # No reversal against the pan direction around the flicker.
    assert np.all(steps[15:25] >= -1e-6)


def test_ease_path_rounds_corners_without_delay_or_overshoot():
    from panstudio.planning import ease_path
    path = np.array([1000.0] * 20 + [1000 + 100 * i for i in range(1, 11)] + [2000.0] * 20)
    eased = ease_path(path, 5.0)
    assert eased.min() >= 1000 - 1e-9 and eased.max() <= 2000 + 1e-9
    assert np.max(np.abs(np.diff(eased, 2))) < np.max(np.abs(np.diff(path, 2)))
    assert abs(eased[25] - path[25]) < 1e-6   # mid-pan is untouched: zero phase


def test_prefetched_preserves_order_and_propagates_errors():
    from panstudio.media import prefetched

    assert list(prefetched(iter(range(20)), depth=2)) == list(range(20))

    def failing():
        yield 1
        raise RuntimeError("decoder failed")

    results = []
    with pytest.raises(RuntimeError, match="decoder failed"):
        for item in prefetched(failing()):
            results.append(item)
    assert results == [1]

    closed = []

    def endless():
        try:
            while True:
                yield 0
        finally:
            closed.append(True)

    for _ in prefetched(endless(), depth=1):
        break
    assert closed == [True]
