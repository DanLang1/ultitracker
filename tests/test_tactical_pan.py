import json
import subprocess
from argparse import Namespace

import numpy as np
import pytest
import cv2

from tactical_pan import (
    apply_deep_coverage,
    apply_crop_geometry,
    apply_manual_corrections,
    analysis_decoder_command,
    identify_deep_pairs,
    letterbox_crop,
    plan_crop_widths,
    render,
    sampled_analysis_frames,
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


def test_analysis_decoder_requests_one_sequential_fixed_rate_pass(tmp_path):
    source = tmp_path / "source.mp4"
    args = Namespace(input=source, start=12.5, sample_rate=2.0)

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
    args = Namespace(input=source, start=0.25, sample_rate=2.0)
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
    assert np.all(np.abs(np.diff(result)) <= 285.0 + 1e-9)


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


def test_deep_preroll_is_limited_to_half_a_second():
    records = [
        {"raw_center_x": 1000.0, "players": []},
        {"raw_center_x": 1000.0, "players": []},
        {"raw_center_x": 1000.0, "players": [player(1, 1400), player(2, 1500)]},
        {"raw_center_x": 1000.0, "players": [player(1, 1490), player(2, 1590)]},
    ]

    identify_deep_pairs(records, sample_rate=2.0)

    assert not records[1].get("deep_active")
    assert records[2]["deep_preroll"]
    assert records[3]["deep_active"]


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


def test_deep_coverage_protects_a_non_deep_player_from_being_cut():
    args = Namespace(
        no_deep_tracking=False, crop_width=2560, sample_rate=2.0,
        deep_lookahead=0.6, frame_width=4608,
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
    result = apply_deep_coverage(args, [record.copy() for _ in range(4)], np.full(4, 1920.0))

    assert np.all(result <= 2060.0 + 1e-9)


def test_deep_coverage_keeps_a_feasible_pair_and_formation_visible():
    args = Namespace(
        no_deep_tracking=False, crop_width=2560, sample_rate=2.0,
        deep_lookahead=0.6, frame_width=4608,
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
    result = apply_deep_coverage(args, records, np.full(3, 1920.0))

    assert records[0]["deep_pair_feasible"]
    assert result[-1] == pytest.approx(2230.0)


def test_deep_coverage_has_no_motion_without_a_deep_signal():
    args = Namespace(no_deep_tracking=False, crop_width=2560, sample_rate=2.0)
    centers = np.array([1800.0, 1900.0, 1850.0])

    result = apply_deep_coverage(args, [{}, {}, {}], centers)

    assert result is centers


@pytest.mark.parametrize("direction", [-1, 1])
def test_deep_coverage_is_symmetric(direction):
    args = Namespace(
        no_deep_tracking=False, crop_width=2560, sample_rate=2.0,
        deep_lookahead=0.6, frame_width=4608,
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
    result = apply_deep_coverage(args, [dict(record) for _ in centers], centers)

    if direction < 0:
        assert result[0] < 2300.0
    else:
        assert result[0] > 2300.0
    assert np.all(np.abs(np.diff(result)) <= 285.0 + 1e-9)


def test_offscreen_formation_detection_does_not_disable_deep_pair():
    args = Namespace(
        no_deep_tracking=False, crop_width=2560, sample_rate=2.0,
        deep_lookahead=0.6, frame_width=4608,
    )
    record = {
        "deep_active": True, "deep_track_ids": [1, 2], "deep_direction": 1,
        "deep_pair_min_x": 3000.0, "deep_pair_max_x": 3400.0,
        "deep_center_x": 3200.0, "deep_leading_x": 3400.0, "deep_velocity_x": 0.0,
        "players": [{"track_id": 1, "center_x": 3000.0}, {"track_id": 2, "center_x": 3400.0},
                    {"track_id": 9, "center_x": 100.0, "box": [50, 1000, 150, 1100], "stationary_factor": 1.0}],
    }
    result = apply_deep_coverage(args, [record], np.array([1920.0]))

    assert result[0] > 1920.0


def test_formation_protection_can_be_disabled_for_ab_comparison():
    args = Namespace(
        no_deep_tracking=False, no_formation_protection=True,
        crop_width=2560, sample_rate=2.0, deep_lookahead=0.6, frame_width=4608,
    )
    record = {
        "deep_active": True, "deep_track_ids": [1, 2], "deep_direction": 1,
        "deep_pair_min_x": 3000.0, "deep_pair_max_x": 3400.0,
        "deep_center_x": 3200.0, "deep_leading_x": 3400.0, "deep_velocity_x": 0.0,
        "players": [{"track_id": 1, "center_x": 3000.0}, {"track_id": 2, "center_x": 3400.0},
                    {"track_id": 9, "center_x": 800.0, "box": [780, 1000, 820, 1100], "stationary_factor": 1.0}],
    }
    result = apply_deep_coverage(args, [record], np.array([1920.0]))

    assert result[0] > 1920.0


def test_protection_plans_for_a_late_opposite_edge_player():
    args = Namespace(
        no_deep_tracking=False, crop_width=2560, sample_rate=2.0,
        deep_lookahead=0.6, frame_width=4608,
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
    result = apply_deep_coverage(args, records, np.full(6, 1920.0))

    assert np.all(np.abs(np.diff(result)) <= 285.0 + 1e-9)
    assert not any(record.get("deep_protection_speed_limited") for record in records)
    assert np.all(result[3:] <= 2060.0 + 1e-9)


def test_deep_margin_is_sacrificed_when_actual_pair_and_formation_fit():
    args = Namespace(
        no_deep_tracking=False, crop_width=2560, sample_rate=2.0,
        deep_lookahead=0.6, frame_width=4608,
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
    result = apply_deep_coverage(args, records, np.full(3, 1920.0))

    assert all(not record["deep_pair_feasible"] for record in records)
    assert all(record["deep_actual_pair_feasible"] for record in records)
    assert result[-1] == pytest.approx(2060.0)


def test_deep_coverage_mirrors_left_and_right_outputs():
    args = Namespace(
        no_deep_tracking=False, crop_width=2560, sample_rate=2.0,
        deep_lookahead=0.6, frame_width=4608,
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
    left = apply_deep_coverage(args, [make_record(True) for _ in centers], centers)
    right = apply_deep_coverage(args, [make_record(False) for _ in centers], centers)

    assert right == pytest.approx(4608.0 - left)


def test_auto_widen_is_capped_and_rate_limited():
    args = Namespace(crop_width=2560, crop_height=1440, sample_rate=2.0, no_auto_widen=False, no_deep_tracking=False)
    records = [{"deep_active": True, "deep_pair_min_x": 1000.0, "deep_pair_max_x": 4500.0}] * 4
    widths = plan_crop_widths(args, records, np.full(4, 2304.0), 4608, 1728)

    assert widths[-1] <= 3456.0 + 1e-9
    assert np.all(np.diff(widths) <= 350.0 + 1e-9)


def test_no_deep_keeps_fixed_crop_geometry_and_aspect_ratio():
    args = Namespace(crop_width=2560, crop_height=1440, sample_rate=2.0, no_auto_widen=False, no_deep_tracking=True)
    records = [{}, {}]
    centers = np.array([1800.0, 1900.0])
    apply_crop_geometry(args, records, centers, 4608, 1728)

    assert [record["crop_width"] for record in records] == [2560.0, 2560.0]
    assert [record["crop_height"] for record in records] == [1440.0, 1440.0]


def widening_args(**overrides):
    values = dict(crop_width=2560, crop_height=1440, sample_rate=2.0,
                  no_auto_widen=False, no_deep_tracking=False)
    return Namespace(**(values | overrides))


def deep_width_event(right=4000.0):
    return {"deep_active": True, "deep_pair_min_x": right - 100,
            "deep_pair_max_x": right}


def test_widening_prepares_for_tight_offscreen_pair_then_holds_and_releases():
    records = [{} for _ in range(24)]
    records[6] = deep_width_event()
    widths = plan_crop_widths(widening_args(), records, np.full(24, 2304.0), 4608, 1728)

    assert widths[0] == 2560
    assert widths[5] > 2560  # Offline ramp begins before the event.
    assert widths[6] == 3456  # No late zoom when the pair arrives.
    assert np.all(widths[6:10] == 3456)  # 1.5 seconds after the event.
    assert widths[10] < widths[9]
    assert widths[-1] == 2560
    assert np.all(np.diff(widths) <= 350 + 1e-9)
    assert np.all(np.diff(widths) >= -110 - 1e-9)


def test_later_small_widening_does_not_inherit_previous_peak():
    records = [{} for _ in range(45)]
    records[4] = deep_width_event()
    records[30] = deep_width_event(3600)
    widths = plan_crop_widths(widening_args(), records, np.full(45, 2304.0), 4608, 1728)

    assert widths[20] == 2560
    assert widths[30] == 2752
    assert widths[33] == 2752
    assert widths[34] < 2752
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

    assert record["crop_width"] == 2980  # Includes the 80px margin, after clamping the center.
    assert 0 <= record["crop_x"] <= 4608 - record["crop_width"]
    assert record["crop_x"] <= center - 1280
    assert record["crop_x"] + record["crop_width"] >= center + 1280
    assert record["crop_y"] + record["crop_height"] == 1728


def test_widening_never_exceeds_source_dimensions():
    records = [deep_width_event(4000)]
    apply_crop_geometry(widening_args(), records, np.array([1400.0]), 3000, 1600)
    assert records[0]["crop_width"] == 3000
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
        "-f", "lavfi", "-i", "color=gray:size=192x72:rate=10:duration=1",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(source),
    ], check=True)
    args = widening_args(input=source, start=0., end=1., crop_width=128,
                         crop_height=72, output_width=128, output_height=72)
    records = [{"time": 0.}, {"time": .5, **deep_width_event(192)}]
    apply_crop_geometry(args, records, np.full(2, 96.), 192, 72)
    render(args, records, 192, 72, 10., output, debug=False)
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
    assert frames[-1][0].mean() < 8
    assert frames[-1][36].mean() > 80


def test_deep_diagnostics_are_json_serializable_and_actual_feasibility_ignores_projection():
    args = widening_args(frame_width=4608, deep_lookahead=.6)
    records = [{
        "deep_active": True, "deep_track_ids": [1, 2], "deep_direction": 1,
        "deep_pair_min_x": 2900., "deep_pair_max_x": 3200.,
        "deep_center_x": 3050., "deep_leading_x": 3200., "deep_velocity_x": 600.,
        "players": [{"track_id": 9, "center_x": 800., "box": [780, 1000, 820, 1100]}],
    }]
    centers = apply_deep_coverage(args, records, np.array([1920.]))
    apply_crop_geometry(args, records, centers, 4608, 1728)
    assert records[0]["deep_actual_pair_feasible"] is True
    assert records[0]["deep_pair_feasible"] is False
    json.dumps(records)


def test_deep_processing_leaves_distant_formation_only_motion_unchanged():
    args = widening_args(frame_width=4608)
    centers = np.array([1800., 1850., 1800., 1850.] * 5)
    records = [{} for _ in centers]
    records[-1] = {"deep_active": True, "deep_center_x": 3500.}
    result = apply_deep_coverage(args, records, centers)
    assert result[:10] == pytest.approx(centers[:10])


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
