#!/usr/bin/env python3
"""CLI entry point and compatibility exports for existing algorithm callers."""
from panstudio.cli import main, parse_args
from panstudio.masking import load_play_area_mask, save_mask_preview
from panstudio.media import analysis_decoder_command, encoder_command, letterbox_crop, render, sampled_analysis_frames
from panstudio.planning import (
    apply_crop_geometry, apply_deep_coverage, apply_disc_anchors, apply_manual_corrections,
    plan_crop_widths, read_anchor_csv, smooth_centers,
)
from panstudio.tracking import (
    calculate_raw_centers, field_bounds, identify_deep_pairs, select_field_players,
    suppress_bystanders, suppress_stationary_tracks, weighted_median,
)


if __name__ == '__main__':
    main()
