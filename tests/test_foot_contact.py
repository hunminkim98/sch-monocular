import math
from pathlib import Path

import numpy as np
import pandas as pd

from utils.foot_contact.contact import (
    decode_c_temporal,
    extract_measured_force_events,
    force_time_to_video_frame,
    load_forceplate_anchored_sync,
    video_frame_to_force_time,
)


def test_walk1_force_event_maps_to_expected_video_frames():
    markerless_start_s = 3.86078711852
    marker_source_start_s = 0.02

    assert (
        force_time_to_video_frame(
            1.017,
            30.0,
            markerless_start_s,
            marker_source_start_s,
        )
        == 147
    )
    assert (
        force_time_to_video_frame(
            1.665,
            30.0,
            markerless_start_s,
            marker_source_start_s,
        )
        == 166
    )


def test_frame_force_mapping_round_trip_is_within_half_frame():
    markerless_start_s = 2.35356834083
    marker_source_start_s = 0.02
    force_time_s = 1.527
    frame = force_time_to_video_frame(
        force_time_s,
        30.0,
        markerless_start_s,
        marker_source_start_s,
    )
    recovered = float(
        video_frame_to_force_time(
            frame,
            30.0,
            markerless_start_s,
            marker_source_start_s,
        )
    )

    assert abs(recovered - force_time_s) <= 0.5 / 30.0


def test_marker_source_origin_is_not_silently_omitted():
    corrected = float(
        video_frame_to_force_time(
            147,
            30.0,
            3.86078711852,
            0.02,
        )
    )
    legacy = float(
        video_frame_to_force_time(
            147,
            30.0,
            3.86078711852,
            0.0,
        )
    )

    assert math.isclose(legacy - corrected, 0.02, abs_tol=1e-12)


def test_forceplate_anchor_is_one_cycle_after_knee_only_sync():
    sync = load_forceplate_anchored_sync(
        Path(__file__).resolve().parents[1] / "utils/foot_contact/sync.csv"
    )

    shifts = sync["forceplate_anchored_start_s"] - sync["knee_only_start_s"]
    assert shifts.between(1.0, 1.1).all()
    assert sync.loc["S18_Walk_1", "first_force_frame"] == 147
    assert sync.loc["S18_Walk_1", "last_force_frame"] == 182
    assert sync.loc["S18_Walk_1", "marker_source_start_s"] == 0.02


def test_force_events_are_extracted_directly_from_thresholded_force():
    force = pd.DataFrame(
        {
            "time": np.arange(200, dtype=float) / 1000.0,
            "ground_force1_vy": np.r_[np.zeros(20), np.full(50, 100.0), np.zeros(130)],
            "ground_force2_vy": np.r_[np.zeros(100), np.full(60, -120.0), np.zeros(40)],
        }
    )

    events = extract_measured_force_events("synthetic", force)

    assert events[["plate", "side"]].values.tolist() == [[1, "left"], [2, "right"]]
    assert events["force_start_s"].tolist() == [0.02, 0.1]
    assert events["force_end_s"].tolist() == [0.069, 0.159]
    assert events["peak_force_n"].tolist() == [100.0, 120.0]


def test_viterbi_bridges_a_short_low_probability_gap():
    probabilities = np.asarray(
        [
            [0.15, 0.90],
            [0.20, 0.85],
            [0.25, 0.45],
            [0.75, 0.35],
            [0.85, 0.20],
        ]
    )

    states = decode_c_temporal(probabilities)

    assert states.shape == (5, 2)
    assert states[0].tolist() == [False, True]
    assert states[-1].tolist() == [True, False]
    assert states.any(axis=1).all()
