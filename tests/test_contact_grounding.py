"""평지 보행용 contact-linear grounding의 영구 회귀 테스트입니다."""

from __future__ import annotations

import numpy as np
import torch

from hmr4d.model.footmr.utils.contact_grounding import (
    CTemporalResult,
    ContactGroundingConfig,
    apply_ground_line,
    estimate_contact_ground_line,
)

FPS = 30.0
FRAME_COUNT = 300


def make_contact(runs: list[tuple[int, int, int]]) -> CTemporalResult:
    states = np.zeros((FRAME_COUNT, 2), dtype=bool)
    probabilities = np.full((FRAME_COUNT, 2), 0.1, dtype=float)
    for side, start, end in runs:
        states[start : end + 1, side] = True
        probabilities[start : end + 1, side] = 0.95
    zeros = np.zeros((FRAME_COUNT, 2), dtype=float)
    return CTemporalResult(
        states=states,
        probabilities=probabilities,
        speed_2d=zeros.copy(),
        speed_3d=zeros.copy(),
        bottom_gap=zeros.copy(),
    )


def make_keypoints() -> np.ndarray:
    keypoints = np.zeros((FRAME_COUNT, 23, 3), dtype=float)
    keypoints[..., 2] = 0.95
    return keypoints


def make_foot_heights(slope_mps: float = -0.03) -> np.ndarray:
    times = np.arange(FRAME_COUNT, dtype=float) / FPS
    return np.column_stack(
        [
            0.12 + slope_mps * times,
            0.14 + slope_mps * times,
        ]
    )


def test_four_stances_in_short_time_range_are_enough() -> None:
    """영상 40% 미만 구간이어도 4 stance와 양발 조건만 만족하면 적용합니다."""
    contact = make_contact(
        [
            (0, 10, 21),
            (1, 25, 36),
            (0, 40, 51),
            (1, 55, 66),
        ]
    )

    line, report = estimate_contact_ground_line(
        contact,
        make_keypoints(),
        make_foot_heights(),
        FPS,
    )

    assert report["applied"] is True
    assert report["contact_event_count"] == 4
    assert report["left_event_count"] == 2
    assert report["right_event_count"] == 2
    assert np.isclose(report["slope_mps"], -0.03, atol=1e-10)
    assert not any("coverage" in key or "span" in key for key in report)
    assert line is not None
    expected = 0.13 - 0.03 * (np.arange(FRAME_COUNT, dtype=float) / FPS)
    assert np.allclose(line, expected, atol=1e-10)


def test_three_stances_fall_back_to_raw() -> None:
    contact = make_contact([(0, 10, 21), (1, 25, 36), (0, 40, 51)])
    line, report = estimate_contact_ground_line(
        contact,
        make_keypoints(),
        make_foot_heights(),
        FPS,
    )
    raw = {
        "transl": torch.randn(FRAME_COUNT, 3),
        "body_pose": torch.randn(FRAME_COUNT, 63),
    }
    grounded = apply_ground_line(raw, line)

    assert line is None
    assert report["applied"] is False
    assert report["fallback_reason"] == "insufficient_stances"
    assert all(torch.equal(raw[name], grounded[name]) for name in raw)


def test_each_side_needs_one_stance() -> None:
    contact = make_contact(
        [
            (0, 10, 21),
            (0, 30, 41),
            (0, 50, 61),
            (0, 70, 81),
        ]
    )
    line, report = estimate_contact_ground_line(
        contact,
        make_keypoints(),
        make_foot_heights(),
        FPS,
    )

    assert line is None
    assert report["applied"] is False
    assert report["fallback_reason"] == "insufficient_stances_per_side"
    assert report["contact_event_count"] == 4
    assert report["right_event_count"] == 0


def test_linear_grounding_does_not_change_acceleration() -> None:
    times = np.arange(FRAME_COUNT, dtype=float) / FPS
    raw_y = 0.9 + 0.04 * times + 0.02 * np.sin(2.0 * np.pi * times)
    line = 0.13 - 0.03 * times
    raw = {"transl": torch.zeros(FRAME_COUNT, 3, dtype=torch.float64)}
    raw["transl"][:, 1] = torch.from_numpy(raw_y)

    grounded = apply_ground_line(raw, line)
    raw_acceleration = np.gradient(np.gradient(raw_y, 1.0 / FPS), 1.0 / FPS)
    grounded_acceleration = np.gradient(
        np.gradient(grounded["transl"][:, 1].numpy(), 1.0 / FPS),
        1.0 / FPS,
    )

    assert np.max(np.abs(raw_acceleration - grounded_acceleration)) < 1e-11
    assert torch.equal(raw["transl"][:, 0], grounded["transl"][:, 0])
    assert torch.equal(raw["transl"][:, 2], grounded["transl"][:, 2])


def test_one_sample_per_stance_still_uses_only_matrix_rank() -> None:
    contact = make_contact(
        [
            (0, 10, 21),
            (1, 25, 36),
            (0, 40, 51),
            (1, 55, 66),
        ]
    )
    config = ContactGroundingConfig(max_samples_per_stance=1)
    line, report = estimate_contact_ground_line(
        contact,
        make_keypoints(),
        make_foot_heights(),
        FPS,
        config,
    )

    assert report["applied"] is True
    assert line is not None
