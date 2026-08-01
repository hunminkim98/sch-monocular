"""C_Temporal 접촉으로 평지 보행의 전역 Y 선형 drift를 보정합니다."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.optimize import least_squares
from scipy.signal import savgol_filter

FOOT_KEYPOINTS = {
    "left": (17, 18, 19),
    "right": (20, 21, 22),
}
VITERBI_STATES = np.asarray(
    [
        (0, 0),  # none
        (1, 0),  # left
        (0, 1),  # right
        (1, 1),  # both
    ],
    dtype=np.int8,
)

# 검증한 C_Temporal state sequence를 재현하는 고정 penalty입니다.
VITERBI_CHANGE_PENALTY = 0.15
VITERBI_NONE_PENALTY = 1.25
VITERBI_BOTH_PENALTY = 0.05
VITERBI_TWO_BIT_PENALTY = 1.35
VITERBI_DIRECT_LEFT_RIGHT_PENALTY = 0.125


@dataclass(frozen=True)
class CTemporalResult:
    """두 발의 C_Temporal 특징과 최종 접촉 상태입니다."""

    states: np.ndarray
    probabilities: np.ndarray
    speed_2d: np.ndarray
    speed_3d: np.ndarray
    bottom_gap: np.ndarray


@dataclass(frozen=True)
class ContactGroundingConfig:
    """평지 보행용 sequence-level 선형 grounding 기준입니다."""

    probability_threshold: float = 0.60
    confidence_threshold: float = 0.50
    central_fraction: float = 0.60
    min_run_frames: int = 8
    max_samples_per_stance: int = 9
    min_total_stances: int = 4
    min_stances_per_side: int = 1
    max_residual_mad_m: float = 0.03
    max_abs_slope_mps: float = 0.10

    def __post_init__(self) -> None:
        if not 0.0 <= self.probability_threshold <= 1.0:
            raise ValueError("probability_threshold는 [0, 1] 범위여야 합니다")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold는 [0, 1] 범위여야 합니다")
        if not 0.0 < self.central_fraction <= 1.0:
            raise ValueError("central_fraction은 (0, 1] 범위여야 합니다")
        if self.min_run_frames < 1 or self.max_samples_per_stance < 1:
            raise ValueError("frame과 stance 표본 수 기준은 1 이상이어야 합니다")
        if self.min_total_stances < 1 or self.min_stances_per_side < 1:
            raise ValueError("stance 수 기준은 1 이상이어야 합니다")
        if self.max_residual_mad_m < 0.0 or self.max_abs_slope_mps < 0.0:
            raise ValueError("residual과 slope 상한은 음수일 수 없습니다")


@dataclass(frozen=True)
class ContactStance:
    """선형 지면 적합에 사용할 하나의 안정된 stance입니다."""

    side_index: int
    start_frame: int
    end_frame: int
    sample_frames: np.ndarray


def _as_numpy(values: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def _smooth(values: np.ndarray) -> np.ndarray:
    """짧은 영상에도 안전한 7-frame Savitzky-Golay smoothing을 적용합니다."""
    if len(values) < 3:
        return values.copy()
    window = min(7, len(values) if len(values) % 2 else len(values) - 1)
    if window < 3:
        return values.copy()
    return savgol_filter(values, window, min(2, window - 1), axis=0, mode="interp")


def decode_c_temporal(probabilities: np.ndarray) -> np.ndarray:
    """Left/right contact 확률을 4-state Viterbi sequence로 변환합니다."""
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
        raise ValueError("probabilities shape은 (frames, 2)여야 합니다")
    if len(probabilities) == 0:
        return np.empty((0, 2), dtype=bool)

    probabilities = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
    emissions = -(
        VITERBI_STATES[None] * np.log(probabilities[:, None])
        + (1 - VITERBI_STATES)[None] * np.log(1.0 - probabilities[:, None])
    ).sum(axis=2)
    emissions[:, 0] += VITERBI_NONE_PENALTY
    emissions[:, 3] += VITERBI_BOTH_PENALTY

    transitions = np.zeros((len(VITERBI_STATES), len(VITERBI_STATES)), dtype=float)
    for source_index, source in enumerate(VITERBI_STATES):
        for target_index, target in enumerate(VITERBI_STATES):
            if source_index == target_index:
                continue
            hamming_distance = int(np.abs(source - target).sum())
            penalty = VITERBI_CHANGE_PENALTY
            if hamming_distance == 2:
                penalty += VITERBI_TWO_BIT_PENALTY
            if (source_index, target_index) in ((1, 2), (2, 1)):
                penalty += VITERBI_DIRECT_LEFT_RIGHT_PENALTY
            transitions[source_index, target_index] = penalty

    costs = np.full((len(probabilities), len(VITERBI_STATES)), np.inf, dtype=float)
    backpointers = np.zeros_like(costs, dtype=np.int16)
    costs[0] = emissions[0]
    for frame in range(1, len(probabilities)):
        candidates = costs[frame - 1, :, None] + transitions
        backpointers[frame] = np.argmin(candidates, axis=0)
        costs[frame] = np.min(candidates, axis=0) + emissions[frame]

    state_indices = np.empty(len(probabilities), dtype=np.int16)
    state_indices[-1] = int(np.argmin(costs[-1]))
    for frame in range(len(probabilities) - 1, 0, -1):
        state_indices[frame - 1] = backpointers[frame, state_indices[frame]]
    return VITERBI_STATES[state_indices].astype(bool)


def compute_c_temporal(
    keypoints: np.ndarray | torch.Tensor,
    boxes_xyxy: np.ndarray | torch.Tensor,
    incam_joints: np.ndarray | torch.Tensor,
    fps: float,
) -> CTemporalResult:
    """2D pose, bbox, camera-space joint에서 C_Temporal을 계산합니다."""
    keypoints = _as_numpy(keypoints).astype(float, copy=False)
    boxes_xyxy = _as_numpy(boxes_xyxy).astype(float, copy=False)
    incam_joints = _as_numpy(incam_joints).astype(float, copy=False)

    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError("fps는 양의 유한값이어야 합니다")
    if keypoints.ndim != 3 or keypoints.shape[1:] != (23, 3):
        raise ValueError("keypoints shape은 (frames, 23, 3)이어야 합니다")
    if boxes_xyxy.shape != (len(keypoints), 4):
        raise ValueError("boxes_xyxy shape은 (frames, 4)여야 합니다")
    if incam_joints.shape != (len(keypoints), 23, 3):
        raise ValueError("incam_joints shape은 (frames, 23, 3)이어야 합니다")
    if len(keypoints) < 2:
        raise ValueError("C_Temporal 계산에는 최소 2 frame이 필요합니다")
    if not (
        np.isfinite(keypoints).all()
        and np.isfinite(boxes_xyxy).all()
        and np.isfinite(incam_joints).all()
    ):
        raise ValueError("C_Temporal 입력에 유한하지 않은 값이 있습니다")

    smoothed_2d = _smooth(keypoints[..., :2])
    smoothed_3d = _smooth(incam_joints)
    bbox_height = np.maximum(boxes_xyxy[:, 3] - boxes_xyxy[:, 1], 1.0)
    speed_2d_values = []
    speed_3d_values = []
    bottom_gap_values = []
    probabilities = []

    for indices in FOOT_KEYPOINTS.values():
        ids = list(indices)
        speed_2d = (
            np.linalg.norm(np.gradient(smoothed_2d[:, ids], axis=0) * fps, axis=-1).mean(axis=1)
            / bbox_height
        )
        speed_3d = np.linalg.norm(
            np.gradient(smoothed_3d[:, ids], axis=0) * fps,
            axis=-1,
        ).mean(axis=1)
        bottom_gap = np.maximum(
            0.0,
            (boxes_xyxy[:, 3] - keypoints[:, ids, 1].max(axis=1)) / bbox_height,
        )
        raw_score = (
            0.5 * np.exp(-0.5 * (speed_2d / 0.22) ** 2)
            + 0.3 * np.exp(-0.5 * (speed_3d / 0.35) ** 2)
            + 0.2 * np.exp(-0.5 * (bottom_gap / 0.055) ** 2)
        )
        probability = 1.0 / (1.0 + np.exp(-7.0 * (raw_score - 0.5)))
        speed_2d_values.append(speed_2d)
        speed_3d_values.append(speed_3d)
        bottom_gap_values.append(bottom_gap)
        probabilities.append(probability)

    probability_array = np.stack(probabilities, axis=1)
    return CTemporalResult(
        states=decode_c_temporal(probability_array),
        probabilities=probability_array,
        speed_2d=np.stack(speed_2d_values, axis=1),
        speed_3d=np.stack(speed_3d_values, axis=1),
        bottom_gap=np.stack(bottom_gap_values, axis=1),
    )


def _boolean_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    active = np.flatnonzero(np.asarray(mask, dtype=bool))
    if len(active) == 0:
        return []
    boundaries = np.flatnonzero(np.diff(active) > 1) + 1
    return [(int(group[0]), int(group[-1])) for group in np.split(active, boundaries)]


def select_contact_stances(
    contact: CTemporalResult,
    keypoints: np.ndarray | torch.Tensor,
    config: ContactGroundingConfig | None = None,
) -> list[ContactStance]:
    """접촉 run에서 중앙부와 pose confidence를 만족하는 stance를 고릅니다."""
    config = config or ContactGroundingConfig()
    keypoints = _as_numpy(keypoints).astype(float, copy=False)
    if keypoints.shape != (len(contact.states), 23, 3):
        raise ValueError("keypoints와 contact의 frame 수 또는 shape이 다릅니다")

    lower_body_confidence = np.median(keypoints[:, 11:23, 2], axis=1)
    stances = []
    for side_index, indices in enumerate(FOOT_KEYPOINTS.values()):
        foot_confidence = np.median(keypoints[:, list(indices), 2], axis=1)
        reliable = (
            contact.states[:, side_index]
            & (contact.probabilities[:, side_index] >= config.probability_threshold)
            & (foot_confidence >= config.confidence_threshold)
            & (lower_body_confidence >= config.confidence_threshold)
        )
        for start, end in _boolean_runs(reliable):
            run_length = end - start + 1
            if run_length < config.min_run_frames:
                continue
            trim = int(np.floor(run_length * (1.0 - config.central_fraction) / 2.0))
            central_start = start + trim
            central_end = end - trim
            central_frames = np.arange(central_start, central_end + 1, dtype=int)
            sample_count = min(config.max_samples_per_stance, len(central_frames))
            sample_indices = np.linspace(0, len(central_frames) - 1, sample_count)
            sample_frames = central_frames[np.rint(sample_indices).astype(int)]
            stances.append(
                ContactStance(
                    side_index=side_index,
                    start_frame=start,
                    end_frame=end,
                    sample_frames=np.unique(sample_frames),
                )
            )
    return sorted(stances, key=lambda stance: stance.start_frame)


def _report(
    *,
    applied: bool,
    reason: str | None,
    stances: list[ContactStance],
    sample_count: int = 0,
    slope_mps: float | None = None,
    ground_intercept_m: float | None = None,
    center_time_s: float | None = None,
    side_offset_m: float | None = None,
    residual_mad_m: float | None = None,
) -> dict[str, bool | int | float | str | None]:
    side_counts = [sum(stance.side_index == side for stance in stances) for side in range(2)]
    return {
        "mode": "contact-linear",
        "applied": applied,
        "fallback_reason": reason,
        "contact_event_count": len(stances),
        "left_event_count": side_counts[0],
        "right_event_count": side_counts[1],
        "sample_count": sample_count,
        "slope_mps": slope_mps,
        "ground_intercept_m": ground_intercept_m,
        "center_time_s": center_time_s,
        "side_offset_m": side_offset_m,
        "residual_mad_m": residual_mad_m,
    }


def estimate_contact_ground_line(
    contact: CTemporalResult,
    keypoints: np.ndarray | torch.Tensor,
    foot_heights: np.ndarray | torch.Tensor,
    fps: float,
    config: ContactGroundingConfig | None = None,
) -> tuple[np.ndarray | None, dict[str, bool | int | float | str | None]]:
    """안정된 stance 높이에 하나의 Huber line을 적합합니다.

    Foot별 상수 높이 차이는 허용하지만 시간 slope는 양발과 전체 sequence가 공유합니다.
    품질 조건을 통과하지 못하면 보정선 대신 ``None``을 반환합니다.
    """
    config = config or ContactGroundingConfig()
    foot_heights = _as_numpy(foot_heights).astype(float, copy=False)
    if foot_heights.shape != (len(contact.states), 2):
        raise ValueError("foot_heights shape은 (frames, 2)여야 합니다")
    if not np.isfinite(foot_heights).all():
        return None, _report(applied=False, reason="non_finite_foot_height", stances=[])
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError("fps는 양의 유한값이어야 합니다")

    stances = select_contact_stances(contact, keypoints, config)
    side_counts = [sum(stance.side_index == side for stance in stances) for side in range(2)]
    if len(stances) < config.min_total_stances:
        return None, _report(applied=False, reason="insufficient_stances", stances=stances)
    if min(side_counts) < config.min_stances_per_side:
        return None, _report(applied=False, reason="insufficient_stances_per_side", stances=stances)

    frames = np.concatenate([stance.sample_frames for stance in stances])
    sides = np.concatenate(
        [np.full(len(stance.sample_frames), stance.side_index, dtype=int) for stance in stances]
    )
    heights = foot_heights[frames, sides]
    times = frames.astype(float) / fps
    center_time = float(times.mean())
    design = np.column_stack(
        [
            np.ones(len(times), dtype=float),
            times - center_time,
            sides.astype(float),
        ]
    )

    # 시간 범위 비율은 검사하지 않습니다. 실제 설계행렬의 rank만 확인합니다.
    if np.linalg.matrix_rank(design) < design.shape[1]:
        return None, _report(
            applied=False,
            reason="rank_deficient",
            stances=stances,
            sample_count=len(frames),
        )

    initial, *_ = np.linalg.lstsq(design, heights, rcond=None)
    try:
        fit = least_squares(
            lambda parameters: design @ parameters - heights,
            initial,
            loss="huber",
            f_scale=0.01,
        )
    except (ValueError, RuntimeError):
        return None, _report(
            applied=False,
            reason="fit_failed",
            stances=stances,
            sample_count=len(frames),
        )
    if not fit.success or not np.isfinite(fit.x).all():
        return None, _report(
            applied=False,
            reason="fit_failed",
            stances=stances,
            sample_count=len(frames),
        )

    intercept, slope, side_offset = (float(value) for value in fit.x)
    residual = heights - design @ fit.x
    residual_median = float(np.median(residual))
    residual_mad = float(np.median(np.abs(residual - residual_median)))
    ground_intercept = intercept + 0.5 * side_offset
    report_values = {
        "stances": stances,
        "sample_count": len(frames),
        "slope_mps": slope,
        "ground_intercept_m": ground_intercept,
        "center_time_s": center_time,
        "side_offset_m": side_offset,
        "residual_mad_m": residual_mad,
    }
    if residual_mad > config.max_residual_mad_m:
        return None, _report(applied=False, reason="residual_too_large", **report_values)
    if abs(slope) > config.max_abs_slope_mps:
        return None, _report(applied=False, reason="slope_out_of_bounds", **report_values)

    all_times = np.arange(len(contact.states), dtype=float) / fps
    ground_line = ground_intercept + slope * (all_times - center_time)
    return ground_line, _report(applied=True, reason=None, **report_values)


def apply_ground_line(
    smpl_params_global: dict[str, torch.Tensor],
    ground_line: np.ndarray | None,
) -> dict[str, torch.Tensor]:
    """SMPL global translation의 Y에만 하나의 보정선을 적용합니다."""
    grounded = {name: value.clone() for name, value in smpl_params_global.items()}
    if ground_line is None:
        return grounded
    transl = grounded["transl"]
    ground_line = np.asarray(ground_line, dtype=float)
    if transl.ndim != 2 or transl.shape[1] != 3 or len(transl) != len(ground_line):
        raise ValueError("transl과 ground_line의 frame 수 또는 shape이 다릅니다")
    line_tensor = torch.as_tensor(ground_line, dtype=transl.dtype, device=transl.device)
    transl[:, 1] = transl[:, 1] - line_tensor
    return grounded


@torch.no_grad()
def apply_contact_linear_grounding(
    smpl_params_global: dict[str, torch.Tensor],
    smpl_params_incam: dict[str, torch.Tensor],
    keypoints: np.ndarray | torch.Tensor,
    boxes_xyxy: np.ndarray | torch.Tensor,
    smplx_model: torch.nn.Module,
    fps: float,
    config: ContactGroundingConfig | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, bool | int | float | str | None]]:
    """Raw FootMR 결과에 선택형 평지 보행 grounding을 적용합니다."""
    _incam_vertices, incam_joints = smplx_model(**smpl_params_incam)
    _global_vertices, global_joints = smplx_model(**smpl_params_global)
    contact = compute_c_temporal(keypoints, boxes_xyxy, incam_joints, fps)
    global_joints = _as_numpy(global_joints)
    foot_heights = np.stack(
        [global_joints[:, list(indices), 1].min(axis=1) for indices in FOOT_KEYPOINTS.values()],
        axis=1,
    )
    ground_line, report = estimate_contact_ground_line(
        contact,
        keypoints,
        foot_heights,
        fps,
        config,
    )
    return apply_ground_line(smpl_params_global, ground_line), report
