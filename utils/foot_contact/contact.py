#!/usr/bin/env python3
"""C_Temporal foot contact와 measured force event의 시간 대응을 계산합니다."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from hmr4d.model.footmr.utils.contact_grounding import (
    FOOT_KEYPOINTS,
    compute_c_temporal,
    decode_c_temporal,
)

FORCE_THRESHOLD_N = 20.0
MINIMUM_FORCE_EVENT_SAMPLES = 30
FORCE_PLATES = (
    (1, "left", "ground_force1_vy"),
    (2, "right", "ground_force2_vy"),
)


def load_forceplate_anchored_sync(path: Path) -> pd.DataFrame:
    """공간적으로 검증한 force-plate 동기화 후보를 읽고 기본 제약을 확인합니다."""
    sync = pd.read_csv(path)
    required = {
        "trial",
        "knee_only_start_s",
        "forceplate_anchored_start_s",
        "cycle_shift_s",
        "marker_source_start_s",
        "first_force_frame",
        "last_force_frame",
    }
    missing = required - set(sync.columns)
    if missing:
        raise ValueError(f"force-plate sync 열 누락: {sorted(missing)}")
    if sync["trial"].duplicated().any() or len(sync) != 5:
        raise ValueError("force-plate sync는 Walk 1-5를 각각 한 번씩 포함해야 합니다")
    shifts = sync["forceplate_anchored_start_s"] - sync["knee_only_start_s"]
    if not np.allclose(shifts, sync["cycle_shift_s"], atol=1e-9, rtol=0.0):
        raise ValueError("cycle_shift_s가 두 start의 차이와 일치하지 않습니다")
    if not shifts.between(1.0, 1.1).all():
        raise ValueError("force-plate anchor가 예상한 한 gait cycle 범위를 벗어났습니다")
    if not sync["marker_source_start_s"].between(0.0, 0.1).all():
        raise ValueError("marker source 시작 시간이 예상 범위를 벗어났습니다")
    return sync.set_index("trial", verify_integrity=True)


def extract_measured_force_events(
    trial: str,
    force: pd.DataFrame,
    threshold_n: float = FORCE_THRESHOLD_N,
    minimum_samples: int = MINIMUM_FORCE_EVENT_SAMPLES,
) -> pd.DataFrame:
    """Trimmed force에서 P1-left, P2-right의 measured stance를 직접 추출합니다."""
    required = {"time", *(column for _plate, _side, column in FORCE_PLATES)}
    missing = required - set(force.columns)
    if missing:
        raise ValueError(f"{trial}: force 열 누락: {sorted(missing)}")

    rows = []
    times = force["time"].to_numpy(dtype=float)
    for plate, side, column in FORCE_PLATES:
        values = np.abs(force[column].to_numpy(dtype=float))
        active = values > threshold_n
        starts = np.flatnonzero(active & ~np.r_[False, active[:-1]])
        ends = np.flatnonzero(active & ~np.r_[active[1:], False])
        valid_runs = [
            (int(start), int(end))
            for start, end in zip(starts, ends)
            if end - start + 1 >= minimum_samples
        ]
        if len(valid_runs) != 1:
            raise ValueError(f"{trial}: P{plate}에서 단일 measured stance를 찾지 못했습니다")
        start, end = valid_runs[0]
        rows.append(
            {
                "trial": trial,
                "plate": plate,
                "side": side,
                "force_start_s": float(times[start]),
                "force_end_s": float(times[end]),
                "force_duration_s": float(times[end] - times[start]),
                "peak_force_n": float(values[start : end + 1].max()),
            }
        )
    return pd.DataFrame(rows).sort_values("force_start_s").reset_index(drop=True)


def compute_full_video_c_temporal(
    video_dir: Path,
    smplx_model: torch.nn.Module,
    fps: float,
) -> pd.DataFrame:
    """영상 전체에서 기존 C_Temporal 특징과 4-state sequence를 재계산합니다."""
    keypoints = torch.load(
        video_dir / "preprocess/vitpose.pt",
        map_location="cpu",
        weights_only=False,
    ).numpy()
    boxes = torch.load(
        video_dir / "preprocess/bbx.pt",
        map_location="cpu",
        weights_only=False,
    )["bbx_xyxy"].numpy()
    result = torch.load(
        video_dir / "hmr4d_results.pt",
        map_location="cpu",
        weights_only=False,
    )
    with torch.inference_mode():
        _vertices, joints = smplx_model(**result["smpl_params_incam"])
    joints = joints.detach().cpu().numpy()

    if len(keypoints) != len(boxes) or len(keypoints) != len(joints):
        raise ValueError(f"{video_dir.name}: pose, bbox, SMPL frame 수가 다릅니다")
    if keypoints.shape[1:] != (23, 3) or joints.shape[1:] != (23, 3):
        raise ValueError(f"{video_dir.name}: 예상하지 못한 23-keypoint shape")

    contact = compute_c_temporal(keypoints, boxes, joints, fps)
    feature_columns: dict[str, np.ndarray] = {}
    for side_index, side in enumerate(FOOT_KEYPOINTS):
        feature_columns[f"speed_2d_body_per_s_{side}"] = contact.speed_2d[:, side_index]
        feature_columns[f"speed_3d_mps_{side}"] = contact.speed_3d[:, side_index]
        feature_columns[f"bbox_bottom_gap_body_{side}"] = contact.bottom_gap[:, side_index]
        feature_columns[f"B_{side}_probability"] = contact.probabilities[:, side_index]

    return pd.DataFrame(
        {
            "video_frame": np.arange(len(keypoints), dtype=int),
            "video_time_s": np.arange(len(keypoints), dtype=float) / fps,
            "C_Temporal_left": contact.states[:, 0],
            "C_Temporal_right": contact.states[:, 1],
            **feature_columns,
        }
    )


@dataclass(frozen=True)
class EventClip:
    """하나의 force-plate event와 대응하는 영상 구간입니다."""

    trial: str
    trial_number: int
    plate: int
    side: str
    force_start_s: float
    force_end_s: float
    peak_force_n: float
    markerless_start_s: float
    marker_source_start_s: float
    start_frame: int
    end_frame: int
    context_start_s: float
    context_end_s: float

    @property
    def event_id(self) -> str:
        return f"{self.trial}_P{self.plate}_{self.side}"

    @property
    def event_duration_s(self) -> float:
        return self.force_end_s - self.force_start_s

    @property
    def frame_count(self) -> int:
        return self.end_frame - self.start_frame + 1


def video_frame_to_force_time(
    video_frame: int | np.ndarray,
    fps: float,
    markerless_start_s: float,
    marker_source_start_s: float,
) -> float | np.ndarray:
    """Video frame timestamp를 trimmed measured-force 시간으로 변환합니다."""
    return np.asarray(video_frame) / fps - markerless_start_s - marker_source_start_s


def force_time_to_video_frame(
    force_time_s: float,
    fps: float,
    markerless_start_s: float,
    marker_source_start_s: float,
) -> int:
    """Trimmed measured-force 시간을 가장 가까운 video frame으로 변환합니다."""
    return int(round((markerless_start_s + marker_source_start_s + float(force_time_s)) * fps))


def contiguous_runs(
    frames: np.ndarray,
    labels: np.ndarray,
) -> list[tuple[int, int]]:
    """Boolean frame label의 연속 구간을 반환합니다."""
    active_indices = np.flatnonzero(labels.astype(bool))
    if len(active_indices) == 0:
        return []
    boundaries = np.flatnonzero(np.diff(active_indices) > 1) + 1
    return [
        (int(frames[group[0]]), int(frames[group[-1]]))
        for group in np.split(active_indices, boundaries)
    ]


def prediction_runs_force_time(
    labels: pd.DataFrame,
    side: str,
    fps: float,
    markerless_start_s: float,
    marker_source_start_s: float,
) -> list[tuple[int, int, float, float]]:
    """C_Temporal 연속 구간을 force-local time interval로 변환합니다."""
    column = f"C_Temporal_{side}"
    frames = labels["video_frame"].to_numpy(dtype=int)
    values = labels[column].to_numpy(dtype=bool)
    runs = []
    half_frame_s = 0.5 / fps
    for first_frame, last_frame in contiguous_runs(frames, values):
        first_time = float(
            video_frame_to_force_time(
                first_frame,
                fps,
                markerless_start_s,
                marker_source_start_s,
            )
        )
        last_time = float(
            video_frame_to_force_time(
                last_frame,
                fps,
                markerless_start_s,
                marker_source_start_s,
            )
        )
        runs.append(
            (
                first_frame,
                last_frame,
                first_time - half_frame_s,
                last_time + half_frame_s,
            )
        )
    return runs


def interval_overlap(
    first_start: float,
    first_end: float,
    second_start: float,
    second_end: float,
) -> float:
    return max(0.0, min(first_end, second_end) - max(first_start, second_start))


def build_event_clips(
    trials: list[object],
    fps: float,
    context_s: float,
) -> list[EventClip]:
    """Force event별 clip frame 범위를 계산합니다."""
    clips = []
    for trial in trials:
        marker_source_start_s = float(trial.marker_source_start_s)
        available_start = int(trial.labels["video_frame"].min())
        available_end = int(trial.labels["video_frame"].max())

        for event in trial.events.sort_values("force_start_s").itertuples(index=False):
            requested_start = force_time_to_video_frame(
                float(event.force_start_s) - context_s,
                fps,
                trial.markerless_start_s,
                marker_source_start_s,
            )
            requested_end = force_time_to_video_frame(
                float(event.force_end_s) + context_s,
                fps,
                trial.markerless_start_s,
                marker_source_start_s,
            )
            start_frame = max(available_start, requested_start)
            end_frame = min(available_end, requested_end)
            if start_frame > end_frame:
                raise ValueError(f"{trial.trial}: force event clip이 비어 있습니다")

            clips.append(
                EventClip(
                    trial=trial.trial,
                    trial_number=trial.trial_number,
                    plate=int(event.plate),
                    side=str(event.side),
                    force_start_s=float(event.force_start_s),
                    force_end_s=float(event.force_end_s),
                    peak_force_n=float(event.peak_force_n),
                    markerless_start_s=float(trial.markerless_start_s),
                    marker_source_start_s=marker_source_start_s,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    context_start_s=float(
                        video_frame_to_force_time(
                            start_frame,
                            fps,
                            trial.markerless_start_s,
                            marker_source_start_s,
                        )
                    ),
                    context_end_s=float(
                        video_frame_to_force_time(
                            end_frame,
                            fps,
                            trial.markerless_start_s,
                            marker_source_start_s,
                        )
                    ),
                )
            )
    return clips


def evaluate_event(
    clip: EventClip,
    labels: pd.DataFrame,
    fps: float,
    marker_source_start_s: float,
) -> dict[str, object]:
    """하나의 measured force pulse와 C_Temporal run을 비교합니다."""
    runs = prediction_runs_force_time(
        labels,
        clip.side,
        fps,
        clip.markerless_start_s,
        marker_source_start_s,
    )
    overlaps = [
        interval_overlap(
            clip.force_start_s,
            clip.force_end_s,
            prediction_start_s,
            prediction_end_s,
        )
        for _first, _last, prediction_start_s, prediction_end_s in runs
    ]
    matched_index = int(np.argmax(overlaps)) if overlaps and max(overlaps) > 0 else None

    frames = labels["video_frame"].to_numpy(dtype=int)
    force_times = np.asarray(
        video_frame_to_force_time(
            frames,
            fps,
            clip.markerless_start_s,
            marker_source_start_s,
        ),
        dtype=float,
    )
    positive_mask = (force_times >= clip.force_start_s) & (force_times <= clip.force_end_s)
    side_labels = labels[f"C_Temporal_{clip.side}"].to_numpy(dtype=bool)
    positive_frames = int(positive_mask.sum())
    recalled_frames = int(np.count_nonzero(positive_mask & side_labels))
    pulse_recall = recalled_frames / positive_frames if positive_frames else math.nan

    result: dict[str, object] = {
        "trial": clip.trial,
        "plate": clip.plate,
        "side": clip.side,
        "force_start_s": clip.force_start_s,
        "force_end_s": clip.force_end_s,
        "force_duration_s": clip.event_duration_s,
        "force_peak_n": clip.peak_force_n,
        "markerless_start_s": clip.markerless_start_s,
        "marker_source_start_s": marker_source_start_s,
        "mapped_force_start_frame": force_time_to_video_frame(
            clip.force_start_s,
            fps,
            clip.markerless_start_s,
            marker_source_start_s,
        ),
        "mapped_force_end_frame": force_time_to_video_frame(
            clip.force_end_s,
            fps,
            clip.markerless_start_s,
            marker_source_start_s,
        ),
        "positive_force_frames_30fps": positive_frames,
        "positive_prediction_frames_30fps": recalled_frames,
        "pulse_recall_30fps": pulse_recall,
        "event_detected": matched_index is not None,
    }

    if matched_index is None:
        result.update(
            {
                "matched_prediction_first_frame": -1,
                "matched_prediction_last_frame": -1,
                "prediction_start_s": math.nan,
                "prediction_end_s": math.nan,
                "prediction_duration_s": math.nan,
                "overlap_s": 0.0,
                "overlap_over_force": 0.0,
                "temporal_iou": 0.0,
                "temporal_dice": 0.0,
                "onset_error_ms": math.nan,
                "offset_error_ms": math.nan,
                "onset_abs_error_ms": math.nan,
                "offset_abs_error_ms": math.nan,
            }
        )
        return result

    first_frame, last_frame, prediction_start_s, prediction_end_s = runs[matched_index]
    overlap_s = overlaps[matched_index]
    prediction_duration_s = prediction_end_s - prediction_start_s
    union_s = clip.event_duration_s + prediction_duration_s - overlap_s
    onset_error_ms = (prediction_start_s - clip.force_start_s) * 1000.0
    offset_error_ms = (prediction_end_s - clip.force_end_s) * 1000.0
    result.update(
        {
            "matched_prediction_first_frame": first_frame,
            "matched_prediction_last_frame": last_frame,
            "prediction_start_s": prediction_start_s,
            "prediction_end_s": prediction_end_s,
            "prediction_duration_s": prediction_duration_s,
            "overlap_s": overlap_s,
            "overlap_over_force": overlap_s / clip.event_duration_s,
            "temporal_iou": overlap_s / union_s if union_s else 0.0,
            "temporal_dice": (2.0 * overlap_s / (clip.event_duration_s + prediction_duration_s)),
            "onset_error_ms": onset_error_ms,
            "offset_error_ms": offset_error_ms,
            "onset_abs_error_ms": abs(onset_error_ms),
            "offset_abs_error_ms": abs(offset_error_ms),
        }
    )
    return result


def evaluate_all_events(
    clips: list[EventClip],
    trials_by_name: dict[str, object],
    fps: float,
) -> pd.DataFrame:
    """10개 force event를 force-plate anchored origin에서 평가합니다."""
    rows = []
    for clip in clips:
        rows.append(
            evaluate_event(
                clip,
                trials_by_name[clip.trial].labels,
                fps,
                clip.marker_source_start_s,
            )
        )
    return pd.DataFrame(rows)


def summarize_events(metrics: pd.DataFrame) -> dict[str, float | int]:
    """Event-only paired 지표를 요약합니다."""
    detected = metrics["event_detected"].astype(bool)
    detected_metrics = metrics[detected]
    return {
        "events": int(len(metrics)),
        "events_detected": int(detected.sum()),
        "event_detection_rate": float(detected.mean()),
        "pulse_recall_macro": float(metrics["pulse_recall_30fps"].mean()),
        "overlap_over_force_macro": float(metrics["overlap_over_force"].mean()),
        "temporal_iou_macro": float(metrics["temporal_iou"].mean()),
        "temporal_dice_macro": float(metrics["temporal_dice"].mean()),
        "onset_mae_ms_detected": float(detected_metrics["onset_abs_error_ms"].mean()),
        "offset_mae_ms_detected": float(detected_metrics["offset_abs_error_ms"].mean()),
        "onset_bias_ms_detected": float(detected_metrics["onset_error_ms"].mean()),
        "offset_bias_ms_detected": float(detected_metrics["offset_error_ms"].mean()),
    }
