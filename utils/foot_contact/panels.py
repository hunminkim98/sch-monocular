#!/usr/bin/env python3
"""Pose, mesh, foot contact를 논문용 3-panel 영상으로 렌더링합니다."""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from utils.foot_contact import visuals as base

FPS = 30.0
WIDTH = 1920
HEIGHT = 1080
PANEL_WIDTH = 612
PANEL_HEIGHT = 1000
PANEL_Y = 40
PANEL_X = (24, 654, 1284)
FORCE_THRESHOLD_N = 20.0
INTRO_FRAMES = 90
TRIAL_CARD_FRAMES = 24
OUTRO_FRAMES = 90

BACKGROUND = (246, 245, 242)
WHITE = (255, 255, 255)
TEXT = (35, 35, 35)
MUTED = (112, 112, 112)
GRID = (221, 221, 221)
LEFT = base.LEFT
RIGHT = base.RIGHT
FORCE = base.FORCE
NONE = base.NONE
ACTIVE_DARK = base.ACTIVE_DARK

POSE_CONFIDENCE_THRESHOLD = 0.35
LEFT_EDGES = (
    (5, 7),
    (7, 9),
    (11, 13),
    (13, 15),
    (15, 17),
    (15, 18),
    (15, 19),
)
RIGHT_EDGES = (
    (6, 8),
    (8, 10),
    (12, 14),
    (14, 16),
    (16, 20),
    (16, 21),
    (16, 22),
)
NEUTRAL_EDGES = (
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
)
LEFT_KEYPOINTS = (1, 3, 5, 7, 9, 11, 13, 15, 17, 18, 19)
RIGHT_KEYPOINTS = (2, 4, 6, 8, 10, 12, 14, 16, 20, 21, 22)


@dataclass
class MeshInfo:
    path: Path
    width: int
    height: int
    frames: int
    fps: float


@dataclass
class ContactPanel:
    static: np.ndarray
    force_plot: tuple[int, int, int, int]
    timeline_plot: tuple[int, int, int, int]
    force_y_max: float


def load_inputs(sync_path: Path, force_dir: Path) -> tuple[
    list[base.TrialData],
    dict[str, MeshInfo],
]:
    trials = base.load_trials(sync_path, force_dir)
    mesh_info: dict[str, MeshInfo] = {}
    for trial in trials:
        path = trial.video_path.parent / "1_incam.mp4"
        probe = base.probe_video(path)
        info = MeshInfo(
            path=path,
            width=int(probe["width"]),
            height=int(probe["height"]),
            frames=int(probe["nb_frames"]),
            fps=base.eval_fraction(str(probe["avg_frame_rate"])),
        )
        if info.frames != trial.source_frames:
            raise ValueError(f"{trial.trial}: raw와 mesh frame 수가 다릅니다")
        if not math.isclose(info.fps, trial.source_fps, abs_tol=1e-6):
            raise ValueError(f"{trial.trial}: raw와 mesh FPS가 다릅니다")
        if (info.width, info.height) != (trial.source_width, trial.source_height):
            raise ValueError(f"{trial.trial}: raw와 mesh 해상도가 다릅니다")
        mesh_info[trial.trial] = info
    return trials, mesh_info


def panel_background(canvas: np.ndarray, x: int) -> None:
    cv2.rectangle(
        canvas,
        (x, PANEL_Y),
        (x + PANEL_WIDTH, PANEL_Y + PANEL_HEIGHT),
        WHITE,
        -1,
        cv2.LINE_AA,
    )
    cv2.rectangle(
        canvas,
        (x, PANEL_Y),
        (x + PANEL_WIDTH, PANEL_Y + PANEL_HEIGHT),
        (205, 205, 205),
        1,
        cv2.LINE_AA,
    )


def centered_panel_text(
    canvas: np.ndarray,
    panel_x: int,
    text: str,
    y: int,
    scale: float,
    color: tuple[int, int, int] = TEXT,
    thickness: int = 1,
) -> None:
    size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x = panel_x + (PANEL_WIDTH - size[0]) // 2
    base.draw_text(canvas, text, (x, y), scale, color, thickness)


def fit_frame(
    canvas: np.ndarray,
    image: np.ndarray,
    panel_x: int,
    top: int = 92,
    bottom: int = 1020,
) -> tuple[int, int, int, int, float]:
    available_width = PANEL_WIDTH - 28
    available_height = bottom - top
    source_height, source_width = image.shape[:2]
    scale = min(available_width / source_width, available_height / source_height)
    target_width = int(round(source_width * scale))
    target_height = int(round(source_height * scale))
    x = panel_x + (PANEL_WIDTH - target_width) // 2
    y = top + (available_height - target_height) // 2
    resized = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)
    canvas[y : y + target_height, x : x + target_width] = resized
    cv2.rectangle(
        canvas,
        (x, y),
        (x + target_width, y + target_height),
        (130, 130, 130),
        1,
        cv2.LINE_AA,
    )
    return x, y, target_width, target_height, scale


def map_keypoint(
    point: np.ndarray,
    image_rect: tuple[int, int, int, int, float],
) -> tuple[int, int]:
    x, y, _width, _height, scale = image_rect
    return int(round(x + float(point[0]) * scale)), int(round(y + float(point[1]) * scale))


def valid_keypoint(keypoints: np.ndarray, index: int) -> bool:
    point = keypoints[index]
    return bool(np.isfinite(point).all() and float(point[2]) >= POSE_CONFIDENCE_THRESHOLD)


def draw_pose_edges(
    canvas: np.ndarray,
    keypoints: np.ndarray,
    image_rect: tuple[int, int, int, int, float],
    edges: tuple[tuple[int, int], ...],
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    for first, second in edges:
        if not (valid_keypoint(keypoints, first) and valid_keypoint(keypoints, second)):
            continue
        cv2.line(
            canvas,
            map_keypoint(keypoints[first], image_rect),
            map_keypoint(keypoints[second], image_rect),
            color,
            thickness,
            cv2.LINE_AA,
        )


def draw_pose_points(
    canvas: np.ndarray,
    keypoints: np.ndarray,
    image_rect: tuple[int, int, int, int, float],
    indices: tuple[int, ...],
    color: tuple[int, int, int],
) -> None:
    for index in indices:
        if not valid_keypoint(keypoints, index):
            continue
        position = map_keypoint(keypoints[index], image_rect)
        cv2.circle(canvas, position, 5, WHITE, -1, cv2.LINE_AA)
        cv2.circle(canvas, position, 3, color, -1, cv2.LINE_AA)


def draw_pose_overlay(
    canvas: np.ndarray,
    keypoints: np.ndarray,
    image_rect: tuple[int, int, int, int, float],
) -> None:
    draw_pose_edges(canvas, keypoints, image_rect, NEUTRAL_EDGES, ACTIVE_DARK, 2)
    draw_pose_edges(canvas, keypoints, image_rect, LEFT_EDGES, LEFT, 3)
    draw_pose_edges(canvas, keypoints, image_rect, RIGHT_EDGES, RIGHT, 3)
    draw_pose_points(canvas, keypoints, image_rect, LEFT_KEYPOINTS, LEFT)
    draw_pose_points(canvas, keypoints, image_rect, RIGHT_KEYPOINTS, RIGHT)


def draw_video_footer(
    canvas: np.ndarray,
    image_rect: tuple[int, int, int, int, float],
    text: str,
) -> None:
    x, y, width, height, _scale = image_rect
    base.alpha_rectangle(
        canvas,
        (x + 10, y + height - 52),
        (x + width - 10, y + height - 10),
        (0, 0, 0),
        0.62,
    )
    base.draw_text(
        canvas,
        text,
        (x + 23, y + height - 25),
        0.40,
        WHITE,
        1,
    )


def make_contact_panel(trial: base.TrialData) -> ContactPanel:
    panel = np.full((PANEL_HEIGHT, PANEL_WIDTH, 3), WHITE, dtype=np.uint8)
    force_plot = (52, 220, 530, 235)
    timeline_plot = (120, 555, 462, 202)
    force_x, force_y, force_w, force_h = force_plot
    timeline_x, timeline_y, timeline_w, _timeline_h = timeline_plot
    duration = trial.aligned_duration_s
    maximum = max(float(trial.force_left.max()), float(trial.force_right.max()))
    force_y_max = max(200.0, math.ceil(maximum / 100.0) * 100.0)

    base.draw_text(panel, "(c) C_Temporal foot contact", (22, 38), 0.66, TEXT, 2)
    base.draw_text(panel, "Measured force - evaluation only", (20, 182), 0.52, TEXT, 2)
    base.draw_text(panel, "Left / P1", (420, 182), 0.39, LEFT, 2)
    base.draw_text(panel, "Right / P2", (505, 182), 0.39, RIGHT, 2)

    for event in trial.events.itertuples(index=False):
        x1 = base.time_to_x(float(event.force_start_s), duration, force_x, force_w)
        x2 = base.time_to_x(float(event.force_end_s), duration, force_x, force_w)
        color = LEFT if event.side == "left" else RIGHT
        base.alpha_rectangle(
            panel,
            (x1, force_y),
            (x2, force_y + force_h),
            color,
            0.07,
        )

    for force_n in np.arange(0.0, force_y_max + 1.0, 200.0):
        y = int(round(force_y + force_h - force_n / force_y_max * force_h))
        cv2.line(panel, (force_x, y), (force_x + force_w, y), GRID, 1, cv2.LINE_AA)
        base.draw_text(panel, f"{force_n:.0f}", (10, y + 4), 0.32, MUTED, 1)

    threshold_y = int(round(force_y + force_h - FORCE_THRESHOLD_N / force_y_max * force_h))
    base.draw_dashed_line(
        panel,
        (force_x, threshold_y),
        (force_x + force_w, threshold_y),
        (135, 135, 135),
        1,
        6,
    )
    base.draw_text(panel, "20 N", (force_x + 5, threshold_y - 5), 0.30, MUTED, 1)

    sampled = np.arange(0, len(trial.force_time), max(1, len(trial.force_time) // 900))
    for values, color in ((trial.force_left, LEFT), (trial.force_right, RIGHT)):
        points = []
        for index in sampled:
            x = base.time_to_x(
                float(trial.force_time[index]),
                duration,
                force_x,
                force_w,
            )
            y = int(round(force_y + force_h - float(values[index]) / force_y_max * force_h))
            points.append((x, int(np.clip(y, force_y, force_y + force_h))))
        cv2.polylines(
            panel,
            [np.asarray(points, dtype=np.int32)],
            False,
            color,
            2,
            cv2.LINE_AA,
        )

    tick = 0.0
    while tick <= duration + 1e-9:
        x = base.time_to_x(tick, duration, force_x, force_w)
        cv2.line(panel, (x, force_y), (x, force_y + force_h), GRID, 1, cv2.LINE_AA)
        base.draw_text(panel, f"{tick:.1f}", (x - 11, force_y + force_h + 18), 0.30, MUTED)
        tick += 0.5
    cv2.rectangle(
        panel,
        (force_x, force_y),
        (force_x + force_w, force_y + force_h),
        (145, 145, 145),
        1,
        cv2.LINE_AA,
    )

    base.draw_text(panel, "Force-aligned passage timeline", (20, 520), 0.52, TEXT, 2)
    rows = {
        "C LEFT": (555, LEFT),
        "FORCE LEFT": (595, FORCE),
        "C RIGHT": (655, RIGHT),
        "FORCE RIGHT": (695, FORCE),
    }
    row_height = 28
    for label, (row_y, _color) in rows.items():
        base.draw_text(panel, label, (16, row_y + 20), 0.33, MUTED, 1)
        cv2.rectangle(
            panel,
            (timeline_x, row_y),
            (timeline_x + timeline_w, row_y + row_height),
            (238, 238, 238),
            -1,
        )

    frames = trial.labels["video_frame"].to_numpy(dtype=int)
    for side, column, row_label, color in (
        ("left", "C_Temporal_left", "C LEFT", LEFT),
        ("right", "C_Temporal_right", "C RIGHT", RIGHT),
    ):
        labels = trial.labels[column].to_numpy(dtype=bool)
        for first_frame, last_frame in base.contiguous_runs(frames, labels):
            start_s = (
                first_frame / FPS
                - trial.markerless_start_s
                - trial.marker_source_start_s
                - 0.5 / FPS
            )
            end_s = (
                last_frame / FPS
                - trial.markerless_start_s
                - trial.marker_source_start_s
                + 0.5 / FPS
            )
            x1 = base.time_to_x(start_s, duration, timeline_x, timeline_w)
            x2 = base.time_to_x(end_s, duration, timeline_x, timeline_w)
            y = rows[row_label][0]
            cv2.rectangle(
                panel,
                (x1, y),
                (x2, y + row_height),
                color,
                -1,
                cv2.LINE_AA,
            )
        force_row = "FORCE LEFT" if side == "left" else "FORCE RIGHT"
        for event in trial.events[trial.events["side"] == side].itertuples(index=False):
            x1 = base.time_to_x(
                float(event.force_start_s),
                duration,
                timeline_x,
                timeline_w,
            )
            x2 = base.time_to_x(
                float(event.force_end_s),
                duration,
                timeline_x,
                timeline_w,
            )
            y = rows[force_row][0]
            cv2.rectangle(
                panel,
                (x1, y),
                (x2, y + row_height),
                FORCE,
                -1,
                cv2.LINE_AA,
            )
            cv2.rectangle(
                panel,
                (x1, y),
                (x2, y + row_height),
                ACTIVE_DARK,
                1,
                cv2.LINE_AA,
            )

    for tick in np.arange(0.0, duration + 0.001, 0.5):
        x = base.time_to_x(float(tick), duration, timeline_x, timeline_w)
        cv2.line(panel, (x, 548), (x, 730), GRID, 1, cv2.LINE_AA)
        base.draw_text(panel, f"{tick:.1f}", (x - 11, 758), 0.30, MUTED, 1)
    base.draw_text(panel, "aligned time (s)", (262, 784), 0.32, MUTED, 1)

    base.draw_text(
        panel,
        "Force used for evaluation only.",
        (20, 910),
        0.43,
        MUTED,
        1,
    )
    base.draw_text(
        panel,
        "Outside measured events is not scored as a negative.",
        (20, 940),
        0.38,
        MUTED,
        1,
    )
    base.draw_text(
        panel,
        "P1 = left | P2 = right | threshold = 20 N",
        (20, 970),
        0.36,
        MUTED,
        1,
    )
    return ContactPanel(
        static=panel,
        force_plot=force_plot,
        timeline_plot=timeline_plot,
        force_y_max=force_y_max,
    )


def update_contact_panel(
    panel_cache: ContactPanel,
    trial: base.TrialData,
    row: pd.Series,
) -> np.ndarray:
    panel = panel_cache.static.copy()
    frame = int(row["video_frame"])
    aligned_time = float(row["aligned_reference_time_s"])
    left_contact = bool(row["C_Temporal_left"])
    right_contact = bool(row["C_Temporal_right"])
    state = base.state_name(left_contact, right_contact)

    base.draw_text(
        panel,
        f"Walk {trial.trial_number} | frame {frame} | aligned {aligned_time:0.3f} s",
        (22, 72),
        0.43,
        MUTED,
        1,
    )
    base.rounded_badge(panel, (20, 88, 172, 50), "LEFT CONTACT", left_contact, LEFT)
    base.rounded_badge(
        panel,
        (207, 88, 182, 50),
        "RIGHT CONTACT",
        right_contact,
        RIGHT,
    )
    state_color = (
        LEFT
        if state == "LEFT"
        else RIGHT if state == "RIGHT" else ACTIVE_DARK if state == "BOTH" else NONE
    )
    base.rounded_badge(
        panel,
        (404, 88, 185, 50),
        f"STATE: {state}",
        state != "NONE",
        state_color,
    )

    force_x, force_y, force_w, force_h = panel_cache.force_plot
    cursor_x = base.time_to_x(
        aligned_time,
        trial.aligned_duration_s,
        force_x,
        force_w,
    )
    cv2.line(
        panel,
        (cursor_x, force_y),
        (cursor_x, force_y + force_h),
        ACTIVE_DARK,
        2,
        cv2.LINE_AA,
    )
    left_force = float(np.interp(aligned_time, trial.force_time, trial.force_left))
    right_force = float(np.interp(aligned_time, trial.force_time, trial.force_right))
    base.draw_text(
        panel,
        f"current: L {left_force:0.1f} N | R {right_force:0.1f} N",
        (20, 204),
        0.36,
        ACTIVE_DARK,
        1,
    )

    timeline_x, timeline_y, timeline_w, timeline_h = panel_cache.timeline_plot
    timeline_cursor = base.time_to_x(
        aligned_time,
        trial.aligned_duration_s,
        timeline_x,
        timeline_w,
    )
    cv2.line(
        panel,
        (timeline_cursor, timeline_y - 8),
        (timeline_cursor, timeline_y + timeline_h),
        ACTIVE_DARK,
        3,
        cv2.LINE_AA,
    )

    measured_now = any(
        float(event.force_start_s) <= aligned_time <= float(event.force_end_s)
        for event in trial.events.itertuples(index=False)
    )
    status = (
        "Measured plate-contact event: included in comparison"
        if measured_now
        else "Outside measured plate-contact events: not scored"
    )
    base.draw_text(
        panel,
        status,
        (20, 845),
        0.38,
        ACTIVE_DARK if measured_now else MUTED,
        1,
    )
    return panel


def compose_frame(
    trial: base.TrialData,
    panel_cache: ContactPanel,
    raw_frame: np.ndarray,
    mesh_frame: np.ndarray,
    row: pd.Series,
    keypoints: np.ndarray,
) -> np.ndarray:
    canvas = np.full((HEIGHT, WIDTH, 3), BACKGROUND, dtype=np.uint8)
    for panel_x in PANEL_X:
        panel_background(canvas, panel_x)

    frame = int(row["video_frame"])
    aligned_time = float(row["aligned_reference_time_s"])

    centered_panel_text(
        canvas,
        PANEL_X[0],
        "(a) Monocular video + 2D pose",
        76,
        0.61,
        TEXT,
        2,
    )
    centered_panel_text(
        canvas,
        PANEL_X[1],
        "(b) FootMR mesh reconstruction",
        76,
        0.61,
        TEXT,
        2,
    )
    raw_rect = fit_frame(canvas, raw_frame, PANEL_X[0])
    mesh_rect = fit_frame(canvas, mesh_frame, PANEL_X[1])
    draw_pose_overlay(
        canvas,
        keypoints,
        raw_rect,
    )
    draw_video_footer(
        canvas,
        raw_rect,
        f"source frame {frame} | aligned {aligned_time:0.3f} s",
    )
    draw_video_footer(
        canvas,
        mesh_rect,
        f"source frame {frame} | synchronized mesh",
    )

    contact_panel = update_contact_panel(panel_cache, trial, row)
    canvas[
        PANEL_Y : PANEL_Y + PANEL_HEIGHT,
        PANEL_X[2] : PANEL_X[2] + PANEL_WIDTH,
    ] = contact_panel
    cv2.rectangle(
        canvas,
        (PANEL_X[2], PANEL_Y),
        (PANEL_X[2] + PANEL_WIDTH, PANEL_Y + PANEL_HEIGHT),
        (205, 205, 205),
        1,
        cv2.LINE_AA,
    )
    return canvas


def make_title_card(
    title: str,
    subtitle: str,
    lines: list[str],
) -> np.ndarray:
    return base.make_title_card(title, subtitle, lines)


def render_videos(
    trials: list[base.TrialData],
    mesh_info: dict[str, MeshInfo],
    c_summary: pd.Series,
    output_dir: Path,
    crf: int,
) -> tuple[list[Path], Path, Path, dict[str, int]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    walks_dir = output_dir / "walks"
    walks_dir.mkdir(parents=True, exist_ok=True)
    panel_cache = {trial.trial: make_contact_panel(trial) for trial in trials}

    combined_temp = output_dir / ".foot_contact.tmp.mp4"
    combined_output = output_dir / "foot_contact_appendix.mp4"
    combined_writer = base.open_writer(combined_temp)
    combined_frames = 0

    intro = make_title_card(
        "C_Temporal Foot-Contact Estimation",
        "2D pose | FootMR mesh | Temporal contact",
        [
            "S18 Walk 1-5 | 30 fps monocular video",
            "Raw pose and mesh use the same source frame index.",
            "Measured force is shown for evaluation only.",
        ],
    )
    for _ in range(INTRO_FRAMES):
        combined_writer.write(intro)
        combined_frames += 1

    output_paths: list[Path] = []
    frame_counts: dict[str, int] = {}
    qc_indices = {"intro": INTRO_FRAMES // 2}
    preview_path = output_dir / "preview.png"
    preview_written = False

    for trial in trials:
        title_card = make_title_card(
            f"Walk {trial.trial_number}",
            (
                f"frames {trial.start_frame}-{trial.end_frame} | "
                f"force-plate passage | two measured stance events"
            ),
            [
                "(a) video-derived 2D pose",
                "(b) FootMR camera-space mesh",
                "(c) C_Temporal contact and measured force",
            ],
        )
        for _ in range(TRIAL_CARD_FRAMES):
            combined_writer.write(title_card)
            combined_frames += 1

        expected_frames = trial.end_frame - trial.start_frame + 1
        qc_indices[f"walk_{trial.trial_number}"] = combined_frames + expected_frames // 2
        temp_path = walks_dir / f".walk_{trial.trial_number}.tmp.mp4"
        output_path = walks_dir / f"walk_{trial.trial_number}.mp4"
        writer = base.open_writer(temp_path)
        raw_capture = cv2.VideoCapture(str(trial.video_path))
        mesh_capture = cv2.VideoCapture(str(mesh_info[trial.trial].path))
        if not raw_capture.isOpened() or not mesh_capture.isOpened():
            raise RuntimeError(f"{trial.trial}: raw 또는 mesh 영상을 열 수 없습니다")

        frame_count = 0
        for row in trial.labels.itertuples(index=False):
            frame_index = int(row.video_frame)
            raw_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            mesh_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            raw_success, raw_frame = raw_capture.read()
            mesh_success, mesh_frame = mesh_capture.read()
            if not raw_success or not mesh_success:
                raise RuntimeError(f"{trial.trial}: source frame {frame_index} 읽기 실패")
            rendered = compose_frame(
                trial,
                panel_cache[trial.trial],
                raw_frame,
                mesh_frame,
                pd.Series(row._asdict()),
                trial.keypoints[frame_index],
            )
            writer.write(rendered)
            combined_writer.write(rendered)
            combined_frames += 1
            frame_count += 1

            if (
                not preview_written
                and trial.trial_number == 3
                and bool(row.C_Temporal_left)
                and float(row.aligned_reference_time_s) >= 1.8
            ):
                if not cv2.imwrite(str(preview_path), rendered):
                    raise RuntimeError("preview 저장 실패")
                preview_written = True

        raw_capture.release()
        mesh_capture.release()
        writer.release()
        if frame_count != expected_frames:
            raise RuntimeError(f"{trial.trial}: 출력 frame 수 불일치")
        base.transcode_h264(temp_path, output_path, crf)
        output_paths.append(output_path)
        frame_counts[trial.trial] = frame_count

    outro = make_title_card(
        "Force-plate paired summary",
        "10 measured events from one participant | internal pilot",
        [
            f"Events detected: {int(c_summary['events_detected'])}/{int(c_summary['events'])}",
            f"Force-pulse recall: {float(c_summary['pulse_recall_macro']):0.3f}",
            f"Temporal IoU: {float(c_summary['temporal_iou_macro']):0.3f}",
            (
                f"Onset MAE: {float(c_summary['onset_mae_ms_detected']):0.1f} ms | "
                f"Offset MAE: {float(c_summary['offset_mae_ms_detected']):0.1f} ms"
            ),
            (
                f"Mean bias: onset {float(c_summary['onset_bias_ms_detected']):+0.1f} ms | "
                f"offset {float(c_summary['offset_bias_ms_detected']):+0.1f} ms"
            ),
        ],
    )
    qc_indices["summary"] = combined_frames + OUTRO_FRAMES // 2
    for _ in range(OUTRO_FRAMES):
        combined_writer.write(outro)
        combined_frames += 1
    combined_writer.release()
    base.transcode_h264(combined_temp, combined_output, crf)
    base.extract_qc_frames(combined_output, output_dir / "qc_frames", qc_indices)

    if not preview_written:
        raise RuntimeError("preview frame을 선택하지 못했습니다")
    frame_counts["combined"] = combined_frames
    return output_paths, combined_output, preview_path, frame_counts


def audio_stream_count(path: Path) -> int:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a",
        "-show_entries",
        "stream=index",
        "-of",
        "csv=p=0",
        str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return len([line for line in result.stdout.splitlines() if line.strip()])


def validate_outputs(
    output_dir: Path,
    trials: list[base.TrialData],
    mesh_info: dict[str, MeshInfo],
    walk_outputs: list[Path],
    combined_output: Path,
    preview_path: Path,
    frame_counts: dict[str, int],
) -> tuple[dict[str, object], dict[str, object]]:
    outputs = [*walk_outputs, combined_output]
    output_probes = {path.name: base.probe_video(path) for path in outputs}
    checks: dict[str, bool] = {}

    for path in outputs:
        probe = output_probes[path.name]
        prefix = path.stem
        checks[f"{prefix}:h264"] = probe["codec_name"] == "h264"
        checks[f"{prefix}:yuv420p"] = probe["pix_fmt"] == "yuv420p"
        checks[f"{prefix}:1920x1080"] = (
            int(probe["width"]) == WIDTH and int(probe["height"]) == HEIGHT
        )
        checks[f"{prefix}:cfr30"] = (
            probe["r_frame_rate"] == "30/1" and probe["avg_frame_rate"] == "30/1"
        )
        checks[f"{prefix}:audio_absent"] = audio_stream_count(path) == 0
        checks[f"{prefix}:decodable_nonempty"] = path.is_file() and path.stat().st_size > 0

    for trial, path in zip(trials, walk_outputs):
        info = mesh_info[trial.trial]
        expected_frames = frame_counts[trial.trial]
        checks[f"{trial.trial}:raw_mesh_frame_match"] = info.frames == trial.source_frames
        checks[f"{trial.trial}:raw_mesh_fps_match"] = math.isclose(
            info.fps,
            trial.source_fps,
            abs_tol=1e-6,
        )
        checks[f"{trial.trial}:raw_mesh_resolution_match"] = (
            info.width,
            info.height,
        ) == (trial.source_width, trial.source_height)
        checks[f"{trial.trial}:output_frame_count"] = (
            int(output_probes[path.name]["nb_frames"]) == expected_frames
        )
        checks[f"{trial.trial}:label_coverage"] = (
            len(trial.labels) == expected_frames
            and trial.labels["C_Temporal_left"].notna().all()
            and trial.labels["C_Temporal_right"].notna().all()
        )
        checks[f"{trial.trial}:pose_frame_coverage"] = len(
            trial.keypoints
        ) == trial.source_frames and trial.keypoints.shape[1:] == (23, 3)
        checks[f"{trial.trial}:force_event_count"] = len(trial.events) == 2

        for event in trial.events.itertuples(index=False):
            values = trial.force_left if event.side == "left" else trial.force_right
            active = values > FORCE_THRESHOLD_N
            starts = np.flatnonzero(active & ~np.r_[False, active[:-1]])
            ends = np.flatnonzero(active & ~np.r_[active[1:], False])
            runs = [
                (float(trial.force_time[start]), float(trial.force_time[end]))
                for start, end in zip(starts, ends)
                if end - start + 1 >= 30
            ]
            checks[f"{trial.trial}:P{int(event.plate)}_{event.side}_boundary"] = any(
                math.isclose(start, float(event.force_start_s), abs_tol=1e-9)
                and math.isclose(end, float(event.force_end_s), abs_tol=1e-9)
                for start, end in runs
            )

    checks["combined:frame_count"] = (
        int(output_probes[combined_output.name]["nb_frames"]) == frame_counts["combined"]
    )
    preview = cv2.imread(str(preview_path))
    checks["preview:valid"] = preview is not None and preview.shape[:2] == (HEIGHT, WIDTH)
    qc_frames = sorted((output_dir / "qc_frames").glob("*.png"))
    checks["qc_frames:seven_present"] = len(qc_frames) == 7
    for path in qc_frames:
        image = cv2.imread(str(path))
        checks[f"qc_frame:{path.stem}:valid"] = image is not None and image.shape[:2] == (
            HEIGHT,
            WIDTH,
        )
    checks["five_walks_present"] = len(walk_outputs) == 5
    checks["no_temporary_render_files"] = not any(output_dir.rglob("*.tmp.mp4"))
    checks = {name: bool(passed) for name, passed in checks.items()}

    failures = [name for name, passed in checks.items() if not passed]
    qc = {
        "status": "PASS" if not failures else "FAIL",
        "checks": checks,
        "failures": failures,
        "frame_sync_policy": (
            "For every output row, raw and mesh VideoCapture objects are both "
            "positioned at the exact trial.labels video_frame before reading."
        ),
        "visual_review_required": [
            "pose follows the subject throughout each accepted interval",
            "anatomical left and right colors are correct",
            "mesh and raw motion are synchronized",
            "contact and force cursors remain legible at playback size",
            "participant publication consent is confirmed before submission",
        ],
    }
    manifest = {
        "title": "C_Temporal Three-Panel Supplementary Video",
        "fps": FPS,
        "resolution": [WIDTH, HEIGHT],
        "force_threshold_n": FORCE_THRESHOLD_N,
        "panels": [
            "(a) Monocular video + 2D pose",
            "(b) FootMR mesh reconstruction",
            "(c) C_Temporal foot contact",
        ],
        "frame_counts": frame_counts,
        "inputs": [
            {
                "trial": trial.trial,
                "raw_video": str(trial.video_path),
                "mesh_video": str(mesh_info[trial.trial].path),
                "pose": str(trial.keypoint_path),
                "force": str(trial.force_path),
                "start_frame": trial.start_frame,
                "end_frame_inclusive": trial.end_frame,
                "markerless_start_s": trial.markerless_start_s,
                "marker_source_start_s": trial.marker_source_start_s,
                "aligned_duration_s": trial.aligned_duration_s,
                "raw_mesh_frame_count": trial.source_frames,
                "raw_mesh_fps": trial.source_fps,
            }
            for trial in trials
        ],
        "outputs": {
            "combined": str(combined_output),
            "walks": [str(path) for path in walk_outputs],
            "preview": str(preview_path),
            "qc_frames": [str(path) for path in qc_frames],
        },
        "ffprobe": output_probes,
    }
    (output_dir / "qc_report.json").write_text(
        json.dumps(qc, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "render_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if failures:
        raise RuntimeError("QC 실패: " + ", ".join(failures))
    return qc, manifest
