#!/usr/bin/env python3
"""Foot-contact appendix가 공유하는 입력 및 렌더링 도구입니다."""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from utils.foot_contact import contact

ROOT = Path(__file__).resolve().parents[2]
FPS = 30.0
WIDTH = 1920
HEIGHT = 1080
FORCE_THRESHOLD_N = 20.0

BACKGROUND = (246, 245, 242)
PANEL = (255, 255, 255)
TEXT = (35, 35, 35)
MUTED = (112, 112, 112)
LEFT = (178, 114, 0)  # Okabe-Ito blue, BGR
RIGHT = (0, 94, 213)  # Okabe-Ito vermilion, BGR
FORCE = (65, 178, 230)  # Okabe-Ito yellow, BGR
NONE = (190, 190, 190)
ACTIVE_DARK = (45, 45, 45)

METADATA_VIDEO_DIRS = {
    "S18_Walk_1": "S18_Walk_1_fps30_f25",
    "S18_Walk_2": "S18_Walk_2_fps30_f27",
    "S18_Walk_3": "S18_Walk_3_fps30_f25",
    "S18_Walk_4": "S18_Walk_4_fps30_f25",
    "S18_Walk_5": "S18_Walk_5_fps30_f25",
}


@dataclass
class TrialData:
    trial: str
    trial_number: int
    video_path: Path
    keypoint_path: Path
    force_path: Path
    start_frame: int
    end_frame: int
    markerless_start_s: float
    marker_source_start_s: float
    aligned_duration_s: float
    labels: pd.DataFrame
    events: pd.DataFrame
    keypoints: np.ndarray
    force_time: np.ndarray
    force_left: np.ndarray
    force_right: np.ndarray
    source_width: int
    source_height: int
    source_frames: int
    source_fps: float


def draw_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float,
    color: tuple[int, int, int] = TEXT,
    thickness: int = 1,
) -> None:
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_centered_text(
    image: np.ndarray,
    text: str,
    y: int,
    scale: float,
    color: tuple[int, int, int] = TEXT,
    thickness: int = 1,
) -> None:
    size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    draw_text(image, text, ((WIDTH - size[0]) // 2, y), scale, color, thickness)


def alpha_rectangle(
    image: np.ndarray,
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
    color: tuple[int, int, int],
    alpha: float,
) -> None:
    overlay = image.copy()
    cv2.rectangle(overlay, top_left, bottom_right, color, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0.0, image)


def draw_dashed_line(
    image: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int = 1,
    dash: int = 8,
) -> None:
    start_array = np.asarray(start, dtype=float)
    end_array = np.asarray(end, dtype=float)
    delta = end_array - start_array
    length = float(np.linalg.norm(delta))
    if length == 0:
        return
    direction = delta / length
    for offset in np.arange(0.0, length, dash * 2):
        segment_start = start_array + direction * offset
        segment_end = start_array + direction * min(offset + dash, length)
        cv2.line(
            image,
            tuple(np.rint(segment_start).astype(int)),
            tuple(np.rint(segment_end).astype(int)),
            color,
            thickness,
            cv2.LINE_AA,
        )


def rounded_badge(
    image: np.ndarray,
    rect: tuple[int, int, int, int],
    label: str,
    active: bool,
    color: tuple[int, int, int],
) -> None:
    x, y, width, height = rect
    fill = color if active else (238, 238, 238)
    cv2.rectangle(image, (x, y), (x + width, y + height), fill, -1, cv2.LINE_AA)
    cv2.rectangle(
        image,
        (x, y),
        (x + width, y + height),
        color if active else (205, 205, 205),
        2,
        cv2.LINE_AA,
    )
    text_color = (255, 255, 255) if active else MUTED
    text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
    text_x = x + (width - text_size[0]) // 2
    text_y = y + (height + text_size[1]) // 2
    draw_text(image, label, (text_x, text_y), 0.62, text_color, 2)


def read_mot(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source):
            if line.strip().lower() == "endheader":
                return pd.read_csv(path, sep=r"\s+", skiprows=line_number + 1)
    raise ValueError(f"endheader를 찾지 못했습니다: {path}")


def probe_video(path: Path) -> dict[str, object]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,pix_fmt,width,height,r_frame_rate,avg_frame_rate,nb_frames,duration",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(result.stdout)["streams"][0]


def eval_fraction(value: str) -> float:
    numerator, denominator = value.split("/")
    return float(numerator) / float(denominator)


def load_trials(sync_path: Path, force_dir: Path) -> list[TrialData]:
    """최종 렌더링 입력을 원본 inference 결과와 measured force에서 직접 읽습니다."""
    sync = contact.load_forceplate_anchored_sync(sync_path)
    metadata_root = ROOT / "outputs/focal_ablation_5_30fps/metadata"
    trials = []

    for trial, video_dir_name in METADATA_VIDEO_DIRS.items():
        video_dir = metadata_root / video_dir_name
        video_path = video_dir / "0_input_video.mp4"
        keypoint_path = video_dir / "preprocess/vitpose.pt"
        force_path = force_dir / f"{trial}_marker_based_grf_synced_trimmed.mot"
        keypoints = torch.load(
            keypoint_path,
            map_location="cpu",
            weights_only=False,
        ).numpy()
        force = read_mot(force_path)
        events = contact.extract_measured_force_events(trial, force)
        source = probe_video(video_path)
        source_frames = int(source["nb_frames"])
        source_fps = eval_fraction(str(source["avg_frame_rate"]))

        if len(keypoints) != source_frames or keypoints.shape[1:] != (23, 3):
            raise ValueError(f"{trial}: 영상과 23-keypoint 입력이 일치하지 않습니다")
        if not math.isclose(source_fps, FPS, abs_tol=1e-6):
            raise ValueError(f"{trial}: 영상이 30fps가 아닙니다")

        trial_sync = sync.loc[trial]
        trials.append(
            TrialData(
                trial=trial,
                trial_number=int(trial.rsplit("_", 1)[1]),
                video_path=video_path,
                keypoint_path=keypoint_path,
                force_path=force_path,
                start_frame=0,
                end_frame=source_frames - 1,
                markerless_start_s=float(trial_sync["forceplate_anchored_start_s"]),
                marker_source_start_s=float(trial_sync["marker_source_start_s"]),
                aligned_duration_s=float(force["time"].iloc[-1]),
                labels=pd.DataFrame(),
                events=events,
                keypoints=keypoints,
                force_time=force["time"].to_numpy(dtype=float),
                force_left=np.abs(force["ground_force1_vy"].to_numpy(dtype=float)),
                force_right=np.abs(force["ground_force2_vy"].to_numpy(dtype=float)),
                source_width=int(source["width"]),
                source_height=int(source["height"]),
                source_frames=source_frames,
                source_fps=source_fps,
            )
        )
    return trials


def time_to_x(time_s: float, duration_s: float, x: int, width: int) -> int:
    fraction = np.clip(time_s / duration_s, 0.0, 1.0)
    return int(round(x + fraction * width))


def contiguous_runs(frames: np.ndarray, labels: np.ndarray) -> list[tuple[int, int]]:
    indices = np.flatnonzero(labels.astype(bool))
    if len(indices) == 0:
        return []
    boundaries = np.flatnonzero(np.diff(indices) > 1) + 1
    return [
        (int(frames[group[0]]), int(frames[group[-1]])) for group in np.split(indices, boundaries)
    ]


def state_name(left_contact: bool, right_contact: bool) -> str:
    if left_contact and right_contact:
        return "BOTH"
    if left_contact:
        return "LEFT"
    if right_contact:
        return "RIGHT"
    return "NONE"


def make_title_card(
    title: str,
    subtitle: str,
    lines: list[str],
) -> np.ndarray:
    image = np.full((HEIGHT, WIDTH, 3), BACKGROUND, dtype=np.uint8)
    cv2.rectangle(image, (90, 85), (1830, 995), PANEL, -1, cv2.LINE_AA)
    cv2.rectangle(image, (90, 85), (110, 995), LEFT, -1)
    draw_centered_text(image, title, 305, 1.38, TEXT, 3)
    draw_centered_text(image, subtitle, 370, 0.74, MUTED, 1)
    y = 520
    for line in lines:
        draw_centered_text(image, line, y, 0.67, TEXT, 1)
        y += 62
    return image


def open_writer(path: Path) -> cv2.VideoWriter:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (WIDTH, HEIGHT),
    )
    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter를 열 수 없습니다: {path}")
    return writer


def transcode_h264(source: Path, destination: Path, crf: int) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        str(source),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-r",
        "30",
        "-fps_mode",
        "cfr",
        "-movflags",
        "+faststart",
        str(destination),
    ]
    subprocess.run(command, check=True)
    source.unlink()


def extract_qc_frames(
    video_path: Path,
    output_dir: Path,
    frame_indices: dict[str, int],
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"QC frame 추출용 영상을 열 수 없습니다: {video_path}")
    outputs = []
    for name, frame_index in frame_indices.items():
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = capture.read()
        if not success:
            raise RuntimeError(f"QC frame {frame_index} 추출 실패")
        path = output_dir / f"{name}.png"
        if not cv2.imwrite(str(path), frame):
            raise RuntimeError(f"QC image 저장 실패: {path}")
        outputs.append(path)
    capture.release()
    return outputs
