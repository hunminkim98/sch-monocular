import math
import os
from pathlib import Path
import shutil
import struct
import subprocess

import cv2
import ffmpeg
import imageio.v3 as iio
import numpy as np
import torch
from tqdm import tqdm

FOCAL_LENGTH_35MM_KEY = "com.apple.quicktime.camera.focal_length.35mm_equivalent"


def get_video_lwh(video_path):
    L, H, W, _ = iio.improps(video_path, plugin="pyav").shape
    return L, W, H


def read_video_np(video_path, start_frame=0, end_frame=-1, scale=1.0):
    """
    인자:
        video_path: str
    반환:
        frames: np.array, (N, H, W, 3) RGB, uint8
    """
    # video 경로가 없으면 FFmpeg가 오류를 발생시킵니다.
    filter_args = []
    should_check_length = False

    # 1. frame 범위 자르기
    if not (start_frame == 0 and end_frame == -1):
        if end_frame == -1:
            filter_args.append(("trim", f"start_frame={start_frame}"))
        else:
            should_check_length = True
            filter_args.append(("trim", f"start_frame={start_frame}:end_frame={end_frame}"))

    # 2. scale 조정
    if scale != 1.0:
        filter_args.append(("scale", f"iw*{scale}:ih*{scale}"))

    # 실행 후 길이를 확인합니다.
    frames = iio.imread(video_path, plugin="pyav", filter_sequence=filter_args)
    if should_check_length:
        assert len(frames) == end_frame - start_frame

    return frames


def get_video_reader(video_path):
    return iio.imiter(video_path, plugin="pyav")


def read_images_np(image_paths, verbose=False):
    """
    인자:
        image_paths: list of str
    반환:
        images: np.array, (N, H, W, 3) RGB, uint8
    """
    if verbose:
        images = [cv2.imread(str(img_path))[..., ::-1] for img_path in tqdm(image_paths)]
    else:
        images = [cv2.imread(str(img_path))[..., ::-1] for img_path in image_paths]
    images = np.stack(images, axis=0)
    return images


def save_video(images, video_path, fps=30, crf=17):
    """
    인자:
        images: (N, H, W, 3) RGB, uint8
        crf: 17은 시각적으로 무손실이며 기본값은 23입니다. 6이 증가할 때마다
            bitrate가 절반으로 줄어듭니다. 0은 완전 무손실입니다.
            https://trac.ffmpeg.org/wiki/Encode/H.264#crf
    """
    if isinstance(images, torch.Tensor):
        images = images.cpu().numpy().astype(np.uint8)
    elif isinstance(images, list):
        images = np.array(images).astype(np.uint8)

    with iio.imopen(video_path, "w", plugin="pyav") as writer:
        writer.init_video_stream("libx264", fps=fps)
        writer._video_stream.options = {"crf": str(crf)}
        writer.write(images)


def _run_video_command(command):
    """FFmpeg 계열 명령을 실행하고 stderr를 포함해 오류를 전달합니다."""
    try:
        return subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as error:
        raise RuntimeError(f"필수 실행 파일을 찾을 수 없습니다: {command[0]}") from error
    except subprocess.CalledProcessError as error:
        stderr = (error.stderr or "").strip()
        raise RuntimeError(f"{command[0]} 실행 실패: {stderr}") from error


def _iter_mov_boxes(file, start, end):
    """지정된 범위의 ISO BMFF box를 순회합니다."""
    offset = start
    while offset + 8 <= end:
        file.seek(offset)
        header = file.read(8)
        if len(header) != 8:
            return
        size, box_type = struct.unpack(">I4s", header)
        header_size = 8
        if size == 1:
            extended_size = file.read(8)
            if len(extended_size) != 8:
                return
            size = struct.unpack(">Q", extended_size)[0]
            header_size = 16
        elif size == 0:
            size = end - offset

        box_end = offset + size
        if size < header_size or box_end > end:
            return
        yield box_type, offset + header_size, box_end
        offset = box_end


def _find_mov_box(file, start, end, box_type):
    for current_type, payload_start, box_end in _iter_mov_boxes(file, start, end):
        if current_type == box_type:
            return payload_start, box_end
    return None


def _iter_mov_meta_boxes(file, start, end):
    for box_type, payload_start, box_end in _iter_mov_boxes(file, start, end):
        if box_type == b"meta":
            yield payload_start, box_end
        elif box_type in {b"trak", b"udta"}:
            yield from _iter_mov_meta_boxes(file, payload_start, box_end)


def _read_mdta_key_index(file, keys_start, keys_end):
    file.seek(keys_start)
    header = file.read(8)
    if len(header) != 8:
        return None
    _, entry_count = struct.unpack(">II", header)
    if entry_count > 4096:
        return None

    for index in range(1, entry_count + 1):
        entry_start = file.tell()
        entry_header = file.read(8)
        if len(entry_header) != 8:
            return None
        entry_size, namespace = struct.unpack(">I4s", entry_header)
        entry_end = entry_start + entry_size
        if entry_size < 8 or entry_end > keys_end:
            return None
        key = file.read(entry_size - 8).decode("utf-8")
        if namespace == b"mdta" and key == FOCAL_LENGTH_35MM_KEY:
            return index
        file.seek(entry_end)
    return None


def _read_utf8_mdta_value(file, ilst_start, ilst_end, key_index):
    item = _find_mov_box(file, ilst_start, ilst_end, struct.pack(">I", key_index))
    if item is None:
        return None
    item_start, item_end = item
    data = _find_mov_box(file, item_start, item_end, b"data")
    if data is None:
        return None
    data_start, data_end = data
    if data_end - data_start < 8:
        return None

    file.seek(data_start)
    data_header = file.read(8)
    if len(data_header) != 8:
        return None
    data_type, _ = struct.unpack(">II", data_header)
    if data_type != 1:
        return None
    return file.read(data_end - data_start - 8).decode("utf-8").strip("\x00").strip()


def read_focal_length_35mm_equivalent(video_path):
    """Apple QuickTime mdta에서 검증된 35mm-equivalent focal을 읽습니다."""
    try:
        path = Path(video_path)
        with path.open("rb") as file:
            file.seek(0, os.SEEK_END)
            file_size = file.tell()
            moov = _find_mov_box(file, 0, file_size, b"moov")
            if moov is None:
                return None
            moov_start, moov_end = moov
            for meta_start, meta_end in _iter_mov_meta_boxes(file, moov_start, moov_end):
                # ISO BMFF meta에는 version/flags가 있지만 Apple QuickTime meta에는 없을 수 있습니다.
                keys = ilst = None
                for children_start in (meta_start, meta_start + 4):
                    keys = _find_mov_box(file, children_start, meta_end, b"keys")
                    ilst = _find_mov_box(file, children_start, meta_end, b"ilst")
                    if keys is not None and ilst is not None:
                        break
                if keys is None or ilst is None:
                    continue
                key_index = _read_mdta_key_index(file, *keys)
                if key_index is None:
                    continue
                value_text = _read_utf8_mdta_value(file, *ilst, key_index)
                if value_text is not None:
                    value = float(value_text)
                    break
            else:
                return None
    except (OSError, UnicodeDecodeError, ValueError, struct.error):
        return None

    if not math.isfinite(value) or not 10.0 <= value <= 300.0:
        return None
    return value


def resolve_focal_length_35mm(video_path, requested_f_mm=None):
    """CLI, 원본 메타데이터, 기본 추정값 순으로 focal을 선택합니다."""
    if requested_f_mm is not None:
        return float(requested_f_mm), "cli"
    metadata_f_mm = read_focal_length_35mm_equivalent(video_path)
    if metadata_f_mm is not None:
        return metadata_f_mm, "metadata"
    return None, "default"


def normalize_video_to_30fps(source_path, output_path, crf=23):
    """원본 재생시간을 유지하며 FootMR용 CFR 30fps 영상으로 정규화합니다."""
    source_path = Path(source_path).resolve()
    output_path = Path(output_path).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"입력 영상을 찾을 수 없습니다: {source_path}")
    if source_path == output_path:
        raise ValueError("원본 영상과 정규화 영상의 경로는 달라야 합니다.")

    if output_path.is_file():
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.parent / f".{output_path.stem}.tmp{output_path.suffix}"
    try:
        _run_video_command(
            [
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-autorotate",
                "1",
                "-i",
                str(source_path),
                "-map",
                "0:v:0",
                "-an",
                "-vf",
                "setpts=PTS-STARTPTS,fps=30,format=yuv420p",
                "-fps_mode",
                "cfr",
                "-c:v",
                "libx264",
                "-crf",
                str(crf),
                "-preset",
                "medium",
                str(temp_path),
            ]
        )
        os.replace(temp_path, output_path)
    finally:
        temp_path.unlink(missing_ok=True)


def get_writer(video_path, fps=30, crf=17):
    """사용 후 반드시 ``.close()``를 호출해야 합니다."""
    writer = iio.imopen(video_path, "w", plugin="pyav")
    writer.init_video_stream("libx264", fps=fps)
    writer._video_stream.options = {"crf": str(crf)}
    return writer


def copy_file(video_path, out_video_path, overwrite=True):
    if not overwrite and Path(out_video_path).exists():
        return
    shutil.copy(video_path, out_video_path)


def merge_videos_horizontal(in_video_paths: list, out_video_path: str):
    if len(in_video_paths) < 2:
        raise ValueError("At least two video paths are required for merging.")
    inputs = [ffmpeg.input(path) for path in in_video_paths]
    merged_video = ffmpeg.filter(inputs, "hstack", inputs=len(inputs))
    output = ffmpeg.output(merged_video, out_video_path)
    try:
        ffmpeg.run(output, overwrite_output=True, quiet=True)
    except ffmpeg.Error as e:
        print(e.stderr)


def merge_videos_vertical(in_video_paths: list, out_video_path: str):
    if len(in_video_paths) < 2:
        raise ValueError("At least two video paths are required for merging.")
    inputs = [ffmpeg.input(path) for path in in_video_paths]
    merged_video = ffmpeg.filter(inputs, "vstack", inputs=len(inputs))
    output = ffmpeg.output(merged_video, out_video_path)
    ffmpeg.run(output, overwrite_output=True, quiet=True)
