import json
import shutil
import subprocess

import numpy as np
import pytest

from hmr4d.utils import video_io_utils
from hmr4d.utils.video_io_utils import (
    FOCAL_LENGTH_35MM_KEY,
    normalize_video_to_30fps,
    read_focal_length_35mm_equivalent,
    resolve_focal_length_35mm,
)

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="FFmpeg와 ffprobe가 필요합니다.",
)


def run_command(command):
    return subprocess.run(command, check=True, capture_output=True, text=True)


def create_test_video(path, fps=60, duration=2, size="160x96", metadata=None):
    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"testsrc=size={size}:rate={fps}:duration={duration}",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
    ]
    if metadata:
        command.extend(["-movflags", "use_metadata_tags"])
        for key, value in metadata.items():
            command.extend(["-metadata", f"{key}={value}"])
    command.append(str(path))
    run_command(command)


def probe_video(path, show_frames=False):
    entries = (
        "stream=avg_frame_rate,nb_read_frames,width,height:"
        "stream_side_data=rotation:format=duration"
    )
    if show_frames:
        entries += ":frame=best_effort_timestamp_time"
    result = run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            entries,
            "-of",
            "json",
            str(path),
        ]
    )
    return json.loads(result.stdout)


def test_normalize_60fps_video_to_cfr_30fps(tmp_path):
    source = tmp_path / "source_60fps.mp4"
    output = tmp_path / "canonical_30fps.mp4"
    create_test_video(source)
    source_bytes = source.read_bytes()

    normalize_video_to_30fps(source, output, crf=23)
    probe = probe_video(output, show_frames=True)
    stream = probe["streams"][0]
    timestamps = np.asarray(
        [float(frame["best_effort_timestamp_time"]) for frame in probe["frames"]]
    )

    assert source.read_bytes() == source_bytes
    assert stream["avg_frame_rate"] == "30/1"
    assert int(stream["nb_read_frames"]) == 60
    assert float(probe["format"]["duration"]) == pytest.approx(2.0, abs=1 / 30)
    assert np.diff(timestamps) == pytest.approx(np.full(59, 1 / 30), abs=1e-6)


def test_normalize_reuses_valid_existing_output(tmp_path):
    source = tmp_path / "source.mp4"
    output = tmp_path / "canonical.mp4"
    create_test_video(source, duration=1)

    normalize_video_to_30fps(source, output)
    original_mtime = output.stat().st_mtime_ns
    normalize_video_to_30fps(source, output)

    assert output.stat().st_mtime_ns == original_mtime


def test_normalize_rejects_missing_source(tmp_path):
    with pytest.raises(FileNotFoundError):
        normalize_video_to_30fps(tmp_path / "missing.mp4", tmp_path / "output.mp4")


def test_normalize_failure_does_not_publish_partial_output(tmp_path):
    source = tmp_path / "invalid.mp4"
    output = tmp_path / "output.mp4"
    source.write_bytes(b"not a video")

    with pytest.raises(RuntimeError):
        normalize_video_to_30fps(source, output)

    assert not output.exists()
    assert list(tmp_path.glob(".*.tmp.mp4")) == []


def test_normalize_applies_rotation_to_pixels(tmp_path):
    landscape = tmp_path / "landscape.mp4"
    rotated = tmp_path / "rotated.mov"
    output = tmp_path / "canonical.mp4"
    create_test_video(landscape, fps=30, duration=1, size="160x96")
    run_command(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-display_rotation",
            "90",
            "-i",
            str(landscape),
            "-c",
            "copy",
            str(rotated),
        ]
    )

    normalize_video_to_30fps(rotated, output)
    probe = probe_video(output)
    stream = probe["streams"][0]

    assert (int(stream["width"]), int(stream["height"])) == (96, 160)
    assert "side_data_list" not in stream


def test_reads_verified_35mm_equivalent_metadata(tmp_path):
    source = tmp_path / "metadata.mov"
    create_test_video(source, metadata={FOCAL_LENGTH_35MM_KEY: "25"})

    assert read_focal_length_35mm_equivalent(source) == pytest.approx(25.0)


def test_rejects_physical_focal_from_lens_model(tmp_path):
    source = tmp_path / "physical_focal_only.mov"
    create_test_video(
        source,
        metadata={"com.apple.quicktime.camera.lens_model": "iPhone back camera 6.86mm f/1.78"},
    )

    assert read_focal_length_35mm_equivalent(source) is None


def test_rejects_out_of_range_35mm_equivalent_metadata(tmp_path):
    source = tmp_path / "invalid_metadata.mov"
    create_test_video(source, metadata={FOCAL_LENGTH_35MM_KEY: "6.86"})

    assert read_focal_length_35mm_equivalent(source) is None


def test_explicit_focal_takes_precedence_over_metadata(monkeypatch, tmp_path):
    source = tmp_path / "source.mov"
    monkeypatch.setattr(video_io_utils, "read_focal_length_35mm_equivalent", lambda path: 25.0)

    assert resolve_focal_length_35mm(source, requested_f_mm=27) == (27.0, "cli")


def test_resolve_focal_uses_original_video_metadata(monkeypatch, tmp_path):
    source = tmp_path / "original.mov"
    received_paths = []

    def read_metadata(path):
        received_paths.append(path)
        return 25.0

    monkeypatch.setattr(video_io_utils, "read_focal_length_35mm_equivalent", read_metadata)

    assert resolve_focal_length_35mm(source) == (25.0, "metadata")
    assert received_paths == [source]


def test_resolve_focal_falls_back_to_default(monkeypatch, tmp_path):
    source = tmp_path / "no_metadata.mp4"
    monkeypatch.setattr(video_io_utils, "read_focal_length_35mm_equivalent", lambda path: None)

    assert resolve_focal_length_35mm(source) == (None, "default")
