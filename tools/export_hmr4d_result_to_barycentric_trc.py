#!/usr/bin/env python3
"""FootMR SMPL 결과를 barycentric 가상 마커의 TRC 형식으로 내보냅니다.

이 exporter는 ``hmr4d_results.pt``에 저장된 SMPL-X 파라미터를 읽고, 생성된
vertex를 SMPL topology로 변환한 뒤 설정된 barycentric 마커를 샘플링하여
TRC, NPZ, JSON 보고서 파일을 작성합니다.

좌표는 결과에 저장된 그대로 출력합니다. 이 스크립트는 FootMR post-processing,
marker grounding, height scaling, 회전, filtering 또는 trial 전용 동기화를
추가로 적용하지 않습니다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_MARKER_YAML = (
    REPO_ROOT
    / "assets"
    / "marker_mappings"
    / "soma_sch_v1_smpl_barycentric.yaml"
)
SMPLX_TO_SMPL = REPO_ROOT / "hmr4d" / "utils" / "body_model" / "smplx2smpl_sparse.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--result",
        type=Path,
        required=True,
        help="FootMR hmr4d_results.pt file.",
    )
    parser.add_argument(
        "--root-key",
        choices=["smpl_params_global", "smpl_params_incam"],
        default="smpl_params_global",
        help="SMPL parameter dictionary to export.",
    )
    parser.add_argument(
        "--marker-yaml",
        type=Path,
        default=DEFAULT_MARKER_YAML,
        help="Barycentric marker mapping YAML.",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help=(
            "Video used to read the TRC frame rate. If omitted, the exporter "
            "looks for 0_input_video.mp4 beside the result."
        ),
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Explicit TRC frame rate. Overrides the video's reported frame rate.",
    )
    parser.add_argument(
        "--allow-video-frame-mismatch",
        action="store_true",
        help="Allow video and result frame counts to differ.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to a trc directory beside the result.",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Output filename prefix. Defaults to the result filename stem.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=96,
        help="Frames per SMPL-X forward pass.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device used for mesh generation.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def safe_torch_load(path: Path) -> Any:
    """임의 객체를 허용하지 않고 tensor 전용 프로젝트 artifact를 불러옵니다."""

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def normalize_result_params(
    data: dict[str, Any], result_path: Path, root_key: str
) -> dict[str, torch.Tensor]:
    if root_key not in data:
        raise KeyError(
            f"{root_key} not found in {result_path}; available keys={list(data)}"
        )
    raw_params = data[root_key]
    if not isinstance(raw_params, dict):
        raise TypeError(f"{root_key} must be a dictionary")

    required = ("body_pose", "betas", "global_orient", "transl")
    missing = [key for key in required if key not in raw_params]
    if missing:
        raise KeyError(f"{root_key} is missing required fields: {missing}")

    params: dict[str, torch.Tensor] = {}
    for key in required:
        value = raw_params[key]
        if not torch.is_tensor(value):
            raise TypeError(f"{root_key}.{key} must be a tensor")
        value = value.detach().cpu().float()
        if value.ndim == 3 and value.shape[0] == 1:
            value = value[0]
        params[key] = value.contiguous()

    body_pose = params["body_pose"]
    if body_pose.ndim != 2 or body_pose.shape[1] != 63:
        raise ValueError(
            f"{root_key}.body_pose must have shape [T, 63], "
            f"got {tuple(body_pose.shape)}"
        )
    frame_count = body_pose.shape[0]
    if frame_count == 0:
        raise ValueError("Cannot export an empty result")

    for key, width in (("global_orient", 3), ("transl", 3)):
        value = params[key]
        if value.ndim != 2 or value.shape != (frame_count, width):
            raise ValueError(
                f"{root_key}.{key} must have shape [{frame_count}, {width}], "
                f"got {tuple(value.shape)}"
            )

    betas = params["betas"]
    if betas.ndim == 1:
        betas = betas.unsqueeze(0)
    if betas.ndim != 2 or betas.shape[1] < 10:
        raise ValueError(
            f"{root_key}.betas must have shape [T, >=10], [1, >=10], or [>=10], "
            f"got {tuple(betas.shape)}"
        )
    if betas.shape[0] == 1:
        betas = betas.expand(frame_count, -1)
    elif betas.shape[0] != frame_count:
        raise ValueError(
            f"{root_key}.betas has {betas.shape[0]} frames; expected {frame_count}"
        )
    params["betas"] = betas[:, :10].contiguous()

    for key, value in params.items():
        if not torch.isfinite(value).all():
            raise ValueError(f"{root_key}.{key} contains non-finite values")
    return params


def load_marker_mapping(
    path: Path,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    require_file(path, "marker mapping YAML")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("markers"), dict):
        raise ValueError(f"Invalid marker mapping structure: {path}")

    labels: list[str] = []
    face_indices: list[int] = []
    face_vertices: list[list[int]] = []
    weights: list[list[float]] = []

    for label, spec in data["markers"].items():
        if not isinstance(spec, dict):
            raise TypeError(f"{label}: marker definition must be a dictionary")
        if spec.get("kind") != "barycentric":
            raise ValueError(f"{label}: only barycentric markers are supported")

        vertices = [int(value) for value in spec["smpl_face_vertices"]]
        marker_weights = [float(value) for value in spec["barycentric_weights"]]
        if len(vertices) != 3 or len(marker_weights) != 3:
            raise ValueError(f"{label}: expected three face vertices and weights")
        if min(vertices) < 0:
            raise ValueError(f"{label}: vertex indices must be non-negative")
        if not np.isfinite(marker_weights).all():
            raise ValueError(f"{label}: barycentric weights must be finite")
        if abs(sum(marker_weights) - 1.0) > 1e-5:
            raise ValueError(
                f"{label}: barycentric weights do not sum to one: {marker_weights}"
            )

        labels.append(str(label))
        face_indices.append(int(spec["smpl_face_index"]))
        face_vertices.append(vertices)
        weights.append(marker_weights)

    if not labels:
        raise ValueError("Marker mapping contains no markers")

    return (
        labels,
        np.asarray(face_indices, dtype=np.int64),
        np.asarray(face_vertices, dtype=np.int64),
        np.asarray(weights, dtype=np.float32),
        data.get("meta", {}),
    )


def video_metadata(path: Path) -> dict[str, Any]:
    require_file(path, "source video")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open source video: {path}")
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if frame_count <= 0:
            raise ValueError(f"Invalid frame count reported for {path}: {frame_count}")
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError(f"Invalid FPS reported for {path}: {fps}")
        return {
            "path": str(path),
            "frames": frame_count,
            "fps": fps,
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "duration_s": float(frame_count / fps),
        }
    finally:
        capture.release()


def resolve_timebase(
    result_path: Path,
    video_arg: Path | None,
    fps_arg: float | None,
    result_frames: int,
    allow_frame_mismatch: bool,
) -> tuple[np.ndarray, dict[str, Any], list[str]]:
    warnings: list[str] = []
    video_path: Path | None = None
    if video_arg is not None:
        video_path = resolve_path(video_arg)
    elif fps_arg is None:
        candidate = result_path.parent / "0_input_video.mp4"
        if candidate.is_file():
            video_path = candidate.resolve()

    video: dict[str, Any] | None = None
    if video_path is not None:
        video = video_metadata(video_path)
        if video["frames"] != result_frames:
            message = (
                f"video has {video['frames']} frames but result has {result_frames}"
            )
            if not allow_frame_mismatch:
                raise ValueError(
                    f"{message}; pass --allow-video-frame-mismatch to override"
                )
            warnings.append(message)

    if fps_arg is not None:
        fps = float(fps_arg)
        source = "explicit --fps"
        if video is not None and not np.isclose(fps, video["fps"], rtol=0, atol=1e-6):
            warnings.append(
                f"explicit FPS {fps} overrides video-reported FPS {video['fps']}"
            )
    elif video is not None:
        fps = float(video["fps"])
        source = "video metadata"
    else:
        raise ValueError(
            "No timebase available. Pass --video or --fps, or place "
            "0_input_video.mp4 beside the result."
        )

    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"FPS must be positive and finite, got {fps}")
    times = np.arange(result_frames, dtype=np.float64) / fps
    return (
        times,
        {
            "fps": fps,
            "source": source,
            "video": video,
            "start_time_s": float(times[0]),
            "end_time_s": float(times[-1]),
        },
        warnings,
    )


def compute_smpl_vertices(
    params: dict[str, torch.Tensor], chunk_size: int, device_name: str
) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    if chunk_size <= 0:
        raise ValueError(f"--chunk-size must be positive, got {chunk_size}")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    from hmr4d.utils.smplx_utils import make_smplx

    import hmr4d

    require_file(SMPLX_TO_SMPL, "SMPL-X-to-SMPL converter")
    smplx_model = make_smplx("supermotion").to(device).eval()
    smpl_model = make_smplx("smpl").eval()
    faces = np.asarray(smpl_model.faces, dtype=np.int64)

    converter = safe_torch_load(SMPLX_TO_SMPL)
    if not torch.is_tensor(converter) or converter.ndim != 2:
        raise TypeError(f"Invalid SMPL-X-to-SMPL converter: {SMPLX_TO_SMPL}")
    if getattr(converter, "is_sparse", False):
        converter = converter.to_dense()
    converter = converter.to(device=device, dtype=torch.float32)

    total_frames = params["body_pose"].shape[0]
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, total_frames, chunk_size):
            end = min(start + chunk_size, total_frames)
            output = smplx_model(
                body_pose=params["body_pose"][start:end].to(device),
                betas=params["betas"][start:end].to(device),
                global_orient=params["global_orient"][start:end].to(device),
                transl=params["transl"][start:end].to(device),
            )
            if converter.shape[1] != output.vertices.shape[1]:
                raise ValueError(
                    "SMPL-X-to-SMPL converter and SMPL-X model vertex counts differ: "
                    f"{converter.shape[1]} != {output.vertices.shape[1]}"
                )
            vertices = torch.einsum("sv,tvc->tsc", converter, output.vertices)
            chunks.append(vertices.detach().cpu().numpy().astype(np.float32))

    smpl_vertices = np.concatenate(chunks, axis=0)
    if faces.max() >= smpl_vertices.shape[1]:
        raise ValueError("SMPL face topology references a missing vertex")
    return (
        smpl_vertices,
        faces,
        {
            "hmr4d_module": str(Path(hmr4d.__file__).resolve()),
            "smplx_to_smpl": str(SMPLX_TO_SMPL),
        },
    )


def validate_marker_topology(
    labels: list[str],
    face_indices: np.ndarray,
    face_vertices: np.ndarray,
    faces: np.ndarray,
) -> None:
    if face_indices.min() < 0 or face_indices.max() >= len(faces):
        raise ValueError("Marker mapping references a face outside the SMPL topology")
    mapped_faces = faces[face_indices]
    mismatches = np.flatnonzero(np.any(mapped_faces != face_vertices, axis=1))
    if len(mismatches):
        index = int(mismatches[0])
        raise ValueError(
            f"{labels[index]}: mapped face vertices {face_vertices[index].tolist()} "
            f"do not match SMPL face {face_indices[index]} "
            f"{mapped_faces[index].tolist()}"
        )


def sample_markers(
    vertices: np.ndarray,
    face_vertices: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    if face_vertices.min() < 0 or face_vertices.max() >= vertices.shape[1]:
        raise ValueError("Marker mapping references a vertex outside the SMPL mesh")
    triangles = vertices[:, face_vertices, :]
    markers = np.einsum(
        "tmvc,mv->tmc", triangles, weights, optimize=True
    ).astype(np.float32)
    if not np.isfinite(markers).all():
        raise RuntimeError("Marker sampling produced non-finite coordinates")
    return markers


def coordinate_bounds(markers: np.ndarray) -> dict[str, list[float]]:
    flattened = markers.reshape(-1, 3)
    return {
        "min_xyz": flattened.min(axis=0).astype(float).tolist(),
        "max_xyz": flattened.max(axis=0).astype(float).tolist(),
    }


def write_trc(
    path: Path,
    labels: list[str],
    markers: np.ndarray,
    times: np.ndarray,
    fps: float,
) -> None:
    frame_count, marker_count, dimensions = markers.shape
    if dimensions != 3:
        raise ValueError(f"TRC markers must have three coordinates, got {dimensions}")
    if len(labels) != marker_count or len(times) != frame_count:
        raise ValueError("TRC labels, markers, and times have incompatible shapes")

    lines = [
        f"PathFileType\t4\t(X/Y/Z)\t{path.name}",
        (
            "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\t"
            "OrigDataRate\tOrigDataStartFrame\tOrigNumFrames"
        ),
        (
            f"{fps:.8f}\t{fps:.8f}\t{frame_count}\t{marker_count}\tm\t"
            f"{fps:.8f}\t1\t{frame_count}"
        ),
        "Frame#\tTime\t" + "\t\t\t".join(labels) + "\t\t",
        "\t\t"
        + "\t".join(
            f"X{index}\tY{index}\tZ{index}"
            for index in range(1, marker_count + 1)
        ),
        "",
    ]
    for frame_number, (time, frame_markers) in enumerate(
        zip(times, markers), start=1
    ):
        values = "\t".join(
            f"{coordinate:.9f}" for coordinate in frame_markers.reshape(-1)
        )
        lines.append(f"{frame_number}\t{time:.9f}\t{values}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_prefix(prefix: str) -> str:
    if not prefix or Path(prefix).name != prefix or prefix in {".", ".."}:
        raise ValueError(f"Invalid output prefix: {prefix!r}")
    return prefix


def main() -> int:
    args = parse_args()
    result_path = resolve_path(args.result)
    marker_yaml = resolve_path(args.marker_yaml)
    require_file(result_path, "FootMR result")

    result_data = safe_torch_load(result_path)
    if not isinstance(result_data, dict):
        raise TypeError(f"FootMR result must be a dictionary: {result_path}")
    params = normalize_result_params(result_data, result_path, args.root_key)
    frame_count = params["body_pose"].shape[0]

    times, timebase, warnings = resolve_timebase(
        result_path=result_path,
        video_arg=args.video,
        fps_arg=args.fps,
        result_frames=frame_count,
        allow_frame_mismatch=args.allow_video_frame_mismatch,
    )
    labels, face_indices, face_vertices, weights, marker_meta = (
        load_marker_mapping(marker_yaml)
    )
    vertices, faces, namespace = compute_smpl_vertices(
        params=params,
        chunk_size=args.chunk_size,
        device_name=args.device,
    )
    validate_marker_topology(
        labels=labels,
        face_indices=face_indices,
        face_vertices=face_vertices,
        faces=faces,
    )
    markers = sample_markers(vertices, face_vertices, weights)

    out_dir = (
        resolve_path(args.out_dir)
        if args.out_dir is not None
        else result_path.parent / "trc"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = validate_prefix(args.prefix or result_path.stem)

    trc_path = out_dir / f"{prefix}.trc"
    npz_path = out_dir / f"{prefix}_markers.npz"
    report_path = out_dir / f"{prefix}_trc_export_report.json"

    write_trc(
        path=trc_path,
        labels=labels,
        markers=markers,
        times=times,
        fps=float(timebase["fps"]),
    )
    np.savez_compressed(
        npz_path,
        marker_labels=np.asarray(labels, dtype=np.str_),
        markers=markers,
        times=times,
        smpl_face_indices=face_indices,
        smpl_face_vertices=face_vertices,
        barycentric_weights=weights,
    )

    report = {
        "status": "pass",
        "result": str(result_path),
        "root_key": args.root_key,
        "marker_yaml": str(marker_yaml),
        "marker_yaml_meta": marker_meta,
        "marker_count": len(labels),
        "markers": labels,
        "frames": frame_count,
        "timebase": timebase,
        "coordinate_units": "m",
        "coordinate_bounds": coordinate_bounds(markers),
        "coordinate_policy": (
            "Exported as stored in hmr4d_results.pt; no additional FootMR "
            "post-processing, marker grounding, height scaling, rotation, "
            "filtering, or synchronization was applied."
        ),
        "postprocess_note": (
            "The standard FootMR demo applies static-joint and IK post-processing "
            "before saving hmr4d_results.pt unless inference used --no_postproc."
        ),
        "mesh": {
            "source_topology": "SMPL-X",
            "export_topology": "SMPL",
            "vertices": int(vertices.shape[1]),
            "faces": int(faces.shape[0]),
        },
        "namespace": namespace,
        "artifacts": {
            "trc": str(trc_path),
            "npz": str(npz_path),
            "report": str(report_path),
        },
        "warnings": warnings,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "frames": report["frames"],
                "marker_count": report["marker_count"],
                "fps": report["timebase"]["fps"],
                "artifacts": report["artifacts"],
                "warnings": warnings,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
