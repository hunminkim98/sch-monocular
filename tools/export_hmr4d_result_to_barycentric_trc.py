#!/usr/bin/env python3
"""저장된 FootMR 결과를 barycentric marker TRC로 내보냅니다."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hmr4d.utils.trc_export import DEFAULT_MARKER_YAML, export_footmr_result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--result",
        type=Path,
        required=True,
        help="FootMR hmr4d_results.pt 파일입니다.",
    )
    parser.add_argument(
        "--root-key",
        choices=("smpl_params_global", "smpl_params_incam"),
        default="smpl_params_global",
        help="독립 export에서 사용할 SMPL parameter dictionary입니다.",
    )
    parser.add_argument(
        "--marker-yaml",
        type=Path,
        default=DEFAULT_MARKER_YAML,
        help="Barycentric marker mapping YAML입니다.",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help=(
            "TRC frame rate를 읽을 영상입니다. 생략하고 --fps도 지정하지 않으면 "
            "result 옆의 0_input_video.mp4를 사용합니다."
        ),
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="영상 metadata보다 우선할 명시적 TRC frame rate입니다.",
    )
    parser.add_argument(
        "--allow-video-frame-mismatch",
        action="store_true",
        help="독립 분석이 필요한 경우 영상과 결과의 frame 수 불일치를 허용합니다.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="출력 디렉터리입니다. 기본값은 result 옆의 trc 디렉터리입니다.",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="출력 파일 prefix입니다. 기본값은 result 파일 stem입니다.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=96,
        help="SMPL-X forward pass당 frame 수입니다.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Mesh 생성에 사용할 Torch device입니다. 기본값은 사용 가능한 CUDA 또는 CPU입니다.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = export_footmr_result(
        args.result,
        root_key=args.root_key,
        marker_yaml=args.marker_yaml,
        video=args.video,
        fps=args.fps,
        allow_video_frame_mismatch=args.allow_video_frame_mismatch,
        out_dir=args.out_dir,
        prefix=args.prefix,
        chunk_size=args.chunk_size,
        device=args.device,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "frames": report["frames"],
                "marker_count": report["marker_count"],
                "fps": report["timebase"]["fps"],
                "artifacts": report["artifacts"],
                "warnings": report["warnings"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
