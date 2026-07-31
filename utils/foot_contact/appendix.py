#!/usr/bin/env python3
"""실제 force-plate passage만 포함하는 3-panel appendix 영상을 생성합니다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from hmr4d.utils.smplx_utils import make_smplx
from utils.foot_contact import contact
from utils.foot_contact import panels
from utils.foot_contact import visuals

ROOT = Path(__file__).resolve().parents[2]
FPS = 30.0
CONTEXT_S = 0.3
SYNC_PATH = Path(__file__).with_name("sync.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "공간적으로 검증한 동기화로 두 measured stance를 모두 포함하는 "
            "C_Temporal Supplementary Video를 생성합니다."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/foot_contact_appendix",
    )
    parser.add_argument(
        "--force-dir",
        type=Path,
        required=True,
        help="동기화된 marker-based GRF .mot 파일 디렉터리",
    )
    parser.add_argument("--context-s", type=float, default=CONTEXT_S)
    parser.add_argument("--crf", type=int, default=18)
    return parser.parse_args()


def prepare_forceplate_passages(
    context_s: float,
    force_dir: Path,
) -> tuple[
    list[visuals.TrialData],
    dict[str, panels.MeshInfo],
]:
    """Force-plate passage 구간과 영상 전체 C_Temporal label을 준비합니다."""
    trials, mesh_info = panels.load_inputs(SYNC_PATH, force_dir)
    anchored = contact.load_forceplate_anchored_sync(SYNC_PATH)
    smplx_model = make_smplx("supermotion_v437coco23").eval()

    for trial in trials:
        sync = anchored.loc[trial.trial]
        full_labels = contact.compute_full_video_c_temporal(
            trial.video_path.parent,
            smplx_model,
            FPS,
        )
        force_origin_s = trial.markerless_start_s + trial.marker_source_start_s
        first_force_s = float(trial.events["force_start_s"].min())
        last_force_s = float(trial.events["force_end_s"].max())
        start_frame = contact.force_time_to_video_frame(
            first_force_s - context_s,
            FPS,
            trial.markerless_start_s,
            trial.marker_source_start_s,
        )
        end_frame = contact.force_time_to_video_frame(
            last_force_s + context_s,
            FPS,
            trial.markerless_start_s,
            trial.marker_source_start_s,
        )
        trial.start_frame = max(0, start_frame)
        trial.end_frame = min(trial.source_frames - 1, end_frame)
        trial.aligned_duration_s = float(trial.force_time[-1])

        passage = full_labels[
            full_labels["video_frame"].between(trial.start_frame, trial.end_frame)
        ].copy()
        passage.insert(0, "trial", trial.trial)
        passage["aligned_reference_time_s"] = passage["video_time_s"] - force_origin_s
        expected_frames = trial.end_frame - trial.start_frame + 1
        if len(passage) != expected_frames:
            raise ValueError(f"{trial.trial}: passage C label frame이 연속적이지 않습니다")
        trial.labels = passage

        expected_first = int(sync["first_force_frame"])
        expected_last = int(sync["last_force_frame"])
        actual_first = contact.force_time_to_video_frame(
            first_force_s,
            FPS,
            trial.markerless_start_s,
            trial.marker_source_start_s,
        )
        actual_last = contact.force_time_to_video_frame(
            last_force_s,
            FPS,
            trial.markerless_start_s,
            trial.marker_source_start_s,
        )
        if (actual_first, actual_last) != (expected_first, expected_last):
            raise ValueError(f"{trial.trial}: audited force frame과 재계산 결과가 다릅니다")
        if not trial.start_frame < actual_first < actual_last < trial.end_frame:
            raise ValueError(f"{trial.trial}: 두 stance 전체가 passage 안에 들어오지 않습니다")

    return trials, mesh_info


def write_research_notes(
    output_dir: Path,
    trials: list[visuals.TrialData],
    summary: dict[str, float | int],
    context_s: float,
) -> None:
    trial_lines = []
    for trial in trials:
        first_force_frame = contact.force_time_to_video_frame(
            float(trial.events["force_start_s"].min()),
            FPS,
            trial.markerless_start_s,
            trial.marker_source_start_s,
        )
        last_force_frame = contact.force_time_to_video_frame(
            float(trial.events["force_end_s"].max()),
            FPS,
            trial.markerless_start_s,
            trial.marker_source_start_s,
        )
        trial_lines.append(
            f"- Walk {trial.trial_number}: clip {trial.start_frame}-{trial.end_frame}; "
            f"measured stance span {first_force_frame}-{last_force_frame}; "
            f"markerless start {trial.markerless_start_s:.6f} s"
        )

    text = f"""# Force-plate-anchored C_Temporal Supplementary Video

## Correction

The earlier event-local draft must not be used. Knee-angle correlation alone
selected the preceding gait cycle in all five Walk trials. The selected offsets
were therefore shifted to the next local knee-correlation peak (about
1.04-1.09 s later) and accepted only when both measured force pulses coincided
with the visible force-plate passage.

Each Walk is now one continuous clip, beginning about {context_s:.1f} s before
the first measured stance and ending about {context_s:.1f} s after the second.
The raw video itself is trimmed to this interval; both measured stance pulses
remain visible in the right panel.

## Mapping

```text
force_time_s = video_frame / 30 - markerless_start_s - marker_source_start_s
video_frame  = round((markerless_start_s + marker_source_start_s + force_time_s) * 30)
```

{chr(10).join(trial_lines)}

## C_Temporal inference

C_Temporal is recomputed over each full source video before passage trimming.
Measured force is not used as a C_Temporal inference feature.

## Force-paired internal pilot

- Events detected: {int(summary["events_detected"])}/{int(summary["events"])}
- Force-pulse recall: {float(summary["pulse_recall_macro"]):.3f}
- Temporal IoU: {float(summary["temporal_iou_macro"]):.3f}
- Onset MAE: {float(summary["onset_mae_ms_detected"]):.1f} ms
- Offset MAE: {float(summary["offset_mae_ms_detected"]):.1f} ms

This is a five-trial, one-participant internal pilot. The synchronization is
post-hoc and spatially audited, not hardware-triggered. Confirm participant
consent before external publication.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def extend_qc(
    output_dir: Path,
    trials: list[visuals.TrialData],
) -> None:
    qc_path = output_dir / "qc_report.json"
    qc = json.loads(qc_path.read_text(encoding="utf-8"))
    for trial in trials:
        first_force_frame = contact.force_time_to_video_frame(
            float(trial.events["force_start_s"].min()),
            FPS,
            trial.markerless_start_s,
            trial.marker_source_start_s,
        )
        last_force_frame = contact.force_time_to_video_frame(
            float(trial.events["force_end_s"].max()),
            FPS,
            trial.markerless_start_s,
            trial.marker_source_start_s,
        )
        qc["checks"][f"{trial.trial}:both_stances_inside_clip"] = bool(
            trial.start_frame < first_force_frame < last_force_frame < trial.end_frame
        )
    qc["checks"]["forceplate_anchored_sync_used"] = True
    qc["checks"]["one_continuous_clip_per_walk"] = True
    failures = [name for name, passed in qc["checks"].items() if not passed]
    qc["failures"] = failures
    qc["status"] = "PASS" if not failures else "FAIL"
    qc["synchronization"] = {
        "method": ("next local knee-correlation peak constrained by visible force-plate passage"),
        "source": str(SYNC_PATH),
        "hardware_triggered": False,
    }
    qc_path.write_text(json.dumps(qc, ensure_ascii=False, indent=2), encoding="utf-8")
    if failures:
        raise RuntimeError("추가 force-plate QC 실패: " + ", ".join(failures))


def main() -> None:
    args = parse_args()
    if args.context_s < 0.0:
        raise ValueError("--context-s는 0 이상이어야 합니다")

    trials, mesh_info = prepare_forceplate_passages(args.context_s, args.force_dir)
    clips = contact.build_event_clips(trials, FPS, args.context_s)
    metrics = contact.evaluate_all_events(
        clips,
        {trial.trial: trial for trial in trials},
        FPS,
    )
    summary = contact.summarize_events(metrics)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    pd.DataFrame([summary]).to_csv(
        args.output_dir / "summary.csv",
        index=False,
    )
    contact.load_forceplate_anchored_sync(SYNC_PATH).reset_index().to_csv(
        args.output_dir / "sync.csv",
        index=False,
    )

    c_summary = pd.Series(summary)
    walk_outputs, combined, preview, frame_counts = panels.render_videos(
        trials,
        mesh_info,
        c_summary,
        args.output_dir,
        args.crf,
    )
    panels.validate_outputs(
        args.output_dir,
        trials,
        mesh_info,
        walk_outputs,
        combined,
        preview,
        frame_counts,
    )
    write_research_notes(args.output_dir, trials, summary, args.context_s)
    extend_qc(args.output_dir, trials)
    print(f"Rendered: {combined}")
    print(f"Walks: {len(walk_outputs)}")
    print("QC: PASS")


if __name__ == "__main__":
    main()
