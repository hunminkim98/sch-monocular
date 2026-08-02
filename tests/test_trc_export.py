"""FootMR TRC production export의 영구 회귀 테스트입니다."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from hmr4d.utils import trc_export


def make_result(path: Path, frame_count: int = 2) -> dict[str, torch.Tensor]:
    params = {
        "body_pose": torch.arange(frame_count * 63, dtype=torch.float32).reshape(frame_count, 63),
        "betas": torch.arange(10, dtype=torch.float32).reshape(1, 10),
        "global_orient": torch.arange(frame_count * 3, dtype=torch.float32).reshape(frame_count, 3),
        "transl": torch.arange(frame_count * 3, dtype=torch.float32).reshape(frame_count, 3),
    }
    torch.save({"smpl_params_global": params}, path)
    return params


def make_marker_yaml(path: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "meta": {"status": "work_in_progress"},
                "markers": {
                    "TEST": {
                        "kind": "barycentric",
                        "smpl_face_index": 0,
                        "smpl_face_vertices": [0, 1, 2],
                        "barycentric_weights": [0.2, 0.3, 0.5],
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def fake_mesh(frame_count: int) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    vertices = np.asarray(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 4.0]],
            [[1.0, 2.0, 3.0], [2.0, 2.0, 3.0], [1.0, 4.0, 7.0]],
        ],
        dtype=np.float32,
    )
    return vertices[:frame_count], np.asarray([[0, 1, 2]]), {"source": "test"}


def test_production_wrapper_forces_global_and_strict_video(monkeypatch) -> None:
    received = {}

    def fake_export(result, **kwargs):
        received["result"] = result
        received.update(kwargs)
        return {"status": "pass"}

    monkeypatch.setattr(trc_export, "export_footmr_result", fake_export)

    report = trc_export.export_global_trc(
        "hmr4d_results.pt",
        "0_input_video.mp4",
        out_dir="trc",
        device="cpu",
    )

    assert report == {"status": "pass"}
    assert received["root_key"] == "smpl_params_global"
    assert received["video"] == "0_input_video.mp4"
    assert received["allow_video_frame_mismatch"] is False


def test_sample_markers_preserves_xyz_without_transform() -> None:
    vertices, _, _ = fake_mesh(frame_count=2)
    face_vertices = np.asarray([[0, 1, 2]], dtype=np.int64)
    weights = np.asarray([[0.2, 0.3, 0.5]], dtype=np.float32)

    markers = trc_export.sample_markers(vertices, face_vertices, weights)

    expected = np.asarray([[[0.3, 1.0, 2.0]], [[1.3, 3.0, 5.0]]], dtype=np.float32)
    assert np.array_equal(markers, expected)


def test_timebase_rejects_video_frame_mismatch(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        trc_export,
        "video_metadata",
        lambda path: {"path": str(path), "frames": 9, "fps": 30.0},
    )

    with pytest.raises(ValueError, match="video has 9 frames but result has 10"):
        trc_export.resolve_timebase(
            result_path=tmp_path / "hmr4d_results.pt",
            video_arg=tmp_path / "0_input_video.mp4",
            fps_arg=None,
            result_frames=10,
            allow_frame_mismatch=False,
        )


def test_write_trc_matches_marker_array(tmp_path) -> None:
    markers = np.asarray(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]],
        ],
        dtype=np.float32,
    )
    times = np.asarray([0.0, 1.0 / 30.0])
    path = tmp_path / "trial.trc"

    trc_export.write_trc(path, ["LEFT", "RIGHT"], markers, times, fps=30.0)
    lines = path.read_text(encoding="utf-8").splitlines()
    metadata = lines[2].split("\t")
    rows = [line.split("\t") for line in lines[6:]]

    assert metadata[:5] == ["30.00000000", "30.00000000", "2", "2", "m"]
    assert [int(row[0]) for row in rows] == [1, 2]
    assert np.asarray([float(row[1]) for row in rows]) == pytest.approx(times)
    written_markers = np.asarray([[float(value) for value in row[2:]] for row in rows])
    assert written_markers.reshape(markers.shape) == pytest.approx(markers)


def test_shared_export_keeps_stored_global_parameters(monkeypatch, tmp_path) -> None:
    result_path = tmp_path / "hmr4d_results.pt"
    marker_yaml = tmp_path / "markers.yaml"
    out_dir = tmp_path / "trc"
    stored_params = make_result(result_path)
    make_marker_yaml(marker_yaml)
    received = {}

    def compute(params, chunk_size, device_name):
        received["params"] = {key: value.clone() for key, value in params.items()}
        return fake_mesh(frame_count=len(params["transl"]))

    monkeypatch.setattr(trc_export, "compute_smpl_vertices", compute)

    report = trc_export.export_footmr_result(
        result_path,
        marker_yaml=marker_yaml,
        fps=30.0,
        out_dir=out_dir,
        prefix="trial",
        device="cpu",
    )

    assert torch.equal(received["params"]["body_pose"], stored_params["body_pose"])
    assert torch.equal(received["params"]["global_orient"], stored_params["global_orient"])
    assert torch.equal(received["params"]["transl"], stored_params["transl"])
    expected_betas = stored_params["betas"].expand(len(stored_params["transl"]), -1)
    assert torch.equal(received["params"]["betas"], expected_betas)
    assert report["root_key"] == "smpl_params_global"
    assert report["frames"] == 2
    assert report["marker_count"] == 1
    assert (out_dir / "trial.trc").is_file()
    assert (out_dir / "trial_markers.npz").is_file()
    assert (out_dir / "trial_trc_export_report.json").is_file()
    with np.load(out_dir / "trial_markers.npz") as marker_data:
        assert marker_data["markers"] == pytest.approx(
            np.asarray([[[0.3, 1.0, 2.0]], [[1.3, 3.0, 5.0]]])
        )


def test_export_failure_keeps_previous_complete_artifacts(monkeypatch, tmp_path) -> None:
    result_path = tmp_path / "hmr4d_results.pt"
    marker_yaml = tmp_path / "markers.yaml"
    out_dir = tmp_path / "trc"
    make_result(result_path)
    make_marker_yaml(marker_yaml)
    out_dir.mkdir()
    previous = {
        out_dir / "trial.trc": b"old trc",
        out_dir / "trial_markers.npz": b"old npz",
        out_dir / "trial_trc_export_report.json": b"old report",
    }
    for path, content in previous.items():
        path.write_bytes(content)

    monkeypatch.setattr(
        trc_export,
        "compute_smpl_vertices",
        lambda params, chunk_size, device_name: fake_mesh(len(params["transl"])),
    )

    def fail_write(*args, **kwargs):
        raise RuntimeError("write failed")

    monkeypatch.setattr(trc_export, "write_trc", fail_write)

    with pytest.raises(RuntimeError, match="write failed"):
        trc_export.export_footmr_result(
            result_path,
            marker_yaml=marker_yaml,
            fps=30.0,
            out_dir=out_dir,
            prefix="trial",
            device="cpu",
        )

    assert all(path.read_bytes() == content for path, content in previous.items())
    assert list(out_dir.glob(".trial_*")) == []


def test_demo_default_does_not_export_trc() -> None:
    from tools import demo

    cfg = SimpleNamespace(export_trc=False)

    assert demo.export_trc_if_requested(cfg) is None


def test_demo_opt_in_calls_production_export(monkeypatch) -> None:
    from tools import demo

    received = {}

    def fake_export(result, video, **kwargs):
        received.update({"result": result, "video": video, **kwargs})
        return {
            "frames": 2,
            "marker_count": 1,
            "timebase": {"fps": 30.0},
            "artifacts": {"trc": "outputs/demo/trc/hmr4d_results.trc"},
        }

    monkeypatch.setattr(trc_export, "export_global_trc", fake_export)
    cfg = SimpleNamespace(
        export_trc=True,
        video_path="outputs/demo/0_input_video.mp4",
        paths=SimpleNamespace(
            hmr4d_results="outputs/demo/hmr4d_results.pt",
            trc_dir="outputs/demo/trc",
        ),
    )

    demo.export_trc_if_requested(cfg)

    assert received == {
        "result": "outputs/demo/hmr4d_results.pt",
        "video": "outputs/demo/0_input_video.mp4",
        "out_dir": "outputs/demo/trc",
        "device": "cpu",
    }


def test_demo_config_keeps_export_disabled_by_default() -> None:
    config_path = Path(__file__).resolve().parents[1] / "hmr4d" / "configs" / "demo.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["export_trc"] is False


def test_standalone_cli_uses_shared_export(monkeypatch, capsys) -> None:
    from tools import export_hmr4d_result_to_barycentric_trc as cli

    received = {}

    def fake_export(result, **kwargs):
        received["result"] = result
        received.update(kwargs)
        return {
            "status": "pass",
            "frames": 2,
            "marker_count": 1,
            "timebase": {"fps": 30.0},
            "artifacts": {"trc": "trc/trial.trc"},
            "warnings": [],
        }

    monkeypatch.setattr(cli, "export_footmr_result", fake_export)

    assert cli.main(["--result", "hmr4d_results.pt", "--fps", "30"]) == 0
    output = json.loads(capsys.readouterr().out)

    assert received["root_key"] == "smpl_params_global"
    assert received["fps"] == 30.0
    assert output["status"] == "pass"
