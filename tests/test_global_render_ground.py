"""Global mesh renderer의 Grounded Y=0 보존 회귀 테스트입니다."""

from __future__ import annotations

import torch

from hmr4d.utils.vis.renderer import get_global_render_y_offset


def make_vertices_with_late_outlier() -> torch.Tensor:
    vertices = torch.zeros(3, 2, 3)
    vertices[0, :, 1] = torch.tensor([0.0, 1.0])
    vertices[1, :, 1] = torch.tensor([0.1, 1.1])
    vertices[2, :, 1] = torch.tensor([-10.0, -9.0])
    return vertices


def test_grounded_result_preserves_zero_despite_late_outlier() -> None:
    vertices = make_vertices_with_late_outlier()

    offset = get_global_render_y_offset(vertices, {"applied": True})

    assert offset.shape == torch.Size([])
    assert offset.dtype == vertices.dtype
    assert offset.device == vertices.device
    assert offset.item() == 0.0


def test_missing_grounding_report_keeps_legacy_sequence_minimum() -> None:
    vertices = make_vertices_with_late_outlier()

    offset = get_global_render_y_offset(vertices)

    assert offset.item() == -10.0


def test_failed_grounding_keeps_legacy_sequence_minimum() -> None:
    vertices = make_vertices_with_late_outlier()

    offset = get_global_render_y_offset(vertices, {"applied": False})

    assert offset.item() == -10.0
