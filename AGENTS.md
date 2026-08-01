# Repository Guidelines

## Project Structure & Module Organization

Core Python code lives in `hmr4d/`. FootMR models and post-processing are under
`hmr4d/model/footmr/`; network definitions, datasets, data modules, and Hydra
configuration live in `hmr4d/network/`, `hmr4d/dataset/`, `hmr4d/datamodule/`,
and `hmr4d/configs/`. Use `tools/` for executable entry points such as
`demo.py`, `train.py`, and the TRC exporter. Versioned static resources belong
in `assets/`; barycentric marker definitions are in `assets/marker_mappings/`.
`inputs/` contains local checkpoints and datasets, while generated artifacts
go to `outputs/`; both are Git-ignored. Treat
`footmr_clean_migration_bundle_2026-07-28/` as read-only reference material,
not as an active source tree.

## Build, Test, and Development Commands

Use Linux, Python 3.10, and the CUDA 12.1-compatible dependencies documented in
`docs/INSTALL.md`.

```bash
conda create -n footmr python=3.10
conda activate footmr
pip install -r requirements.txt
pip install -e .
python tools/demo.py --video docs/example_video/stepdance.mp4
python tools/train.py exp=footmr/mixed/mixed exp_name_var=my_run
black --check hmr4d tools
```

Export stored mesh results without additional scaling or grounding:

```bash
python tools/export_hmr4d_result_to_barycentric_trc.py \
  --result outputs/demo/<video>/hmr4d_results.pt
```

## Coding Style & Naming Conventions

Use four-space indentation and Black's 100-character line limit. Use
`snake_case` for modules, functions, variables, and Hydra keys; use
`PascalCase` for classes and `UPPER_SNAKE_CASE` for constants. Keep CLI scripts
thin and reusable logic in `hmr4d/`. Write new comments, explanatory
docstrings, TODOs, and developer notes in Korean by default. Keep identifiers,
CLI flags, external APIs, and standard technical terms in English; do not
translate untouched upstream comments. Preserve line endings and avoid
unrelated formatting.

## Testing Guidelines

No first-party automated test suite is currently established. Add new tests as
`tests/test_<feature>.py` using pytest, with lightweight synthetic tensors where
possible. Run `python -m pytest -q` when tests exist. For pipeline changes,
also run the relevant CLI smoke test and report the video, checkpoint,
configuration overrides, and produced artifact paths. Validate exporters with
finite-coordinate, shape, frame-rate, and topology checks.

## Experiment Lifecycle & Cleanup

Production과 직접 관련 없는 일회성 분석 코드, 실험 전용 테스트, 임시 CLI,
notebook, CSV, plot, render, cache 및 `outputs/` 산출물은 영구 소스가 아니다.
실험이 진행 중일 때만 유지하고 결론이 확정되면 다음 순서로 정리한다.

1. 실험 질문, 입력과 설정, 핵심 수치, 해석, 한계 및 최종 결정을 LLM wiki에
   기록한다. 수치 근거와 원본 실험 기록은 `raw/`에 보존하고, 정리된 지식은
   `wiki/` article과 index/log에 반영한다.
2. 기록이 완료되면 production과 무관한 실험 코드, 실험만을 위한 테스트,
   생성 산출물과 cache를 모두 폐기한다. 실험을 위해 production 또는 shared
   utility에 추가한 hook, option, parameter도 다른 사용처가 없으면 되돌린다.
3. 삭제 후 repository 전체에서 제거한 module, path, CLI 및 output reference를
   검색하고, production 회귀 테스트와 wiki evidence/link 검사를 실행한다.

명시적으로 production 기능으로 채택된 코드, production 동작을 보호하는 영구
regression test, 사용자가 보존을 지정한 publication asset만 예외로 유지한다.
결론이 wiki/raw에 보존되기 전에는 재현에 필요한 실험 자료를 삭제하지 않는다.

## Commit & Pull Request Guidelines

History uses short lowercase action summaries such as `updated data download
link`. Prefer focused messages such as `add marker topology validation`; do not
mix checkpoints or generated outputs into code commits. Pull requests should
explain the purpose, affected configs, checkpoint/data assumptions, and exact
verification commands. Link related issues and include before/after renders or
metrics when visual or numerical behavior changes.

## Data & Configuration Safety

Never commit model weights, licensed body models, datasets, videos, or
`outputs/`. Keep machine-specific paths and credentials out of Hydra configs.
Initialize optional DPVO and Sapiens submodules only when the selected workflow
requires them.
