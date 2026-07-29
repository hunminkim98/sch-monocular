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
