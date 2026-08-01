# FootMR production 산출물 균형 정리 감사 기록

> Source: Local repository filesystem cleanup audit
> Collected: 2026-08-01
> Published: Unknown

Contact-linear grounding 결론이 확정된 뒤 production 코드, 영구 회귀 테스트,
LLM wiki 근거 기록, 사용자가 보존을 지정한 publication asset만 남기는 균형 정리를
수행했다.

Appendix 재렌더링에 필요한 Walk 1-5 metadata는
`outputs/focal_ablation_5_30fps/metadata/`에서
`inputs/foot_contact_appendix/`로 이동했다. 이동 뒤 크기는 28M이며, Walk별
`0_input_video.mp4`, `1_incam.mp4`, `hmr4d_results.pt`와 preprocess 입력을
유지했다. `utils/foot_contact/visuals.py`의 입력 경로도 새 위치로 변경했다.

다음 실험 전용 항목은 제거했다.

- `outputs/focal_ablation_5_30fps/`에 남아 있던 default inference, comparison,
  reference marker, 분석 script와 cache
- C_Temporal appendix의 Walk별 clip, QC frame, preview, CSV/JSON 중간 결과
- Raw vs Grounded 시각화의 Walk별 clip
- repository 작업 과정에서 생성된 `__pycache__`, `.pytest_cache`, `.pyc`

정리 후 `outputs/`에는 다음 최종 통합 영상 두 개만 남겼다.

- `outputs/c_temporal_forceplate_passage_supplementary_video_5_30fps/C_Temporal_Three_Panel_Supplementary_Video.mp4` — 4567253 bytes
- `outputs/contact_grounding_raw_vs_grounded_5_30fps/S18_Walk_1_5_Raw_vs_Grounded_Global.mp4` — 2760878 bytes

Walk별 frame 수와 수치 검증 결과는 기존 immutable raw 기록과 wiki에 유지하지만,
Walk별 영상 파일 자체는 최종 통합본 검증 후 중간 산출물로 제거했다. Production
grounding 구현과 합성 regression test는 결과 생성용 일회성 코드가 아니므로
유지했다.
