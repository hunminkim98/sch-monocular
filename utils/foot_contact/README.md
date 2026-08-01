# Foot-contact utilities

이 디렉터리는 FootMR production inference와 분리된 실험·검증용 도구입니다.

- `contact.py`: C_Temporal contact와 measured force event 계산
- `visuals.py`: 영상, pose, force 입력과 공통 그리기 도구
- `panels.py`: 논문용 3-panel 렌더링
- `appendix.py`: force-plate 구간을 준비하고 최종 appendix 생성
- `sync.csv`: Walk 1-5의 검증된 동기화 파라미터

재렌더링에 필요한 Walk별 영상, mesh, pose 입력은 실험 결과와 분리해
`inputs/foot_contact_appendix/`에 보관합니다.

저장소 루트에서 다음과 같이 실행합니다.

```bash
python -m utils.foot_contact.appendix --force-dir <marker-based-grf-directory>
```
