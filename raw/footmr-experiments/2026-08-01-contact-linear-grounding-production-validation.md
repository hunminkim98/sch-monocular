# C_Temporal contact-linear grounding production 구현 검증

> Source: Local FootMR production implementation and Walk 1-5 regression
> Collected: 2026-08-01
> Published: Unknown

## 구현 범위

- `contact-linear`은 선택형 평지 보행 mode이며 기본값은 `none`이다.
- `contact-linear`을 선택하면 FootMR anti-sliding 전 Raw global motion을 사용한다.
  기존 FootMR anti-sliding 구현 파일과 논리는 수정하지 않았다.
- 기존 C_Temporal 계산을 `hmr4d/model/footmr/utils/contact_grounding.py`의 순수
  production core로 옮겼고, `utils/foot_contact/contact.py`는 force-plate 평가용
  wrapper로 유지했다.
- 출력의 `smpl_params_global.transl[:, 1]`에서 sequence 전체에 공통인 직선 하나만
  뺀다. Pose, X, Z와 `smpl_params_incam`은 바꾸지 않는다.
- 새 mode는 출력 이름에 `_gcontact`를 추가해 기존 결과 cache와 분리한다.
- 적용 여부, fallback 이유, stance 수, slope, intercept, 양발 상수 offset,
  residual MAD를 `hmr4d_results.pt`의 `grounding`에 저장한다.

## 고정 접촉 및 품질 기준

- Contact probability: 0.60 이상
- Foot 및 lower-body pose confidence: 0.50 이상
- Contact run 중앙부: 60%
- 최소 run 길이: 8 frame
- Stance별 최대 표본: 9 frame
- 전체 stance: 4개 이상
- 좌우 stance: 각각 1개 이상
- Contact-line residual MAD: 0.03 m 이하
- Ground slope 절댓값: 0.10 m/s 이하
- 설계행렬 rank: 3

접촉 표본이 영상 시간 범위의 일정 비율을 차지해야 한다는 조건은 사용하지 않는다.
구현과 저장 report에 time coverage 또는 span ratio 항목이 없다. 시간 표본에 대해서는
실제 Huber 설계행렬이 rank 3인지 여부만 검사한다.

## 합성 회귀 테스트

300 frame 영상에서 frame 10-66 안에 4개 stance를 배치했다. 이 범위는 전체 frame
범위의 18.7%이므로 40%보다 작지만, 전체 4개와 좌우 각각 2개 조건 및 rank 3을
만족해 보정이 적용됐다. 이 테스트는 시간 범위 40% 조건이 다시 추가되는 것을
방지한다.

추가로 전체 3개 stance에서는 `insufficient_stances`, 4개가 모두 왼발이면
`insufficient_stances_per_side`로 Raw를 반환하는 것을 확인했다. Fallback 결과의
모든 tensor 값은 Raw와 정확히 같았다. 선형 보정의 이차 미분과 pose 및 X/Z 보존도
검증했다. 새 grounding 테스트 5개와 기존 foot-contact 테스트 6개를 합쳐 11개가
통과했다.

## 실제 Walk 1-5 회귀

Metadata focal, 30 fps CFR Walk 1-5의 저장 model output에서 `static_cam=true` Raw
global root를 재구성한 뒤 production 함수 자체를 CPU에서 실행했다.

| Walk | Applied | Stance (L/R) | Slope (cm/s) | Residual MAD (cm) |
|---|---|---:|---:|---:|
| 1 | True | 10 (5/5) | -3.016291 | 2.306844 |
| 2 | True | 9 (4/5) | -2.590464 | 0.817968 |
| 3 | True | 10 (5/5) | -3.060453 | 1.921341 |
| 4 | True | 10 (5/5) | -2.716428 | 1.899656 |
| 5 | True | 10 (5/5) | -3.487621 | 1.517180 |

다섯 Walk 모두 품질 조건을 통과했다. 이전 production-independent A/B slope와의
최대 차이는 0.000007 cm/s였다. 모든 Walk에서 body pose와 X/Z translation은
bitwise 동일했다. Float32 output에 보정선을 적용한 뒤 수치 미분한 보정선의 최대
이차 미분은 1.073e-04 m/s²였다.

## 실행 환경 제한

새 CLI의 help, 30 fps 변환, metadata focal 선택, `_gcontact` 출력 경로 생성은
확인했다. 현재 WSL 실행 세션에서는 NVIDIA driver가 노출되지 않아 checkpoint를
사용한 새 GPU end-to-end inference는 시작 전에 중단됐다. 저장된 실제 inference의
Raw 재구성, production 함수 실행, 합성 회귀에는 GPU를 사용하지 않았다.

## 최종 결정

1. `contact-linear`을 평지 보행용 선택형 production mode로 유지한다.
2. 전체 stance 4개 이상, 좌우 각각 1개 이상을 요구한다.
3. 접촉 표본 시간 범위 40% 조건은 사용하지 않는다.
4. 품질 조건을 통과하지 못하면 Raw global motion을 그대로 반환한다.
5. 기존 FootMR anti-sliding 경로는 기본 경로로 보존하며 수정하지 않는다.
