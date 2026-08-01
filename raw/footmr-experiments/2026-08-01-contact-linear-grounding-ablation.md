# C_Temporal 기반 sequence-level 선형 grounding A/B 실험 기록

> Source: Local Walk 1-5 contact-aware linear-grounding ablation
> Collected: 2026-08-01
> Published: Unknown

## 질문

평지 보행에서 Raw FootMR의 수직 drift를 제거할 때, force plate나 marker-based GT를
입력으로 사용하지 않고 기존 C_Temporal foot contact만으로 하나의 선형 지면을
추정할 수 있는지 확인했다. 보정 목표는 position과 velocity를 개선하면서 기존
FootMR pose와 acceleration waveform을 바꾸지 않는 것이다.

## Production과 분리한 실험 조건

- Metadata focal을 사용한 30 fps CFR Walk 1-5의 저장 결과를 사용했다.
- 저장된 post-processed `smpl_params_global`은 사용하지 않았다.
  `net_outputs.model_output.pred_x`를 `MM_V1_AMASS_LOCAL_BEDLAM_CAM` 통계로
  decode하고, `static_cam=true`의 identity camera angular velocity에서
  anti-sliding 전 global root를 재구성했다.
- FootMR ankle refinement는 유지하고 root anti-sliding과 `process_ik`만 배제했다.
- Contact는 production-independent `utils/foot_contact/contact.py`의 C_Temporal을
  그대로 재계산했다. Force plate와 marker-based GT는 contact 검출이나 지면 적합에
  사용하지 않았다.
- 지면 높이 proxy는 각 발에 대응하는 C_Temporal 3개 foot vertex의 Y 최솟값이다.
- C_Temporal contact run 중 중앙 60%만 사용했다. Contact probability는 0.60 이상,
  해당 발과 lower-body pose confidence는 0.50 이상이어야 했다.
- 긴 정지 구간이 결과를 독점하지 않도록 stance마다 최대 9개 frame을 균등 표본화했다.
  Walk별 유효 stance는 10, 9, 10, 10, 10개였다.
- 양발 표면의 상수 높이 차이는 허용하되 시간 slope는 전 sequence에 하나만 허용했다.

## 선형 지면 모델

안정된 contact sample의 발 높이 `h`를 다음 Huber line에 적합했다.

```text
h(t, side) = a + b (t - tc) + c I(right)
```

- `a`: left-foot 기준 상수 높이
- `b`: 모든 frame과 양발이 공유하는 ground drift slope
- `c`: right-foot의 상수 surface offset
- `tc`: 안정된 contact sample의 평균 시간

전체 body의 모든 Y 좌표에 적용할 보정은 다음 한 개의 직선이다.

```text
ground(t)      = a + c/2 + b (t - tc)
Y_corrected(t) = Y_raw(t) - ground(t)
```

Frame별 offset, stance별 offset, quadratic term은 사용하지 않았다. 따라서 보정의
이차 미분은 0이고 이론상 새로운 acceleration을 만들지 않는다.

## Ground slope 검증

`+Y`는 위 방향이다. Full reliable sequence에서 contact가 추정한 slope와, 기존
GT overlap에서 Raw ASIS midpoint와 marker-based `pelvis_ty` 잔차가 보인 oracle
slope는 다음과 같다.

| Walk | Full-sequence contact (cm/s) | Same-window contact (cm/s) | GT residual oracle (cm/s) | 보정 후 residual (cm/s) |
|---|---:|---:|---:|---:|
| 1 | -3.016288 | -4.014717 | -4.552748 | -1.536460 |
| 2 | -2.590457 | -3.577680 | -4.088207 | -1.497750 |
| 3 | -3.060459 | -3.675296 | -4.022538 | -0.962079 |
| 4 | -2.716428 | -3.214943 | -3.882411 | -1.165983 |
| 5 | -3.487615 | -3.741889 | -4.496437 | -1.008821 |

Full-sequence contact fit과 GT oracle은 서로 다른 시간창이므로 contact slope가 더
작았다. 동일 GT 시간창의 contact만 진단적으로 선택했을 때 oracle과의 최대 차이는
0.754548 cm/s였다. 이 same-window 선택은 비교용일 뿐 production 보정에는 사용하지
않았다.

## Pelvis Y A/B 결과

Reference는 marker-based OpenSim MOT `pelvis_ty`이고 prediction은 barycentric
`R_ASIS`와 `L_ASIS` midpoint다. 기존 `accepted_sync_trim.csv` 공통 구간에서
GT +1 video frame sensitivity correction을 주 분석으로 사용했다. Position에는
평가 시작 0.20초의 상수 offset만 적용했고, 30 Hz `numpy.gradient`로 직접
미분한 뒤 양 끝 0.10초를 제외했다. Pooled 표본은 364 frame이다.

| Vertical 지표 | Raw | Contact-linear grounded | 변화 |
|---|---:|---:|---:|
| Position RMSE (m) | 0.056321 | 0.015653 | 72.208351% 감소 |
| Position r | 0.305130 | 0.720228 | 증가 |
| Velocity RMSE (m/s) | 0.067812 | 0.054376 | 19.812780% 감소 |
| Velocity r | 0.944291 | 0.944217 | 사실상 동일 |
| Acceleration RMSE (m/s²) | 0.867214 | 0.867214 | 변화 없음 |
| Acceleration r | 0.885027 | 0.885027 | 변화 없음 |

Position과 velocity RMSE는 Walk 1-5 각각에서 모두 감소했다.

| Walk | Position Raw → Grounded (m) | Velocity Raw → Grounded (m/s) | Acceleration Raw → Grounded (m/s²) |
|---|---:|---:|---:|
| 1 | 0.056192 → 0.016939 | 0.075139 → 0.062157 | 0.796684 → 0.796684 |
| 2 | 0.052801 → 0.022755 | 0.066405 → 0.053974 | 0.859495 → 0.859495 |
| 3 | 0.055589 → 0.012437 | 0.058849 → 0.042374 | 0.752278 → 0.752278 |
| 4 | 0.054821 → 0.014654 | 0.062994 → 0.050354 | 0.876723 → 0.876723 |
| 5 | 0.061042 → 0.010641 | 0.074325 → 0.060916 | 1.013148 → 1.013148 |

Acceleration sample의 보정 전후 최대 절댓값 차이는 4.996004e-14 m/s²였다.
Bland–Altman bias와 limits도 Raw -0.010606 m/s²,
[-1.712559, +1.691346] m/s²에서 바뀌지 않았다. 전체 sequence correction line을
두 번 미분한 최대 절댓값은 9.992007e-14 m/s²였다.

## Sensitivity와 화면 이탈 처리

- Sync 0 frame에서도 position RMSE는 0.052704 m에서 0.013661 m로, velocity
  RMSE는 0.092641 m/s에서 0.084683 m/s로 감소했다. Acceleration RMSE는
  1.045105 m/s²로 동일했다.
- Foot proxy를 3개 vertex 최솟값 대신 평균으로 바꾼 slope는 본 분석과 Walk별
  최대 0.083152 cm/s 차이였다.
- 더 엄격한 probability 0.75, confidence 0.60, 중앙 50% 조건의 slope는 Walk
  1부터 -2.765588, -2.482056, -2.552349, -2.262890, -2.722210 cm/s였다.
- Leave-one-stance-out slope는 모든 trial에서 음수 방향을 유지했다. 전체 범위는
  -4.128527~-2.198158 cm/s였다.
- Contact-line residual MAD는 Walk별 2.306860, 0.817951, 1.921365, 1.899657,
  1.517180 cm였다.
- Pose confidence gate 때문에 안정된 마지막 contact sample은 영상 종료보다 최소
  1.733333초 먼저 끝났다. 따라서 피험자가 화면 밖으로 나가며 발생한 마지막
  noise 구간은 지면 적합에 포함되지 않았다.

## 해석 제한

- 한 피험자의 평지 Walk 5개만 검증했다. 경사로, 계단, 달리기, 점프에는 이 모델을
  적용할 수 없다.
- C_Temporal contact와 동일한 foot vertex에서 ground를 추정했으므로 contact 오검출은
  slope 오차로 이어질 수 있다. 유효 stance 수, 양발 coverage, 시간 span, confidence,
  slope magnitude를 검사하고 실패 시 Raw로 되돌아가야 한다.
- Full-sequence slope는 GT가 존재하는 짧은 평가창의 slope보다 작았다. 이것은 단일
  선형 모델이 sequence 전체 drift를 근사한 결과이며 완전한 drift 제거를 보장하지 않는다.
- ASIS midpoint는 whole-body COM이 아니고 reference도 raw marker가 아닌
  marker-based OpenSim IK pelvis다. 이 실험은 실제 GaitDynamics 또는 GRF를
  force plate에 직접 비교한 force-estimation validation이 아니다.

## 결정

1. 평지 보행 전용 mode에서는 C_Temporal contact로 추정한 sequence-level 선형
   grounding을 production 후보로 채택한다.
2. 원본 FootMR pose, X/Z, foot-lock, anti-sliding 논리는 바꾸지 않는다. 모든 global
   Y에 한 개의 직선만 빼고 frame별 또는 stance별 grounding은 금지한다.
3. 최소 품질 조건을 만족하지 못하면 보정하지 않고 Raw를 반환한다.
4. 현재 결과는 구현 승인이 아니라 production 구현 전 A/B 근거다. 구현 뒤에는 같은
   Walk 1-5 regression과 synthetic linear-drift test를 영구 테스트로 남긴다.
