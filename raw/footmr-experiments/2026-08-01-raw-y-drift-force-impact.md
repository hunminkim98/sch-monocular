# Raw FootMR Y drift와 force-estimation 영향 실험 기록

> Source: Local Walk 1-5 Raw FootMR vertical-drift diagnostic (`outputs/y_drift_force_check`)
> Collected: 2026-08-01
> Published: Unknown

## 질문

Raw FootMR pelvis Y가 시간에 따라 이동하는 현상이 위치 오차에 그치는지, 아니면
곡률을 가져 가짜 수직 가속도와 force-estimation bias를 만드는지 확인했다.

## 데이터와 고정 조건

- 영상은 metadata focal을 사용한 30 fps CFR Walk 1-5 결과다.
- 저장된 `smpl_params_global`은 이미 후처리된 값이므로 Raw로 사용하지 않았다.
  `net_outputs.model_output.pred_x`를 `MM_V1_AMASS_LOCAL_BEDLAM_CAM` 통계로 다시
  decode하고, 저장 당시 조건인 `static_cam=true`의 identity camera angular
  velocity로 global translation을 재구성했다.
- Raw pelvis proxy는 barycentric `R_ASIS`와 `L_ASIS`의 midpoint다. Raw root
  translation Y도 별도 sensitivity로 계산했다.
- reference는 같은 trial의 marker-based OpenSim MOT `pelvis_ty`다. ASIS midpoint와
  OpenSim pelvis origin은 같은 해부학적 점이 아니므로 position에는 평가 시작
  0.20초의 상수 offset만 허용했다.
- `accepted_sync_trim.csv`의 공통 구간을 사용했다. 주 분석은 GT를 영상 시간축에서
  +1 video frame 뒤로 둔 기존 pelvis-kinematics sensitivity correction이고,
  0 frame을 별도 sensitivity로 계산했다.
- 피험자가 영상 밖으로 나가는 마지막 noise 구간은 공통 MOT overlap에 포함되지
  않았다. 평가 source frame 범위는 Walk 1부터 순서대로 85-160, 71-137,
  64-144, 63-147, 55-139다. 전 구간에서 hip confidence 최솟값은 0.747261,
  lower-body median confidence 최솟값은 0.764721이었다.
- 30 Hz에서 직접 미분했고, 공통 구간 양 끝 0.10초를 제외했다. 주 분석에는 추가
  filtering을 적용하지 않았다. FootMR 위치에 4th-order zero-phase Butterworth
  6.0 Hz low-pass를 적용한 결과를 sensitivity로 계산했다.

## Drift 분해

각 Walk의 평가 구간에서 다음 수직 잔차를 계산했다.

```text
e(t) = Raw ASIS midpoint Y(t) - marker-based pelvis_ty(t)
```

시간 중심을 `τ=0`으로 두고 Huber robust regression으로 선형식과 이차식을 각각
적합했다.

```text
linear:    e(τ) = b0 + b1 τ
quadratic: e(τ) = b0 + b1 τ + b2 τ²
```

이차항이 만드는 가짜 가속도와 whole-body COM으로 가정했을 때의 등가 힘은 다음과
같이 계산했다.

```text
a_false       = 2 b2
Delta F / BW  = a_false / 9.80665
```

`quadratic detrending`은 `b1 τ + b2 τ²`만 Raw Y에서 제거했다. 이 보정은 GT
잔차를 보고 적합한 oracle diagnostic이며 production 후처리 후보가 아니다.

## Walk별 주 결과

`+Y`는 위 방향이다. 다섯 Walk의 slope가 모두 음수이므로 이번 공식 좌표계에서
관찰된 것은 위쪽이 아니라 reference 대비 아래쪽 drift다.

| Walk | 평가 시간 (s) | 중심 slope (cm/s) | trend net change (m) | a_false (m/s²) | 등가 힘 (% BW) |
|---|---:|---:|---:|---:|---:|
| 1 | 2.300000 | -4.597972 | -0.105753 | -0.019353 | -0.197347 |
| 2 | 2.000000 | -4.057600 | -0.081152 | +0.009570 | +0.097587 |
| 3 | 2.466667 | -4.024600 | -0.099274 | -0.015220 | -0.155204 |
| 4 | 2.600000 | -3.855375 | -0.100240 | -0.014872 | -0.151648 |
| 5 | 2.600000 | -4.492500 | -0.116806 | -0.009882 | -0.100772 |

중심 slope 평균은 -0.042056 m/s이고 범위는 -0.045980~-0.038554 m/s다.
즉 reference 대비 약 -4.598~-3.855 cm/s의 일관된 저주파 drift가 있다.
등가 힘 평균은 -0.101477% BW, 범위는 -0.197347~+0.097587% BW이며 최대
절댓값은 0.197347% BW다.

선형 trend residual RMSE 대비 이차 trend residual RMSE 감소율은 Walk별
22.900310%, 6.070224%, 29.657975%, 28.118364%, 13.559529%였다. 완전히 직선은
아니지만 곡률의 가속도 규모는 중력 대비 작다.

## GT-informed detrending 전후

주 분석의 5개 Walk, 364 frame을 합친 결과다.

| 지표 | Raw | Quadratic detrended | 변화 |
|---|---:|---:|---:|
| Vertical position RMSE (m) | 0.054737 | 0.042777 | 21.850281% 감소 |
| Vertical position r | 0.303961 | 0.882132 | 증가 |
| Vertical velocity RMSE (m/s) | 0.067871 | 0.052869 | 22.103464% 감소 |
| Vertical velocity r | 0.944091 | 0.944317 | 사실상 동일 |
| Vertical acceleration RMSE (m/s²) | 0.868498 | 0.868460 | 0.004317% 감소 |
| Vertical acceleration r | 0.884658 | 0.884650 | 사실상 동일 |

Position과 velocity는 저주파 trend 제거로 개선됐지만 acceleration RMSE와
correlation은 변하지 않았다. 이 결과는 drift가 주로 상수 속도 또는 매우 작은
곡률 성분임을 뜻한다.

## Acceleration agreement와 force 해석

Raw acceleration Bland-Altman bias는 -0.010543 m/s²이고 descriptive 95%
limits of agreement는 [-1.715016, +1.693930] m/s²였다. Detrending 후 bias는
+0.000083 m/s²이고 limits는 [-1.704442, +1.704608] m/s²였다. 평균 bias만
없어졌을 뿐 agreement 폭은 줄지 않았다.

Pelvis를 whole-body COM이라고 단순 가정해 중력으로 나누면 Raw acceleration
RMSE 0.868498 m/s²는 8.856212% BW, limits of agreement는
[-17.488298, +17.273275]% BW에 해당한다. 이것은 실제 GRF 오차가 아니라 pelvis
acceleration proxy의 크기 환산이다.

따라서 수직 drift 곡률이 만드는 systematic force bias는 최대 0.197347% BW로
작지만, frame별 acceleration disagreement는 작지 않다. Drift가 무해하다는 말과
Raw kinematics가 실제 force estimation에 검증됐다는 말은 같지 않다.

## Sensitivity

- Sync 0 frame에서 acceleration RMSE/r은 1.044981 m/s² / 0.832135였고,
  +1 frame에서는 0.868498 m/s² / 0.884658이었다. +1 frame이 이번 진단에서도
  derivative agreement를 높였다.
- FootMR 위치에 6.0 Hz low-pass를 추가한 +1 frame 결과는 acceleration RMSE/r
  0.854854 m/s² / 0.888306이었다. 최대 등가 힘 절댓값은 0.195380% BW로 주
  분석의 0.197347% BW와 사실상 같았다.
- ASIS midpoint 대신 정확히 재구성한 Raw root translation을 사용했을 때 최대
  등가 힘 절댓값은 0.214152% BW였다. 6.0 Hz sensitivity에서는 0.213468% BW였다.
  따라서 pelvis surface marker 선택이나 filtering이 핵심 결론을 바꾸지 않았다.

## 해석 제한

- ASIS midpoint는 whole-body COM이 아니다. `a/g` 환산은 drift 규모를 BW로
  이해하기 위한 proxy이며 GRF estimation 또는 GaitDynamics 검증이 아니다.
- Reference MOT도 raw marker가 아니라 marker-based OpenSim IK 결과다.
- Detrending은 GT 잔차로 적합했으므로 실사용 가능한 blind correction이 아니다.
- 한 피험자의 Walk 5개이고 frame은 독립 표본이 아니다. Bland-Altman limits는
  descriptive 진단으로만 해석한다.
- 이 sub-analysis는 Raw vertical residual과 직접 30 Hz 미분만 다룬다. 기존
  네 조건 X/Y/Z 비교의 절대 수치를 대체하지 않고 결론의 방향을 검증한다.

## 결정

1. Raw Y의 일관된 position drift는 인정한다.
2. 현재 5개 Walk에서는 drift 곡률의 최대 영향이 0.197347% BW이므로 drift만을
   이유로 force-estimation 입력에서 Raw를 배제하지 않는다.
3. GT-informed detrending은 acceleration을 개선하지 않으므로 production에
   추가하지 않는다. 특히 매 frame grounding이나 trend 제거를 자동 적용하지 않는다.
4. 실제 force 유효성은 provenance가 확인된 subject model의 whole-body segment
   kinematics와 GaitDynamics/GRF output을 force plate에 직접 비교해 판단한다.
