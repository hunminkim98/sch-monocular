# FootMR pelvis kinematics, sync, agreement 실험 기록

> Source: Local experiment artifacts in `outputs/pelvis_trc_vs_gt/` (`report.md`, `sync_report.md`, and `sync_optima.csv`)
> Collected: 2026-08-01
> Published: Unknown

# FootMR TRC pelvis proxy vs marker-based OpenSim pelvis

## 비교 질문

현재 FootMR 결과를 legacy markerless OpenSim 모델에 넣지 않고, TRC의 `(R_ASIS + L_ASIS) / 2`를 marker-based OpenSim MOT의 `pelvis_tx/pelvis_ty/pelvis_tz`와 직접 비교했다.

## 고정 조건

- Walk 1-5와 Raw, Camera only, FootMR baseline, C contact를 비교했다.
- `utils/foot_contact/sync.csv`의 확정된 video↔marker-based 오프셋만 사용했다.
- lag fitting과 scale fitting은 하지 않았다(`scale=1.0`).
- 네 조건의 평균 궤적으로 계산한 trial별 수평 yaw 하나를 네 조건에 동일하게 적용했다.
- 위치에는 ASIS midpoint와 OpenSim pelvis origin의 고정 차이를 제거하기 위해 평가 시작 0.20초의 조건별 상수 offset만 적용했다.
- 상수 위치 offset은 velocity와 acceleration에 영향을 주지 않는다.
- 입력은 기존 6 Hz low-pass 결과이므로 추가 필터링하지 않았다.
- 수치 평가는 양 끝 0.10초를 제외했다.

## Pooled 결과

### Position (offset-aligned, m)

| Condition | X/ML RMSE | Y/Vertical RMSE | Z/AP RMSE | 3D-vector RMSE |
|---|---:|---:|---:|---:|
| Raw | 0.0327 | 0.0615 | 0.1119 | 0.1318 |
| Camera only | 0.0856 | 0.0615 | 0.1441 | 0.1785 |
| FootMR baseline | 0.0489 | 0.0614 | 0.0397 | 0.0880 |
| C contact | 0.0589 | 0.0615 | 0.0565 | 0.1022 |

### Velocity (m/s)

| Condition | X/ML RMSE | Y/Vertical RMSE | Z/AP RMSE | 3D-vector RMSE |
|---|---:|---:|---:|---:|
| Raw | 0.0608 | 0.1075 | 0.1201 | 0.1722 |
| Camera only | 0.1062 | 0.1075 | 0.1720 | 0.2289 |
| FootMR baseline | 0.1122 | 0.1074 | 0.1342 | 0.2052 |
| C contact | 0.1300 | 0.1073 | 0.1375 | 0.2175 |

### Acceleration (m/s²)

| Condition | X/ML RMSE | Y/Vertical RMSE | Z/AP RMSE | 3D-vector RMSE |
|---|---:|---:|---:|---:|
| Raw | 0.6682 | 1.1424 | 0.9830 | 1.6486 |
| Camera only | 1.0612 | 1.1424 | 1.9272 | 2.4790 |
| FootMR baseline | 1.6435 | 1.1411 | 1.8901 | 2.7524 |
| C contact | 1.8713 | 1.1400 | 2.3313 | 3.1994 |

## 핵심 관찰

- 3D acceleration RMSE는 Raw 1.6486 m/s², FootMR baseline 2.7524 m/s², C contact 3.1994 m/s²였다.
- C contact의 3D acceleration RMSE는 Raw보다 94.1% 높고, FootMR baseline보다 16.2% 높았다.
- 이 3D acceleration 악화는 Walk 1-5 모두에서 반복됐다.
- 반면 Y/Vertical acceleration은 C contact가 Raw 대비 0.2% 낮아 사실상 동일했다. 차이는 주로 X/ML 및 Z/AP에서 발생했다.
- position은 FootMR baseline이 가장 낮았지만, velocity와 acceleration은 Raw가 가장 낮았다. 따라서 후처리는 위치 궤적을 맞추는 대신 미분 신호를 악화시킬 수 있다.

## 해석 제한

- 이것은 같은 해부학적 점의 비교가 아니다. ASIS midpoint는 pelvis proxy이고, MOT translation은 OpenSim pelvis body origin이다.
- GT MOT도 marker-based IK 모델의 산출물이므로 raw marker Ground Truth 자체는 아니다.
- 따라서 절대 position 오차보다 velocity/acceleration waveform과 조건 간 상대 차이를 우선 해석해야 한다.
- 이 실험은 현재 FootMR TRC에 OpenSim IK, BodyKinematics, GaitDynamics 또는 GRF 추정을 적용하지 않았다.

# Pelvis sync sensitivity

확정 sync를 기준으로 marker-based GT timestamp를 -15~+15 video frame (±0.5초) 이동했다. 모든 shift가 동일한 FootMR sample 구간을 사용하며 lag, scale, yaw를 각 shift에 재적합하지 않았다.

Positive shift는 GT를 영상 시간축에서 뒤로 이동한다는 뜻이다.

| Metric | Accepted r | Best local shift | Best r | Gain |
|---|---:|---:|---:|---:|
| Vertical position | 0.5036 | +1 frame (+0.033 s) | 0.5310 | +0.0274 |
| Vertical velocity | 0.8303 | +1 frame (+0.033 s) | 0.9566 | +0.1263 |
| Vertical acceleration | 0.7962 | +1 frame (+0.033 s) | 0.9234 | +0.1272 |
| XYZ velocity | 0.8205 | +1 frame (+0.033 s) | 0.9135 | +0.0930 |
| XYZ acceleration | 0.7290 | +1 frame (+0.033 s) | 0.8393 | +0.1103 |

## +1 frame 적용 시 기존 조건 비교

| Condition | 3D RMSE @ 0 | Mean-axis r @ 0 | 3D RMSE @ +1 | Mean-axis r @ +1 | r(X/Y/Z) @ +1 |
|---|---:|---:|---:|---:|---:|
| Raw | 1.6418 | 0.715 | 1.2815 | 0.820 | 0.620 / 0.915 / 0.828 |
| Camera only | 2.4663 | 0.487 | 2.2172 | 0.633 | 0.430 / 0.915 / 0.217 |
| FootMR baseline | 2.7399 | 0.498 | 2.4695 | 0.656 | 0.313 / 0.915 / 0.443 |
| C contact | 3.1965 | 0.451 | 3.1011 | 0.583 | 0.297 / 0.914 / 0.139 |

## Acceleration Bland–Altman (+1 frame)

각 셀은 `bias [95% lower, upper LoA]`이며 단위는 m/s²이다.

| Condition | X/ML | Y/Vertical | Z/AP |
|---|---:|---:|---:|
| Raw | +0.010 [-1.191, +1.211] | +0.002 [-1.394, +1.399] | -0.005 [-1.718, +1.708] |
| Camera only | +0.003 [-2.027, +2.033] | +0.002 [-1.394, +1.398] | -0.010 [-3.597, +3.578] |
| FootMR baseline | -0.019 [-3.109, +3.071] | +0.003 [-1.391, +1.397] | -0.010 [-3.474, +3.454] |
| C contact | -0.059 [-3.624, +3.507] | +0.003 [-1.397, +1.402] | +0.027 [-4.701, +4.756] |

## 해석 원칙

- 주 지표는 후처리 영향을 거의 받지 않는 Raw Y/Vertical velocity와 acceleration이다.
- 보조 지표는 X/Y/Z 축별 correlation을 Fisher 평균한 velocity와 acceleration이다.
- 보행 파형은 주기적이므로 ±0.5초 끝의 다른 gait cycle peak는 동기화 근거로 단독 채택하지 않는다.
- 최종 sync 판정은 현재점 주변 ±3프레임에서 수행했다.
- trial별 최적 shift의 방향이 일관적인지와 0-shift 대비 gain을 함께 본다.
- Bland–Altman은 frame을 독립 표본으로 간주한 descriptive 진단이다. 동일 피험자의 반복 시계열이므로 population-level 95% confidence interval로 해석하지 않는다.

## Trial-level local optima extract

Columns: `trial,metric,accepted_shift_r,best_shift_frames,best_shift_s,best_shift_r,correlation_gain`

```csv
S18_Walk_1,vertical_velocity,0.7345667813029055,2,0.06666666666666667,0.9597344996374599,0.2251677183345544
S18_Walk_2,vertical_velocity,0.8353465023998898,1,0.03333333333333333,0.9645186602903483,0.12917215789045844
S18_Walk_3,vertical_velocity,0.871350504613262,1,0.03333333333333333,0.9716870780247223,0.1003365734114603
S18_Walk_4,vertical_velocity,0.8662670211180749,1,0.03333333333333333,0.9556141402749557,0.08934711915688087
S18_Walk_5,vertical_velocity,0.8161771547204668,2,0.06666666666666667,0.963396413114106,0.14721925839363914
S18_Walk_1,vertical_acceleration,0.7148854391236853,2,0.06666666666666667,0.9144292934432738,0.19954385431958854
S18_Walk_2,vertical_acceleration,0.8246364300140125,1,0.03333333333333333,0.9329279491634861,0.10829151914947355
S18_Walk_3,vertical_acceleration,0.8120797728349441,1,0.03333333333333333,0.946212631512798,0.13413285867785385
S18_Walk_4,vertical_acceleration,0.8203026663977003,1,0.03333333333333333,0.9053351520376139,0.08503248563991361
S18_Walk_5,vertical_acceleration,0.7936912633516636,1,0.03333333333333333,0.9333510988448729,0.13965983549320926
S18_Walk_1,all_axis_acceleration,0.6425567830761706,2,0.06666666666666667,0.8180698209544601,0.17551303787828954
S18_Walk_2,all_axis_acceleration,0.7387794382506899,1,0.03333333333333333,0.8740661909196928,0.13528675266900292
S18_Walk_3,all_axis_acceleration,0.7615667895412371,1,0.03333333333333333,0.8635783177156074,0.10201152817437031
S18_Walk_4,all_axis_acceleration,0.7854880027700141,1,0.03333333333333333,0.8307359794531036,0.04524797668308955
S18_Walk_5,all_axis_acceleration,0.6980839550725374,1,0.03333333333333333,0.8230129222545186,0.12492896718198121
```
