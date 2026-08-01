# FootMR pelvis kinematics 직접 검증

> Sources: FootMR local pelvis kinematics experiment, 2026-08-01
> Raw: [FootMR pelvis kinematics, sync, agreement 실험 기록](../../raw/footmr-experiments/2026-08-01-pelvis-kinematics-sync-agreement.md)
> Updated: 2026-08-01

## Overview

FootMR TRC의 ASIS midpoint를 marker-based OpenSim MOT의 pelvis translation과
직접 비교했다. 현재 FootMR 출력에는 legacy markerless OpenSim model을 다시
적용하지 않았다. Position은 FootMR baseline이 가장 가까웠지만 velocity와
acceleration은 Raw가 가장 안정적이었다. Pelvis kinematics 기준으로는 기존
동기화보다 GT를 +1 frame 뒤로 옮긴 결과가 더 일관됐으며, 이 보정 뒤에도 Raw가
가장 낮은 acceleration RMSE와 가장 높은 correlation을 보였다.

## 비교 경계와 고정 조건

FootMR 측 pelvis proxy는 `(R_ASIS + L_ASIS) / 2`이고 reference는 marker-based
OpenSim MOT의 `pelvis_tx/pelvis_ty/pelvis_tz`다. 두 점은 정확히 같은 해부학적
점이 아니며 reference도 raw marker가 아니라 marker-based IK 결과다. 그러므로
절대 position보다 velocity, acceleration waveform과 조건 간 상대 비교를 우선한다.

Walk 1-5의 Raw, Camera only, FootMR baseline, C contact를 비교했다. Scale과 lag는
fit하지 않았고 `scale=1.0`을 유지했다. Trial별 수평 yaw 하나를 네 조건에 공통으로
적용했다. Position에만 평가 시작 0.20초의 상수 offset을 적용했으며, 이 offset은
velocity와 acceleration에 영향을 주지 않는다. 기존 6 Hz low-pass 결과를 다시
filtering하지 않았고 양 끝 0.10초를 평가에서 제외했다.

## 최초 고정 sync 결과

| Condition | Position 3D RMSE (m) | Velocity 3D RMSE (m/s) | Acceleration 3D RMSE (m/s²) |
|---|---:|---:|---:|
| Raw | 0.1318 | 0.1722 | 1.6486 |
| Camera only | 0.1785 | 0.2289 | 2.4790 |
| FootMR baseline | 0.0880 | 0.2052 | 2.7524 |
| C contact | 0.1022 | 0.2175 | 3.1994 |

FootMR baseline은 position 오차가 가장 작았지만 Raw는 velocity와 acceleration
오차가 가장 작았다. C contact acceleration RMSE는 Raw보다 94.1%, FootMR
baseline보다 16.2% 높았고 Walk 1-5 모두에서 악화가 반복됐다. Y/Vertical
acceleration은 Raw와 C contact가 사실상 같았으며 차이는 주로 X/ML과 Z/AP에서
발생했다. Contact-aware root correction이 위치를 개선하면서 미분 신호를
악화시킬 수 있음을 보여준다.

## Correlation과 sync 민감도

확정 sync 주변에서 marker-based GT timestamp를 -15~+15 video frame 이동했다.
모든 shift는 동일한 FootMR sample을 사용했고 shift마다 lag, scale, yaw를 다시
fit하지 않았다. 보행 주기의 반복 peak를 잘못 선택하지 않도록 최종 판정은
현재점 주변 ±3프레임에서 수행했다.

| Metric | Accepted r | Best local shift | Best r |
|---|---:|---:|---:|
| Vertical velocity | 0.8303 | +1 frame (+0.033 s) | 0.9566 |
| Vertical acceleration | 0.7962 | +1 frame (+0.033 s) | 0.9234 |
| XYZ velocity | 0.8205 | +1 frame (+0.033 s) | 0.9135 |
| XYZ acceleration | 0.7290 | +1 frame (+0.033 s) | 0.8393 |

Positive shift는 GT를 영상 시간축에서 뒤로 옮긴다는 뜻이다. Walk별 local optimum도
대부분 +1 frame이었고 일부는 +2 frame이었다. 따라서 pelvis kinematics에는 공통
+1 frame이 가장 합리적인 sensitivity correction이다. 다만 이 결과만으로
force-plate passage에 고정한 공유 `sync.csv`를 변경하지 않는다. Contact/force
동기화와 pelvis kinematics의 한-frame 차이는 독립적으로 관리하고 물리 event를
통해 다시 확인해야 한다.

## +1 frame 결과

Mean-axis r은 X/Y/Z Pearson correlation의 Fisher 평균이다.

| Condition | 3D acceleration RMSE (m/s²) | Mean-axis r | r(X/Y/Z) |
|---|---:|---:|---:|
| Raw | 1.2815 | 0.820 | 0.620 / 0.915 / 0.828 |
| Camera only | 2.2172 | 0.633 | 0.430 / 0.915 / 0.217 |
| FootMR baseline | 2.4695 | 0.656 | 0.313 / 0.915 / 0.443 |
| C contact | 3.1011 | 0.583 | 0.297 / 0.914 / 0.139 |

Sync correction 뒤에도 Raw가 acceleration RMSE와 correlation 모두 가장 좋았다.
모든 조건의 Y correlation이 거의 같은 반면 X/Z correlation은 후처리 조건에서
낮아졌다. 따라서 contact 정확도와 foot sliding 감소가 pelvis dynamics 개선을
보장하지 않는다.

## Bland–Altman agreement

Acceleration의 mean bias는 모든 조건과 축에서 거의 0이었지만 95% limits of
agreement 폭은 크게 달랐다. Raw의 X/ML LoA는 [-1.191, +1.211] m/s²,
Z/AP LoA는 [-1.718, +1.708] m/s²였다. C contact의 X/ML LoA는
[-3.624, +3.507] m/s², Z/AP LoA는 [-4.701, +4.756] m/s²로 훨씬 넓었다.
Raw의 Y/Vertical LoA는 [-1.394, +1.399] m/s²였고 다른 조건도 유사했다.

Near-zero bias는 높은 agreement를 뜻하지 않는다. C contact는 평균 오차는 작지만
frame별 horizontal acceleration 오차의 분산이 크다. Bland–Altman은 한 피험자의
반복 frame을 독립 표본처럼 표시한 descriptive 진단이며 population-level 95%
confidence interval로 해석하지 않는다.

## 현재 결정

1. Force-estimation용 pelvis kinematics의 잠정 입력은 Raw로 둔다.
2. FootMR baseline과 C contact anti-sliding은 시각적 sliding 완화용 결과로
   분리하고 dynamics 입력에는 자동 적용하지 않는다.
3. Pelvis 비교에는 +1 frame sensitivity correction을 보고하되 공유 force/contact
   sync를 자동 변경하지 않는다.
4. Raw의 실제 force-estimation 우위는 아직 확정하지 않는다. Subject scaling과
   provenance가 검증된 model에서 실제 GaitDynamics 또는 inverse dynamics output을
   force plate와 비교해야 한다.

## See Also

- [FootMR contact 후처리와 OpenSim 검증 경계](contact-postprocess-dynamics-and-opensim-validation.md)
- [FootMR foot-contact 시각화 표준](foot-contact.md)
