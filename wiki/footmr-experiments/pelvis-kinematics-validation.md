# FootMR pelvis kinematics 직접 검증

> Sources: FootMR local pelvis kinematics experiment, 2026-08-01; FootMR Raw Y drift diagnostic, 2026-08-01; FootMR C_Temporal linear grounding ablation, 2026-08-01; FootMR contact-linear grounding production validation, 2026-08-01; FootMR contact-linear grounding GPU end-to-end validation, 2026-08-01; FootMR contact-linear grounding visualization, 2026-08-01; FootMR local production artifact cleanup audit, 2026-08-01
> Raw: [FootMR pelvis kinematics, sync, agreement 실험 기록](../../raw/footmr-experiments/2026-08-01-pelvis-kinematics-sync-agreement.md); [Raw FootMR Y drift와 force-estimation 영향 실험 기록](../../raw/footmr-experiments/2026-08-01-raw-y-drift-force-impact.md); [C_Temporal 기반 sequence-level 선형 grounding A/B 실험 기록](../../raw/footmr-experiments/2026-08-01-contact-linear-grounding-ablation.md); [C_Temporal contact-linear grounding production 구현 검증](../../raw/footmr-experiments/2026-08-01-contact-linear-grounding-production-validation.md); [Contact-linear grounding Walk 1-5 GPU end-to-end 검증](../../raw/footmr-experiments/2026-08-01-contact-linear-grounding-gpu-e2e.md); [Contact-linear grounding Raw vs Grounded global 시각화](../../raw/footmr-experiments/2026-08-01-contact-linear-grounding-visualization.md); [FootMR production 산출물 균형 정리 감사 기록](../../raw/footmr-experiments/2026-08-01-production-artifact-cleanup.md)
> Updated: 2026-08-01

## Overview

FootMR TRC의 ASIS midpoint를 marker-based OpenSim MOT의 pelvis translation과
직접 비교했다. 현재 FootMR 출력에는 legacy markerless OpenSim model을 다시
적용하지 않았다. Position은 FootMR baseline이 가장 가까웠지만 velocity와
acceleration은 Raw가 가장 안정적이었다. Pelvis kinematics 기준으로는 기존
동기화보다 GT를 +1 frame 뒤로 옮긴 결과가 더 일관됐으며, 이 보정 뒤에도 Raw가
가장 낮은 acceleration RMSE와 가장 높은 correlation을 보였다. 별도 Y-drift
진단에서는 Raw가 reference 대비 약 -4 cm/s로 이동했지만, 곡률이 만드는 등가
힘은 최대 0.197347% BW였다. Drift 제거는 position과 velocity만 개선하고
acceleration은 개선하지 않았다. 후속 blind A/B에서는 C_Temporal contact로만
추정한 sequence-level 선형 ground가 position RMSE를 72.208351%, velocity RMSE를
19.812780% 줄이면서 acceleration을 바꾸지 않았다. 따라서 이 방식은 평지 보행
전용 선택형 production mode로 구현했으며, 일반 detrending이나 frame별 grounding과는
구분한다.

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

## Raw Y drift는 크지만 acceleration bias의 주원인은 아니다

Raw 저장 결과의 `smpl_params_global`은 이미 후처리됐으므로 그대로 사용하지
않았다. 저장된 model output을 다시 decode하고 `static_cam=true` 조건에서
contact-aware root correction 전 global translation을 재구성했다. Barycentric
ASIS midpoint와 marker-based `pelvis_ty`의 잔차에 Huber robust quadratic trend를
적합했다. 피험자가 화면 밖으로 나가는 마지막 구간은 common MOT overlap과 pose
confidence 기준으로 제외했다.

`+Y`가 위 방향인 공식 좌표계에서 다섯 Walk 모두 slope가 음수였다. 따라서 이번
측정에서 보이는 것은 위쪽이 아니라 reference 대비 아래쪽 drift다.

| Walk | 중심 slope (cm/s) | trend net change (m) | 곡률 등가 힘 (% BW) |
|---|---:|---:|---:|
| 1 | -4.597972 | -0.105753 | -0.197347 |
| 2 | -4.057600 | -0.081152 | +0.097587 |
| 3 | -4.024600 | -0.099274 | -0.155204 |
| 4 | -3.855375 | -0.100240 | -0.151648 |
| 5 | -4.492500 | -0.116806 | -0.100772 |

Slope 범위는 -4.598~-3.855 cm/s였지만 곡률의 최대 절댓값은 0.197347% BW였다.
즉 position drift는 명확하나, 그 drift가 만드는 systematic acceleration은 작다.

GT 잔차에서 적합한 quadratic trend를 제거한 전후 비교는 다음과 같다. 이 보정은
oracle diagnostic이며 production에서 사용할 수 있는 blind correction이 아니다.

| Vertical 지표 | Raw | Detrended | 해석 |
|---|---:|---:|---|
| Position RMSE (m) | 0.054737 | 0.042777 | 21.850281% 감소 |
| Velocity RMSE (m/s) | 0.067871 | 0.052869 | 22.103464% 감소 |
| Acceleration RMSE (m/s²) | 0.868498 | 0.868460 | 0.004317% 감소 |
| Acceleration r | 0.884658 | 0.884650 | 사실상 동일 |

Acceleration Bland–Altman bias는 -0.010543 m/s²였지만 limits of agreement는
[-1.715016, +1.693930] m/s²였다. Pelvis를 whole-body COM이라고 단순 가정해
환산하면 RMSE는 8.856212% BW, limits는 [-17.488298, +17.273275]% BW다.
이 값은 실제 GRF 오차가 아니라 pelvis acceleration proxy다. 결론은 drift 곡률의
systematic bias가 작다는 것이지, frame별 force 오차가 충분히 작거나 실제
GaitDynamics가 검증됐다는 뜻이 아니다.

이 결론은 6.0 Hz low-pass sensitivity에서 최대 0.195380% BW, Raw root
translation sensitivity에서 최대 0.214152% BW로 유지됐다. Sync 0 frame 대비
+1 frame에서도 acceleration RMSE가 1.044981에서 0.868498 m/s²로 낮아져 기존
sync 방향과 일치했다.

## 평지 보행에서는 contact로 한 개의 ground line을 추정할 수 있다

GT-informed detrending의 실사용 불가능성을 해결하기 위해 force plate와
marker-based GT를 보정 입력에서 완전히 제외했다. 기존 C_Temporal이 검출한 stance의
중앙부와 3개 foot vertex의 최저 Y를 사용하고, 양발의 상수 surface offset만 허용한
Huber line 하나를 sequence 전체에 적합했다.

```text
h(t, side) = a + b (t - tc) + c I(right)
ground(t) = a + c/2 + b (t - tc)
Y_corrected(t) = Y_raw(t) - ground(t)
```

Frame별·stance별 offset과 quadratic term은 없다. 따라서 이 보정은 Raw pose와
X/Z를 보존하고, Y position에서 하나의 직선만 제거한다.

| Vertical 지표, GT +1 frame | Raw | Contact-linear grounded | 변화 |
|---|---:|---:|---:|
| Position RMSE (m) | 0.056321 | 0.015653 | 72.208351% 감소 |
| Velocity RMSE (m/s) | 0.067812 | 0.054376 | 19.812780% 감소 |
| Acceleration RMSE (m/s²) | 0.867214 | 0.867214 | 변화 없음 |
| Acceleration r | 0.885027 | 0.885027 | 변화 없음 |

Position과 velocity RMSE는 Walk 1-5 모두에서 감소했다. 보정 전후 acceleration과
correction line의 이차 미분은 부동소수점 오차 수준에서 0이었다. 즉 이전의 매 frame
grounding 실수처럼 frame마다 다른 offset을 만들어 가짜 수직 acceleration을
추가하지 않는다.

Full reliable sequence에서 contact slope는 -3.487615~-2.590457 cm/s였고 모두
GT residual과 같은 하강 방향이었다. GT가 존재하는 동일 시간창만 진단적으로 비교하면
contact와 oracle slope의 최대 차이는 0.754548 cm/s였다. Same-window 선택은 검증에만
사용했고 실제 보정 line은 GT 없이 full reliable sequence에서 적합했다.

Sync 0 frame에서도 position RMSE는 0.052704에서 0.013661 m, velocity RMSE는
0.092641에서 0.084683 m/s로 감소했고 acceleration RMSE 1.045105 m/s²는 동일했다.
Foot proxy 평균 sensitivity와 본 slope의 최대 차이는 0.083152 cm/s였으며,
leave-one-stance-out에서도 모든 slope가 음수 방향을 유지했다. Pose confidence gate는
화면 이탈 noise가 있는 영상 마지막 부분을 최소 1.733333초 이상 지면 적합에서
제외했다.

이 결과는 한 피험자의 평지 Walk 5개에 한정된다. 경사로·계단·달리기·점프에는
적용하지 않으며, 양발 stance 수와 temporal coverage, pose confidence, residual,
slope magnitude가 부족하면 보정 없이 Raw로 되돌아가야 한다. 또한 실제 force
estimation 검증이 아니라 pelvis kinematics 검증이다.

> **Status: Outdated** (2026-08-01)
> A/B 당시 제안했던 temporal coverage 비율 guard는 최종 production 기준에서
> 제거했다. 시간축에는 임의의 영상 비율을 요구하지 않고 실제 Huber 설계행렬의
> rank만 검사한다.

## Production 구현과 최종 guard

`contact-linear`을 선택형 평지 보행 mode로 추가했고 기본값은 `none`으로 유지했다.
새 mode는 FootMR anti-sliding 전 Raw motion을 사용한다. 기존 FootMR anti-sliding
구현 파일과 논리는 수정하지 않았으며, pose와 X/Z도 바꾸지 않는다. 결과 cache가
섞이지 않도록 새 출력 이름에는 `_gcontact`가 붙는다.

최종 적용 조건은 contact probability 0.60 이상, foot 및 lower-body confidence
0.50 이상, contact run 중앙 60%, 최소 8 frame, stance별 최대 9 frame이다. 전체
stance는 4개 이상, 좌우는 각각 1개 이상이어야 한다. Residual MAD는 0.03 m 이하,
slope 절댓값은 0.10 m/s 이하여야 하며 설계행렬 rank는 3이어야 한다. 접촉 표본의
time coverage 또는 span ratio는 계산하거나 저장하지 않는다.

300 frame 합성 sequence의 frame 10-66에만 4개 stance를 둔 테스트는 전체 범위의
18.7%만 사용하지만 정상 적용됐다. 반대로 stance 3개 또는 한쪽 발 stance만 있는
경우에는 Raw tensor 값을 정확히 보존했다. 새 grounding 테스트 5개와 기존 contact
테스트 6개를 합친 11개가 통과했다.

저장된 실제 Walk 1-5의 Raw root를 재구성해 production 함수를 실행한 회귀 결과는
다음과 같다.

| Walk | Stance (L/R) | Slope (cm/s) | Residual MAD (cm) |
|---|---:|---:|---:|
| 1 | 10 (5/5) | -3.016291 | 2.306844 |
| 2 | 9 (4/5) | -2.590464 | 0.817968 |
| 3 | 10 (5/5) | -3.060453 | 1.921341 |
| 4 | 10 (5/5) | -2.716428 | 1.899656 |
| 5 | 10 (5/5) | -3.487621 | 1.517180 |

다섯 Walk 모두 적용됐고 이전 A/B slope와의 최대 차이는 0.000007 cm/s였다. Pose와
X/Z translation은 bitwise 동일했고 Float32에서 보정선의 수치 이차 미분도 매우
작았다. 현재 WSL 세션에서는 NVIDIA driver가 보이지 않아 새 checkpoint GPU
end-to-end 실행은 시작 전에 중단됐지만, 실제 저장 inference의 production 함수
회귀와 CLI help·전처리 경로는 확인했다.

> **Status: Outdated** (2026-08-01)
> NVIDIA driver가 보이지 않은 것은 sandbox 내부 실행에 한정된 상태였다. GPU 접근을
> 허용한 실행에서는 Walk 1-5의 checkpoint-to-output 경로가 모두 완료됐다.

## Walk 1-5 GPU checkpoint-to-output 검증

RTX 5090 Laptop GPU에서 `--static_cam --grounding contact-linear`로 다섯 Walk를 다시
추론했다. 현재 PyTorch 2.10과 기존 Ultralytics checkpoint의 `weights_only` 기본값이
충돌해 YOLO tracker를 새로 돌릴 수 없었으므로, 동일 Walk와 focal에서 이미 검증한
bbox, ViTPose, ViT feature만 재사용했다. FootMR checkpoint 이후의 Raw 선택,
C_Temporal grounding, 저장, incam/global render와 영상 병합은 새 CLI 경로가 수행했다.

| Walk | Stance (L/R) | Slope (cm/s) | Residual MAD (cm) | Output frames |
|---|---:|---:|---:|---:|
| 1 | 10 (5/5) | -3.016285 | 2.306828 | 270 |
| 2 | 9 (4/5) | -2.590460 | 0.817925 | 240 |
| 3 | 10 (5/5) | -3.060461 | 1.921363 | 240 |
| 4 | 10 (5/5) | -2.716432 | 1.899667 | 270 |
| 5 | 10 (5/5) | -3.487622 | 1.517167 | 271 |

모두 `applied=True`였고 fallback은 없었다. Raw 대비 body pose와 X/Z translation은
Walk 1-5 모두 bitwise 동일했고, 병합 영상은 모두 30/1 fps였다. Fresh GPU slope와
CPU production-function 회귀의 최대 차이는 0.000008 cm/s였다.

Global renderer는 일부 frame에서 PyTorch3D coarse rasterization bin overflow
warning을 냈다. 영상 파일과 frame 수는 생성됐지만 시각적 완전성은 별도 검토해야
한다. 이 경고는 저장된 SMPL grounding 수치에는 영향을 주지 않는다.

## Raw vs Grounded global 시각화

최종 grounding 효과를 직접 검토할 수 있도록 Walk 1-5의 Raw와 Grounded global
mesh를 나란히 렌더링해 `outputs/contact_grounding_raw_vs_grounded_5_30fps/`에
로컬 영구 보관했다. 두 panel은 같은 XZ 원점, yaw transform, 고정 camera와 실제
`Y=0 m` 지면을 공유하며 panel별 Y 정렬은 하지 않았다. Raw가 지면 아래에 있어도
몸 전체가 보이도록 지면은 얇은 line grid로 표시했다.

검증 당시 생성한 Walk별 중간 영상은 1280x800, 30/1 fps이며 frame 수는
270, 240, 240, 270, 271이다.
통합 영상
`S18_Walk_1_5_Raw_vs_Grounded_Global.mp4`는 1291 frame, 43.033333초다.
영상에 표시한 slope는 Walk 1부터 -3.016285, -2.590460, -3.060461, -2.716432,
-3.487622 cm/s로 production GPU 회귀값과 일치했다.

Walk 1의 시작·중간·끝과 통합 영상의 Walk 2-5 전환 frame을 육안 검수했다. Raw와
Grounded mesh 전체, `Y=0` grid, label이 정상이고 Grounded의 발이 grid 부근에
놓였다. 이 영상은 global vertical offset과 선형 drift 보정의 시각화이며 contact
정확도나 force-estimation 성능의 추가 검증은 아니다.

검증 완료 후 production 산출물 균형 정리를 적용했다. Walk별 중간 영상은 제거하고
위 통합 영상만 영구 보관했다. Appendix 재현용 metadata는 실험 output과 분리해
`inputs/foot_contact_appendix/`로 이동했다.

## 현재 결정

1. Dynamics 입력의 기반은 FootMR anti-sliding 전 Raw로 유지한다. 평지 보행
   mode에서는 품질 조건을 통과한 C_Temporal contact로 sequence-level 선형 Y
   grounding 하나만 적용하는 방식을 production 후보로 채택한다.
2. GT-informed detrending, quadratic correction, 매 frame grounding, stance별
   offset은 production에 추가하지 않는다.
3. FootMR baseline과 C contact anti-sliding은 시각적 sliding 완화용 결과로
   분리하고 dynamics 입력에는 자동 적용하지 않는다.
4. Pelvis 비교에는 +1 frame sensitivity correction을 보고하되 공유 force/contact
   sync를 자동 변경하지 않는다.
5. Raw의 실제 force-estimation 우위는 아직 확정하지 않는다. Subject scaling과
   provenance가 검증된 model에서 실제 GaitDynamics 또는 inverse dynamics output을
   force plate와 비교해야 한다.
6. Contact-linear grounding은 아직 production에 통합되지 않았다. 구현 시 같은
   Walk 1-5 regression과 synthetic linear-drift test, 품질 실패 시 Raw fallback을
   영구 검증한다.

> **Status: Outdated** (2026-08-01)
> 선택형 `contact-linear` mode 통합과 Walk 1-5 및 합성 회귀 검증을 완료했다. 최종
> stance 기준은 전체 4개 이상, 좌우 각각 1개 이상이며 time coverage 비율은 없다.

7. 기본 `none` 경로는 기존 FootMR anti-sliding을 보존한다. `contact-linear`은 평지
   보행에 명시적으로 선택하며 품질 실패 시 Raw global motion을 반환한다.

## See Also

- [FootMR contact 후처리와 OpenSim 검증 경계](contact-postprocess-dynamics-and-opensim-validation.md)
- [FootMR foot-contact 시각화 표준](foot-contact.md)
