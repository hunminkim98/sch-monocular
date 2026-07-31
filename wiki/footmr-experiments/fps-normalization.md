# FootMR 입력 FPS 정규화

> Sources: FootMR FPS ablation experiment, 2026-07-29
> Raw: [FootMR FPS 정규화 실험 기록](../../raw/footmr-experiments/fps-normalization.md)
> Updated: 2026-07-29

## Overview

FootMR의 고정 시간축과 입력 영상을 맞추기 위해 production 입력을 30 fps CFR로
정규화하는 결정은 현재 증거로 지지된다. 다섯 걷기 영상에서 pose 정합도, 시간축
안정성, vertical grounding은 전반적으로 개선됐다. 다만 실제 지면 근접 구간의 foot
sliding은 영상에 따라 개선과 악화가 섞였으므로, 정규화를 발 미끄러짐까지 해결하는
방법으로 해석해서는 안 된다.

## 실험 설계

동일 세션의 nominal 60 fps 걷기 영상 `S18_Walk_1.mov`부터 `S18_Walk_5.mov`까지를
두 조건으로 비교했다. native 조건은 원본 시간 간격과 537, 479, 479, 540, 542개
frame을 유지했고, normalized 조건은 FFmpeg `fps=30`과 CFR을 적용해 270, 240, 240,
270, 271개 frame으로 변환했다.

두 조건에는 동일한 FootMR checkpoint, preprocessing, `static_cam`, 기본
post-processing을 사용했다. normalized 30 Hz timestamp에 가장 가까운 native
frame을 선택했으며 모든 대응 timestamp 오차는 0 ms였다. 열 개 결과 tensor는 모두
finite였다.

## Pose 정합도와 시간축 안정성

정규화 조건은 다섯 영상 모두에서 재투영 오차 중앙값, root-aligned joint jerk p95,
global root jerk 중앙값을 낮췄다. 영상별 개선율 중앙값은 각각 5.1%, 19.6%, 35.9%다.
confidence-weighted 재투영 평균은 세 영상에서 개선됐고 영상별 개선율 중앙값은
3.0%였다.

반면 tail metric은 일관되지 않았다. global root jerk p95는 두 영상에서만 개선됐고
영상별 개선율 중앙값은 -51.8%였다. 재투영 오차 p95도 한 영상에서만 개선됐으며
영상별 개선율 중앙값은 -7.0%였다. 따라서 대표적인 정합도와 smoothness 개선은
관찰됐지만 모든 outlier가 줄었다고 볼 수는 없다.

## Grounding

각 frame의 최저 mesh vertex와 고정 ground plane 사이 거리 중앙값은 다섯 영상 모두
감소했다.

| 영상 | Native 중앙값 | Normalized 중앙값 | 거리 감소 | 20 cm 이내 frame 비율 |
|---|---:|---:|---:|---:|
| Walk 1 | 1.95 m | 0.53 m | 73.0% | 9% → 14% |
| Walk 2 | 0.48 m | 0.22 m | 53.4% | 18% → 44% |
| Walk 3 | 2.18 m | 1.04 m | 52.2% | 2% → 5% |
| Walk 4 | 2.79 m | 1.20 m | 56.9% | 2% → 7% |
| Walk 5 | 0.21 m | 0.09 m | 56.3% | 49% → 100% |

이 결과는 vertical grounding 개선을 지지하지만 문제 해결을 뜻하지는 않는다.
Walk 1, 3, 4의 normalized 결과도 mesh-ground 중앙값이 0.53~1.20 m여서 여전히
상당히 떠 있었다.

## Foot sliding 정정

초기 contact-logit 기반 분석은 실제 foot sliding이 모든 영상에서 악화됐다고
해석했지만, 이 결론은 폐기됐다. native Walk 1, 3, 4에서 mesh가 공중에 떠 있는
구간도 모델 contact logit이 접촉으로 판정했기 때문이다. 이때 낮은 수평 발 속도는
좋은 접촉이 아니라 global vertical drift가 만든 잘못된 이점이다.

발 관절이 ground에서 20 cm 이내이고 수직 속도가 0.25 m/s 미만인 구간만 다시
평가한 결과는 다음과 같다.

| 영상 | Native | Normalized | 판정 |
|---|---:|---:|---|
| Walk 1 | 139.0 mm/s | 52.0 mm/s | normalized 개선 |
| Walk 2 | 51.5 mm/s | 439.4 mm/s | normalized 악화 |
| Walk 3 | 접촉 없음 | 접촉 없음 | 판정 불가 |
| Walk 4 | 접촉 없음 | 접촉 없음 | 판정 불가 |
| Walk 5 | 16.6 mm/s | 67.0 mm/s | normalized 악화 |

평가 가능한 세 영상 중 Walk 1은 개선됐고 Walk 2와 Walk 5는 악화됐다. Walk 5의
normalized 마지막 stance에서는 약 0.67 m, Walk 2의 한 stance에서는 약 0.24 m의
누적 이동이 확인됐다.

## Global trajectory 영향

정규화 결과의 수평 path length는 native보다 영상별 10.0~29.8% 길었고 net
displacement는 22.1~35.3% 컸다. 공통 시간축의 start-aligned global root 위치 차이
중앙값은 영상별 340~787 mm인 반면, root-aligned COCO23 관절 차이 중앙값은 17~26
mm였다. 입력 FPS 선택은 국소 pose보다 global 이동 궤적에 훨씬 큰 영향을 주었다.

native 조건의 낮은 발 속도가 실제 정확도를 뜻하는지, FootMR가 60 fps를 30 fps로
해석하면서 contact post-processing을 실제 시간 기준으로 더 자주 적용해 이동
궤적을 과도하게 억제한 것인지는 현재 증거로 구분할 수 없다.

## 결정과 후속 과제

Production 경로에서는 30 fps CFR 정규화를 유지한다. FootMR의 static-joint 기준이
30 fps(`0.033`초)를 가정하고 global translation도 frame 단위 velocity를 누적하므로,
입력 시간축을 모델 가정과 맞추는 것이 타당하다.

다음 병목은 FPS 전처리보다 global post-processing이다. frame별 ground constraint와
stance 동안의 foot-lock constraint를 별도로 보강하고, marker 또는 force-plate가
있는 3D ground truth에서 root trajectory error, foot-contact precision/recall,
foot-skating을 검증해야 한다.

## 한계

사용한 영상은 같은 세션의 유사한 걷기 다섯 개뿐이어서 사람, 카메라, 동작의
다양성을 대표하지 않는다. 평가 지표는 3D ground truth가 없는 self-consistency
proxy이며 절대적인 3D 정확도, 실제 보행 거리, 접촉 정확도를 검증하지 못한다.
