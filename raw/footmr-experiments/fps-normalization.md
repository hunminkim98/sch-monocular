# FootMR FPS 정규화 실험 기록

> Source: Local experiment artifacts: outputs/fps_ablation_5/comparison/report.md; outputs/fps_ablation_5/comparison/visual_review.md
> Collected: 2026-07-29
> Published: 2026-07-29

## 정량 비교 보고서 원문

30 fps 정규화는 FootMR 출력의 **대표적인 2D 정합도와 시간축 안정성**을 개선했습니다.
5개 영상 모두에서 재투영 오차 중앙값, 관절 jerk p95, root jerk 중앙값이 낮아졌습니다.

후속 시각 검토에서 초기 contact-logit 기반 foot sliding 지표가 공중에 떠 있는 발까지
접촉으로 집계한 사실을 확인했습니다. 따라서 **실제 foot sliding이 5개 모두
악화됐다는 초기 해석은 폐기합니다.** 지면 근접 구간으로 다시 평가하면 Walk 1은
개선, Walk 2와 Walk 5는 악화, Walk 3과 Walk 4는 접촉 부족으로 판정 불가였습니다.
자세한 내용은 `visual_review.md`에 정리했습니다.

따라서 현재 5개 영상만으로 내릴 수 있는 결론은 다음과 같습니다.

- FootMR의 고정 30 Hz 시간축에는 입력을 30 fps CFR로 정규화하는 편이 더 타당합니다.
- 일반적인 pose 정합도와 smoothness는 정규화 조건이 더 좋았습니다.
- 발 미끄러짐과 이동 거리의 절대 정확도는 3D ground truth가 없으므로 개선을 입증하지
  못했습니다. 이 부분은 후속 ground-truth 평가가 필요합니다.

### 실험 조건

- 입력: `S18_Walk_1.mov`부터 `S18_Walk_5.mov`까지, nominal 60 fps
- native 조건: 원본 시간 간격을 유지한 537, 479, 479, 540, 542 frame
- normalized 조건: FFmpeg `fps=30`, CFR로 변환한 270, 240, 240, 270, 271 frame
- 모델: `inputs/checkpoints/footmr/footmr_checkpoint.ckpt`
- 설정: 두 조건 모두 동일한 preprocessing, `static_cam`, 기본 post-processing
- 비교 시간축: normalized 30 Hz frame timestamp에 대응하는 native frame을 nearest
  timestamp로 선택
- 검증: 모든 대응 timestamp 오차는 0 ms였으며, 10개 결과 tensor는 모두 finite

렌더 영상은 품질 지표 계산에 사용하지 않았습니다. 비교는 저장된 FootMR 결과 tensor와
ViTPose keypoint를 직접 사용했습니다.

### 영상별 개선율

양수는 normalized 30 fps가 개선된 경우이며, 모든 지표는 낮을수록 좋습니다.

| 영상 | frame 수 native → 30 fps | 재투영 중앙값 | 동일 접촉 발 속도 중앙값 | 관절 jerk p95 | root jerk 중앙값 |
|---|---:|---:|---:|---:|---:|
| Walk 1 | 537 → 270 | +9.3% | -241.3% | +42.1% | +42.1% |
| Walk 2 | 479 → 240 | +1.9% | -19.4% | +19.6% | +75.1% |
| Walk 3 | 479 → 240 | +2.2% | -519.4% | +37.4% | +35.9% |
| Walk 4 | 540 → 270 | +5.1% | -161.6% | +11.5% | +18.0% |
| Walk 5 | 542 → 271 | +23.9% | -294.2% | +5.3% | +22.8% |
| **영상별 개선율 중앙값** | — | **+5.1%** | **-241.3%** | **+19.6%** | **+35.9%** |

### 종합 결과

#### 개선된 항목

- 재투영 오차 중앙값: 5/5 개선, 영상별 개선율 중앙값 **5.1%**
- root-aligned joint jerk p95: 5/5 개선, 중앙값 **19.6%**
- global root jerk 중앙값: 5/5 개선, 중앙값 **35.9%**
- confidence-weighted 재투영 평균: 3/5 개선, 중앙값 **3.0%**
  - Walk 5의 큰 outlier 감소로 산술 평균 개선율은 14.7%였으나, 표본이 작고 한 영상의
    영향이 크므로 중앙값을 대표값으로 사용했습니다.

#### 악화되거나 혼재된 항목

- 동일 접촉 구간 발 속도 중앙값: 0/5 개선
  - 영상별 normalized/native 비율 중앙값 **3.41배**
- 모델 자체 접촉 판정을 사용한 발 미끄러짐 중앙값: 1/5 개선
- global root jerk p95: 2/5 개선, 영상별 개선율 중앙값 **-51.8%**
- 재투영 오차 p95: 1/5 개선, 영상별 개선율 중앙값 **-7.0%**
  - 다만 Walk 5의 큰 outlier는 정규화 조건에서 크게 감소했습니다.

### global trajectory 차이

정규화 결과는 native 결과보다 수평 path length가 영상별 10.0~29.8% 길었고
net displacement가 22.1~35.3% 컸습니다. 공통 시간축에서 두 결과의 start-aligned
global root 위치 차이 중앙값은 영상별 340~787 mm였지만, root-aligned COCO23 관절
차이 중앙값은 17~26 mm였습니다.

즉, fps 선택은 국소 pose보다 **global 이동 궤적**에 훨씬 큰 영향을 주었습니다.
native 60 fps 입력에서 낮게 나온 발 속도는 실제 정확도 향상일 수도 있지만,
FootMR가 60 fps를 30 fps로 해석하고 contact post-processing을 실제 시간 기준으로
두 배 자주 적용하면서 이동 궤적을 더 강하게 억제한 결과일 수도 있습니다.
3D ground truth 없이는 두 설명을 분리할 수 없습니다.

### 지표 정의와 한계

- 재투영 오차: 예측한 incam COCO23 관절을 영상에 투영한 뒤 ViTPose keypoint와 비교
- 관절 jerk: hip-center로 root 정렬한 COCO23 관절의 3차 시간 차분
- root jerk: global translation의 3차 시간 차분
- 발 미끄러짐: global 좌표의 좌·우 발 중심 수평 속도
- 동일 접촉 구간: native와 normalized 양쪽이 모두 해당 발을 접촉으로 판정한 interval

이 지표들은 3D ground truth가 없는 상태의 self-consistency proxy입니다. 또한 5개
영상은 같은 세션의 유사한 걷기 영상이므로 사람·카메라·동작 다양성을 대표하지 않습니다.
절대적인 3D 정확도, 실제 보행 거리, 접촉 정확도는 평가하지 못합니다.

### 권장 사항

현재 production 경로에서는 30 fps CFR 정규화를 유지하는 것이 합리적입니다. 코드의
static-joint 기준이 30 fps(`0.033`초)를 가정하고, global translation도 frame 단위
velocity를 누적하기 때문입니다.

연구 결과로 “정확도가 개선됐다”고 주장하려면 다음 중 하나가 추가로 필요합니다.

1. marker/force-plate가 있는 3D ground truth에서 MPJPE, root trajectory error,
   foot-contact precision/recall, foot-skating을 비교
2. 최소한 실제 보행 거리와 지면 접촉 구간을 수동 annotation한 검증 세트로 비교

## 시각 검토 보고서 원문

### 정정된 결론

초기 contact-logit 기반 지표만으로는 실제 foot sliding을 판단할 수 없습니다.
시각화 결과 native 조건의 Walk 1, 3, 4에서 인체가 지면으로부터 크게 떠 있는데도
FootMR contact logit이 접촉으로 판정한 구간이 확인됐습니다. 이 구간의 낮은 수평 발
속도는 좋은 접촉이 아니라 global vertical drift로 인한 잘못된 이점입니다.

실제 mesh 지면 근접도와 수직 발 속도를 적용해 다시 확인한 결과는 다음과 같습니다.

- Walk 1: normalized 30 fps가 foot sliding과 grounding 모두 개선
- Walk 2: normalized가 더 잘 grounding되지만, grounding된 구간의 sliding은 악화
- Walk 3: 두 조건 모두 대부분 공중에 떠 있어 foot sliding 판정 불가
- Walk 4: 두 조건 모두 대부분 공중에 떠 있어 foot sliding 판정 불가
- Walk 5: normalized가 더 잘 grounding되지만, 마지막 stance의 sliding은 명확히 악화

따라서 **실제 foot sliding이 5개 모두 악화됐다는 결론은 성립하지 않습니다.**
평가 가능한 3개 중 Walk 1은 개선됐고 Walk 2와 5는 악화됐습니다.

### Grounding

`mesh-ground`는 각 frame에서 가장 낮은 mesh vertex와 고정 ground plane 사이의
거리입니다. 낮을수록 좋습니다.

| 영상 | Native 중앙값 | Normalized 중앙값 | 거리 감소 | 20 cm 이내 frame 비율 |
|---|---:|---:|---:|---:|
| Walk 1 | 1.95 m | 0.53 m | 73.0% | 9% → 14% |
| Walk 2 | 0.48 m | 0.22 m | 53.4% | 18% → 44% |
| Walk 3 | 2.18 m | 1.04 m | 52.2% | 2% → 5% |
| Walk 4 | 2.79 m | 1.20 m | 56.9% | 2% → 7% |
| Walk 5 | 0.21 m | 0.09 m | 56.3% | 49% → 100% |

30 fps 정규화는 5개 모두에서 vertical grounding을 크게 개선했습니다. 다만 Walk 1,
3, 4는 normalized 결과도 중앙값 기준 0.53~1.20 m 떠 있어 global vertical
trajectory 문제는 해결되지 않았습니다.

### 실제 지면 근접 구간의 foot sliding

접촉 interval은 발 관절이 ground에서 20 cm 이내이고 수직 속도가 0.25 m/s 미만인
경우로 제한했습니다. 표의 값은 해당 구간의 발 수평 속도 중앙값입니다.

| 영상 | Native | Normalized | 판정 |
|---|---:|---:|---|
| Walk 1 | 139.0 mm/s | 52.0 mm/s | normalized 개선 |
| Walk 2 | 51.5 mm/s | 439.4 mm/s | normalized 악화 |
| Walk 3 | 접촉 없음 | 접촉 없음 | 판정 불가 |
| Walk 4 | 접촉 없음 | 접촉 없음 | 판정 불가 |
| Walk 5 | 16.6 mm/s | 67.0 mm/s | normalized 악화 |

Walk 5 normalized 결과에서는 마지막 왼발 stance 동안 약 0.67 m의 누적 이동이
확인돼 시각적으로도 명확한 sliding입니다. Walk 2 normalized에도 한 stance에서 약
0.24 m의 이동이 있습니다. 반면 Walk 1의 대표 grounded stance에서는 normalized
궤적이 더 짧았습니다.

### 시각화 읽는 법

- 왼쪽 파란색: native 약 60 fps 입력
- 오른쪽 주황색: normalized 30 fps 입력
- checkerboard: 고정 world ground
- `mesh-ground`: 현재 mesh의 ground clearance
- 청록색/자홍색: 각각 왼발/오른발이 실제 ground 근처에 있는 동안의 누적 궤적
- 궤적이 길수록 해당 stance에서 foot sliding이 큼

### 최종 판단

30 fps 정규화는 모델의 30 Hz 시간축에 맞고, pose smoothness와 grounding을 확실히
개선했습니다. 그러나 Walk 2와 Walk 5에서는 접촉 중 horizontal foot locking이
약해지는 trade-off가 실제로 관찰됐습니다.

Production 입력 정규화는 유지하되, 다음 개선 대상은 FPS 전처리보다 global
post-processing입니다. 특히 frame별 ground constraint와 stance 동안의 foot-lock
constraint를 별도로 보강해야 합니다.
