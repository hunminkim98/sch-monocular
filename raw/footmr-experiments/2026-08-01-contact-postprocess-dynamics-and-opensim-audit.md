# FootMR contact 후처리, dynamics, OpenSim provenance 감사 기록

> Source: Local experiment artifacts in outputs/contact_force_impact; current repository exporter; legacy footmr-baseline-5b/Sub18 OpenSim files; user clarification and model-name audit
> Collected: 2026-08-01
> Published: 2026-08-01

## 조사 범위

S18 Walk 1-5를 대상으로 FootMR의 contact-aware anti-sliding과 C_Temporal
contact 대체가 foot sliding 및 dynamics 관련 신호에 미치는 영향을 조사했다.
Production 코드는 변경하지 않았고 실험 파일은 `outputs/contact_force_impact`에
두었다.

비교 조건은 다음 네 가지였다.

- `raw_pre_contact`: contact-aware post-processing 전 FootMR 3D motion
- `camera_only`: camera-consistency correction과 마지막 constant Y shift만 유지하고,
  contact 기반 누적 root correction과 `process_ik`는 끈 조건
- `baseline`: 원래 FootMR contact-aware post-processing
- `c_contact_only`: 원래 FootMR 후처리 논리는 유지하고 contact label만
  C_Temporal로 교체한 조건

## Foot sliding 결과

Force-plate에서 실제 힘이 측정된 stance 10개만 사용했다. Sliding은 각 발의
`HEEL`, `MH1`, `MH5`, `TOE` marker centroid가 수평면에서 이동한 누적 경로다.
따라서 시작점과 끝점 사이의 직선거리만을 뜻하지 않는다.

| Condition | Mean speed (m/s) | Total sliding (m) |
|---|---:|---:|
| Raw | 0.2578 | 1.7275 |
| Camera-only | 0.2812 | 1.8839 |
| FootMR baseline | 0.1965 | 1.3166 |
| C-contact-only | 0.1191 | 0.7977 |

FootMR anti-sliding과 C-contact-only는 발 미끄러짐을 줄였다. 그러나
`pp_static_joint_cam`은 contact foot의 frame-to-frame displacement를 root
translation에서 빼고 이후 frame에 누적하므로, contact 전환과 marker noise가
root velocity 및 acceleration correction으로 바뀔 수 있다. C-contact-only와
baseline의 whole-body COM acceleration 차이와 root translation acceleration
차이의 상관계수는 0.9969였다.

## COM-equivalent force 보조 실험

FootMR은 force를 직접 출력하지 않는다. 아래 수치는 GaitDynamics 결과나
FootMR force prediction이 아니다. FootMR marker를 legacy OpenSim model의 IK에
넣고, BodyKinematics의 whole-body COM velocity를 미분하여 acceleration을 얻은
뒤 다음 식으로 계산한 COM-equivalent total GRF proxy다.

```text
Fx = m * ax
Fy = m * (ay + g)
Fz = m * az
```

`m = 59.7 kg`, `g = 9.80665 m/s^2`를 사용했다. Force plate 1-3을 합산하고
raw vertical force가 20.0 N보다 큰 sample만 선택했다. 양쪽 force를 body weight로
정규화한 뒤 다음 3D-vector RMSE를 계산했다.

```text
sqrt(mean(delta_Fx^2 + delta_Fy^2 + delta_Fz^2))
```

| Condition | COM-equivalent force vs force plate 3D RMSE (BW) |
|---|---:|
| Marker-based COM | 0.3394 |
| Raw | 0.3635 |
| Camera-only | 0.4126 |
| FootMR baseline | 0.4301 |
| C-contact-only | 0.4920 |

이 표를 `Force RMSE` 또는 실제 GRF estimation accuracy라고 부른 이전 표현은
잘못이다. Force plate가 모든 발의 지면반력을 항상 측정하지 않았고,
marker-based COM도 force plate 대비 0.3394 BW의 오차가 있었다. 이 결과는
legacy model 아래에서 수행한 보조 sanity check일 뿐이다. `Raw가 실제
GaitDynamics force estimation에 가장 적합하다`는 결론은 확정되지 않았다.

## 사용된 OpenSim model과 parameter

현재 저장소에는 사용자가 제공한 OpenSim model이 없었다. 실험은 사용자가
조사를 요청했던 과거 프로젝트의 다음 파일을 별도 고지 없이 재사용했다.

```text
C:/Users/gnsal/OneDrive/Desktop/Projects/SideProjects/footmr-baseline-5b/
sch-research/Sub18/armless_model/Sub18_Scaled_monocular.osim
```

현재 평가 스크립트의 `MODEL_PATH`도 이 외부 절대 경로를 가리킨다. 모델 파일의
확인된 특성은 다음과 같다.

- OpenSim document version: 40500
- credits/publication: Rajagopal et al. full-body musculoskeletal gait model
- arms가 제거된 14-body variant
- 25 coordinates
- summed body mass: 59.700000000000 kg
- gravity: `0 -9.8066499999999994 0`
- file name: `Sub18_Scaled_monocular.osim`
- internal model name: `Sub03`

파일명과 내부 subject name이 불일치한다. 이것만으로 scaling이 틀렸다고 단정할
수는 없지만 provenance warning이다. 이번 실험에서는 ScaleTool을 실행하지 않고
과거에 이미 scaled된 모델을 그대로 사용했다.

IK setup은 48 marker tasks 중 24개를 활성화했다. `R_ASIS`, `L_ASIS`,
`R_PSIS`, `L_PSIS` weight는 5이고 나머지 20개 active marker weight는 1이다.
`coordinate_file`은 `Unassigned`, `report_errors`는 `true`였다. IK coordinate는
4th-order zero-phase Butterworth 6.0 Hz low-pass filter를 적용했다.
BodyKinematics는 모든 body와 whole-body COM을 global frame에서 기록했다.

Solver accuracy와 실행 당시 OpenSim runtime version은 별도로 기록되지 않았다.
XML에 명시되지 않은 solver 설정은 완전하게 재현할 수 없다.

Marker-based reference도 과거 프로젝트의
`11_BODY_KINEMATICS_COM/body_kinematics/marker_based` 결과와
`12_GRF_SYNC_TRIMMED/marker_based` force files를 사용했다. 현재 저장소만으로는
OpenSim 비교를 독립적으로 재현할 수 없다.

## AMES 명칭 정정

OpenSim 공식 문서 및 관련 문헌 검색에서 표준 OpenSim 인체 모델 이름으로
`AMES`는 확인되지 않았다. 현재 사용한 `.osim` 파일의 credits와 publication은
Rajagopal, Dembia, DeMers, Delp, Hicks, Delp의 2016 full-body model을 가리킨다.
따라서 이 기록에서는 `AMES model`이라는 표현을 사용하지 않는다.

현재 파일의 정확한 표현은 다음이다.

```text
legacy subject-scaled armless Rajagopal 2016 model
```

## Scaling 관련 사용자 정정

사용자는 과거 workflow에서 피험자 키를 근거로 markerless 3D coordinates를
확대 또는 축소하여 좌표를 과소평가하거나 과대평가한 문제가 있었다고
확인했다. 현재 원본 exporter는 `hmr4d_results.pt`에 저장된 좌표를 그대로
출력하며 marker grounding, height scaling, rotation, filtering, trial-specific
synchronization을 추가하지 않는다.

따라서 과거 height-based coordinate scaling의 영향을 포함할 수 있는 legacy
scaled OpenSim model과 현재 무보정 FootMR coordinates를 결합한 결과는 공식
validation에 사용할 수 없다. 같은 모델을 모든 condition에 적용했다는 사실은
condition 간 exploratory comparison에는 도움이 되지만 model-coordinate scale
mismatch를 제거하지 않는다.

## 결정

1. 기존 OpenSim COM 및 COM-equivalent force 수치는 provisional로 분류하고
   production 또는 논문의 force validation 근거로 사용하지 않는다.
2. FootMR/C_Temporal contact 정확도와 동일 좌표계 안의 foot-sliding 상대 비교는
   유지한다.
3. FootMR coordinates를 피험자 키에 맞추기 위해 사후 확대·축소하지 않는다.
4. OpenSim model scaling과 FootMR coordinate scale을 별개의 문제로 관리한다.
5. 실제 force validation은 출처가 검증된 base model, 독립적인 subject scaling
   protocol, 고정된 model/IK parameters, 실제 GaitDynamics output을 사용해
   force plate와 다시 비교한다.
6. Raw, baseline, C-contact-only 중 무엇이 GaitDynamics에 가장 적절한지는 위
   재검증 전까지 미결정으로 둔다.
