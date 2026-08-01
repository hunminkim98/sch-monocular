# FootMR contact 후처리와 OpenSim 검증 경계

> Sources: FootMR local experiment and provenance audit, 2026-08-01; FootMR local pelvis kinematics experiment, 2026-08-01
> Raw: [FootMR contact 후처리, dynamics, OpenSim provenance 감사 기록](../../raw/footmr-experiments/2026-08-01-contact-postprocess-dynamics-and-opensim-audit.md); [FootMR pelvis kinematics, sync, agreement 실험 기록](../../raw/footmr-experiments/2026-08-01-pelvis-kinematics-sync-agreement.md)
> Updated: 2026-08-01

## Overview

FootMR의 contact-aware anti-sliding은 foot sliding을 줄이지만 root trajectory를
수정하므로 dynamics 신호를 별도로 검증해야 한다. 지금까지의 OpenSim COM 및
COM-equivalent force 비교는 과거 프로젝트의 subject-scaled model과 결과를
재사용한 exploratory test이며 공식 force validation이 아니다. FootMR coordinate
scale과 OpenSim model scaling은 분리하고, 실제 GaitDynamics 출력으로 다시
검증해야 한다. 별도의 pelvis 직접 비교에서는 Raw가 velocity와 acceleration을
가장 잘 보존했지만, 이는 실제 force output 검증과 구분한다.

## 확인된 anti-sliding trade-off

Force-plate에서 힘이 측정된 10개 stance의 foot-marker centroid 누적 수평 경로는
Raw 1.7275 m, Camera-only 1.8839 m, FootMR baseline 1.3166 m,
C-contact-only 0.7977 m였다. 따라서 baseline과 C-contact-only가 같은 FootMR
좌표계 안에서 sliding을 줄인다는 상대적 경향은 유지한다.

그러나 FootMR 후처리는 contact foot의 frame 간 displacement를 root
translation에서 제거하고 이후 frame에 누적한다. Contact label이 더 정확해도
marker noise까지 stationary constraint로 해석하면 root velocity와 acceleration을
오염시킬 수 있다. Foot contact 검출 정확도와 dynamics 품질을 같은 지표로
취급하지 않는다.

## Force RMSE 해석 정정

FootMR은 force를 직접 출력하지 않는다. 기존 표는 whole-body COM acceleration에
질량과 중력을 적용한 `m(a + g)`와 합산 force plate를 비교한 COM-equivalent
force sanity check다. GaitDynamics, inverse dynamics 또는 FootMR force prediction
결과가 아니다.

`Raw 0.3635 BW가 가장 낮으므로 실제 force estimation에 가장 적합하다`는 이전
해석은 확정된 결론으로 사용하지 않는다.

> **Status: Disputed**
> Legacy model을 사용한 proxy table 안에서는 Raw가 가장 낮았지만, force plate가
> 전체 GRF를 항상 측정하지 않았고 marker-based COM도 0.3394 BW의 오차를 보였다.
> 실제 GaitDynamics 출력에 대한 검증이 아니므로 Raw의 force-estimation 우위를
> 입증하지 않는다.

## OpenSim 재적용 없는 pelvis kinematics 검증

FootMR TRC의 `(R_ASIS + L_ASIS) / 2`를 marker-based OpenSim MOT의 pelvis
translation과 직접 비교했다. 현재 FootMR 출력에는 legacy markerless OpenSim
model, BodyKinematics, GaitDynamics 또는 GRF 추정을 적용하지 않았다.

Pelvis sync를 +1 frame 보정했을 때 3D acceleration RMSE는 Raw 1.2815 m/s²,
FootMR baseline 2.4695 m/s², C contact 3.1011 m/s²였다. X/Y/Z correlation의
Fisher 평균은 각각 0.820, 0.656, 0.583이었다. 따라서 Raw가 이 비교에서는
가장 낮은 acceleration 오차와 가장 높은 waveform correlation을 보였다.

이 결과는 legacy scaled model을 사용한 COM-equivalent force 표와 독립적인
kinematics 근거다. 다만 ASIS midpoint와 OpenSim pelvis origin은 같은 해부학적
점이 아니고 reference도 marker-based IK 결과다. Raw를 dynamics 입력의 잠정
선택으로 사용할 수는 있지만 실제 force-estimation 우위를 확정하지 않는다.

## Legacy OpenSim model 감사

실험은 현재 저장소가 아니라 과거 `footmr-baseline-5b` 프로젝트에 남아 있던
`Sub18_Scaled_monocular.osim`을 사용했다. 이 파일은 14 bodies, 25 coordinates,
총질량 59.700000000000 kg인 armless Rajagopal 계열 모델이다. 파일명은 Sub18을
가리키지만 내부 model name은 `Sub03`이다. 이번 실험에서 ScaleTool은 실행하지
않았고 과거 scaling 결과를 그대로 사용했다.

IK는 48 tasks 중 24 markers를 활성화했다. 네 pelvis markers의 weight는 5,
나머지 20 active markers는 1이었다. 이후 coordinate에는 4th-order zero-phase
6.0 Hz low-pass filter를 적용하고 BodyKinematics의 global whole-body COM을
사용했다. Solver accuracy와 실행 OpenSim runtime version은 기록되지 않아 현재
저장소만으로 완전하게 재현할 수 없다.

`AMES model`이라는 명칭은 현재 모델을 가리키는 표현으로 사용하지 않는다.

> **Status: Outdated** (2026-08-01)
> 현재 `.osim` 파일의 credits와 publication은 Rajagopal et al. 2016 full-body
> gait model을 가리킨다. 검증된 표현은 `legacy subject-scaled armless Rajagopal
> 2016 model`이다.

## Scaling 원칙

현재 FootMR exporter는 저장된 coordinates를 그대로 내보내며 height scaling을
추가하지 않는다. 과거에는 피험자 키를 이용해 markerless 3D coordinates를
확대·축소하여 metric scale을 과소 또는 과대평가한 문제가 있었다. 그 영향을
포함할 수 있는 legacy scaled model을 현재 무보정 coordinates와 결합한 결과는
공식 validation에 사용하지 않는다.

앞으로 다음 원칙을 적용한다.

1. FootMR coordinates를 목표 키에 맞춰 사후 확대·축소하지 않는다.
2. OpenSim model의 subject scaling은 FootMR coordinate rescaling과 분리한다.
3. Model scaling의 입력과 provenance를 독립적으로 기록한다.
4. 모든 비교 condition에 하나의 검증된 model과 동일한 IK parameter를 적용한다.
5. Force 평가는 실제 GaitDynamics output과 force plate를 비교한다.
6. Pelvis kinematics에는 Raw를 잠정 입력으로 두되, 실제 force estimation용
   입력은 GaitDynamics 또는 inverse dynamics 재검증 전까지 확정하지 않는다.

## 유지하는 결과와 폐기하는 결과

유지하는 결과는 C_Temporal contact의 force-paired 검출 성능, foot-contact
시각화 방식, 동일 FootMR 좌표계 안의 foot-sliding 상대 비교다.

Provisional로 내리는 결과는 legacy OpenSim model로 계산한 whole-body COM RMSE,
COM-equivalent force RMSE, 그리고 이를 근거로 한 Raw의 실제 GaitDynamics 우위
주장이다. 반면 OpenSim을 FootMR 출력에 재적용하지 않은 pelvis velocity와
acceleration 비교는 독립적인 kinematics 근거로 유지한다.

## See Also

- [FootMR foot-contact 시각화 표준](foot-contact.md)
- [FootMR pelvis kinematics 직접 검증](pelvis-kinematics-validation.md)
- [FootMR 입력 FPS 정규화](fps-normalization.md)
