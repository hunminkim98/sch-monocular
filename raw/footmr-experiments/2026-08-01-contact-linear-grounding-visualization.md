# Contact-linear grounding Raw vs Grounded global 시각화

> Source: Local FootMR Walk 1-5 persistent visualization render
> Collected: 2026-08-01
> Published: Unknown

## 목적

Contact-linear grounding이 Raw FootMR의 pose와 수평 이동을 바꾸지 않고 전체 사람을
고정된 지면에 배치하는 효과를 시각적으로 확인하기 위해 Walk 1-5의 global mesh를
동일한 조건에서 나란히 렌더링했다.

## 렌더링 조건

- 입력은 metadata focal과 30 fps CFR 조건의 Walk 1-5다.
- FootMR checkpoint를 `static_cam=true`, `no_postproc=true`로 다시 실행해
  anti-sliding 전 Raw motion을 얻었다.
- Grounded panel은 같은 Raw에 production `contact-linear` 함수만 적용했다.
- 각 Walk의 Raw와 Grounded는 같은 XZ 원점, yaw transform, 고정 global camera를
  공유한다. Panel별 Y 정렬은 하지 않았다.
- 두 panel은 실제 `Y=0 m` 위치의 같은 지면을 사용한다. Raw mesh가 지면 아래에 있어도
  몸 전체를 확인할 수 있도록 불투명 checkerboard 대신 얇은 line grid를 사용했다.
- Raw는 orange, Grounded는 blue이며 각 frame에 Grounded-Raw Y shift, 추정 slope,
  stance 수를 표시했다.
- PyTorch3D renderer는 `bin_size=0`으로 실행해 coarse rasterization bin overflow를
  피했다.

## 보관 경로

통합 영상:

`outputs/contact_grounding_raw_vs_grounded_5_30fps/S18_Walk_1_5_Raw_vs_Grounded_Global.mp4`

Walk별 영상:

- `outputs/contact_grounding_raw_vs_grounded_5_30fps/walks/S18_Walk_1_Raw_vs_Grounded_Global.mp4`
- `outputs/contact_grounding_raw_vs_grounded_5_30fps/walks/S18_Walk_2_Raw_vs_Grounded_Global.mp4`
- `outputs/contact_grounding_raw_vs_grounded_5_30fps/walks/S18_Walk_3_Raw_vs_Grounded_Global.mp4`
- `outputs/contact_grounding_raw_vs_grounded_5_30fps/walks/S18_Walk_4_Raw_vs_Grounded_Global.mp4`
- `outputs/contact_grounding_raw_vs_grounded_5_30fps/walks/S18_Walk_5_Raw_vs_Grounded_Global.mp4`

## 수치 회귀와 영상 규격

| Walk | Slope (cm/s) | Frames | Duration (s) | Resolution | FPS |
|---|---:|---:|---:|---:|---:|
| 1 | -3.016285 | 270 | 9.000000 | 1280x800 | 30/1 |
| 2 | -2.590460 | 240 | 8.000000 | 1280x800 | 30/1 |
| 3 | -3.060461 | 240 | 8.000000 | 1280x800 | 30/1 |
| 4 | -2.716432 | 270 | 9.000000 | 1280x800 | 30/1 |
| 5 | -3.487622 | 271 | 9.033333 | 1280x800 | 30/1 |

다섯 slope는 production GPU end-to-end 회귀값과 일치했다. 통합 영상은 1291 frame,
43.033333초, 1280x800, 30/1 fps다.

## 육안 검수

- Walk 1의 0.50초, 4.50초, 8.50초 frame에서 Raw와 Grounded mesh 전체가 보이고
  Grounded의 발이 `Y=0` grid 부근에 놓이는 것을 확인했다.
- 통합 영상의 9.50초, 17.50초, 25.50초, 34.50초 frame에서 Walk 2-5 전환과 label이
  정상인 것을 확인했다.
- Raw panel에서 line grid가 몸을 가로지르는 것은 별도 Y 정렬을 하지 않았기 때문이다.
  이는 Raw global height가 지면 아래에 있음을 숨기지 않고 보여주는 의도된 표현이다.
- 이 영상은 global vertical offset과 선형 drift 보정을 시각화한다. Foot-contact 자체의
  정확도나 force-estimation 성능을 추가로 검증하는 자료는 아니다.

영상은 Git-ignored `outputs/`에 로컬 영구 보관한다. 재현에만 사용한 일회성 renderer
script와 검수용 still image는 기록 후 제거한다.
