# FootMR foot-contact 시각화 표준

> Sources: FootMR local experiment, 2026-07-31
> Raw: [FootMR foot-contact 시각화 실험 기록](../../raw/footmr-experiments/foot-contact.md)
> Updated: 2026-07-31

## Overview

FootMR의 foot-contact supplementary visualization은 raw video와 2D pose,
동일 프레임의 mesh reconstruction, 그리고 C_Temporal contact/force 정보를
나란히 보여주는 3-panel 형식으로 고정한다. 발 위의 큰 원형 `L`·`R` 배지는
사용하지 않으며, contact 판독 정보는 오른쪽 panel에만 모은다.

## 고정 레이아웃

왼쪽부터 다음 순서를 사용한다.

1. `(a) Monocular video + 2D pose`
2. `(b) FootMR mesh reconstruction`
3. `(c) C_Temporal foot contact`

첫 번째 panel에는 색으로 좌우를 구분한 2D pose만 표시한다. 발 위의 큰 원형
`L`·`R` 배지와 `L/R CONTACT` 꼬리표는 제거한다. 세 번째 panel은
`LEFT CONTACT`, `RIGHT CONTACT`, `STATE` 상태 표시, measured-force plot,
force-aligned passage timeline을 포함한다.

Raw video와 mesh는 항상 동일한 source frame을 사용한다. Measured force는
평가용으로만 표시하며 C_Temporal inference 입력으로 사용하지 않는다.

## 표시 구간과 동기화

각 Walk는 첫 measured stance보다 0.3 s 앞에서 시작해 두 번째 measured
stance보다 0.3 s 뒤에서 끝나는 하나의 연속 clip으로 만든다. Force가 존재하는
두 stance가 모두 영상 안에 있어야 하며, force event 밖의 구간은
non-contact Ground Truth로 해석하지 않는다.

```text
force_time_s = video_frame / 30 - markerless_start_s - marker_source_start_s
video_frame  = round((markerless_start_s + marker_source_start_s + force_time_s) * 30)
```

무릎 각도 상관만 사용한 이전 동기화는 반복되는 보행 주기 때문에 앞선 주기를
선택했다. 최종 방식은 다음 local peak를 선택하고 영상에서 실제 force-plate
통과와 두 force pulse가 일치하는지 확인한다. 이 교정은 1.04-1.09 s의
주기 이동에 해당한다.

## Force 해석

Measured stance는 absolute vertical force가 20.0 N을 넘는 연속 구간에서
직접 추출한다. 이 실험에서는 P1을 왼발, P2를 오른발로 대응한다. 별도의
과거 A/B 결과나 저장된 C_Temporal label 파일을 최종 렌더링 입력으로
사용하지 않는다.

## 검증된 결과

최종 합본은 H.264/yuv420p, constant 30.0 fps, 1920 x 1080이며 573
frames이다. Walk별 clip은 각각 54, 54, 55, 55, 55 frames이다.

내부 force-paired 결과는 10/10 events 검출, force-pulse recall 0.830,
Temporal IoU 0.832, onset MAE 61.1 ms, offset MAE 51.6 ms였다. 다만 이는
five-trial, one-participant internal pilot이며, 동기화도 hardware-trigger가
아닌 post-hoc spatial audit이다.

## 유지해야 할 구현

Canonical entry point는 `utils/foot_contact/appendix.py`이다. 다음
supporting files만 최종 시각화 구현으로 유지한다.

- `utils/foot_contact/contact.py`
- `utils/foot_contact/visuals.py`
- `utils/foot_contact/panels.py`
- `utils/foot_contact/sync.csv`

외부 공개 전에는 identifiable video에 대한 participant consent를 반드시
확인한다.
