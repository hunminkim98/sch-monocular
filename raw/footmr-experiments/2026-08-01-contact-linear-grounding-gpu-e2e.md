# Contact-linear grounding Walk 1-5 GPU end-to-end 검증

> Source: Local FootMR Walk 1-5 checkpoint-to-output GPU regression
> Collected: 2026-08-01
> Published: Unknown

## 실행 조건

- GPU: NVIDIA GeForce RTX 5090 Laptop GPU
- 입력: `videos/S18_Walk_1`부터 `videos/S18_Walk_5`
- Mode: `--static_cam --grounding contact-linear`
- 원본 영상 metadata focal과 30 fps CFR 정규화를 사용했다.
- 현재 PyTorch 2.10의 `torch.load(weights_only=True)` 기본값과 기존 Ultralytics
  checkpoint가 충돌해 YOLO tracker를 새로 실행할 수 없었다. Grounding 변경과
  무관한 전처리 호환성 문제이므로, 같은 Walk와 focal 조건에서 이미 검증한 bbox,
  ViTPose, ViT feature를 임시 출력 경로에 복사해 재사용했다.
- FootMR checkpoint 추론, Raw mode 선택, C_Temporal grounding, 결과 저장, incam/global
  render와 두 영상 병합은 새 production CLI 경로에서 실행했다.

## 결과

| Walk | Applied | Stance (L/R) | Slope (cm/s) | Residual MAD (cm) | Output frames |
|---|---|---:|---:|---:|---:|
| 1 | True | 10 (5/5) | -3.016285 | 2.306828 | 270 |
| 2 | True | 9 (4/5) | -2.590460 | 0.817925 | 240 |
| 3 | True | 10 (5/5) | -3.060461 | 1.921363 | 240 |
| 4 | True | 10 (5/5) | -2.716432 | 1.899667 | 270 |
| 5 | True | 10 (5/5) | -3.487622 | 1.517167 | 271 |

다섯 Walk 모두 `applied=True`였고 fallback은 없었다. 저장된 top-level grounded 결과와
`net_outputs.pred_smpl_params_global`의 Raw를 비교했을 때 body pose와 X/Z translation은
Walk 1-5 모두 bitwise 동일했다. 병합 영상은 모두 30/1 fps였고 frame 수는 표와 같다.

Fresh GPU slope와 이전 CPU production-function 회귀 slope의 최대 차이는
0.000008 cm/s였다. 따라서 CPU 재구성 검증과 실제 checkpoint 경로가 같은 contact와
ground line을 재현했다.

## 제한

- 이번 검증은 기존 전처리를 재사용했으므로 raw video에서 YOLO/VitPose를 새로 실행하는
  전체 전처리 회귀는 아니다. PyTorch와 Ultralytics 호환성은 별도 환경 문제다.
- Global renderer는 일부 frame에서 PyTorch3D coarse rasterization bin overflow
  warning을 출력했다. 결과 영상 파일과 frame 수는 정상 생성됐지만 시각적 완전성은
  별도로 검토해야 한다. 이 warning은 저장된 SMPL grounding 수치에는 영향을 주지 않는다.
- 검증 산출물은 `/tmp/footmr-contact-grounding-e2e`에만 생성했으며 wiki 기록 후 삭제한다.
