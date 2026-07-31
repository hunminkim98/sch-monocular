# FootMR foot-contact 시각화 실험 기록

> Source: Local final artifacts in outputs/c_temporal_forceplate_passage_supplementary_video_5_30fps and the user-approved visualization decision
> Collected: 2026-07-31
> Published: 2026-07-31

## Final visualization decision

The fixed supplementary visualization is a three-panel 1920 x 1080 video:

1. `(a) Monocular video + 2D pose`
2. `(b) FootMR mesh reconstruction`
3. `(c) C_Temporal foot contact`

The monocular-video panel keeps the color-coded 2D pose overlay. Large circular
`L` and `R` foot badges and their attached `L/R CONTACT` labels are not shown.
Foot-contact state is communicated only in the right panel through the
`LEFT CONTACT`, `RIGHT CONTACT`, and `STATE` badges, the measured-force plot,
and the force-aligned passage timeline.

Raw video and mesh use the same source frame. Measured force is displayed for
evaluation only and is not a C_Temporal inference feature.

## Force-plate passage policy

The earlier event-local draft must not be used. Knee-angle correlation alone
selected the preceding gait cycle in all five Walk trials. The accepted offsets
were shifted to the next local knee-correlation peak, 1.04-1.09 s later, and
accepted only when both measured force pulses coincided with the visible
force-plate passage.

Each Walk is one continuous clip beginning 0.3 s before the first measured
stance and ending 0.3 s after the second measured stance. Both measured stance
pulses remain visible in the right panel. Time outside a measured force event
is not interpreted as verified non-contact Ground Truth.

The mapping is:

```text
force_time_s = video_frame / 30 - markerless_start_s - marker_source_start_s
video_frame  = round((markerless_start_s + marker_source_start_s + force_time_s) * 30)
```

The audited frame ranges are:

- Walk 1: clip 138-191; measured stance span 147-182
- Walk 2: clip 124-177; measured stance span 133-168
- Walk 3: clip 118-172; measured stance span 127-163
- Walk 4: clip 119-173; measured stance span 128-164
- Walk 5: clip 113-167; measured stance span 122-158

Measured stance is extracted directly from absolute vertical force above
20.0 N. P1 is assigned to the left foot and P2 to the right foot for these
trials.

## Final output and verification

The combined output is `foot_contact_appendix.mp4`. It is H.264, yuv420p,
constant 30.0 fps, 1920 x 1080, and 573 frames. The five individual Walk clips
contain 54, 54, 55, 55, and 55 frames.

The internal force-paired result contains 10 measured events:

- Events detected: 10/10
- Force-pulse recall: 0.830
- Temporal IoU: 0.832
- Onset MAE: 61.1 ms
- Offset MAE: 51.6 ms

The render QC status is `PASS`. This is a five-trial, one-participant internal
pilot. Synchronization is post-hoc and spatially audited, not
hardware-triggered. Participant consent must be confirmed before external
publication.

## Canonical implementation

The canonical entry point is `utils/foot_contact/appendix.py`.

Its maintained supporting files are:

- `utils/foot_contact/contact.py`
- `utils/foot_contact/visuals.py`
- `utils/foot_contact/panels.py`
- `utils/foot_contact/sync.csv`

Measured stance events are reconstructed directly from the synchronized force
files. The final renderer does not depend on the earlier A/B ablation output,
the earlier paired-summary output, or an archived C_Temporal label file.
