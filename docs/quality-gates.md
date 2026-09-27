# Quality gates

Review fixed, authorized examples at original resolution and at the same frames/timecodes across variants. Two reviewers or repeat review can reduce subjective drift; a similarity score alone cannot declare success. Define thresholds only after a baseline and labeled examples exist.

| Gate | Pass condition | How to check |
| --- | --- | --- |
| Input and identity selection | Source/target are valid; correct person selected, including multi-face and re-entry cases | Input validation, contact sheet, face ID trace |
| Coverage | Every expected frame accounted for; no missed swap or unprocessed segment | Count frames and inspect sampled + transition frames |
| Geometry | Eyes/nose/mouth align without shape distortion | Paired crops at frontal, profile, pitch and expression extremes |
| Identity | Intended likeness recognizable across pose/lighting; no drift toward target or other person | Blind side-by-side human ranking; optional embedding score as supporting signal |
| Composite | Mask edge, hairline, skin tone, lighting and occluders plausible; teeth/lips/eyes retained | Zoomed crops, before/after flicker toggle |
| Temporal | No identity jumps, mask shimmer, frozen frames or progressive blur | Full short clip playback and frame-difference/contact sheet review |
| Stability | Repeatable outputs or explained variation; no crash/OOM; measurable runtime/VRAM | Repeat same case, log environment and timings |
| Provenance | Model/source use permitted; exported synthetic media handled transparently | Rights and disclosure checklist |

**Release rule:** any wrong-person swap, lost frames, unexplained crash or severe persistent artifact blocks promotion. Record per-case pass/fail and severity; an experiment that improves one metric while worsening another stays a candidate. Final identity and visual thresholds require owner-reviewed examples and are currently [UNKNOWN].
