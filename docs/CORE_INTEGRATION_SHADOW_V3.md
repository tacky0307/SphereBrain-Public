# Core Integration Shadow v3

## Purpose

Measure whether structural assistance can be prepared for Core integration without overriding strong baseline decisions or changing the selected candidate set.

## Safety design

- Core and `brain.json` remain read-only.
- Structural assistance is evaluated only when the top two baseline candidates are within `tie_margin`.
- Margins at or below `near_zero_margin` are classified as genuine numerical ties.
- Near-zero ties use only an absolute modulation cap.
- Meaningful non-zero margins use the smaller of the absolute cap and the relative cap.
- A changed winner in a near-zero tie is recorded as `tie_resolved_by_structure`.
- A changed winner outside the near-zero range is recorded as `strong_decision_overridden`.
- The selected Edge set, route set, activated Node set, and step count are compared directly.

## Default parameters

- gain: `0.02`
- tie margin: `0.0025`
- near-zero margin: `1e-8`
- relative cap ratio: `0.35`
- absolute cap: `5e-5`

## Pass condition

`safe_for_core_feature_flag` becomes true only when every test region is deterministic and read-only, all caps are respected, no strong decision or selected candidate set is changed, and route/Node/step outcomes remain identical.

Passing this experiment permits only an initial-value-OFF Core feature flag. It does not automatically enable structural assistance in normal learning runs.
