# Known Issues / v1.2.1 Backlog

Items identified during the pre-v1.2.0 code audit that were intentionally
**deferred** because addressing them requires changing validated numerical
behavior and/or non-trivial new testing. None of these block v1.2.0; the
v1.2.0 behavior is the version validated against real-world data.

## Deferred — require re-validation before changing

### 1. `masked_convolve` normalization of zero-sum (differential) kernels
- **Where:** `src/RoninConvolutions.jl` — `masked_convolve`.
- **What:** Every kernel, including the zero-sum `laplacian` and `sobel_*`
  stencils, is normalized by `weighted_sum / sum(abs.(kernel))` over valid
  neighbors. For smoothing kernels this is a weighted average; for
  differential kernels it scales the edge response and renormalizes near
  edges/missing gates by the valid footprint.
- **Open question:** whether differential kernels should instead return the
  raw masked weighted sum (no division) so the edge response is independent
  of missing-neighbor count.
- **Why deferred:** this is the validated v1.2.0 behavior and feeds every
  edge/texture feature to the classifier. Changing it shifts feature values
  and would require retraining + re-validation. Needs a fixture pinning the
  intended differential-kernel output before any change.

### 2. `remove_validation` split policy
- **Where:** `src/Io.jl` — `remove_validation`.
- **What:** Deterministic every-10th-row stride (always starting at row 1);
  `STEP = 10` hard-coded; `remove_original=true` deletes the input
  unconditionally with no guard against `training_output`/`validation_output`
  colliding with `input_dataset`.
- **Follow-up:** expose `validation_fraction`/`STEP`; offer a seeded random
  split (`rng`/`seed` kwarg) for reproducibility; guard the `rm` against path
  collisions. The exact-partition test added in v1.2.0
  (`test/unit_tests.jl`, "remove_validation split") pins the current behavior.

## Deferred — performance / cleanup (no behavior change)

### 3. `@eval` in the per-task feature loop
- **Where:** `src/RoninFeatures.jl` — `process_single_file` (spatial/derived
  task dispatch).
- **What:** Dynamic dispatch via `@eval` in the per-file/per-task hot loop
  incurs world-age/compilation overhead and returns `Any` (type-unstable).
- **Follow-up:** replace with an explicit `if/elseif` (or a `Dict` of
  concrete functions) over the small fixed task set. Pure refactor; should be
  output-identical and covered by the existing feature tests.

### 4. Thread-unsafe global `REPLACE_MISSING_WITH_FILL`
- **Where:** `src/RoninFeatures.jl` (global), set from threaded mask-gen in
  `src/Ronin.jl`.
- **What:** Module-global mutated inside `Threads.@threads` regions. Currently
  benign (all threads write the same constant for a given run) but genuinely
  unsafe.
- **Follow-up:** thread the flag through as a function argument.

## Fixed in v1.2.0 (for reference)

These were found in the same audit and fixed because they were unambiguous
bugs / bit-identical perf wins:
- `get_task_params` never skipped `#` comment lines (Char-vs-String compare).
- `composite_prediction` skip-path re-flattened the feature mask every
  iteration (O(n²) allocation).
- `masked_convolve` re-summed the loop-invariant `total_abs_weight` per gate.
- `_detect_cfrad_layout` `:ok` branch computed an unused, throw-prone dim tuple.
- `mask_features=true` silently ignored caller-supplied weight matrices.
- `predict_log_proba` used the matrix logarithm instead of elementwise `log.`.
