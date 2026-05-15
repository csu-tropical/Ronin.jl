# Concepts

This page is the shared mental model behind Ronin. It applies to both feature
modes (convolution and legacy hand-tuned).

## Gates and the classification problem

A radar sweep is a grid of **gates** (range × time/azimuth). Each gate is
either **meteorological** (real weather echo we want to keep) or
**non-meteorological** (ground clutter, noise, sidelobes — we want to remove
it). Ronin trains a Random Forest to make that binary call per gate, then
blanks the non-meteorological gates in the raw fields.

### MD / NMD label convention

- **Meteorological data (MD)** is `1` / `true`.
- **Non-meteorological data (NMD)** is `0` / `false`.

This convention holds everywhere: training targets, predictions, masks.

## `ModelConfig` is the single source of truth

Ronin is built around one configuration struct, [`ModelConfig`](@ref).
Everything needed to define, train, evaluate, and apply a QC model lives in it:
data paths, the number of cascade passes, feature settings, RF hyperparameters,
per-pass thresholds, and QC output variables. Construct it with
[`make_config`](@ref) (which auto-derives paths and `mask_names`), persist it
with [`save_config`](@ref), and reload it with [`load_config`](@ref).

`config.task_mode` selects the feature regime — `"convolution"` for the
recommended kernel-bank mode, `""` for the legacy hand-tuned predictors. The
behavior is **not** agnostic to this value; it branches across training,
prediction, QC, and mask generation.

## The pipeline: features → RF → met-prob → mask

For each pass:

1. **Features** are computed per gate (a kernel bank in convolution mode, or
   hand-tuned spatial predictors in legacy mode).
2. A **Random Forest** is trained / applied to those features.
3. The RF emits a **meteorological probability** per gate — the model's
   confidence that the gate is weather. In training this is written back into
   the CfRadial files as `met_prob_pass_<N>`.
4. A **mask** is derived by keeping gates whose met-prob falls in the pass's
   `(low, high)` threshold window.

## The multi-pass cascade

`num_models` sets the number of passes. With `num_models > 1`, each pass after
the first only sees the gates that survived the previous pass's mask. Pass `i`
(for `i < num_models`) writes `met_prob_pass_<i>` and the mask
`mask_names[i+1]`; the next pass trains/predicts only within that mask. This
lets later passes specialize on the harder, ambiguous gates.

- `met_probs::Vector{Tuple{Float32,Float32}}` — one `(low, high)` window per
  pass.
- `mask_names` — default `["mask_pass_0", "mask_pass_1", ...]` from
  `make_config`; `mask_pass_0` is the (empty) starting mask.

Two passes is the recommended starting point.

## `FILL_VAL` and missing data

Missing / blanked gates are represented by `Ronin.FILL_VAL`
(`typemin(Int16) = -32768`). Windowed spatial reducers exclude `missing` gates
from their neighbourhoods; a fully-missing window collapses to `FILL_VAL`.

## Variable name glossary

Ronin works with multiple radar conventions. The QC'd (interactively edited)
fields serve as ground truth for training.

**ELDORA**

| Quantity | Raw | QC'd (ground truth) |
|---|---|---|
| Velocity | `VV` | `VG` |
| Reflectivity | `ZZ` | `DBZ` |
| Signal quality (NCP) | `NCP` | — |

**NOAA TDR**

| Quantity | Raw | QC'd (ground truth) |
|---|---|---|
| Velocity | `VEL` | `VE` |
| Reflectivity | `DBZ` | `DZ` |
| Signal quality (SQI) | `SQI` | — |
