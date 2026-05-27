# Convolution Feature Mode

Set `task_mode = "convolution"`. This is the recommended feature regime as of
1.2.0. Instead of a hand-tuned predictor file, features are derived by
convolving a **kernel bank** over a set of radar variables.

## Kernel bank

A [`ConvolutionKernel`](@ref) is a named `Matrix{Float32}` of weights.
[`build_kernel_bank`](@ref)`(kernel_sizes)` builds the bank: for each size `k`
it adds a `mean_<k>x<k>` kernel; it always adds `laplacian_3x3`,
`sobel_range_3x3`, and `sobel_azi_3x3`; and for each `k ≥ 5` it adds a
`gaussian_<k>x<k>`.

With the default `conv_kernel_sizes = [3, 5, 7]` this yields **8 kernels**:
`mean_3`, `mean_5`, `mean_7`, `laplacian_3x3`, `sobel_range_3x3`,
`sobel_azi_3x3`, `gaussian_5x5`, `gaussian_7x7`.

## Feature layout

[`compute_convolution_features`](@ref) produces, per *conv-variable × kernel*,
**two columns**:

- `<var>_<kern>` — the convolved value.
- `<var>_<kern>_vfrac` — the valid (non-missing) fraction within the kernel
  footprint.

If a met-prob mask is supplied, the experimental masked-convolution columns are
appended next (see below). Finally, **four scalar features are appended, in this
order: AHT, ELV, RNG, NRG**.

The total feature count is given by
[`get_convolution_feature_count`](@ref):

```
n_vars * n_kernels * 2  +  n_masked_vars * n_masked_kernels * 2  +  4
```

## `conv_variables`

`config.conv_variables` lists the inputs the kernel bank operates on. The
resolver (`load_conv_variable`) accepts:

- Raw CfRadial fields (e.g. `"DBZ"`, `"VEL"`, `"WIDTH"`).
- Derived variables `PGG` (probability of ground gate) and `SIG` (signal
  quality).
- Spatial reducers `AVG(var)`, `ISO(var)`, `STD(var)` — resolved recursively,
  so `ISO(PGG)` works.
- Prior-pass output `met_prob_pass_<N>` (used by cascade pass `N+1`).

`conv_variables` can differ per pass via `PASS_CONFIG` — a common pattern is to
append `"met_prob_pass_1"` to Pass 2's variables.

## Computing features

[`calculate_features_conv`](@ref)`(config, output_file; …)` is the
convolution-mode replacement for `calculate_features`. In the workflow this is
driven by `02_train.jl` / `run_workflow.jl`; you rarely call it directly.

## Feature selection / importance

Training on all features then computing permutation importance
([`compute_importance`](@ref)) writes `recommended_features` (and
`importances`) into the model's JLD2. Inspect with
[`inspect_model_configuration`](@ref), set `config.selected_features` to the
recommended indices, and retrain.

Note: you must **retrain** on the selected feature subset — you cannot train on
N features and prune to K at inference, because the Random Forest's tree splits
reference specific column indices. The model's `selected_features` is reloaded
at inference so the feature matrix matches what the trees expect.

## Met-prob-masked convolutions

!!! warning "Experimental"

    Met-prob-masked convolutions are experimental and may be revisited or
    changed in a future release. They are **not** part of the recommended
    workflow path. Enable them only if you are deliberately experimenting.

When enabled, an additional set of convolution columns is computed where the
kernel only accumulates contributions from neighbours that the previous pass
considered meteorological. [`masked_convolve`](@ref) distinguishes the *valid*
set (which gates get an output) from the *neighbor mask* (valid ∧ met-prob
mask — which gates contribute to the sum).

Configuration fields:

- `masked_conv_variables` — variables to compute masked-conv columns for.
- `masked_conv_kernel_types` — subset of
  `{mean, gaussian, laplacian, sobel_range, sobel_azi}`.
- `masked_conv_kernel_sizes` — kernel sizes.
- `masked_conv_threshold` — met-prob cutoff (default `0.1f0`).
- `masked_conv_met_prob_field` — which `met_prob_pass_<N>` field to mask with.

The filtered kernel bank is the Cartesian product of types × sizes, with the
same structural constraints as the main bank (`laplacian`/`sobel` only at
`k == 3`, `gaussian` only at `k ≥ 5`); other combinations are silently skipped
and the skipped set is reported once at training time.
