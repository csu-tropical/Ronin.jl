# Changelog

All notable changes to Ronin.jl. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.2.0] - 2026

First Julia General Registry release. Project.toml had been at `0.1.1` while
git tags ran ahead to `v1.0.0` / `v1.1.0`; this release reconciles
`Project.toml` to a number that reflects everything since the v1.1.0 tag and
adds the registry-readiness work needed for AutoMerge.

### Added

- **Convolution feature mode.** `task_mode = "convolution"`, with a kernel
  bank, `calculate_features_conv`, `compute_convolution_features`,
  `ConvolutionKernel`, `build_kernel_bank`, `get_convolution_feature_count`,
  `select_features`, `compute_rf_feature_importance`. Replaces the
  hand-crafted predictor file pattern as the recommended path.
- **Multi-pass cascade.** `train_multi_model`, `train_single_pass`,
  `regenerate_masks`, `generate_pass_masks`, `composite_prediction`,
  `composite_QC`. ModelConfig grows `met_probs`, `mask_names`,
  `selected_features`, per-pass `conv_variables`.
- **Met-prob-masked convolutions.** Optional. `masked_conv_variables`,
  `masked_conv_kernel_types`, `masked_conv_kernel_sizes`,
  `masked_conv_threshold`, `masked_conv_met_prob_field`. Currently
  experimental; may be revisited in a future release.
- **Hyperparameter sweep.** `run_hypertuning` over a `(n_trees, max_depth)`
  grid with AUC-ROC, HSS, F1, NMD/MD hit-rate metrics.
- **Pass-2 threshold sweep.** `sweep_pass2_met_probs` to tune
  `met_prob` cuts between cascade passes.
- **Permutation feature importance.** `compute_importance` with
  configurable `n_importance_repeats` and `importance_subsample_fraction`.
- **Dict-keyed config persistence (fixes #34).** `save_config(path, config)`
  writes one JLD2 key per ModelConfig field; `load_config(path)` reads the
  new format **or** legacy `save_object` files inline. Adding new fields to
  ModelConfig in the future no longer breaks existing config files.
  `migrate_model_config` is available as a one-shot upgrade utility.
- **Model file inspection.** `inspect_model_configuration`,
  `load_model_with_metadata`, `met_prob_histogram`, `compute_auc_roc`.
- **Validation-set splitting.** `split_training_testing_validation!`.
- **Workflow scripts.** `workflow/` directory with `00_config.jl`,
  `01_split_data.jl` … `06_qc.jl`, plus `run_workflow.jl` orchestrator.
  `run_workflow.jl` now auto-persists the in-memory ModelConfig to
  `model_config_<EXPERIMENT_NAME>.jld2` so inference / operational scripts
  can reload it without re-executing the workflow.
- **Convolution-mode evaluation API.** `run_evaluation`,
  `evaluate_model(::ModelConfig)`.
- **CI matrix on Julia 1.10 + 1.11.** Switched to
  `julia-actions/julia-runtest@v1` against the new `test/runtests.jl` /
  `unit_tests.jl` suite (1600+ lines of `@testset` coverage).

### Changed (non-breaking via deprecation shims)

The following v1.1.0 names continue to work but emit `Base.depwarn`:

- `process_single_file(...; REMOVE_LOW_NCP=)` → `REMOVE_LOW_SIG_QUALITY=`
- `calculate_features(...; REMOVE_LOW_NCP=)` → `REMOVE_LOW_SIG_QUALITY=`
- `make_config(; REMOVE_LOW_NCP=)` → `REMOVE_LOW_SIG_QUALITY=`
- `ModelConfig.REMOVE_LOW_NCP` field access (read and write) → forwards to
  `REMOVE_LOW_SIG_QUALITY` via `Base.getproperty` / `Base.setproperty!`

### Removed (breaking — no shim)

- `predict_with_model` removed entirely. Use `composite_prediction`,
  `run_evaluation`, or `QC_scan` depending on use case.
- `evaluate_model(::String, ::String, ::String)` 3-arg overload removed.
  Use `evaluate_model(::ModelConfig)`.
- Ghost exports `multipass_uncertain` and `error_characteristics` removed
  from the export list. Both pointed at undefined / commented-out
  functions and would `UndefVarError` if called, so this is a
  documentation-only correction.
- `BenchmarkTools` and `Documenter` no longer in the package's `[deps]`.
  BenchmarkTools was unused in `src/`; Documenter belongs in
  `docs/Project.toml`.

### Changed (breaking — type narrowing, generally absorbed by Julia auto-promotion)

- `ModelConfig.met_probs` field: `Vector{Tuple{Float64,Float64}}` →
  `Vector{Tuple{Float32,Float32}}`.
- `weight_matrixes` keyword: inner element type narrowed from
  `Float64` to `Float32` across `process_single_file` and
  `calculate_features`.
- `task_weights` and other Float64-typed internals migrated to Float32
  for memory / throughput.

### Fixed

- **#34** — Loading legacy ModelConfig JLD2 files no longer breaks every
  time a field is added to the struct. `load_config` handles both formats
  and returns a real, mutable `ModelConfig` so subsequent `setproperty!`
  calls (e.g. `config.task_paths = [...]`) work.
- `train_model` is exported again (function still existed but had been
  dropped from the export list in this development cycle).
- Documentation build: `docs/src/api.md` updated to match current API
  surface, removing references to functions removed in this release.
- Documentation deploy URL fixed: `irslushy/Ronin.jl` →
  `csu-tropical/Ronin.jl`.

### Tests

- `test/runtests.jl` is now a thin includer; the legacy 762-line script
  with `@assert` and benchmark-CFRadial dependencies is retired.
- `test/unit_tests.jl` covers RoninConstants, RoninFeatures, Core Ronin,
  DecisionTree, Training Pipeline, I/O, Reproducibility, Edge Cases,
  RoninConvolutions, model inspection, save_config / load_config, and
  the v1.1.0 deprecation shims. **741 assertions, 14 testsets, 0
  failures** locally and in CI on Julia 1.10 and 1.11.

[1.2.0]: https://github.com/csu-tropical/Ronin.jl/releases/tag/v1.2.0
