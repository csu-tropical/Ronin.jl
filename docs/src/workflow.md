# Workflow Guide

The `workflow/` directory is the **recommended** way to use Ronin. It is a set
of small, ordered scripts that all share one in-memory `config::ModelConfig`
defined in `00_config.jl`. You edit configuration in exactly one place, then run
the stages you need.

Every script begins with `include("00_config.jl")`. Run them with threads
enabled:

```julia
julia --threads=auto workflow/02_train.jl
```

## The scripts

| Script | Purpose |
|---|---|
| `00_config.jl` | Shared configuration — *included by every other script*. Edit here. |
| `01_split_data.jl` | One-time `split_training_testing_validation!(CASE_PATHS, …)`. |
| `02_train.jl` | `train_multi_model(config)` on **all** features. |
| `02a_evaluate.jl` | `inspect_model_configuration` + `run_evaluation` on the testing set (optionally validation). No retraining. |
| `03_importance.jl` | `compute_importance(config)`; rewrites the JLD2 with `recommended_features` and prints indices to paste into `SELECTED_FEATURES`. |
| `04_retrain.jl` | `train_multi_model(config)` with the pruned `selected_features` (errors if empty). |
| `05_sweep_pass2.jl` | `sweep_pass2_met_probs(…)` — freeze Pass 1, sweep Pass 2 `met_prob` thresholds (errors if `num_models < 2`). |
| `06_qc.jl` | Points `config.input_path` at `QC_PATH`, sets `config.met_probs`, calls `QC_scan(config)` to write QC'd fields. |
| `run_workflow.jl` | Single orchestrator that runs any subset of the above in order, via boolean flags. |
| `kitchen_sink_config.jl` | A worked-example alternative to `00_config.jl` (a real `selected_features` list, finer sweep grids). |

A typical first run: `01_split_data.jl` → `02_train.jl` → `02a_evaluate.jl` →
`03_importance.jl` → set `SELECTED_FEATURES` → `04_retrain.jl` →
(optionally `05_sweep_pass2.jl`) → `06_qc.jl`.

## `00_config.jl` sections

This file is organized into clearly delimited sections. The fields you actually
edit are:

- **S1 — Experiment.** `EXPERIMENT_NAME`, `EXPERIMENT_NOTES`. The name drives
  auto-generated paths and the persisted `model_config_<EXPERIMENT_NAME>.jld2`.
- **S2 — Data paths.** `CASE_PATHS` (input cfradial directories),
  `TRAINING_PATH`, `TESTING_PATH`, `VALIDATION_PATH`.
- **S3 — Model parameters.**
  - `num_models` (cascade passes; 2 recommended).
  - `met_probs_train` / `met_probs_test` — per-pass `(low, high)` thresholds.
  - Signal-quality filtering: `SIG_QUALITY_VAR_NAME`,
    `SIG_QUALITY_THRESHOLD`, `REMOVE_LOW_SIG_QUALITY`.
  - PGG ground-gate filtering: `REMOVE_HIGH_PGG`, `PGG_THRESHOLD`.
  - RF hyperparameters: `n_trees`, `max_depth`, `class_weights`,
    `max_training_threads`.
  - Feature mode: `task_mode` and (for convolution) `conv_variables`,
    `conv_kernel_sizes`, `feature_importance_threshold`,
    `n_importance_repeats`, `importance_subsample_fraction`.
  - `PASS_CONFIG` — per-pass overrides (`conv_variables`,
    `selected_features`, and the experimental masked-conv block).
  - QC output: `HAS_INTERACTIVE_QC`, `QC_var`, `remove_var`, `VARS_TO_QC`,
    `QC_SUFFIX`.
- **S4 — Sweep parameters** (used by `05_sweep_pass2.jl`): the
  `SWEEP_*` / `INFER_*` grids, `NMD_TARGET`, `SECONDARY_METRIC`.
- **S5 — QC output path** (`QC_PATH`, used by `06_qc.jl`).
- **S6 — Hypertuning parameters** (used by `run_workflow.jl`): the
  `HYPERTUNE_*` grids.

The bottom of the file (do not edit) assembles `config_kwargs`, calls
`make_config(...)`, and defines `configure_pass!(config, pass)`, which applies
the relevant `PASS_CONFIG` entry (conv variables, selected features, masked-conv
settings) to `config` before each pass operation.

`kitchen_sink_config.jl` is a drop-in replacement showing non-trivial settings;
its `configure_pass!` omits the masked-conv block and it has no S6 section.

## Per-pass configuration

`PASS_CONFIG` is a `Dict` mapping pass number → a NamedTuple of overrides. Any
pass without an entry falls back to the base `conv_variables` with all features
and no masked convolutions. Pass 2+ typically appends `"met_prob_pass_1"` as a
predictor and may add an experimental masked-conv block. Set a pass's
`selected_features` after running importance for that pass.

## `run_workflow.jl` orchestrator

`run_workflow.jl` runs stages in **file order, top to bottom**, each guarded by
a boolean flag (all default `false`). Set the flags you want at the top of the
file and run it once. Selectors `MASK_PASS` (default `1`) and `TRAIN_PASS`
(default `2`) target the pass-specific stages.

High-level phase flags:

- `RUN_SPLIT_DATA` — one-time data split (run before anything else).
- `RUN_FULL_TRAINING` — train → evaluate → importance, all passes.
- `RUN_FULL_RETRAIN` — retrain with `selected_features` → evaluate.
- `USE_PRECOMPUTED_FEATURES` — skip feature calculation, use saved HDF5.
- `RUN_VALIDATION` — also evaluate on the validation set.

Incremental / fine-grained flags (run in this order):

- `RUN_CALCULATE_FEATURES` — calculate and save features, no training.
- `RUN_TRAINING` — train all passes from scratch.
- `RUN_EVALUATION` — evaluate on the testing set.
- `SKIP_EXISTING_MET_PROBS` — reuse `met_prob_pass_N` already in the CfRadials.
- `RUN_IMPORTANCE` — compute feature importance for `TRAIN_PASS`.
- `RUN_RETRAIN` — retrain `TRAIN_PASS` with pruned features.
- `RUN_RETRAIN_EVALUATION` — evaluate the retrained model.
- `RUN_HISTOGRAM` — show the met-prob distribution for `MASK_PASS`.
- `RUN_GENERATE_MASKS` — (re)generate met-prob/masks from `MASK_PASS`.
- `RUN_TRAIN_NEXT_PASS` — train `TRAIN_PASS` using existing prior-pass masks.
- `RUN_PASS2_SWEEP` — sweep Pass 2 met-prob thresholds.
- `RUN_HYPERTUNING` — sweep `n_trees` / `max_depth`, evaluate AUC on testing.
- `RUN_QC` — apply QC and write corrected fields.

(Re-read `run_workflow.jl` for the authoritative, current stage list — the
flags above are stable but the file evolves; reason about flags, not line
numbers.)

## Config auto-persistence

`run_workflow.jl` calls [`save_config`](@ref) to write
`model_config_<EXPERIMENT_NAME>.jld2` at the start and end of a run. Inference
and operational drivers can then reload the exact trained configuration without
re-running the workflow:

```julia
config = load_config("model_config_aft_training.jld2")
```

This is the bridge to the operational entry points — see
[Choosing a QC Entry Point](entrypoints.md).
