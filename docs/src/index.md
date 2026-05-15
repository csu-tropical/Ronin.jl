# Ronin.jl

**Ronin** (Random forest Optimized Nonmeteorological IdentificatioN) removes
non-meteorological gates from Doppler radar scans using the machine-learning
methodology of [Dr. Alex DesRosiers and Dr. Michael
Bell](https://journals.ametsoc.org/view/journals/aies/aop/AIES-D-23-0064.1/AIES-D-23-0064.1.xml).
It derives features from raw radar data, trains a Random Forest classifier on
them, and applies that model to clean the raw fields in radar scans, with
built-in evaluation tooling.

If you use Ronin in published work, please cite DesRosiers and Bell (2023),
*Artificial Intelligence for the Earth Systems*.

## Installation

Ronin is in the Julia General Registry:

```julia
using Pkg
Pkg.add("Ronin")
```

## Fastest path

The recommended way to train and apply a model is the shipped `workflow/`
pipeline. From a clone of the repository:

```julia
# edit workflow/00_config.jl (experiment name, data paths, model params)
julia --threads=auto workflow/01_split_data.jl   # one-time train/test/val split
julia --threads=auto workflow/02_train.jl        # train the multi-pass cascade
julia --threads=auto workflow/02a_evaluate.jl    # evaluate on the testing set
julia --threads=auto workflow/06_qc.jl           # write QC'd fields
```

See the [Workflow Guide](workflow.md) for the full pipeline, the
[Concepts](concepts.md) page for the mental model, and [Choosing a QC Entry
Point](entrypoints.md) if you are integrating Ronin into your own driver.

## What changed in 1.2.0

Version 1.2.0 is the first Julia General Registry release. The documentation
below now covers the full current surface; the legacy single-model,
hand-tuned-predictor flow is preserved on the [Legacy Hand-Tuned
Mode](legacy.md) page but is no longer the recommended path. Highlights of the
1.1.0 → 1.2.0 delta (see `CHANGELOG.md` for the authoritative list):

- **Convolution feature mode** (`task_mode = "convolution"`) — a kernel bank
  replaces the hand-crafted predictor file as the recommended feature set.
- **Multi-pass cascade** — `train_multi_model` / `composite_prediction` /
  `composite_QC`, with per-pass `met_probs`, `mask_names`, and
  `selected_features`.
- **Met-prob-masked convolutions** — optional and *experimental*.
- **Hyperparameter and Pass-2 threshold sweeps** — `run_hypertuning`,
  `sweep_pass2_met_probs`.
- **Permutation feature importance** — `compute_importance`.
- **Robust config persistence** — `save_config` / `load_config` survive future
  `ModelConfig` field additions (fixes #34).
- **Workflow scripts** — the `workflow/` directory and `run_workflow.jl`
  orchestrator, with auto-persisted `model_config_<EXPERIMENT_NAME>.jld2`.
- **Validation-set splitting** — `split_training_testing_validation!`.

Non-breaking deprecations: the v1.1.0 `REMOVE_LOW_NCP` keyword and
`ModelConfig.REMOVE_LOW_NCP` field still work but emit a deprecation warning;
use `REMOVE_LOW_SIG_QUALITY`. Breaking removals: `predict_with_model` and the
3-arg `evaluate_model(::String, ::String, ::String)` are gone — use
`composite_prediction`, `run_evaluation`, or `QC_scan`.
