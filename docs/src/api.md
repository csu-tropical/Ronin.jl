# API Reference

Complete reference for Ronin's exported API, grouped by topic. See the
[Concepts](concepts.md) and [Workflow Guide](workflow.md) pages for how these
fit together.

## Configuration

The `ModelConfig` struct is the single source of truth; the rest construct,
persist, reload, and migrate it.

```@docs
ModelConfig
make_config
save_config
load_config
migrate_model_config
```

## Data preparation

Splitting cfradial directories into train/test/(validation) sets and related
directory utilities.

```@docs
split_training_testing!
split_training_testing_validation!
remove_validation
parse_directory
```

## Feature calculation — convolution mode

The recommended feature regime: a kernel bank convolved over `conv_variables`.

```@docs
calculate_features_conv
ConvolutionKernel
build_kernel_bank
compute_convolution_features
masked_convolve
get_convolution_feature_count
```

## Feature calculation — legacy hand-tuned mode

The v1.1.0 hand-crafted predictor pipeline and its spatial reducers.

```@docs
calculate_features
process_single_file
get_task_params
get_num_tasks
calc_avg
calc_std
calc_iso
airborne_ht
prob_groundgate
```

## Training

Fitting Random Forest classifiers, single-pass and multi-pass.

```@docs
train_model
train_multi_model
train_single_pass
```

## Multi-pass masking

Generating and regenerating the inter-pass met-prob masks.

```@docs
generate_pass_masks
regenerate_masks
```

## Feature importance / selection

Permutation and RF-native importance, and feature subsetting.

```@docs
compute_importance
get_feature_importance
select_features
compute_rf_feature_importance
```

## Hyperparameter tuning

Sweeping RF hyperparameters and inter-pass thresholds.

```@docs
run_hypertuning
sweep_pass2_met_probs
```

## Prediction & QC

Applying a trained cascade to radar data. See
[Choosing a QC Entry Point](entrypoints.md) for which to use.

```@docs
composite_prediction
composite_QC
QC_scan
write_field
```

## Evaluation & inspection

Scoring models and inspecting saved model files.

```@docs
evaluate_model
run_evaluation
get_contingency
compute_auc_roc
met_prob_histogram
characterize_misclassified_gates
inspect_model_configuration
load_model_with_metadata
```

## Utilities

```@docs
compute_balanced_class_weights
```
