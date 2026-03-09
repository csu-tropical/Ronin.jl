##=============================================================================
## Step 2: Train Multi-Pass Model
##=============================================================================
## Trains the full multi-pass RF cascade on ALL features.
##
## Prerequisites: Data must exist at TRAINING_PATH (run 01_split_data.jl first)
##
## After this step:
##   - Model files:   trained_model_<experiment>_<pass>.jld2
##   - Feature files:  output_features_<experiment>_<pass>.h5
##
## Next step: 02a_evaluate.jl to check skill, then 03_importance.jl (optional)
##=============================================================================

include("00_config.jl")

## Safety check: initial training should use all features
if !isempty(SELECTED_FEATURES)
    @warn "SELECTED_FEATURES is non-empty ($(length(SELECTED_FEATURES)) features). " *
          "Initial training uses all features. " *
          "To retrain with a pruned set, use 04_retrain.jl instead."
end

println("\n", "="^70)
println("TRAINING: $(EXPERIMENT_NAME)")
println("  num_models = $(num_models), task_mode = $(task_mode)")
println("  n_trees = $(n_trees), max_depth = $(max_depth), class_weights = $(class_weights)")
println("  SIG_QUALITY = $(REMOVE_LOW_SIG_QUALITY ? SIG_QUALITY_THRESHOLD : "off")")
println("  PGG = $(REMOVE_HIGH_PGG ? PGG_THRESHOLD : "off")")
println("  met_probs_train = $(met_probs_train)")
if task_mode == "convolution"
    println("  conv_variables = $(config.conv_variables)")
    println("  conv_kernel_sizes = $(config.conv_kernel_sizes)")
end
println("="^70)

train_multi_model(config)
println("Training complete.")
println("Next: run 02a_evaluate.jl to check skill.")
