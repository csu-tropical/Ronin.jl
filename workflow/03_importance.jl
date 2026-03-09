##=============================================================================
## Step 3: Compute Feature Importance
##=============================================================================
## Computes permutation-based feature importance for each pass.
##
## This is computationally expensive: for each feature, the column is shuffled
## n_repeats times and accuracy is re-evaluated. The cost scales linearly with
## the number of features and the size of the training set.
##
## Prerequisites:
##   - 02_train.jl must have been run (models and feature H5 files must exist)
##   - SELECTED_FEATURES should be empty (importance is computed on the full
##     feature set to identify which features matter)
##
## After this step:
##   - Each JLD2 model file contains:
##       recommended_features — indices above the importance threshold
##       importances          — raw importance scores per feature
##       feature_names        — names corresponding to each index
##   - Use inspect_model_configuration() to review
##   - Copy recommended indices to SELECTED_FEATURES in 00_config.jl
##   - Then run 04_retrain.jl
##
## NOTE: This step retrains from cached features (fast) because the importance
## computation is integrated into the training pipeline. The retrained model
## will have identical skill to the original since it uses the same features.
##=============================================================================

include("00_config.jl")

if !isempty(SELECTED_FEATURES)
    @warn "SELECTED_FEATURES is non-empty. Feature importance should be " *
          "computed on the full feature set. Continuing, but results may " *
          "not reflect the importance of all available features."
end

## Enable importance computation; use cached features to avoid recomputing
config.compute_feature_importance = true
config.file_preprocessed = fill(true, num_models)

println("\n", "="^70)
println("COMPUTING FEATURE IMPORTANCE: $(EXPERIMENT_NAME)")
println("  This will take a while — shuffling each feature column and re-evaluating...")
println("="^70)

train_multi_model(config)

## Inspect each model and print recommended features
println("\n", "="^70)
println("RESULTS")
println("="^70)

for (i, path) in enumerate(config.model_output_paths)
    println("\n--- Pass $(i) ---")
    inspect_model_configuration(path)

    md = load_model_with_metadata(path, config.task_mode)
    if !isempty(md.recommended_features)
        println("\nTo retrain Pass $(i) with pruned features, add to 00_config.jl:")
        println("  SELECTED_FEATURES = $(md.recommended_features)")
    end
end

println("\n", "="^70)
println("NEXT STEPS:")
println("  1. Review the recommended features above")
println("  2. Copy the recommended indices to SELECTED_FEATURES in 00_config.jl")
println("  3. Run 04_retrain.jl to retrain with the pruned feature set")
println("  4. Run 02a_evaluate.jl (set EVAL_SUFFIX = \"_pruned\") to compare skill")
println("="^70)
