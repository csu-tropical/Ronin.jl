##=============================================================================
## Step 3: Compute Feature Importance
##=============================================================================
## Computes permutation-based feature importance for each pass.
##
## This loads existing trained models and cached features — no retraining.
## For each feature, the column is shuffled n_repeats times and accuracy is
## re-evaluated. The cost scales linearly with the number of features and
## the evaluation sample size.
##
## Performance tuning (set in 00_config.jl):
##   n_importance_repeats         — shuffles per feature (default 3)
##   importance_subsample_fraction — fraction of gates to evaluate on (default 1.0)
##   Start julia with JULIA_NUM_THREADS=N for parallel feature evaluation
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
##=============================================================================

include("00_config.jl")

if !isempty(SELECTED_FEATURES)
    @warn "SELECTED_FEATURES is non-empty. Feature importance should be " *
          "computed on the full feature set. Continuing, but results may " *
          "not reflect the importance of all available features."
end

println("\n", "="^70)
println("COMPUTING FEATURE IMPORTANCE: $(EXPERIMENT_NAME)")
println("  n_repeats = $(config.n_importance_repeats)")
println("  subsample_fraction = $(config.importance_subsample_fraction)")
println("  threads = $(Threads.nthreads())")
println("="^70)

compute_importance(config)

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
