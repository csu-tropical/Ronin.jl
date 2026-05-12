##=============================================================================
## Step 4: Retrain with Pruned Features
##=============================================================================
## Retrains the multi-pass cascade using only the selected feature subset.
##
## WHY RETRAIN (instead of just pruning at inference):
##   RF tree nodes store splits like "if feature[5] < 0.3". If you remove
##   columns, index 5 points to a different variable and predictions are wrong.
##   Retraining with fewer, more informative features also improves
##   generalization: each random feature draw at a split is higher quality.
##
## Prerequisites:
##   - 02_train.jl and 03_importance.jl must have been run
##   - SELECTED_FEATURES in 00_config.jl must be set to the recommended indices
##
## After this step:
##   - Model files are overwritten with the pruned-feature models
##   - JLD2 files contain selected_features (what the model was trained on)
##   - Run 02a_evaluate.jl to check skill (set EVAL_SUFFIX = "_pruned")
##   - Compare to initial training results
##   - If skill dropped, widen SELECTED_FEATURES or revert to full set
##=============================================================================

include("00_config.jl")

if isempty(SELECTED_FEATURES)
    error("SELECTED_FEATURES is empty in 00_config.jl.\n" *
          "Run 03_importance.jl first, then set SELECTED_FEATURES " *
          "to the recommended indices before running this script.")
end

println("\n", "="^70)
println("RETRAINING WITH PRUNED FEATURES: $(EXPERIMENT_NAME)")
println("  Using $(length(SELECTED_FEATURES)) selected features (out of full set)")
println("  Indices: $(SELECTED_FEATURES)")
println("="^70)

train_multi_model(config)
println("Retrain complete.")
println("Next: run 02a_evaluate.jl (set EVAL_SUFFIX = \"_pruned\") to compare skill.")
