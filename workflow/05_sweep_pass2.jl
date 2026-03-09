##=============================================================================
## Step 5: Pass 2 Met-Prob Threshold Sweep
##=============================================================================
## Freezes the Pass 1 model and sweeps met_prob thresholds for Pass 2.
##
## For each (low, high) threshold combination:
##   1. Regenerates the pass mask from saved met_prob predictions (fast)
##   2. Recalculates features on the filtered subset (spatial features change
##      because removing gates changes the neighborhood)
##   3. Retrains Pass 2 on the "hard" data
##   4. Evaluates the full cascade on the testing set
##
## Prerequisites:
##   - 02_train.jl (or 04_retrain.jl) must have been run
##   - Pass 1 model and met_prob fields must exist in CfRadials
##   - num_models >= 2
##
## Sweep parameters are configured in Section 4 of 00_config.jl.
##=============================================================================

include("00_config.jl")

if num_models < 2
    error("Pass 2 sweep requires num_models >= 2 (currently $(num_models))")
end

println("\n", "="^70)
println("PASS 2 THRESHOLD SWEEP: $(EXPERIMENT_NAME)")
println("  Low grid:  $(SWEEP_MET_PROB_LOW_GRID)")
println("  High grid: $(SWEEP_MET_PROB_HIGH_GRID)")
println("  Use met_prob as feature: $(USE_MET_PROB_AS_FEATURE)")
println("  Sweep inference: $(SWEEP_INFERENCE)")
println("="^70)

sweep_results = sweep_pass2_met_probs(
    config, TRAINING_PATH, TESTING_PATH;
    experiment_name        = EXPERIMENT_NAME,
    met_prob_low_grid      = SWEEP_MET_PROB_LOW_GRID,
    met_prob_high_grid     = SWEEP_MET_PROB_HIGH_GRID,
    use_met_prob_as_feature = USE_MET_PROB_AS_FEATURE,
    sweep_inference        = SWEEP_INFERENCE,
    infer_low_grid         = INFER_LOW_GRID,
    infer_high_grid        = INFER_HIGH_GRID,
    nmd_target             = NMD_TARGET,
    secondary_metric       = SECONDARY_METRIC,
)
