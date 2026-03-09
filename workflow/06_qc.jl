##=============================================================================
## Step 6: Apply QC
##=============================================================================
## Applies the trained model to write QC'd fields to CfRadial files.
## Run after you are satisfied with the evaluation results.
##
## Prerequisites:
##   - A trained model must exist (from 02_train.jl or 04_retrain.jl)
##   - QC_PATH in 00_config.jl points to the data to QC
##=============================================================================

include("00_config.jl")

println("\n", "="^70)
println("APPLYING QC: $(EXPERIMENT_NAME)")
println("  Input: $(QC_PATH)")
println("  Variables: $(VARS_TO_QC)")
println("  Suffix: $(QC_SUFFIX)")
println("="^70)

config.input_path = QC_PATH
config.met_probs  = met_probs_test[1:num_models]

QC_scan(config)

println("QC complete. Check output files in $(QC_PATH)")
