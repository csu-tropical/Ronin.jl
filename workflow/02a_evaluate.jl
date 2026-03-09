##=============================================================================
## Step 2a/4a: Evaluate Model
##=============================================================================
## Runs composite_prediction on the testing set and prints classification
## metrics. Can be run any time a trained model exists — does not retrain.
##
## Use this after:
##   - 02_train.jl  (initial training)
##   - 04_retrain.jl (retrained with pruned features)
##   - 05_sweep_pass2.jl (after finding best Pass 2 thresholds)
##   - Or any time you want to re-evaluate an existing model
##
## Set EVAL_SUFFIX below to distinguish output files between runs.
## Set EVAL_VALIDATION = true to also evaluate on the validation set.
##=============================================================================

include("00_config.jl")

##-----------------------------------------------------------------------------
## Evaluation settings (edit these as needed)
##-----------------------------------------------------------------------------

## Suffix appended to prediction output filename to distinguish runs.
## Examples: "" (default), "_pruned", "_sweep_best"
EVAL_SUFFIX = ""

## Set to true to also evaluate on the validation set (use sparingly)
EVAL_VALIDATION = false

##-----------------------------------------------------------------------------
## Execution
##-----------------------------------------------------------------------------

println("\n", "="^70)
println("EVALUATING: $(EXPERIMENT_NAME)$(EVAL_SUFFIX)")
println("  Models: $(config.model_output_paths)")
println("  met_probs_test = $(met_probs_test)")
if !isempty(config.selected_features)
    println("  selected_features: $(length(config.selected_features)) features")
end
println("="^70)

## Inspect each model
for (i, path) in enumerate(config.model_output_paths)
    if isfile(path)
        println("\n--- Pass $(i) ---")
        inspect_model_configuration(path)
    else
        @warn "Model file not found: $(path). Run training first."
    end
end

## Evaluate on testing set
println("\nEvaluating on TESTING set...")
test_results = run_evaluation(
    config, "TESTING", TESTING_PATH, met_probs_test;
    prediction_outfile = "predictions_test_$(EXPERIMENT_NAME)$(EVAL_SUFFIX).h5",
)

## Evaluate on validation set (optional)
if EVAL_VALIDATION
    println("\nEvaluating on VALIDATION set...")
    val_results = run_evaluation(
        config, "VALIDATION", VALIDATION_PATH, met_probs_test;
        prediction_outfile = "predictions_val_$(EXPERIMENT_NAME)$(EVAL_SUFFIX).h5",
    )
end
