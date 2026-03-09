##=============================================================================
## RONIN Parameter Tuning
##=============================================================================
## Clean interface for training and evaluating RONIN multi-pass models.
##
## HOW TO USE:
##   1. Edit Sections 1-4 below (experiment name, data paths, parameters, workflow)
##   2. Run the entire script — functions in Ronin handle the rest
##   3. Change parameters and re-run to compare experiments
##
## Everything the user needs to set is in Sections 1-4.
## Section 5 is the execution block — do not modify it.
##=============================================================================

using Ronin
using Dates

##=============================================================================
## SECTION 1: EXPERIMENT
##=============================================================================
## Give each experiment a unique name so results don't overwrite each other.
## Notes are printed in logs for your reference.

EXPERIMENT_NAME = "explore_default"
EXPERIMENT_NOTES = "Initial baseline run with default parameters"

##=============================================================================
## SECTION 2: DATA PATHS
##=============================================================================
## Point these to your CfRadial data. Use absolute paths.

CASE_PATHS = [
    "/Users/mmbell/Science/ronin_testing/tm_swps",
]

TRAINING_PATH   = "/Users/mmbell/Science/ronin_testing/tm_swps/TRAINING/"
TESTING_PATH    = "/Users/mmbell/Science/ronin_testing/tm_swps/TESTING/"
VALIDATION_PATH = "/Users/mmbell/Science/ronin_testing/tm_swps/VALIDATION/"

## Set to true the FIRST time to split data into train/test/val sets.
## Set to false on subsequent runs to keep splits consistent across experiments.
SPLIT_DATA = false

##=============================================================================
## SECTION 3: MODEL PARAMETERS
##=============================================================================

##-----------------------------------------------------------------------------
## 3a. Model architecture
##     num_models: number of passes in the multi-pass cascade
##       1 = single RF classifier
##       2 = two-pass (recommended): Pass 1 classifies easy data, Pass 2
##           retrains on the uncertain remainder
##       3+ = additional refinement passes
##-----------------------------------------------------------------------------
num_models = 2

##-----------------------------------------------------------------------------
## 3b. Meteorological probability thresholds
##     Format: (low_threshold, high_threshold) per pass
##     Gates with met probability in [low, high] are "uncertain" and passed
##     to the next model for re-classification.
##
##     met_probs_train: thresholds used during training to create masks
##       between passes — controls WHAT DATA the next pass trains on
##     met_probs_test: thresholds used during inference/evaluation
##       (can differ from training thresholds)
##
##     For single-pass models, these are still required but only the first
##     entry matters and no mask is created.
##
##     Tuning guidance:
##       - Narrower range (0.3, 0.7) = Pass 2 gets fewer, harder gates
##       - Wider range (0.1, 0.9) = Pass 2 gets more gates, easier subset
##       - The Pass 2 sweep (Section 4) automates finding the best threshold
##-----------------------------------------------------------------------------
met_probs_train = [(0.1f0, 0.8f0), (0.1f0, 0.999f0)]
met_probs_test  = [(0.1f0, 0.8f0), (0.1f0, 0.999f0)]

##-----------------------------------------------------------------------------
## 3c. Signal quality filtering
##     SIG_QUALITY_VAR: variable name in CfRadial (SQI for TDR, NCP for ELDORA)
##     SIG_QUALITY_THRESHOLD: remove gates at or below this value
##     REMOVE_LOW_SIG_QUALITY: toggle filtering on/off
##
##     Tuning: try 0.1, 0.2, 0.3 — higher = more aggressive noise removal
##-----------------------------------------------------------------------------
SIG_QUALITY_VAR_NAME    = "SQI"
SIG_QUALITY_THRESHOLD   = 0.2f0
REMOVE_LOW_SIG_QUALITY  = true

##-----------------------------------------------------------------------------
## 3d. Ground gate filtering (PGG)
##     PGG_THRESHOLD: probability threshold for ground gate removal
##     REMOVE_HIGH_PGG: toggle filtering on/off
##
##     Tuning: 1.0 = conservative, 0.8 = moderate, 0.5 = aggressive
##-----------------------------------------------------------------------------
REMOVE_HIGH_PGG = true
PGG_THRESHOLD   = 1.0f0

##-----------------------------------------------------------------------------
## 3e. Random forest hyperparameters
##     n_trees: more trees = smoother, slower (try 11, 21, 51, 101)
##     max_depth: deeper = more capacity, overfitting risk (try 8, 10, 14, 18)
##     class_weights: "" (uniform) or "balanced" (weight by class frequency)
##-----------------------------------------------------------------------------
n_trees       = 51
max_depth     = 14
class_weights = "balanced"

##-----------------------------------------------------------------------------
## 3f. Feature mode — choose ONE of the two approaches below
##
##     Option A: Convolution pre-processor (recommended)
##       Automatically generates features by convolving radar variables with
##       a bank of kernels at multiple scales. Set task_mode = "convolution".
##
##       conv_variables: radar fields and derived quantities to convolve
##         - Raw fields: "DBZ", "VEL", "WIDTH", etc. (any CfRadial variable)
##         - Derived: "PGG" (ground gate prob), "SIG" (signal quality)
##         - Spatial: "AVG(var)", "ISO(var)", "STD(var)" — computes the
##           spatial statistic first, then convolves the result. This enables
##           detection of patterns like clusters of isolated gates.
##           Inner variables can be derived: "ISO(PGG)", "AVG(SIG)", etc.
##         - Prior-pass: "met_prob_pass_1" (Pass 1 confidence, for Pass 2+)
##       conv_kernel_sizes: spatial scales [3, 5, 7] = fine to coarse
##       feature_importance_threshold: prune features below this fraction
##         of max importance (0.01 = 1% of max)
##
##     Option B: Hand-crafted features
##       Set task_mode = "" and configure task_paths/task_weights below.
##-----------------------------------------------------------------------------

## --- Option A: Convolution mode ---
task_mode = "convolution"
conv_variables = ["DBZ", "VEL", "SIG", "PGG", "WIDTH", "ISO(DBZ)", "ISO(VEL)"]
conv_kernel_sizes = [3, 5, 7]
feature_importance_threshold = 0.01

## --- Option B: Hand-crafted features (uncomment to use instead) ---
# task_mode = ""
# pass_1_tasks = ["DBZ", "STD(DBZ)", "ISO(DBZ)", "PGG", "SIG"]
# pass_2_tasks = ["DBZ", "STD(DBZ)", "ISO(DBZ)", "STD(VEL)", "ISO(VEL)"]
# task_paths = Union{String, Vector{String}}[pass_1_tasks, pass_2_tasks]
# sw = Ronin.standard_window
# aw = Ronin.azi_window
# rw = Ronin.range_window
# pw = Ronin.placeholder_window
# task_1_weights = [pw, rw, aw, pw, pw]
# task_2_weights = [pw, rw, aw, rw, aw]
# weights_tot = [task_1_weights, task_2_weights]

##-----------------------------------------------------------------------------
## 3g. QC and output settings
##     HAS_INTERACTIVE_QC: true if CfRadials have human-QC'd fields for training
##     QC_var: the QC'd variable name (e.g., "VG" for ELDORA, "VE" for TDR)
##     remove_var: the raw variable name (e.g., "VV" for ELDORA, "DBZ" for TDR)
##     VARS_TO_QC: which variables to apply QC corrections to
##-----------------------------------------------------------------------------
HAS_INTERACTIVE_QC = true
QC_var     = "VE"
remove_var = "DBZ"
VARS_TO_QC = ["DBZ", "VEL"]
QC_SUFFIX  = "_QC"

##=============================================================================
## SECTION 4: WORKFLOW
##=============================================================================
## Toggle which steps to run. Each step depends on the previous ones.

##-----------------------------------------------------------------------------
## 4a. Training — train the full multi-pass cascade
##     Set to true for the first run or when changing model parameters.
##     Set to false to skip training and just evaluate existing models.
##-----------------------------------------------------------------------------
RUN_TRAINING = false

##-----------------------------------------------------------------------------
## 4b. Feature importance (convolution mode only)
##     Computes permutation-based feature importance after training, then
##     prunes features below feature_importance_threshold and re-saves the
##     model with the selected feature indices.
##     This is computationally expensive (shuffles each feature column
##     n_repeats times and re-evaluates accuracy). Set to false to skip
##     during rapid iteration, then enable for a final optimized model.
##     Only applies when task_mode = "convolution".
##-----------------------------------------------------------------------------
COMPUTE_FEATURE_IMPORTANCE = false

##-----------------------------------------------------------------------------
## 4c. Evaluation — evaluate trained models on the testing set
##     Runs composite_prediction and prints classification metrics.
##-----------------------------------------------------------------------------
RUN_EVALUATION = true

##-----------------------------------------------------------------------------
## 4d. Evaluation on validation set
##     Use sparingly to avoid overfitting to the validation set.
##-----------------------------------------------------------------------------
RUN_VALIDATION = false

##-----------------------------------------------------------------------------
## 4e. Pass 2 met_prob sweep — find the best Pass 1→2 threshold
##     Freezes Pass 1 model and sweeps met_prob thresholds for Pass 2.
##     For each threshold:
##       1. Regenerates mask from saved met_prob predictions (fast)
##       2. Recalculates features on the filtered subset (spatial features
##          change because removing gates changes the neighborhood)
##       3. Retrains Pass 2 on the "hard" data
##       4. Evaluates the full cascade
##
##     Requires num_models >= 2 and RUN_TRAINING to have been done at least
##     once (Pass 1 model and met_prob fields must exist in CfRadials).
##
##     If USE_MET_PROB_AS_FEATURE is true (convolution mode only), the Pass 1
##     met probability field is added as a convolution variable for Pass 2,
##     giving it the confidence landscape as a predictor.
##-----------------------------------------------------------------------------
RUN_PASS2_SWEEP = false

## Sweep parameters (only used if RUN_PASS2_SWEEP = true)
SWEEP_MET_PROB_LOW_GRID  = Float32[0.1, 0.2, 0.3, 0.4]
SWEEP_MET_PROB_HIGH_GRID = Float32[0.6, 0.7, 0.8, 0.9]
USE_MET_PROB_AS_FEATURE  = true

## Also sweep inference thresholds for the best training config (cheap)
SWEEP_INFERENCE          = true
INFER_LOW_GRID           = Float32[0.1, 0.2, 0.3]
INFER_HIGH_GRID          = Float32[0.98, 0.99, 0.999]

## Sweep quality targets
NMD_TARGET       = 0.99f0    # minimum NMD hit rate to consider acceptable
SECONDARY_METRIC = :hss      # rank by :hss or :md_hit_rate

##-----------------------------------------------------------------------------
## 4f. QC output — apply the trained model to write QC'd fields
##     Writes corrected fields to CfRadial files. Run after you're satisfied
##     with the evaluation results.
##-----------------------------------------------------------------------------
RUN_QC = false

## Path to apply QC to (defaults to TESTING_PATH)
QC_PATH = TESTING_PATH

##=============================================================================
## SECTION 5: EXECUTION (do not modify below this line)
##=============================================================================
## This section builds the config and runs the selected workflow steps.

## Split data if requested
if SPLIT_DATA
    Ronin.split_training_testing_validation!(CASE_PATHS, TRAINING_PATH, TESTING_PATH, VALIDATION_PATH)
    println("Data split complete: train=$(TRAINING_PATH), test=$(TESTING_PATH), val=$(VALIDATION_PATH)")
end

## Build model config using make_config (auto-generates paths and mask_names)
config_kwargs = Dict{Symbol,Any}(
    :met_probs              => met_probs_train,
    :task_mode              => task_mode,
    :verbose                => true,
    :REMOVE_HIGH_PGG        => REMOVE_HIGH_PGG,
    :PGG_THRESHOLD          => PGG_THRESHOLD,
    :REMOVE_LOW_SIG_QUALITY => REMOVE_LOW_SIG_QUALITY,
    :SIG_QUALITY_THRESHOLD  => SIG_QUALITY_THRESHOLD,
    :SIG_QUALITY_VAR        => SIG_QUALITY_VAR_NAME,
    :remove_var             => remove_var,
    :QC_var                 => QC_var,
    :FILL_VAL               => Ronin.FILL_VAL,
    :HAS_INTERACTIVE_QC     => HAS_INTERACTIVE_QC,
    :replace_missing        => false,
    :write_out              => true,
    :QC_mask                => false,
    :VARS_TO_QC             => VARS_TO_QC,
    :QC_SUFFIX              => QC_SUFFIX,
    :n_trees                => n_trees,
    :max_depth              => max_depth,
    :overwrite_output       => true,
    :class_weights          => class_weights,
    :compute_feature_importance => COMPUTE_FEATURE_IMPORTANCE,
)

## Add feature mode parameters
if task_mode == "convolution"
    config_kwargs[:conv_variables] = conv_variables
    config_kwargs[:conv_kernel_sizes] = conv_kernel_sizes
    config_kwargs[:feature_importance_threshold] = feature_importance_threshold
else
    config_kwargs[:task_paths] = task_paths
    config_kwargs[:task_weights] = weights_tot
end

config = make_config(;
    num_models = num_models,
    input_path = TRAINING_PATH,
    experiment_name = EXPERIMENT_NAME,
    config_kwargs...
)

## --- Training ---
if RUN_TRAINING
    println("\n", "="^70)
    println("TRAINING: $(EXPERIMENT_NAME)")
    println("  num_models=$(num_models), task_mode=$(task_mode)")
    println("  n_trees=$(n_trees), max_depth=$(max_depth), class_weights=$(class_weights)")
    println("  SIG_QUALITY=$(REMOVE_LOW_SIG_QUALITY ? SIG_QUALITY_THRESHOLD : "off"), PGG=$(REMOVE_HIGH_PGG ? PGG_THRESHOLD : "off")")
    println("  met_probs_train=$(met_probs_train)")
    if task_mode == "convolution"
        println("  conv_variables=$(config.conv_variables)")
        println("  conv_kernel_sizes=$(config.conv_kernel_sizes)")
    end
    println("="^70)

    train_multi_model(config)
    println("Training complete.")
end

## --- Evaluation on testing set ---
if RUN_EVALUATION
    println("\nEvaluating on TESTING set...")
    test_results = run_evaluation(
        config, "TESTING", TESTING_PATH, met_probs_test;
        prediction_outfile = "predictions_test_$(EXPERIMENT_NAME).h5",
    )
end

## --- Evaluation on validation set ---
if RUN_VALIDATION
    println("\nEvaluating on VALIDATION set...")
    val_results = run_evaluation(
        config, "VALIDATION", VALIDATION_PATH, met_probs_test;
        prediction_outfile = "predictions_val_$(EXPERIMENT_NAME).h5",
    )
end

## --- Pass 2 met_prob sweep ---
if RUN_PASS2_SWEEP
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
end

## --- QC output ---
if RUN_QC
    config.input_path = QC_PATH
    config.met_probs  = met_probs_test
    QC_scan(config)
    println("QC complete. Check output files in $(QC_PATH)")
end
