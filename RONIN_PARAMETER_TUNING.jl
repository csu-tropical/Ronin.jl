##=============================================================================
## RONIN Parameter Tuning Template
##=============================================================================
## This script provides a structured workflow for training, testing, and
## evaluating RONIN models with different parameter configurations.
##
## WORKFLOW:
##   1. Set your data paths and experiment name
##   2. Configure the parameters you want to test in the EXPERIMENT section
##   3. Run the script -- results are printed and logged to a DataFrame
##   4. Change parameters and re-run to compare experiments
##
## The script is divided into sections that you can run incrementally
## in the REPL or as a complete script.
##=============================================================================

using Ronin
using DataFrames
using Dates

##=============================================================================
## SECTION 1: DATA PATHS
##=============================================================================
## Edit these to point to your data. Use absolute paths.

CASE_PATHS = [
    "/Users/mmbell/Science/ronin_testing/CFRADS/BERYL",
    "/Users/mmbell/Science/ronin_testing/CFRADS/EARL",
]

TRAINING_PATH   = "/Users/mmbell/Science/ronin_testing/CFRADS/TRAINING/"
TESTING_PATH    = "/Users/mmbell/Science/ronin_testing/CFRADS/TESTING/"
VALIDATION_PATH = "/Users/mmbell/Science/ronin_testing/CFRADS/VALIDATION/"

## Set to true the FIRST time to split data. Set to false on subsequent runs
## so you don't re-shuffle each time (keeps splits consistent across experiments).
SPLIT_DATA = false

if SPLIT_DATA
    Ronin.split_training_testing_validation!(CASE_PATHS, TRAINING_PATH, TESTING_PATH, VALIDATION_PATH)
    println("Data split complete: train=$(TRAINING_PATH), test=$(TESTING_PATH), val=$(VALIDATION_PATH)")
end

##=============================================================================
## SECTION 2: EXPERIMENT CONFIGURATION
##=============================================================================
## Give each experiment a unique name so you can track what you changed.

EXPERIMENT_NAME = "explore_default"
EXPERIMENT_NOTES = "Initial baseline run with default parameters"

##-----------------------------------------------------------------------------
## 2a. Model architecture -- number of passes in the multi-pass cascade
##-----------------------------------------------------------------------------
num_models = 2

##-----------------------------------------------------------------------------
## 2b. Meteorological probability thresholds
##     Format: (low_threshold, high_threshold) per pass
##     Gates with met probability in [low, high] are classified as "uncertain"
##     and passed to the next model in the cascade.
##     Tighter range = more gates re-examined; wider = more aggressive filtering.
##
##     Tuning guidance:
##       - Increase high_threshold (e.g., .95) to remove more NMD
##       - Decrease low_threshold (e.g., .05) to retain more MD
##       - Final pass thresholds matter most for the output quality
##-----------------------------------------------------------------------------
met_probs_train = [(0.1f0, 0.8f0), (0.1f0, 0.999f0)]
met_probs_test  = [(0.1f0, 0.8f0), (0.1f0, 0.999f0)]

##-----------------------------------------------------------------------------
## 2c. Signal quality filtering
##     SIG_QUALITY_VAR: variable name in CfRadial (SQI for TDR, NCP for ELDORA)
##     SIG_QUALITY_THRESHOLD: remove gates at or below this value
##     REMOVE_LOW_SIG_QUALITY: toggle filtering on/off
##
##     Tuning guidance:
##       - Higher threshold = more aggressive removal of noisy gates
##       - Try 0.1, 0.2, 0.3 to see impact on MD retention vs NMD removal
##-----------------------------------------------------------------------------
SIG_QUALITY_VAR_NAME    = "SQI"
SIG_QUALITY_THRESHOLD   = 0.2f0
REMOVE_LOW_SIG_QUALITY  = true

##-----------------------------------------------------------------------------
## 2d. Ground gate filtering (PGG)
##     PGG_THRESHOLD: probability threshold for ground gate removal
##     REMOVE_HIGH_PGG: toggle filtering on/off
##
##     Tuning guidance:
##       - 1.0 = only remove gates certain to be ground (conservative)
##       - 0.8 = also remove gates likely to be ground (more aggressive)
##       - Try 0.5, 0.8, 1.0 to see the trade-off
##-----------------------------------------------------------------------------
REMOVE_HIGH_PGG = true
PGG_THRESHOLD   = 0.8f0

##-----------------------------------------------------------------------------
## 2e. Random forest hyperparameters
##     n_trees: number of trees in each random forest
##     max_depth: maximum depth of each tree
##     class_weights: "" (uniform) or "balanced" (weight by class frequency)
##
##     Tuning guidance:
##       n_trees:  more trees = smoother predictions, slower training
##                 try 11, 21, 51, 101
##       max_depth: deeper = more capacity, risk of overfitting
##                  try 8, 10, 14, 18, -1 (unlimited)
##       class_weights: "balanced" recommended when MD >> NMD or vice versa
##-----------------------------------------------------------------------------
n_trees       = 51
max_depth     = 14
class_weights = "balanced"

##-----------------------------------------------------------------------------
## 2f-alt. Convolution pre-processor mode (alternative to hand-crafted features)
##     Set task_mode = "convolution" to use the convolution bank instead of
##     hand-crafted features. In this mode, task_paths and task_weights are
##     ignored -- features are generated automatically.
##
##     conv_variables: which radar fields to convolve
##     conv_kernel_sizes: spatial scales for convolution kernels
##     feature_importance_threshold: fraction of max importance below which
##                                   features are pruned after training
##
##     Tuning guidance:
##       - Start with ["DBZ", "VEL"] and add "PGG", "SIG" to see impact
##       - Kernel sizes [3, 5, 7] cover fine to coarse texture
##       - Lower threshold (0.005) retains more features; higher (0.05) prunes more
##-----------------------------------------------------------------------------
## Uncomment below to use convolution mode instead of hand-crafted features:
 task_mode = "convolution"
 conv_variables = ["DBZ", "VEL", "SIG"]
 conv_kernel_sizes = [3, 5, 7]
 feature_importance_threshold = 0.01

#task_mode = ""

##-----------------------------------------------------------------------------
## 2f. Feature configuration (task lists and spatial windows)
##     Each pass gets a list of features to calculate. You can specify these
##     inline as Vector{String}, or as a file path (String) for backward
##     compatibility. Inline lists are preferred since they keep the full
##     experiment configuration in one place.
##
##     Available spatial features: AVG(var), ISO(var), STD(var)
##     Available derived features: AHT, PGG, RNG, NRG, ELV, SIG
##
##     Window options from RoninConstants.jl:
##       standard_window   -- 7x7 with weights in both dimensions
##       azi_window        -- 7x7 with weights only in azimuth
##       range_window      -- 7x7 with weights only in range
##       placeholder_window -- for features that don't need spatial context
##
##     Each task list must have the same number of entries as its
##     corresponding weight vector. One weight matrix per feature.
##
##     Tuning guidance:
##       - Add/remove spatial features to see which contribute most
##       - Try different window types (range vs azi vs standard)
##       - Derived features (PGG, AHT, ELV) add physical context
##-----------------------------------------------------------------------------

## Pass 1: initial classification features
pass_1_tasks = ["DBZ", "STD(DBZ)", "ISO(DBZ)", "PGG", "SIG"]
## Pass 2: refined classification on uncertain gates
pass_2_tasks = ["DBZ", "STD(DBZ)", "ISO(DBZ)", "STD(VEL)", "ISO(VEL)"]

## task_paths accepts either inline Vector{String} or file path String per pass
task_paths = Union{String, Vector{String}}[pass_1_tasks, pass_2_tasks]

## To use legacy task files instead, uncomment below:
# task_paths = Union{String, Vector{String}}[
#     "./MODEL_SETUP/MODEL_TASKS/tasks_1.txt",
#     "./MODEL_SETUP/MODEL_TASKS/tasks_2.txt",
# ]

sw = Ronin.standard_window
aw = Ronin.azi_window
rw = Ronin.range_window
pw = Ronin.placeholder_window

task_1_weights = [pw, rw, aw, pw, pw]
task_2_weights = [pw, rw, aw, rw, aw]
weights_tot = [task_1_weights, task_2_weights]

##-----------------------------------------------------------------------------
## 2g. QC and output settings
##-----------------------------------------------------------------------------
HAS_INTERACTIVE_QC = true
QC_var     = "VE"
remove_var = "DBZ"
VARS_TO_QC = ["DBZ", "VEL"]
QC_SUFFIX  = "_QC"
mask_names = ["mask_pass_0", "mask_pass_1"]

verbose          = true
replace_missing  = false
write_out        = true
QC_mask          = false
overwrite_output = true

##=============================================================================
## SECTION 3: BUILD MODEL CONFIG
##=============================================================================
## You generally don't need to edit this section.

base_name          = "trained_model_$(EXPERIMENT_NAME)"
base_name_features = "output_features_$(EXPERIMENT_NAME)"

model_output_paths   = [base_name * "_$(i).jld2" for i in 1:num_models]
feature_output_paths = [base_name_features * "_$(i-1).h5" for i in 1:num_models]

config_kwargs = Dict{Symbol,Any}(
    :num_models             => num_models,
    :met_probs              => met_probs_train,
    :model_output_paths     => model_output_paths,
    :feature_output_paths   => feature_output_paths,
    :input_path             => TRAINING_PATH,
    :task_mode              => task_mode,
    :file_preprocessed      => [false for _ in 1:num_models],
    :task_paths             => task_paths,
    :task_weights           => weights_tot,
    :verbose                => verbose,
    :REMOVE_HIGH_PGG        => REMOVE_HIGH_PGG,
    :PGG_THRESHOLD          => PGG_THRESHOLD,
    :REMOVE_LOW_SIG_QUALITY => REMOVE_LOW_SIG_QUALITY,
    :SIG_QUALITY_THRESHOLD  => SIG_QUALITY_THRESHOLD,
    :SIG_QUALITY_VAR        => SIG_QUALITY_VAR_NAME,
    :remove_var             => remove_var,
    :QC_var                 => QC_var,
    :FILL_VAL               => Ronin.FILL_VAL,
    :HAS_INTERACTIVE_QC     => HAS_INTERACTIVE_QC,
    :replace_missing        => replace_missing,
    :write_out              => write_out,
    :QC_mask                => QC_mask,
    :mask_names             => mask_names,
    :VARS_TO_QC             => VARS_TO_QC,
    :QC_SUFFIX              => QC_SUFFIX,
    :n_trees                => n_trees,
    :max_depth              => max_depth,
    :overwrite_output       => overwrite_output,
    :class_weights          => class_weights,
)

## Add convolution-mode parameters if active
if task_mode == "convolution"
    config_kwargs[:conv_variables] = @isdefined(conv_variables) ? conv_variables : ["DBZ", "VEL"]
    config_kwargs[:conv_kernel_sizes] = @isdefined(conv_kernel_sizes) ? conv_kernel_sizes : [3, 5, 7]
    config_kwargs[:feature_importance_threshold] = @isdefined(feature_importance_threshold) ? feature_importance_threshold : 0.01
end

config = ModelConfig(; config_kwargs...)

##=============================================================================
## SECTION 4: TRAIN
##=============================================================================

println("\n", "="^70)
println("TRAINING: $(EXPERIMENT_NAME)")
println("  task_mode=$(task_mode)")
println("  n_trees=$(n_trees), max_depth=$(max_depth), class_weights=$(class_weights)")
println("  SIG_QUALITY=$(REMOVE_LOW_SIG_QUALITY ? SIG_QUALITY_THRESHOLD : "off"), PGG=$(REMOVE_HIGH_PGG ? PGG_THRESHOLD : "off")")
println("  met_probs_train=$(met_probs_train)")
if task_mode == "convolution"
    println("  conv_variables=$(config.conv_variables)")
    println("  conv_kernel_sizes=$(config.conv_kernel_sizes)")
    println("  feature_importance_threshold=$(config.feature_importance_threshold)")
end
println("="^70)

train_multi_model(config)
println("Training complete.")

##=============================================================================
## SECTION 5: EVALUATE ON TESTING DATA
##=============================================================================
## Helper function to run prediction and print full evaluation metrics

function run_evaluation(config::ModelConfig, dataset_name::String, dataset_path::String,
                        met_probs::Vector{Tuple{Float32, Float32}};
                        prediction_outfile::String = "")

    config.input_path = dataset_path
    config.met_probs  = met_probs

    write_preds = prediction_outfile != ""
    outfile = write_preds ? prediction_outfile : "model_predictions.h5"

    predictions, verification, indexers = composite_prediction(
        config;
        write_predictions_out = write_preds,
        prediction_outfile    = outfile,
    )

    targets = Vector{Bool}(verification[:])
    preds   = Vector{Bool}(predictions)

    ## Contingency table
    contingency = get_contingency(preds, targets)

    ## Scalar metrics
    prec, recall, f1, tp, fp, tn, fn, n = evaluate_model(preds, targets)

    ## Contingency hit rates
    md_hit_rate  = Float32(tp / (tp + fn))   # fraction of true MD correctly identified
    nmd_hit_rate = Float32(tn / (tn + fp))   # fraction of true NMD correctly identified

    ## Heidke Skill Score (HSS)
    expected = ((tp + fn) * (tp + fp) + (tn + fp) * (tn + fn)) / n
    hss = (tp + tn - expected) / (n - expected)

    ## Print results
    println("\n", "-"^70)
    println("RESULTS on $(dataset_name): $(config.input_path)")
    println("-"^70)
    println("  Met probability thresholds: $(met_probs)")
    println()
    println(contingency)
    println()
    println("  Counts:    TP=$(tp)  FP=$(fp)  TN=$(tn)  FN=$(fn)  Total=$(n)")
    println("  MD hit rate:  $(round(md_hit_rate, digits=4))  (TP / [TP+FN])")
    println("  NMD hit rate: $(round(nmd_hit_rate, digits=4))  (TN / [TN+FP])")
    println("  Precision:    $(round(prec, digits=4))")
    println("  Recall:       $(round(recall, digits=4))")
    println("  F1 Score:     $(round(f1, digits=4))")
    println("  HSS:          $(round(hss, digits=4))")
    println("  Accuracy:     $(round((tp + tn) / n, digits=4))")
    println("-"^70)

    return (predictions=preds, targets=targets,
            precision=prec, recall=recall, f1=f1, hss=hss,
            md_hit_rate=md_hit_rate, nmd_hit_rate=nmd_hit_rate,
            tp=tp, fp=fp, tn=tn, fn=fn, n=n)
end

## Run on testing data
println("\nEvaluating on TESTING set...")
test_results = run_evaluation(
    config, "TESTING", TESTING_PATH, met_probs_test;
    prediction_outfile = "predictions_test_$(EXPERIMENT_NAME).h5",
)

##=============================================================================
## SECTION 6: EVALUATE ON VALIDATION DATA (optional)
##=============================================================================
## Uncomment to also evaluate on the validation set.
## This should be done sparingly to avoid overfitting to the validation set.

#=
println("\nEvaluating on VALIDATION set...")
val_results = run_evaluation(
    config, "VALIDATION", VALIDATION_PATH, met_probs_test;
    prediction_outfile = "predictions_val_$(EXPERIMENT_NAME).h5",
)
=#

##=============================================================================
## SECTION 7: LOG EXPERIMENT TO RESULTS TABLE
##=============================================================================
## Accumulates results across experiments in a DataFrame.
## Run multiple experiments in the same session to build a comparison table.

if !@isdefined(experiment_log)
    experiment_log = DataFrame(
        experiment    = String[],
        timestamp     = String[],
        dataset       = String[],
        num_models    = Int[],
        n_trees       = Int[],
        max_depth     = Int[],
        class_weights = String[],
        sig_quality   = String[],
        pgg_threshold = String[],
        met_probs     = String[],
        md_hit_rate   = Float32[],
        nmd_hit_rate  = Float32[],
        precision     = Float32[],
        recall        = Float32[],
        f1            = Float32[],
        hss           = Float32[],
        accuracy      = Float32[],
        tp = Int[], fp = Int[], tn = Int[], fn = Int[],
        task_mode     = String[],
        n_selected_features = Int[],
        notes         = String[],
    )
end

function log_experiment!(log::DataFrame, name::String, dataset::String, r, config::ModelConfig, notes::String)
    push!(log, (
        experiment    = name,
        timestamp     = Dates.format(now(), "yyyy-mm-dd HH:MM:SS"),
        dataset       = dataset,
        num_models    = config.num_models,
        n_trees       = config.n_trees,
        max_depth     = config.max_depth,
        class_weights = config.class_weights,
        sig_quality   = config.REMOVE_LOW_SIG_QUALITY ? string(config.SIG_QUALITY_THRESHOLD) : "off",
        pgg_threshold = config.REMOVE_HIGH_PGG ? string(config.PGG_THRESHOLD) : "off",
        met_probs     = string(config.met_probs),
        md_hit_rate   = r.md_hit_rate,
        nmd_hit_rate  = r.nmd_hit_rate,
        precision     = r.precision,
        recall        = r.recall,
        f1            = r.f1,
        hss           = r.hss,
        accuracy      = Float32((r.tp + r.tn) / r.n),
        tp = r.tp, fp = r.fp, tn = r.tn, fn = r.fn,
        task_mode     = config.task_mode,
        n_selected_features = length(config.selected_features),
        notes         = notes,
    ))
end

log_experiment!(experiment_log, EXPERIMENT_NAME, "test", test_results, config, EXPERIMENT_NOTES)
## Uncomment if running validation:
# log_experiment!(experiment_log, EXPERIMENT_NAME, "val", val_results, config, EXPERIMENT_NOTES)

println("\n", "="^70)
println("EXPERIMENT LOG (all experiments this session)")
println("="^70)
println(experiment_log[:, [:experiment, :dataset, :nmd_hit_rate, :md_hit_rate, :f1, :hss, :precision, :recall]])

##=============================================================================
## SECTION 8: MET_PROBS THRESHOLD SWEEP (auto-tune)
##=============================================================================
## Sweeps met_probs_test thresholds WITHOUT retraining to find configurations
## that meet the NMD hit-rate target while maximizing a secondary metric.
##
## How it works:
##   1. Define grids for low_threshold and high_threshold per pass
##   2. Evaluate every combination on the testing set
##   3. Filter results to NMD hit rate >= target
##   4. Rank by secondary metric (HSS by default, or MD hit rate)
##
## This is cheap to run since composite_prediction reuses the trained model.
## Uncomment the block below to run.

#=
NMD_TARGET = 0.99f0                        # hard constraint
SECONDARY_METRIC = :hss                    # :hss or :md_hit_rate

## Threshold grids -- adjust ranges to focus on promising regions
## Pass 1: first-pass thresholds (coarser, sets up the cascade)
pass1_low_grid  = [0.1f0, 0.15f0, 0.2f0]
pass1_high_grid = [0.9f0]

## Pass 2 (final): these matter most for output quality
pass2_low_grid  = [0.1f0, 0.15f0, 0.2f0, 0.3f0]
pass2_high_grid = [0.98f0, 0.99f0, 0.995f0, 0.999f0]

## Build the sweep log
if !@isdefined(sweep_log)
    sweep_log = DataFrame(
        pass1_low     = Float32[],
        pass1_high    = Float32[],
        pass2_low     = Float32[],
        pass2_high    = Float32[],
        nmd_hit_rate  = Float32[],
        md_hit_rate   = Float32[],
        precision     = Float32[],
        recall        = Float32[],
        f1            = Float32[],
        hss           = Float32[],
        accuracy      = Float32[],
        tp = Int[], fp = Int[], tn = Int[], fn = Int[],
    )
end

n_combos = length(pass1_low_grid) * length(pass1_high_grid) *
           length(pass2_low_grid) * length(pass2_high_grid)
println("\n", "="^70)
println("THRESHOLD SWEEP: $(n_combos) combinations")
println("  NMD target: $(NMD_TARGET)")
println("  Secondary metric: $(SECONDARY_METRIC)")
println("="^70)

combo_idx = 0
for p1_lo in pass1_low_grid, p1_hi in pass1_high_grid
    for p2_lo in pass2_low_grid, p2_hi in pass2_high_grid
        combo_idx += 1
        sweep_probs = [(p1_lo, p1_hi), (p2_lo, p2_hi)]
        print("  [$(combo_idx)/$(n_combos)] met_probs=$(sweep_probs) ... ")

        r = run_evaluation(config, "SWEEP", TESTING_PATH, sweep_probs)

        push!(sweep_log, (
            pass1_low    = p1_lo,
            pass1_high   = p1_hi,
            pass2_low    = p2_lo,
            pass2_high   = p2_hi,
            nmd_hit_rate = r.nmd_hit_rate,
            md_hit_rate  = r.md_hit_rate,
            precision    = r.precision,
            recall       = r.recall,
            f1           = r.f1,
            hss          = r.hss,
            accuracy     = Float32((r.tp + r.tn) / r.n),
            tp = r.tp, fp = r.fp, tn = r.tn, fn = r.fn,
        ))
        println("NMD=$(round(r.nmd_hit_rate, digits=4))  MD=$(round(r.md_hit_rate, digits=4))  HSS=$(round(r.hss, digits=4))")
    end
end

## Filter and rank
println("\n", "="^70)
println("ALL SWEEP RESULTS (sorted by $(SECONDARY_METRIC))")
println("="^70)
sorted_all = sort(sweep_log, SECONDARY_METRIC, rev=true)
println(sorted_all[:, [:pass1_low, :pass1_high, :pass2_low, :pass2_high,
                        :nmd_hit_rate, :md_hit_rate, :hss, :f1]])

passing = filter(row -> row.nmd_hit_rate >= NMD_TARGET, sweep_log)

if nrow(passing) > 0
    ranked = sort(passing, SECONDARY_METRIC, rev=true)
    println("\n", "="^70)
    println("CONFIGURATIONS MEETING NMD >= $(NMD_TARGET) (ranked by $(SECONDARY_METRIC))")
    println("="^70)
    println(ranked[:, [:pass1_low, :pass1_high, :pass2_low, :pass2_high,
                        :nmd_hit_rate, :md_hit_rate, :hss, :f1]])

    best = ranked[1, :]
    println("\n>> BEST: met_probs = [($(best.pass1_low), $(best.pass1_high)), ($(best.pass2_low), $(best.pass2_high))]")
    println("   NMD hit rate = $(round(best.nmd_hit_rate, digits=4))")
    println("   MD hit rate  = $(round(best.md_hit_rate, digits=4))")
    println("   HSS          = $(round(best.hss, digits=4))")
    println("   F1           = $(round(best.f1, digits=4))")
else
    println("\nNo configurations met NMD >= $(NMD_TARGET).")
    println("Consider: increasing n_trees, adding a 3rd pass, or raising SIG_QUALITY_THRESHOLD.")
    best_nmd = sort(sweep_log, :nmd_hit_rate, rev=true)[1, :]
    println("Closest: NMD=$(round(best_nmd.nmd_hit_rate, digits=4)) at ",
            "met_probs=[($(best_nmd.pass1_low), $(best_nmd.pass1_high)), ($(best_nmd.pass2_low), $(best_nmd.pass2_high))]")
end
=#

##=============================================================================
## SECTION 9: QC OUTPUT (optional)
##=============================================================================
## Once satisfied with results, apply QC to write corrected fields to CfRadials.
## If you found a good configuration from the sweep, update met_probs_test above
## (or set it here) before running QC.
## Uncomment the lines below to run.

#=
config.input_path = TESTING_PATH
config.met_probs  = met_probs_test
QC_scan(config)
println("QC complete. Check output files in $(TESTING_PATH)")
=#

##=============================================================================
## SECTION 10: QUICK REFERENCE -- Parameter combinations to try
##=============================================================================
## Copy-paste one of these blocks into Section 2 to run a different experiment.
##
## --- Experiment: more_trees ---
## EXPERIMENT_NAME  = "more_trees"
## EXPERIMENT_NOTES = "Double the number of trees"
## n_trees = 51
##
## --- Experiment: deeper_trees ---
## EXPERIMENT_NAME  = "deeper_trees"
## EXPERIMENT_NOTES = "Increase max depth to 18"
## max_depth = 18
##
## --- Experiment: aggressive_sig ---
## EXPERIMENT_NAME  = "aggressive_sig"
## EXPERIMENT_NOTES = "Higher signal quality threshold"
## SIG_QUALITY_THRESHOLD = 0.3f0
##
## --- Experiment: aggressive_pgg ---
## EXPERIMENT_NAME  = "aggressive_pgg"
## EXPERIMENT_NOTES = "Lower PGG threshold to catch more ground clutter"
## PGG_THRESHOLD = 0.8f0
##
## --- Experiment: tighter_final_pass ---
## EXPERIMENT_NAME  = "tighter_final_pass"
## EXPERIMENT_NOTES = "Raise final pass met threshold to .97"
## met_probs_test = [(0.1f0, 0.9f0), (0.1f0, 0.97f0)]
##
## --- Experiment: three_pass ---
## EXPERIMENT_NAME  = "three_pass"
## EXPERIMENT_NOTES = "Add a third pass to the cascade"
## num_models = 3
## met_probs_train = [(0.1f0, 0.9f0), (0.1f0, 0.9f0), (0.1f0, 0.9f0)]
## met_probs_test  = [(0.1f0, 0.9f0), (0.1f0, 0.9f0), (0.1f0, 0.95f0)]
## task_paths = ["./MODEL_SETUP/MODEL_TASKS/tasks_1.txt",
##               "./MODEL_SETUP/MODEL_TASKS/tasks_2.txt",
##               "./MODEL_SETUP/MODEL_TASKS/tasks_3.txt"]
## task_3_weights = [pw, rw, aw, rw, aw]
## weights_tot = [task_1_weights, task_2_weights, task_3_weights]
## mask_names = ["mask_pass_0", "mask_pass_1", "mask_pass_2"]
##
## --- Experiment: no_class_weights ---
## EXPERIMENT_NAME  = "no_class_weights"
## EXPERIMENT_NOTES = "Uniform class weights instead of balanced"
## class_weights = ""
##
## --- Experiment: convolution_bank ---
## EXPERIMENT_NAME  = "conv_bank"
## EXPERIMENT_NOTES = "Convolution pre-processor with automatic feature selection"
## task_mode = "convolution"
## conv_variables = ["DBZ", "VEL"]
## conv_kernel_sizes = [3, 5, 7]
## feature_importance_threshold = 0.01
##
## --- Experiment: conv_with_pgg ---
## EXPERIMENT_NAME  = "conv_with_pgg"
## EXPERIMENT_NOTES = "Convolution bank including PGG as convolved variable"
## task_mode = "convolution"
## conv_variables = ["DBZ", "VEL", "PGG"]
## conv_kernel_sizes = [3, 5, 7]
## feature_importance_threshold = 0.01
