##=============================================================================
## RONIN Shared Configuration
##=============================================================================
## This file is included by all workflow scripts. Edit parameters here ONCE.
##
## Individual scripts (can be run standalone):
##   01_split_data.jl   — one-time data split into train/test/val
##   02_train.jl        — train multi-pass cascade on all features
##   02a_evaluate.jl    — evaluate trained model on testing (and optionally val) set
##   03_importance.jl   — compute feature importance, print recommendations
##   04_retrain.jl      — retrain with pruned feature set
##   05_sweep_pass2.jl  — freeze Pass 1, sweep Pass 2 met_prob thresholds
##   06_qc.jl           — apply final model to write QC'd fields
##
## Or use run_workflow.jl with boolean flags to run multiple steps in sequence.
##=============================================================================

using Ronin
using Dates

##=============================================================================
## SECTION 1: EXPERIMENT
##=============================================================================

EXPERIMENT_NAME = "aft_training"
EXPERIMENT_NOTES = "Aft only"

##=============================================================================
## SECTION 2: DATA PATHS
##=============================================================================

CASE_PATHS = [
    "/Users/mmbell/Science/ronin_testing/aft_training",
]

TRAINING_PATH   = "/Users/mmbell/Science/ronin_testing/aft_training/training/"
TESTING_PATH    = "/Users/mmbell/Science/ronin_testing/aft_training/testing/"
VALIDATION_PATH = "/Users/mmbell/Science/ronin_testing/aft_training/validation/"

##=============================================================================
## SECTION 3: MODEL PARAMETERS
##=============================================================================

##-----------------------------------------------------------------------------
## 3a. Model architecture
##     num_models: number of passes in the multi-pass cascade
##       1 = single RF classifier
##       2 = two-pass (recommended)
##       3+ = additional refinement passes
##-----------------------------------------------------------------------------
num_models = 2

##-----------------------------------------------------------------------------
## 3b. Meteorological probability thresholds
##     met_probs_train: thresholds used during training (controls pass-to-pass masks)
##     met_probs_test:  thresholds used during inference/evaluation
##-----------------------------------------------------------------------------
met_probs_train = [(0.0f0, 1.0f0), (0.0f0, 0.99f0)]
met_probs_test  = [(0.0f0, 1.0f0), (0.0f0, 0.99f0)]

##-----------------------------------------------------------------------------
## 3c. Signal quality filtering
##-----------------------------------------------------------------------------
SIG_QUALITY_VAR_NAME    = "SQI"
SIG_QUALITY_THRESHOLD   = 0.2f0
REMOVE_LOW_SIG_QUALITY  = true

##-----------------------------------------------------------------------------
## 3d. Ground gate filtering (PGG)
##-----------------------------------------------------------------------------
REMOVE_HIGH_PGG = true
PGG_THRESHOLD   = 1.0f0

##-----------------------------------------------------------------------------
## 3e. Random forest hyperparameters
##-----------------------------------------------------------------------------
n_trees       = 51
max_depth     = 14
class_weights = "balanced"
max_training_threads = Threads.nthreads()   # use all available Julia threads for RF training

##-----------------------------------------------------------------------------
## 3f. Feature mode
##
##     Convolution mode (recommended): set task_mode = "convolution"
##       conv_variables supports raw CfRadial fields, derived variables
##       (PGG, SIG), spatial features (AVG/ISO/STD), and prior-pass
##       met probability (met_prob_pass_1).
##
##     Hand-crafted mode: set task_mode = "" and uncomment task_paths/weights
##-----------------------------------------------------------------------------
task_mode = "convolution"
conv_variables = ["DBZ", "VEL", "SIG", "PGG", "WIDTH", "AVG(VEL)", "STD(VEL)", "ISO(VEL)"]
conv_kernel_sizes = [3, 5, 7]
feature_importance_threshold = 0.01

##-----------------------------------------------------------------------------
## 3f-ii. Feature importance performance tuning
##
##     n_importance_repeats: number of random shuffles per feature (default 3)
##       Higher = more stable estimates, lower = faster. 3 is sufficient for
##       screening; increase to 5-10 for final feature selection.
##
##     importance_subsample_fraction: fraction of training gates to evaluate on
##       (default 1.0 = all gates). For exploration use 0.1-0.3, for final
##       decisions use 0.5-1.0. 100K+ gates is statistically sufficient.
##
##     For multi-threaded speedup, start Julia with:
##       julia --threads=auto
##       or: JULIA_NUM_THREADS=8 julia
##-----------------------------------------------------------------------------
n_importance_repeats = 3
importance_subsample_fraction = 0.3

## --- Hand-crafted features (uncomment to use instead) ---
# task_mode = ""
# pass_1_tasks = ["DBZ", "STD(DBZ)", "ISO(DBZ)", "PGG", "SIG"]
# pass_2_tasks = ["DBZ", "STD(DBZ)", "ISO(DBZ)", "STD(VEL)", "ISO(VEL)"]
# task_paths = Union{String, Vector{String}}[pass_1_tasks, pass_2_tasks]
# sw = Ronin.standard_window; aw = Ronin.azi_window
# rw = Ronin.range_window; pw = Ronin.placeholder_window
# task_1_weights = [pw, rw, aw, pw, pw]
# task_2_weights = [pw, rw, aw, rw, aw]
# weights_tot = [task_1_weights, task_2_weights]

##-----------------------------------------------------------------------------
## 3g. Per-pass feature configuration
##
##     Each pass can have its own conv_variables and selected_features.
##     Pass 2+ typically adds "met_prob_pass_N" as a predictor.
##
##     PASS_CONFIG is a Dict mapping pass number → overrides.
##     Any pass not listed uses the defaults (conv_variables, Int[]).
##     Set selected_features after running importance for that pass.
##
##     The config.conv_variables and config.selected_features are updated
##     automatically by configure_pass! before each pass operation.
##-----------------------------------------------------------------------------
PASS_CONFIG = Dict(
    1 => (
        conv_variables = conv_variables,
        selected_features = []
    ),
    2 => (
        conv_variables = vcat(conv_variables, ["met_prob_pass_1"]),
        selected_features = [],
        masked_conv_variables = ["DBZ", "VEL", "SIG"],
        masked_conv_kernel_types = ["mean", "gaussian", "laplacian"],
        masked_conv_kernel_sizes = [3, 5, 7],
        masked_conv_threshold = 0.1f0,
    ),
)

## For backward compatibility: SELECTED_FEATURES sets Pass 1 if PASS_CONFIG[1]
## is not defined. Prefer using PASS_CONFIG above for new configurations.
SELECTED_FEATURES = get(PASS_CONFIG, 1, (selected_features=Int[],)).selected_features

##-----------------------------------------------------------------------------
## 3h. QC and output settings
##-----------------------------------------------------------------------------
HAS_INTERACTIVE_QC = true
QC_var     = "VE"
remove_var = "DBZ"
VARS_TO_QC = ["DBZ", "VEL"]
QC_SUFFIX  = "_QC"

##=============================================================================
## SECTION 4: SWEEP PARAMETERS (used by 05_sweep_pass2.jl)
##=============================================================================
SWEEP_MET_PROB_LOW_GRID  = Float32[0.1, 0.2, 0.3, 0.4]
SWEEP_MET_PROB_HIGH_GRID = Float32[0.6, 0.7, 0.8, 0.9]
USE_MET_PROB_AS_FEATURE  = true

SWEEP_INFERENCE          = true
INFER_LOW_GRID           = Float32[0.1, 0.2, 0.3]
INFER_HIGH_GRID          = Float32[0.98, 0.99, 0.999]

NMD_TARGET       = 0.99f0
SECONDARY_METRIC = :hss
SKIP_EXISTING_SWEEP = false   # Reuse trained models/features from prior sweep runs

##=============================================================================
## SECTION 5: QC OUTPUT PATH (used by 06_qc.jl)
##=============================================================================
QC_PATH = TESTING_PATH

##=============================================================================
## SECTION 6: HYPERTUNING PARAMETERS (used by run_workflow.jl)
##=============================================================================
HYPERTUNE_PASS            = 1
HYPERTUNE_N_TREES_GRID    = [11, 21, 51, 101, 151]
HYPERTUNE_MAX_DEPTH_GRID  = [8, 10, 12, 14, 16, 18, 20]
HYPERTUNE_MET_THRESHOLD   = 0.5f0         # threshold for confusion matrix metrics (AUC is threshold-independent)
HYPERTUNE_SKIP_EXISTING   = false
HYPERTUNE_TEST_IMPORTANCE = true

##=============================================================================
## BUILD CONFIG (do not modify below this line)
##=============================================================================

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
    :max_training_threads   => max_training_threads,
    :compute_feature_importance => false,
    :max_training_threads   => max_training_threads,
)

if task_mode == "convolution"
    config_kwargs[:conv_variables] = conv_variables
    config_kwargs[:conv_kernel_sizes] = conv_kernel_sizes
    config_kwargs[:feature_importance_threshold] = feature_importance_threshold
    config_kwargs[:selected_features] = SELECTED_FEATURES
    config_kwargs[:n_importance_repeats] = n_importance_repeats
    config_kwargs[:importance_subsample_fraction] = importance_subsample_fraction
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

"""
    configure_pass!(config, pass; pass_config=PASS_CONFIG)

Apply per-pass settings (conv_variables, selected_features, masked conv) to config
before running a pass-specific operation. Reads from the PASS_CONFIG dict defined
in 00_config.jl.

If the pass has no entry in PASS_CONFIG, defaults to the base conv_variables
with empty selected_features and no masked convolutions.
"""
function configure_pass!(config, pass; pass_config=PASS_CONFIG)
    if haskey(pass_config, pass)
        pc = pass_config[pass]
        if hasproperty(pc, :conv_variables)
            config.conv_variables = pc.conv_variables
        end
        if hasproperty(pc, :selected_features)
            config.selected_features = pc.selected_features
        end
        if hasproperty(pc, :masked_conv_variables)
            config.masked_conv_variables = pc.masked_conv_variables
            config.masked_conv_kernel_types = pc.masked_conv_kernel_types
            config.masked_conv_kernel_sizes = pc.masked_conv_kernel_sizes
            config.masked_conv_threshold = pc.masked_conv_threshold
            config.masked_conv_met_prob_field = "met_prob_pass_$(pass - 1)"
        else
            config.masked_conv_variables = String[]
            config.masked_conv_kernel_types = String[]
            config.masked_conv_kernel_sizes = Int[]
            config.masked_conv_threshold = 0.1f0
            config.masked_conv_met_prob_field = ""
        end
    else
        config.conv_variables = conv_variables
        config.selected_features = Int[]
        config.masked_conv_variables = String[]
        config.masked_conv_kernel_types = String[]
        config.masked_conv_kernel_sizes = Int[]
        config.masked_conv_threshold = 0.1f0
        config.masked_conv_met_prob_field = ""
    end
    masked_info = isempty(config.masked_conv_variables) ? "" : ", $(length(config.masked_conv_variables)) masked_conv_vars (thresh=$(config.masked_conv_threshold))"
    printstyled("  Pass $(pass) config: $(length(config.conv_variables)) conv_variables, " *
                "$(isempty(config.selected_features) ? "all" : "$(length(config.selected_features))") features$(masked_info)\n",
                color=:cyan)
end
