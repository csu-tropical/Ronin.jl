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

EXPERIMENT_NAME = "kitchen_sink"
EXPERIMENT_NOTES = "Lots of parameters and trees"

##=============================================================================
## SECTION 2: DATA PATHS
##=============================================================================

CASE_PATHS = [
    "/path/to/sweeps",
]

TRAINING_PATH   = "/path/to/sweeps/TRAINING/"
TESTING_PATH    = "/path/to/sweeps/TESTING/"
VALIDATION_PATH = "/path/to/sweeps/VALIDATION/"

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
num_models = 1

##-----------------------------------------------------------------------------
## 3b. Meteorological probability thresholds
##     met_probs_train: thresholds used during training (controls pass-to-pass masks)
##     met_probs_test:  thresholds used during inference/evaluation
##-----------------------------------------------------------------------------
met_probs_train = [(0.1f0, 0.8f0), (0.1f0, 0.999f0)]
met_probs_test  = [(0.1f0, 0.99f0), (0.1f0, 0.999f0)]

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
## 3g. Selected features (for retrain step)
##
##     Leave empty for initial training (02_train.jl uses all features).
##     After running 03_importance.jl and reviewing the recommended features,
##     paste the recommended indices here, then run 04_retrain.jl.
##
##     Example: SELECTED_FEATURES = [1, 3, 5, 7, 12, 14, 18, 22, 25, 30]
##-----------------------------------------------------------------------------
SELECTED_FEATURES = Int[]

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

##=============================================================================
## SECTION 5: QC OUTPUT PATH (used by 06_qc.jl)
##=============================================================================
QC_PATH = TESTING_PATH

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
    :compute_feature_importance => false,
)

if task_mode == "convolution"
    config_kwargs[:conv_variables] = conv_variables
    config_kwargs[:conv_kernel_sizes] = conv_kernel_sizes
    config_kwargs[:feature_importance_threshold] = feature_importance_threshold
    config_kwargs[:selected_features] = SELECTED_FEATURES
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
