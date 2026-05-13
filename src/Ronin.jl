module Ronin

    include("./RoninFeatures.jl")
    include("./RoninConvolutions.jl")
    include("./Io.jl")
    include("./DecisionTree/DecisionTree.jl")


    using NCDatasets
    using ImageFiltering
    using Statistics
    using Images
    using Missings
    using HDF5
    using MLJ, MLJLinearModels, CategoricalArrays
    using DataFrames
    using JLD2
    using DataStructures


    export get_NCP, airborne_ht, prob_groundgate
    export calc_avg, calc_std, calc_iso, process_single_file
    export parse_directory, get_num_tasks, get_task_params, remove_validation
    export calculate_features, calculate_features_conv
    export split_training_testing!, split_training_testing_validation!
    export QC_scan, get_QC_mask
    export evaluate_model, get_feature_importance, train_model
    export train_multi_model, train_single_pass, regenerate_masks, ModelConfig, make_config, composite_prediction, get_contingency, compute_balanced_class_weights
    export write_field, characterize_misclassified_gates, composite_QC
    export ConvolutionKernel, build_kernel_bank, masked_convolve, compute_convolution_features
    export select_features, compute_rf_feature_importance, get_convolution_feature_count
    export run_evaluation, sweep_pass2_met_probs, run_hypertuning, compute_auc_roc
    export load_model_with_metadata, inspect_model_configuration
    export compute_importance, generate_pass_masks, met_prob_histogram
    export save_config, load_config, migrate_model_config

    """
        load_model(path::String, task_mode::String)

    Load a trained model from a JLD2 file, handling both storage formats:
      - `save_object` format (plain model, no keys)
      - `JLD2.jldsave` format (keyed: "model", "selected_features", etc.)

    Returns the model object regardless of which format was used.
    """
    function load_model(path::String, task_mode::String)
        if task_mode == "convolution"
            data = JLD2.load(path)
            if data isa Dict && haskey(data, "model")
                return data["model"]
            elseif data isa Dict && haskey(data, "single_stored_object")
                # Saved with save_object (wraps in "single_stored_object" key)
                return data["single_stored_object"]
            else
                return data
            end
        else
            return load_object(path)
        end
    end

    """
        load_model_with_metadata(path::String, task_mode::String)

    Load a trained model and its metadata from a JLD2 file.

    Returns a NamedTuple with fields:
      - `model`: the trained RF ensemble
      - `selected_features`: Vector{Int} of feature indices the model was trained on (empty = all features)
      - `recommended_features`: Vector{Int} of features recommended by importance analysis (empty if not computed)
      - `feature_names`: Vector{String} of feature names (empty if not saved)
      - `importances`: Vector{Float64} of feature importance scores (empty if not saved)
    """
    function load_model_with_metadata(path::String, task_mode::String)
        if task_mode == "convolution"
            data = JLD2.load(path)
            if data isa Dict
                model = if haskey(data, "model")
                    data["model"]
                elseif haskey(data, "single_stored_object")
                    data["single_stored_object"]
                else
                    data
                end
                selected = get(data, "selected_features", Int[])
                recommended = get(data, "recommended_features", Int[])
                feat_names = get(data, "feature_names", String[])
                imps = get(data, "importances", Float64[])
                conv_vars = get(data, "conv_variables", String[])
                masked_conv_vars = get(data, "masked_conv_variables", String[])
                masked_conv_ktypes = get(data, "masked_conv_kernel_types", String[])
                masked_conv_ksizes = get(data, "masked_conv_kernel_sizes", Int[])
                masked_conv_thresh = get(data, "masked_conv_threshold", 0.1f0)
                masked_conv_mpf = get(data, "masked_conv_met_prob_field", "")
                return (model=model, selected_features=selected,
                        recommended_features=recommended,
                        feature_names=feat_names, importances=imps,
                        conv_variables=conv_vars,
                        masked_conv_variables=masked_conv_vars,
                        masked_conv_kernel_types=masked_conv_ktypes,
                        masked_conv_kernel_sizes=masked_conv_ksizes,
                        masked_conv_threshold=masked_conv_thresh,
                        masked_conv_met_prob_field=masked_conv_mpf)
            else
                return (model=data, selected_features=Int[],
                        recommended_features=Int[],
                        feature_names=String[], importances=Float64[],
                        conv_variables=String[],
                        masked_conv_variables=String[],
                        masked_conv_kernel_types=String[],
                        masked_conv_kernel_sizes=Int[],
                        masked_conv_threshold=0.1f0,
                        masked_conv_met_prob_field="")
            end
        else
            model = load_object(path)
            return (model=model, selected_features=Int[],
                    recommended_features=Int[],
                    feature_names=String[], importances=Float64[],
                    conv_variables=String[],
                    masked_conv_variables=String[],
                    masked_conv_kernel_types=String[],
                    masked_conv_kernel_sizes=Int[],
                    masked_conv_threshold=0.1f0,
                    masked_conv_met_prob_field="")
        end
    end

    """
        inspect_model_configuration(path::String; io::IO=stdout)

    Pretty-print the contents of a JLD2 model file without exposing raw model weights.
    Shows feature names, importances, selected features, and any other saved metadata.
    Pass `io` keyword to redirect output (e.g., to an IOBuffer for testing).
    """
    function inspect_model_configuration(path::String; io::IO=stdout)
        if !isfile(path)
            println(io, "File not found: $(path)")
            return
        end

        data = JLD2.load(path)

        printstyled(io, "\n" * "="^70 * "\n", color=:cyan)
        printstyled(io, "  MODEL CONFIGURATION: $(basename(path))\n", color=:cyan)
        printstyled(io, "="^70 * "\n", color=:cyan)
        println(io, "  File: $(path)")

        if !(data isa Dict)
            printstyled(io, "  Format: save_object (plain model, no metadata)\n", color=:yellow)
            printstyled(io, "  Model type: $(typeof(data))\n", color=:white)
            printstyled(io, "="^70 * "\n", color=:cyan)
            return
        end

        # Determine format
        if haskey(data, "model")
            printstyled(io, "  Format: jldsave (keyed with metadata)\n", color=:green)
        elseif haskey(data, "single_stored_object")
            printstyled(io, "  Format: save_object (no metadata)\n", color=:yellow)
            printstyled(io, "  Model type: $(typeof(data["single_stored_object"]))\n", color=:white)
            printstyled(io, "="^70 * "\n", color=:cyan)
            return
        end

        # Keys present
        println(io, "  Keys: $(join(sort(collect(keys(data))), ", "))")

        # Model info
        if haskey(data, "model")
            m = data["model"]
            printstyled(io, "\n  MODEL:\n", color=:cyan)
            println(io, "    Type: $(typeof(m))")
            if hasproperty(m, :trees)
                println(io, "    Trees: $(length(m.trees))")
            end
        end

        # Convolution variables
        if haskey(data, "conv_variables")
            cv = data["conv_variables"]
            if !isempty(cv)
                printstyled(io, "\n  CONV VARIABLES ($(length(cv))):\n", color=:cyan)
                println(io, "    $(cv)")
            end
        end

        # Masked convolution config
        if haskey(data, "masked_conv_variables")
            mcv = data["masked_conv_variables"]
            if !isempty(mcv)
                printstyled(io, "\n  MASKED CONV CONFIG:\n", color=:cyan)
                println(io, "    Variables: $(mcv)")
                println(io, "    Kernel types: $(get(data, "masked_conv_kernel_types", String[]))")
                println(io, "    Kernel sizes: $(get(data, "masked_conv_kernel_sizes", Int[]))")
                println(io, "    Threshold: $(get(data, "masked_conv_threshold", 0.1f0))")
                println(io, "    Met prob field: $(get(data, "masked_conv_met_prob_field", ""))")
            end
        end

        # Feature names
        if haskey(data, "feature_names")
            names = data["feature_names"]
            printstyled(io, "\n  FEATURE NAMES ($(length(names)) total):\n", color=:cyan)
            for (i, name) in enumerate(names)
                println(io, "    $(lpad(i, 4)). $(name)")
            end
        end

        # Feature importances
        if haskey(data, "importances")
            imps = data["importances"]
            printstyled(io, "\n  FEATURE IMPORTANCES:\n", color=:cyan)
            if haskey(data, "feature_names")
                names = data["feature_names"]
                sorted_idx = sortperm(imps, rev=true)
                for (rank, idx) in enumerate(sorted_idx)
                    name = idx <= length(names) ? names[idx] : "feature_$(idx)"
                    bar_len = imps[idx] > 0 ? round(Int, imps[idx] / maximum(imps) * 30) : 0
                    bar = repeat("█", bar_len)
                    println(io, "    $(lpad(rank, 4)). $(rpad(name, 30)) $(round(imps[idx], digits=6))  $(bar)")
                end
            else
                for (i, imp) in enumerate(imps)
                    println(io, "    $(lpad(i, 4)). $(round(imp, digits=6))")
                end
            end
        end

        # Selected features (what the model was actually trained on)
        if haskey(data, "selected_features")
            sel = data["selected_features"]
            if isempty(sel)
                printstyled(io, "\n  TRAINED ON: all features\n", color=:green)
            else
                printstyled(io, "\n  TRAINED ON FEATURE SUBSET: $(length(sel))", color=:green)
                if haskey(data, "feature_names")
                    names = data["feature_names"]
                    total = length(names)
                    printstyled(io, " / $(total) ($(round(length(sel)/total*100, digits=1))% retained)\n", color=:green)
                    println(io, "    Indices: $(sel)")
                    println(io, "    Names:")
                    for idx in sel
                        name = idx <= length(names) ? names[idx] : "feature_$(idx)"
                        println(io, "      $(lpad(idx, 4)). $(name)")
                    end
                else
                    println(io)
                    println(io, "    Indices: $(sel)")
                end
            end
        end

        # Recommended features (from importance analysis, for future retrain)
        if haskey(data, "recommended_features")
            rec = data["recommended_features"]
            if !isempty(rec)
                printstyled(io, "\n  RECOMMENDED FEATURES (for retrain): $(length(rec))", color=:yellow)
                if haskey(data, "feature_names")
                    names = data["feature_names"]
                    total = length(names)
                    printstyled(io, " / $(total) ($(round(length(rec)/total*100, digits=1))% retained)\n", color=:yellow)
                    println(io, "    Indices: $(rec)")
                    println(io, "    Names:")
                    for idx in rec
                        name = idx <= length(names) ? names[idx] : "feature_$(idx)"
                        println(io, "      $(lpad(idx, 4)). $(name)")
                    end
                else
                    println(io)
                    println(io, "    Indices: $(rec)")
                end
            end
        end

        printstyled(io, "="^70 * "\n", color=:cyan)
    end

    """
    ## Stuct used to store configuration information for a given model

    # Required arguments
    ```julia
    num_models:::Int64
    ```
    Number of ML models in the model chain. Can be one or more.

    ```julia
    model_output_paths::Vector{String}
    ```
    Vector containing paths to each model in the model chain. Should be same length as the number of models

    ```julia
    met_probs::Vector{Tuple{Float32,Float32}}
    ```
    Vector containing the decision range for a gate to be considered meteorological in each model in the chain. Example, if set to (.9, 1),
        > 90% of trees in the random forest must assign a gate a label of meteorological for it to be considered meteorological.
        The range is exclusive on both ends. That is, for a gate to be classified as non-meteorological, it must have
        a probability LESS THAN the low threshold, and for a gate to be classified as meteorological it must have
        a probability GREATER THAN the high threshold. For multi-pass models, gates between these thresholds (inclusive) will
        be sent on to the next pass. Form is (low_threshold, high_threshold)

    ```julia
    feature_output_paths::Vector{String}
    ```
    Vector containing paths representing the locations to output calculated features to for each model in the chain.

    ```julia
    input_path::String
    ```
    Directory containing input radar data

    ```julia
    task_mode::String
    ```
    Whether to obtain feature tasks from a set of input files or user specified vector of strings. Planned to be implemented in a future release.
    For now, codebase behavior is agnostic to its value.

    ```julia
    file_preprocessed::Vector{Bool}
    ```
    For each model in the chain, contains a boolean value signifying if the correspondant feature output path has already been processed. If true,
    will open the file at this path instead of re-calculating input features.

    # Optional arguments

    ## Input tasks and weights
    ## The following arguments are only quasi-optional, one of them must be set.
    ``` task_paths::Vector{String} = [""]
        task_list::Vector{String} = [""]
        task_weights::Vector{Vector} = [[Matrix{Union{Float32, Missing}}(undef, 0,0)]]
    ```
        Currently only `task_paths` are supported. Contains a vector of the same length as the number of
        models, with each entry being the path to a file contianing the tasks for the pass. Future plans involve
        allowing a usesr to specify vectors of tasks in `task_list`.

        `task_weights` must be a vector of vectors, with the first dimension the same length as the number of models in the
        chain. The second dimension much either be 1, containing the default weight matrix `Matrix{Union{Float32, Missing}}(undef, 0,0)`,
        or a secondary vector of matrixes - one matrix for each task in the passs. Sample weight matrixes are defined in RoninConstants.jl

    ```julia
    verbose::Bool = true
    ```
    Whether to print out timing information, etc.

    ```julia
    REMOVE_LOW_SIG_QUALITY::Bool = true
    ```
    Whether to automatically remove gates that do not meet a basic Signal Quality threshold.
    Variable used to determine this specified in `SIG_QUALITY_VAR`

    ```julia
    REMOVE_HIGH_PGG::Bool = true
    ```
    Whether to automatically remove gates that do not meet a basic PGG threshold

    ```julia
    HAS_INTERACTIVE_QC::Bool = false
    ```
    Whether the radar data has already had interactive QC applied to it

    ```julia
    QC_var::String = "VG"
    ```
    If radar data has interactive QC already applied, the name of a variable that the QC has been applied to

    ```julia
    remove_var::String = "VV"
    ```
    Name of a raw variable in the radar data that can be used to determine the location of missing gates
    ```julia
    FILL_VAL::Float32 = RoninConstants.FILL_VAL
    ```
    Fill value for output cfradials
    ```julia
    replace_missing::Bool = false
    ```
    For spatial feature (AVG, STD, etc.) calculation, whether or not to replace MISSING gates in the mask area with FILL_VAL

    ```julia
    write_out::Bool = true
    ```
    Whether or not to write the calculated input features to disk, paths specified in feature_output_paths

    ```julia
    QC_mask::Bool = false
    ```
    For the first model in the chain, whether or not to mask gates considered for feature calculation using a mask specified by `mask_name`
    More details elsewhere in the documentation.

    ```julia
    mask_names::Vector{String} = [""]
    ```
    List of names for masks in the model. Must be of same length as number of models in the chain.
    In the case of a model with `QC_mask` set to `true`, the first mask name in this vector should contain
    a string denoting the name of a field in all cfradial files that is dimensioned the same as the radar sweeps
    and contains values of missing where data is not to be considred, and values of float otherwise.

    ```julia
    VARS_TO_QC::Vector{String} = ["VV", "ZZ"]
    ```
    List of variables to apply QC to to get mask for next model in chain

    ```julia
    QC_SUFFIX::String
    ```
    Postfix to apply to variable name once QC has been applied.

    ```julia
    class_weights::String = ""
    ```
    Class weighting scheme to apply in the training of RF model. Currently only "balanced" is implemented.

    ```julia
    n_trees::Int = 21
    ```
    Number of trees in the random forest

    ```julia
    max_depth::Int = 14
    ```
    Maximum depth of any one tree in the random forest

    ```julia
    overwrite_output::Bool = false
    ```
    If true, will remove/overwrite existing files when internal functionality attempts to write new data to them

    ```julia
    SIG_QUALITY_THRESHOLD::Float32 = .2
    ```
    If REMOVE_LOW_NCP is set to true, threshold at or below which to remove data.

    ```julia
    PGG_THRESHOLD::Float32 = 1.
    ```
    If REMOVE_HIGH_PGG is set to true, threshold at or above which to remove data.

    ```julia
    SIGNAL_QUALITY_VAR::String = "NCP"
    ```
    Name of variable in cfradial file representing signal quality. Most commonly
    "NCP" or "SQI"

    """
    Base.@kwdef mutable struct ModelConfig

        num_models::Int64
        model_output_paths::Vector{String}
        met_probs::Vector{Tuple{Float32, Float32}}

        feature_output_paths::Vector{String}

        input_path::String

        task_mode::String

        file_preprocessed::Vector{Bool}

        task_paths::Vector{Union{String, Vector{String}}} = [""]
        task_list::Vector{String} = [""]
        task_weights::Vector{Vector{Matrix{Union{Float32, Missing}}}} = [[Matrix{Union{Float32, Missing}}(undef, 0,0)]]

        verbose::Bool = true
        REMOVE_LOW_SIG_QUALITY::Bool = true
        REMOVE_HIGH_PGG::Bool = true
        HAS_INTERACTIVE_QC::Bool = false
        QC_var::String = "VG"
        remove_var::String = "VV"
        FILL_VAL::Real = FILL_VAL
        replace_missing::Bool = false
        write_out::Bool = true
        QC_mask::Bool = false
        mask_names::Vector{String} = [""]

        VARS_TO_QC::Vector{String} = ["VV", "ZZ"]
        QC_SUFFIX::String = "_QC"

        ###options are "" or "balanced"
        class_weights::String = ""

        n_trees::Int = 21
        max_depth::Int=14

        overwrite_output::Bool = false

        SIG_QUALITY_THRESHOLD::Float32 = .2f0
        PGG_THRESHOLD::Float32 = 1.f0

        SIG_QUALITY_VAR::String = "NCP"

        conv_variables::Vector{String} = ["DBZ", "VEL"]
        conv_kernel_sizes::Vector{Int} = [3, 5, 7]
        selected_features::Vector{Int} = Int[]
        feature_importance_threshold::Float64 = 0.01
        compute_feature_importance::Bool = true
        n_importance_repeats::Int = 3
        importance_subsample_fraction::Float64 = 1.0
        max_training_threads::Int = 1

        masked_conv_variables::Vector{String} = String[]
        masked_conv_kernel_types::Vector{String} = String[]
        masked_conv_kernel_sizes::Vector{Int} = Int[]
        masked_conv_threshold::Float32 = 0.1f0
        masked_conv_met_prob_field::String = ""

    end

    ## Backward-compat property shim: in v1.1.0 the field was named REMOVE_LOW_NCP.
    ## Reading or writing the old name still works but emits a deprecation warning.
    ## Tests for `propertynames` / Tab-completion still see only the real fields.
    function Base.getproperty(c::ModelConfig, name::Symbol)
        if name === :REMOVE_LOW_NCP
            Base.depwarn("`ModelConfig.REMOVE_LOW_NCP` is deprecated; use `REMOVE_LOW_SIG_QUALITY`.",
                         :getproperty)
            return getfield(c, :REMOVE_LOW_SIG_QUALITY)
        end
        return getfield(c, name)
    end

    function Base.setproperty!(c::ModelConfig, name::Symbol, x)
        if name === :REMOVE_LOW_NCP
            Base.depwarn("`ModelConfig.REMOVE_LOW_NCP` is deprecated; use `REMOVE_LOW_SIG_QUALITY`.",
                         :setproperty!)
            return setfield!(c, :REMOVE_LOW_SIG_QUALITY, convert(Bool, x))
        end
        return setfield!(c, name, convert(fieldtype(ModelConfig, name), x))
    end

    """
        make_config(; num_models, input_path, experiment_name="experiment", kwargs...)

    Convenience constructor for `ModelConfig` that auto-generates paths and
    mask names from `num_models` and `experiment_name`, reducing boilerplate.

    Auto-generated fields (can be overridden via kwargs):
      - `model_output_paths`:   `["trained_model_<name>_<i>.jld2" for i in 1:num_models]`
      - `feature_output_paths`: `["output_features_<name>_<i-1>.h5" for i in 1:num_models]`
      - `mask_names`:           `["mask_pass_<i-1>" for i in 1:num_models]`
      - `met_probs`:            `[(0.1f0, 0.9f0) for _ in 1:num_models]`
      - `file_preprocessed`:    `[false for _ in 1:num_models]`

    All other `ModelConfig` fields can be passed as keyword arguments.
    """
    function make_config(; num_models::Int, input_path::String,
                           experiment_name::String="experiment", kwargs...)

        ## Backward-compat: rewrite the v1.1.0 keyword name to its v1.2.0 replacement.
        kwargs_dict = Dict{Symbol,Any}(kwargs)
        if haskey(kwargs_dict, :REMOVE_LOW_NCP)
            Base.depwarn("`REMOVE_LOW_NCP` is deprecated; use `REMOVE_LOW_SIG_QUALITY`.",
                         :make_config)
            kwargs_dict[:REMOVE_LOW_SIG_QUALITY] = pop!(kwargs_dict, :REMOVE_LOW_NCP)
        end

        defaults = Dict{Symbol,Any}(
            :model_output_paths   => ["trained_model_$(experiment_name)_$(i).jld2" for i in 1:num_models],
            :feature_output_paths => ["output_features_$(experiment_name)_$(i-1).h5" for i in 1:num_models],
            :mask_names           => ["mask_pass_$(i-1)" for i in 1:num_models],
            :met_probs            => [(0.1f0, 0.9f0) for _ in 1:num_models],
            :file_preprocessed    => [false for _ in 1:num_models],
        )

        # User kwargs override defaults
        merged = merge(defaults, kwargs_dict)

        # Truncate per-pass vectors to num_models length so that e.g.
        # met_probs with 2 entries still works when num_models = 1
        per_pass_keys = [:model_output_paths, :feature_output_paths, :mask_names,
                         :met_probs, :file_preprocessed, :task_paths, :task_weights]
        for k in per_pass_keys
            if haskey(merged, k) && merged[k] isa AbstractVector && length(merged[k]) > num_models
                merged[k] = merged[k][1:num_models]
            end
        end

        ModelConfig(; num_models=num_models, input_path=input_path, merged...)
    end

    ## Required ModelConfig fields (no @kwdef defaults). Used by load_config to
    ## validate that legacy single_stored_object files actually yielded enough
    ## values to construct a ModelConfig.
    const _MODELCONFIG_REQUIRED_FIELDS = (:num_models, :model_output_paths, :met_probs,
                                          :feature_output_paths, :input_path, :task_mode,
                                          :file_preprocessed)

    """
        save_config(path::String, config::ModelConfig)

    Save a `ModelConfig` to a JLD2 file in dict-keyed format: one key per field.

    This format is robust to struct evolution. Adding a field to `ModelConfig` later
    just means a new key for fresh files; old files still load via `load_config`,
    which fills any missing key with the struct's default. No migration required
    on every shape change.

    Replaces the legacy pattern `JLD2.save_object(path, config)`, which pickled the
    struct shape and broke on every field addition (see issue #34). The legacy
    format is still readable by `load_config`.
    """
    function save_config(path::String, config::ModelConfig)
        ## Symbol-keyed Dict so the `;pairs...` splat lands as JLD2 keyword args.
        ## JLD2 serializes keyword names as string keys on disk, which `load_config`
        ## then reads back via `data["field_name"]`.
        pairs = Dict{Symbol,Any}()
        for fld in fieldnames(ModelConfig)
            pairs[fld] = getproperty(config, fld)
        end
        ## Format marker so future loaders can detect / version this layout.
        pairs[:__ronin_config_format__] = "dict_v1"
        jldsave(path; pairs...)
        return path
    end

    """
        load_config(path::String) -> ModelConfig

    Load a `ModelConfig` from a JLD2 file, transparently handling both storage formats:

    - **Dict-keyed format** (written by `save_config`): each ModelConfig field is
      stored under its own key. Missing keys fall back to the struct's `@kwdef`
      defaults, so future field additions do not break old files.

    - **Legacy `save_object` format** (single `single_stored_object` key holding a
      pickled struct): JLD2 may load these as a `ReconstructedMutable` shadow type
      when the saved struct shape no longer matches the current `ModelConfig`.
      Each accessible field is copied via `getproperty` into a fresh, real
      `ModelConfig` instance; defaults fill in any new fields the saved file lacks.

    A real, mutable `ModelConfig` is returned regardless of source format —
    callers can then assign to fields (`config.task_paths = [...]`) without
    hitting `setproperty!` errors on `ReconstructedMutable`.
    """
    function load_config(path::String)
        isfile(path) || error("Config file not found: $(path)")
        data = JLD2.load(path)
        data isa Dict || error("Unexpected JLD2 contents in $(path): $(typeof(data))")

        ## Legacy save_object: one key, "single_stored_object", holding a pickled struct.
        if haskey(data, "single_stored_object")
            return _load_legacy_config(data["single_stored_object"], path)
        end

        ## Dict-keyed format. Match keys against current ModelConfig fields and fill
        ## defaults for anything missing.
        field_names = fieldnames(ModelConfig)
        matched = count(f -> haskey(data, String(f)), field_names)
        if matched < length(_MODELCONFIG_REQUIRED_FIELDS)
            error("$(path) does not look like a saved ModelConfig (only $(matched) " *
                  "matching field keys; expected at least $(length(_MODELCONFIG_REQUIRED_FIELDS)) " *
                  "required). Confirm this is a config file written by `save_config`, " *
                  "not a trained-model file.")
        end

        kwargs = Dict{Symbol,Any}()
        for fld in field_names
            key = String(fld)
            haskey(data, key) || continue
            kwargs[fld] = data[key]
        end
        return ModelConfig(; kwargs...)
    end

    ## Internal: rebuild a ModelConfig from a (possibly Reconstructed) legacy object.
    function _load_legacy_config(old, path::String)
        type_name = string(typeof(old).name.name)
        if !occursin("ModelConfig", type_name)
            @warn "load_config: $(path) is in legacy save_object format but the stored " *
                  "object type does not look like a ModelConfig (got $(type_name)). " *
                  "Proceeding to copy whatever fields match by name."
        end

        kwargs = Dict{Symbol,Any}()
        copied = 0
        for fld in fieldnames(ModelConfig)
            ## getproperty works on ReconstructedMutable for fields that exist in the
            ## saved struct; we use try/catch instead of hasproperty because some JLD2
            ## reconstruction wrappers do not override propertynames.
            val = try
                getproperty(old, fld)
            catch
                continue
            end
            kwargs[fld] = val
            copied += 1
        end

        missing_required = [r for r in _MODELCONFIG_REQUIRED_FIELDS if !haskey(kwargs, r)]
        if !isempty(missing_required)
            error("Cannot construct ModelConfig from $(path): legacy file is missing " *
                  "required field(s) $(missing_required). The file may be from a much " *
                  "older Ronin version or may not be a ModelConfig at all.")
        end

        @info "load_config: $(path) is in legacy save_object format. Copied $(copied)/" *
              "$(length(fieldnames(ModelConfig))) fields; defaults fill the rest. " *
              "Re-save with `save_config(path, config)` to migrate to the dict-keyed " *
              "format that survives future struct changes."

        return ModelConfig(; kwargs...)
    end

    """
        migrate_model_config(infile::String, outfile::String; key=nothing)

    One-shot utility: load a legacy `save_object`-format ModelConfig file and re-save
    it in the new dict-keyed format written by `save_config`. Returns the rebuilt
    `ModelConfig`.

    Migration is **not required for loading** — `load_config` reads legacy files
    inline. Use this helper only when you want to upgrade a long-lived file on disk
    so future loads skip the legacy path (and the deprecation `@info`).

    `key` may be passed to select a specific JLD2 key when the file contains
    multiple stored objects; defaults to the standard `single_stored_object` key
    used by `JLD2.save_object`.
    """
    function migrate_model_config(infile::String, outfile::String; key=nothing)
        config = if key === nothing
            load_config(infile)
        else
            ## Caller specified a non-standard key — load that object and recover.
            old = JLD2.load(infile, String(key))
            _load_legacy_config(old, infile)
        end
        save_config(outfile, config)
        @info "Migrated ModelConfig: $(infile) → $(outfile) (dict-keyed format)"
        return config
    end

    """
        Helper function to compute balanced weights according to the
        algoirthm described in https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html

    """
    function compute_balanced_class_weights(samples::Vector{<:Real})
        classes = unique(samples)
        n_classes = length(classes)
        n_samples = length(samples)
        weight_dict = Dict()


        for class in classes
            weight_dict[class] = (n_samples/(n_classes * sum(samples .== class)))
        end

        return(weight_dict)

    end

    """

    Function to process a set of cfradial files and produce input features for training/evaluating a model

    # Required arguments

    ```julia
    input_loc::String
    ```

    Path to input cfradial or directory of input cfradials

    ```julia
    argument_file::String
    ```

    Path to configuration file containing which features to calculate

    ```julia
    output_file::String
    ```

    Path to output calculated features to (generally ends in .h5)

    ```julia
    HAS_INTERACTIVE_QC::Bool
    ```
    Specifies whether or not the file(s) have already undergone a interactive QC procedure.
    If true, function will also output a `Y` array used to verify where interactive QC removed gates. This array is
    formed by considering where gates with non-missing data in raw scans (specified by `remove_variable`) are
    set to missing after QC is performed.

    # Optional keyword arguments

    ```julia
    verbose::Bool=false
    ```
    If true, will print out timing information as each file is processed

    ```julia
    REMOVE_LOW_SIG_QUALITY::Bool=false
    ```

    If true, will ignore gates with Normalized Coherent Power/Signal Quality Index below a threshold specified in RQCFeatures.jl

    ```julia
    SIG_QUALITY_THRESHOLD::Float32 = .2
    ```
    Theshold at or below which to remove data

    ```julia
    SIG_QUALITY_VAR::String = "NCP"
    ```
    Name of variable containing signal quality parameter

    ```julia
    REMOVE_HIGH_PGG::Bool=false
    ```
    If true, will ignore gates with Probability of Ground Gate (PGG) values at or above a threshold specified in RQCFeatures.jl

    ```julia
    PGG_THRESHOLD
    ```
    Threshold at or above which to remove data

    ```julia
    QC_variable::String="VG"
    ```
    Name of variable in input NetCDF files that has been quality-controlled.

    ```julia
    remove_variable::String="VV"
    ```

    Name of a raw variable in input NetCDF files. Used to determine where missing data exists in the input sweeps.
    Data at these locations will be removed from the outputted features.

    ```julia
    replace_missing::Bool=false
    ```
    Whether or not to replace MISSING values with FILL_VAL in spatial parameter calculations
    Default value: False

    ```julia
    write_out::Bool=true
    ```
    Whether or not to write features out to file

    ```julia
    return_idxer::Bool = false
    ```
    If true, will return IDXER, where IDXER is a

    ```julia
    weight_matrixes::Vector{Matrix{Union{Missing, Float32}}} = [(undef, 0,0)]
    ```
    Vector containing a weight matrix for every task in the argument file. For non-spatial parameters, the
        weights are discarded, and so dummy/placeholder matrixes may be used.
    """
    function calculate_features(input_loc::String, argument_file::String, output_file::String, HAS_INTERACTIVE_QC::Bool;
        verbose::Bool=false, REMOVE_LOW_SIG_QUALITY::Bool = false, SIG_QUALITY_THRESHOLD::Float32 = .2f0, SIG_QUALITY_VAR::String="NCP",
        REMOVE_HIGH_PGG::Bool = false, PGG_THRESHOLD::Float32=1.f0,
        QC_variable::String = "VG", remove_variable::String = "VV",
        replace_missing::Bool = false, write_out::Bool=true, QC_mask::Bool = false, mask_name::String = "", return_idxer::Bool=false,
        weight_matrixes::Vector{Matrix{Union{Missing, Float32}}}= [Matrix{Union{Missing, Float32}}(undef, 0,0)],
        REMOVE_LOW_NCP=nothing)

        ## Backward-compat: REMOVE_LOW_NCP is the v1.1.0 name for REMOVE_LOW_SIG_QUALITY.
        if REMOVE_LOW_NCP !== nothing
            Base.depwarn("`REMOVE_LOW_NCP` is deprecated; use `REMOVE_LOW_SIG_QUALITY`.",
                         :calculate_features)
            REMOVE_LOW_SIG_QUALITY = REMOVE_LOW_NCP
        end

        ##If this is a directory, things get a little more complicated
        paths = Vector{String}()

        if isdir(input_loc)
            paths = parse_directory(input_loc)
        else
            paths = [input_loc]
        end

        ###Setup h5 file for outputting mined parameters
        ###processing will proceed in order of the tasks, so
        ###add these as an attribute akin to column headers in the H5 dataset
        ###Also specify the fill value used

        ##Instantiate Matrixes to hold calculated features and verification data
        output_cols = get_num_tasks(argument_file)

        newX = X = Matrix{Float32}(undef,0,output_cols)
        newY = Y = Matrix{Int64}(undef, 0,1)
        idxs = Vector{}(undef,0)


        starttime = time()

        for (i, path) in enumerate(paths)
            dims = (0,0)
            newIdx = Matrix{}(undef, 0,0)
            cfrad = Dataset(path)
            try
                pathstarttime=time()
                dims = (cfrad.dim["range"], cfrad.dim["time"])

                if QC_mask
                    ###We wish to calculate features on where the mask is NON MISSING
                    currmask = Matrix{Bool}(.! map(ismissing, cfrad[mask_name][:,:]))
                    (newX, newY, newIdx) = process_single_file(cfrad, argument_file;
                                                HAS_INTERACTIVE_QC = HAS_INTERACTIVE_QC,
                                                REMOVE_LOW_SIG_QUALITY = REMOVE_LOW_SIG_QUALITY, SIG_QUALITY_THRESHOLD = SIG_QUALITY_THRESHOLD, SIG_QUALITY_VAR=SIG_QUALITY_VAR,
                                                REMOVE_HIGH_PGG = REMOVE_HIGH_PGG, PGG_THRESHOLD=PGG_THRESHOLD, QC_variable = QC_variable, remove_variable = remove_variable,
                                                replace_missing=replace_missing, feature_mask = currmask, mask_features = true, weight_matrixes=weight_matrixes)

                else
                    (newX, newY, newIdx) = process_single_file(cfrad, argument_file;
                                                HAS_INTERACTIVE_QC = HAS_INTERACTIVE_QC,
                                                REMOVE_LOW_SIG_QUALITY = REMOVE_LOW_SIG_QUALITY, SIG_QUALITY_THRESHOLD = SIG_QUALITY_THRESHOLD, SIG_QUALITY_VAR=SIG_QUALITY_VAR,
                                                REMOVE_HIGH_PGG = REMOVE_HIGH_PGG, PGG_THRESHOLD=PGG_THRESHOLD, QC_variable = QC_variable, remove_variable = remove_variable,
                                                replace_missing=replace_missing, weight_matrixes=weight_matrixes)
                end

                close(cfrad)

                if verbose
                    println("Processed $(path) in $(time()-pathstarttime) seconds")
                end

            catch e
                if isa(e, DimensionMismatch)
                    printstyled(Base.stderr, "POSSIBLE ERRONEOUS CFRAD DIMENSIONS... SKIPPING $(path)\n"; color=:red)
                    continue
                else
                    printstyled(Base.stderr, "UNRECOVERABLE ERROR\n"; color=:red)
                    close(cfrad)
                    throw(e)

                ##@TODO CATCH exception handling for invalid task
                end
            end

            X = vcat(X, newX)::Matrix{Float32}
            Y = vcat(Y, newY)::Matrix{Int64}
            newIdx = reshape(newIdx, dims)
            push!(idxs, newIdx)
        end

        println("COMPLETED PROCESSING $(length(paths)) FILES IN $(round((time() - starttime), digits = 2)) SECONDS")

        ###Get verification information
        ###0 indicates NON METEOROLOGICAL data that was removed during interactive QC
        ###1 indicates METEOROLOGICAL data that was retained during interactive QC

        ##Probably only want to write once, I/O is very slow
        if write_out

            println("OUTPUTTING DATA IN HDF5 FORMAT TO FILE: $(output_file)")
            fid = h5open(output_file, "w")

            ###Add information to output h5 file
            attributes(fid)["Parameters"] = get_task_params(argument_file)
            attributes(fid)["MISSING_FILL_VALUE"] = FILL_VAL
            println()
            println("WRITING DATA TO FILE OF SHAPE $(size(X))")
            println("X TYPE: $(typeof(X))")

            write_dataset(fid, "X", X)
            write_dataset(fid, "Y", Y)

            close(fid)
            if return_idxer
                return X, Y, idxs
            else
                return X, Y
            end
        else

            if return_idxer
                return X, Y, idxs
            else
                return X, Y
            end
        end

    end


    """

    Function to process a set of cfradial files and produce input features for training/evaluating a model.
        Allows for user-specified tasks and weight matrices, otherwise the same as above.

    # Required arguments

    ```julia
    input_loc::String
    ```

    Path to input cfradial or directory of input cfradials

    ```julia
    tasks::Vector{String}
    ```

    Vector containing the features to be calculated for each cfradial. Example `[DBZ, ISO(DBZ)]`

    ```julia
    weight_matrixes::Vector{Matrix{Union{Missing, Float32}}}
    ```

    For each task, a weight matrix specifying how much each gate in a spatial calculation will be given.
    Required to be the same size as `tasks`

    ```julia
    output_file::String
    ```

    Location to output the calculated feature data to.

    ```julia
    HAS_INTERACTIVE_QC::Bool
    ```
    Specifies whether or not the file(s) have already undergone a interactive QC procedure.
    If true, function will also output a `Y` array used to verify where interactive QC removed gates. This array is
    formed by considering where gates with non-missing data in raw scans (specified by `remove_variable`) are
    set to missing after QC is performed.

    # Optional keyword arguments

    ```julia
    verbose::Bool = false
    ```
    If true, will print out timing information as each file is processed

    ```julia
    REMOVE_LOW_SIG_QUALITY::Bool = false
    ```
    If true, will ignore gates with Normalized Coherent Power/Signal Quality Index below a threshold specified in RQCFeatures.jl

    ```julia
    SIG_QUALITY_THRESHOLD::Float32 = .2
    ```
    Theshold at or below which to remove data

    ```
    SIG_QUALITY_VAR::String = "NCP"
    ```
    Name of variable containin signal quality information

    ```julia
    REMOVE_HIGH_PGG::Bool = false
    ```

    If true, will ignore gates with Probability of Ground Gate (PGG) values at or above a threshold specified in RQCFeatures.jl

    ```julia
    PGG_THRESHOLD
    ```
    Threshold at or above which to remove data

    ```julia
    QC_variable::String = "VG"
    ```
    Name of variable in input NetCDF files that has been quality-controlled.

    ```julia
    remove_variable::String = "VV"

    Name of a raw variable in input NetCDF files. Used to determine where missing data exists in the input sweeps.
    Data at these locations will be removed from the outputted features.

    ```
    replace_missing::Bool = false
    ```
    Whether or not to replace MISSING values with FILL_VAL in spatial parameter calculations
    Default value: False

    ```julia
    write_out::Bool = true
    ```
    Whether or not to write out to file.
    """
    ###Dispatch for when tasks are provided directly as a Vector{String} instead of a file path.
    ###Matches the same positional argument layout as the file-path version so that
    ###config.task_paths entries of either type dispatch correctly.
    function calculate_features(input_loc::String, tasks::Vector{String}, output_file::String, HAS_INTERACTIVE_QC::Bool;
        verbose::Bool=false, REMOVE_LOW_SIG_QUALITY::Bool = false, SIG_QUALITY_THRESHOLD::Float32 = .2f0, SIG_QUALITY_VAR::String="NCP",
        REMOVE_HIGH_PGG::Bool = false, PGG_THRESHOLD::Float32=1.f0,
        QC_variable::String = "VG", remove_variable::String = "VV",
        replace_missing::Bool = false, write_out::Bool=true, QC_mask::Bool = false, mask_name::String = "", return_idxer::Bool=false,
        weight_matrixes::Vector{Matrix{Union{Missing, Float32}}}= [Matrix{Union{Missing, Float32}}(undef, 0,0)],
        REMOVE_LOW_NCP=nothing)

        if REMOVE_LOW_NCP !== nothing
            Base.depwarn("`REMOVE_LOW_NCP` is deprecated; use `REMOVE_LOW_SIG_QUALITY`.",
                         :calculate_features)
            REMOVE_LOW_SIG_QUALITY = REMOVE_LOW_NCP
        end

        calculate_features(input_loc, tasks, weight_matrixes, output_file, HAS_INTERACTIVE_QC;
            verbose=verbose, REMOVE_LOW_SIG_QUALITY=REMOVE_LOW_SIG_QUALITY, SIG_QUALITY_THRESHOLD=SIG_QUALITY_THRESHOLD,
            SIG_QUALITY_VAR=SIG_QUALITY_VAR, REMOVE_HIGH_PGG=REMOVE_HIGH_PGG, PGG_THRESHOLD=PGG_THRESHOLD,
            QC_variable=QC_variable, remove_variable=remove_variable, replace_missing=replace_missing,
            write_out=write_out, QC_mask=QC_mask, mask_name=mask_name, return_idxer=return_idxer)
    end

    ## Backward-compat: accept Float64 weight_matrixes (the old default). Converts
    ## to Float32 once and forwards to the canonical method below. Common in
    ## downstream code that builds matrices via `ones(...)` or `allowmissing(ones(...))`.
    function calculate_features(input_loc::String, tasks::Vector{String},
        weight_matrixes::Vector{Matrix{Union{Missing, Float64}}},
        output_file::String, HAS_INTERACTIVE_QC::Bool; kwargs...)

        Base.depwarn("`weight_matrixes` with Float64 inner type is deprecated; " *
                     "convert via `Float32.(matrix)` for best performance.",
                     :calculate_features)
        wm32 = Vector{Matrix{Union{Missing, Float32}}}([
            convert(Matrix{Union{Missing, Float32}}, m) for m in weight_matrixes])
        return calculate_features(input_loc, tasks, wm32, output_file, HAS_INTERACTIVE_QC; kwargs...)
    end

    function calculate_features(input_loc::String, tasks::Vector{String}, weight_matrixes::Vector{Matrix{Union{Missing, Float32}}}
        ,output_file::String, HAS_INTERACTIVE_QC::Bool; verbose::Bool=false,
         REMOVE_LOW_SIG_QUALITY = false, SIG_QUALITY_THRESHOLD::Float32 = .2f0, SIG_QUALITY_VAR::String="NCP", REMOVE_HIGH_PGG = false, PGG_THRESHOLD::Float32=.1f0, QC_variable::String = "VG", remove_variable::String = "VV",
         replace_missing::Bool=false, write_out::Bool=true, QC_mask::Bool = false, mask_name::String="", return_idxer::Bool =false,
         REMOVE_LOW_NCP=nothing)

        if REMOVE_LOW_NCP !== nothing
            Base.depwarn("`REMOVE_LOW_NCP` is deprecated; use `REMOVE_LOW_SIG_QUALITY`.",
                         :calculate_features)
            REMOVE_LOW_SIG_QUALITY = REMOVE_LOW_NCP
        end

        ##If this is a directory, things get a little more complicated
        paths = Vector{String}()

        if isdir(input_loc)
            paths = parse_directory(input_loc)
        else
            paths = [input_loc]
        end

        ###Setup h5 file for outputting mined parameters
        ###processing will proceed in order of the tasks, so
        ###add these as an attribute akin to column headers in the H5 dataset
        ###Also specify the fill value used



        ##Instantiate Matrixes to hold calculated features and verification data
        output_cols = length(tasks)

        newX = X = Matrix{Float32}(undef,0,output_cols)
        newY = Y = Matrix{Int64}(undef, 0,1)
        idxs = Vector{}(undef,0)

        starttime = time()

        for (i, path) in enumerate(paths)
            dims = (0,0)
            indexer = Matrix{}(undef, 0,0)
            cfrad = Dataset(path)
            try
                pathstarttime=time()
                dims = (cfrad.dim["range"], cfrad.dim["time"])

                if QC_mask

                    currmask = Matrix{Bool}(.! map(ismissing, cfrad[mask_name][:,:]))
                    (newX, newY, indexer) = process_single_file(cfrad, tasks;
                                                HAS_INTERACTIVE_QC = HAS_INTERACTIVE_QC,
                                                REMOVE_LOW_SIG_QUALITY = REMOVE_LOW_SIG_QUALITY, SIG_QUALITY_THRESHOLD = SIG_QUALITY_THRESHOLD, SIG_QUALITY_VAR = SIG_QUALITY_VAR,
                                                REMOVE_HIGH_PGG = REMOVE_HIGH_PGG, PGG_THRESHOLD = PGG_THRESHOLD, QC_variable = QC_variable, remove_variable = remove_variable,
                                                replace_missing=replace_missing, feature_mask = currmask, mask_features = true, weight_matrixes=weight_matrixes)

                else
                    (newX, newY, indexer) = process_single_file(cfrad, tasks;
                                                HAS_INTERACTIVE_QC = HAS_INTERACTIVE_QC,
                                                REMOVE_LOW_SIG_QUALITY = REMOVE_LOW_SIG_QUALITY, SIG_QUALITY_THRESHOLD = SIG_QUALITY_THRESHOLD, SIG_QUALITY_VAR = SIG_QUALITY_VAR,
                                                REMOVE_HIGH_PGG = REMOVE_HIGH_PGG, PGG_THRESHOLD=PGG_THRESHOLD, QC_variable = QC_variable, remove_variable = remove_variable,
                                                replace_missing=replace_missing, weight_matrixes=weight_matrixes)
                end


                close(cfrad)

                if verbose
                    println("Processed $(path) in $(time()-pathstarttime) seconds")
                end

            catch e
                if isa(e, DimensionMismatch)
                    printstyled(Base.stderr, "POSSIBLE ERRONEOUS CFRAD DIMENSIONS... SKIPPING $(path)\n"; color=:red)
                    continue
                else
                    printstyled(Base.stderr, "UNRECOVERABLE ERROR\n"; color=:red)
                    printstyled(Base.stderr, "ERROR: $(e)")
                    close(cfrad)
                    throw(e)
                ##@TODO CATCH exception handling for invalid task
                end
            end

            X = vcat(X, newX)::Matrix{Float32}
            Y = vcat(Y, newY)::Matrix{Int64}
            newIdx = reshape(indexer, dims)
            push!(idxs, newIdx)
        end

        println("COMPLETED PROCESSING $(length(paths)) FILES IN $(round((time() - starttime), digits = 2)) SECONDS")

        ###Get verification information
        ###0 indicates NON METEOROLOGICAL data that was removed during interactive QC
        ###1 indicates METEOROLOGICAL data that was retained during interactive QC

        ##Probably only want to write once, I/O is very slow
        if write_out
            println("OUTPUTTING DATA IN HDF5 FORMAT TO FILE: $(output_file)")
            fid = h5open(output_file, "w")

            ###Add information to output h5 file
            attributes(fid)["Parameters"] = tasks
            attributes(fid)["MISSING_FILL_VALUE"] = FILL_VAL
            println()
            println("WRITING DATA TO FILE OF SHAPE $(size(X))")
            println("X TYPE: $(typeof(X))")
            write_dataset(fid, "X", X)
            write_dataset(fid, "Y", Y)
            close(fid)
            if return_idxer
                return X, Y, idxs
            else
                return X, Y
            end
        else
            if return_idxer
                return X, Y, idxs
            else
                return X, Y
            end
        end

    end


    """
        process_single_file_conv(cfrad::NCDataset, config::ModelConfig, kernel_bank::Vector{ConvolutionKernel};
                                  feature_mask::Matrix{Bool}=placeholder_mask, mask_features::Bool=false)

    Convolution-mode equivalent of `process_single_file`. Computes convolution features
    for a single sweep, builds the validity mask and INDEXER, and returns (X, Y, INDEXER).
    """
    function process_single_file_conv(cfrad::NCDataset, config::ModelConfig, kernel_bank::Vector{ConvolutionKernel};
                                       feature_mask::AbstractMatrix{Bool}=placeholder_mask, mask_features::Bool=false)

        cfrad_dims = (cfrad.dim["range"], cfrad.dim["time"])
        ngates = cfrad_dims[1] * cfrad_dims[2]

        # Build INDEXER: valid where remove_var is non-missing
        VT = cfrad[config.remove_var][:]
        INDEXER = [!ismissing(x) for x in VT]

        if mask_features
            INDEXER = [INDEXER[i] ? maskval : false for (i, maskval) in enumerate(feature_mask[:])]
        end

        # PGG thresholding
        PGG = nothing
        if config.REMOVE_HIGH_PGG
            PGG = [ismissing(x) || isnan(x) ? Float32(FILL_VAL) : Float32(x) for x in calc_pgg(cfrad)[:]]
            INDEXER[INDEXER] = [x >= config.PGG_THRESHOLD ? false : true for x in PGG[INDEXER]]
        end

        # Signal quality thresholding
        if config.REMOVE_LOW_SIG_QUALITY
            SIG = [ismissing(x) || isnan(x) ? Float32(FILL_VAL) : Float32(x) for x in calc_sig(cfrad, config.SIG_QUALITY_VAR)[:]]
            INDEXER[INDEXER] = [x <= config.SIG_QUALITY_THRESHOLD ? false : true for x in SIG[INDEXER]]
        end

        # Build 2D valid mask for convolutions
        valid_mask = reshape(copy(INDEXER), cfrad_dims)

        # Determine which variables to convolve (include PGG if not in conv_variables)
        conv_vars = copy(config.conv_variables)

        # Build met_prob mask for masked convolutions (if configured)
        met_prob_mask = nothing
        masked_conv_kb = ConvolutionKernel[]
        if !isempty(config.masked_conv_variables) && config.masked_conv_met_prob_field != ""
            masked_conv_kb = build_filtered_kernel_bank(config.masked_conv_kernel_types,
                                                         config.masked_conv_kernel_sizes)
            if config.masked_conv_met_prob_field in keys(cfrad)
                mp_raw = load_conv_variable(cfrad, config.masked_conv_met_prob_field,
                                             valid_mask, config.SIG_QUALITY_VAR)
                mp_f32 = Matrix{Float32}(undef, cfrad_dims...)
                for j in 1:cfrad_dims[2], i in 1:cfrad_dims[1]
                    v = mp_raw[i, j]
                    mp_f32[i, j] = (ismissing(v) || (v isa AbstractFloat && isnan(v))) ? 0.0f0 : Float32(v)
                end
                met_prob_mask = mp_f32 .>= config.masked_conv_threshold
            else
                @warn "Masked conv field $(config.masked_conv_met_prob_field) not found in CfRadial — skipping masked conv features"
            end
        end

        # Compute convolution features
        X_full, feature_names = compute_convolution_features(cfrad, conv_vars, kernel_bank, valid_mask, config.SIG_QUALITY_VAR;
            masked_conv_variables = config.masked_conv_variables,
            masked_conv_kernel_bank = masked_conv_kb,
            masked_conv_threshold = config.masked_conv_threshold,
            met_prob_mask = met_prob_mask)

        # If selected_features is set (inference), subset columns
        if !isempty(config.selected_features)
            X_full = X_full[:, config.selected_features]
        end

        # Subset rows by INDEXER
        X = X_full[INDEXER, :]

        # Build Y if interactive QC
        if config.HAS_INTERACTIVE_QC
            VG = cfrad[config.QC_var][:][INDEXER]
            VV = cfrad[config.remove_var][:][INDEXER]
            Y = reshape([ismissing(x) ? 0 : 1 for x in VG .- VV][:], (:, 1))
            return (X, Y, INDEXER, feature_names)
        else
            return (X, false, INDEXER, feature_names)
        end
    end


    """
        calculate_features_conv(config::ModelConfig, output_file::String;
                                 QC_mask::Bool=false, mask_name::String="",
                                 write_out::Bool=true, return_idxer::Bool=false)

    Convolution-mode feature calculation. Iterates over all files in `config.input_path`
    and computes convolution features for each sweep.
    """
    function calculate_features_conv(config::ModelConfig, output_file::String;
                                      QC_mask::Bool=false, mask_name::String="",
                                      write_out::Bool=true, return_idxer::Bool=false)

        paths = isdir(config.input_path) ? parse_directory(config.input_path) : [config.input_path]

        kernel_bank = build_kernel_bank(config.conv_kernel_sizes)
        masked_conv_kb = if !isempty(config.masked_conv_variables)
            build_filtered_kernel_bank(config.masked_conv_kernel_types, config.masked_conv_kernel_sizes)
        else
            ConvolutionKernel[]
        end
        n_features = get_convolution_feature_count(config.conv_variables, kernel_bank;
            masked_conv_variables=config.masked_conv_variables,
            masked_conv_kernel_bank=masked_conv_kb)
        if !isempty(config.selected_features)
            n_features = length(config.selected_features)
        end

        X = Matrix{Float32}(undef, 0, n_features)
        Y = Matrix{Int64}(undef, 0, 1)
        idxs = Vector{}(undef, 0)
        all_feature_names = String[]

        starttime = time()

        for path in paths
            cfrad = Dataset(path)
            try
                pathstarttime = time()
                dims = (cfrad.dim["range"], cfrad.dim["time"])

                fm = if QC_mask && mask_name != ""
                    Matrix{Bool}(.!map(ismissing, cfrad[mask_name][:, :]))
                else
                    trues(dims)
                end

                result = process_single_file_conv(cfrad, config, kernel_bank;
                                                   feature_mask=fm, mask_features=QC_mask)
                newX = result[1]
                newY = result[2]
                indexer = result[3]
                feature_names = result[4]

                if isempty(all_feature_names)
                    all_feature_names = feature_names
                end

                close(cfrad)

                if config.verbose
                    println("Processed $(path) in $(round(time() - pathstarttime, digits=2)) seconds [conv mode]")
                end

                X = vcat(X, newX)::Matrix{Float32}
                if newY !== false
                    Y = vcat(Y, newY)::Matrix{Int64}
                end
                push!(idxs, reshape(indexer, dims))

            catch e
                if isa(e, DimensionMismatch)
                    printstyled(Base.stderr, "POSSIBLE ERRONEOUS CFRAD DIMENSIONS... SKIPPING $(path)\n"; color=:red)
                    continue
                else
                    close(cfrad)
                    throw(e)
                end
            end
        end

        println("COMPLETED PROCESSING $(length(paths)) FILES IN $(round((time() - starttime), digits=2)) SECONDS [conv mode]")

        if write_out
            println("OUTPUTTING DATA IN HDF5 FORMAT TO FILE: $(output_file)")
            fid = h5open(output_file, "w")
            if !isempty(config.selected_features)
                attributes(fid)["Parameters"] = all_feature_names[config.selected_features]
            else
                attributes(fid)["Parameters"] = all_feature_names
            end
            attributes(fid)["MISSING_FILL_VALUE"] = FILL_VAL
            println("WRITING DATA TO FILE OF SHAPE $(size(X))")
            write_dataset(fid, "X", X)
            write_dataset(fid, "Y", Y)
            close(fid)
        end

        if return_idxer
            return X, Y, idxs
        else
            return X, Y
        end
    end


    """

    Function to train a random forest model using a precalculated set of input and output features (usually output from
    `calculate_features`). Returns nothing.

    # Required arguments
    ```julia
    input_h5::String
    ```
    Location of input features/targets. Input features are expected to have the name "X", and targets the name "Y". This should be
    taken care of automatically if they are outputs from `calculate_features`

    ```julia
    model_location::String
    ```
    Path to save the trained model out to. Typically should end in `.jld2`

    # Optional keyword arguments
    ```julia
    verify::Bool = false
    ```
    Whether or not to output a separate .h5 file containing the trained models predictions on the training set
    (`Y_PREDICTED`) as well as the targets for the training set (`Y_ACTUAL`)

    ```julia
    verify_out::String="model_verification.h5"
    ```
    If `verify`, the location to output this verification to.

    ```julia
    col_subset=:
    ```
    Set of columns from `input_h5` to train model on. Useful if one wishes to train a model while excluding some features from a training set.

    ```julia
    row_subset=:
    ```
    Set of rows from `input_h5` to train on.

    ```julia
    n_trees::Int = 21
    ```
    Number of trees in the Random Forest ensemble

    ```julia
    max_depth::Int = 14
    ```
    Maximum node depth in each tree in RF ensemble

    ```julia
    class_weights::Vector{Float32} = Vector{Float32}([1.,2.])
    ```
    Vector of class weights to apply to each observation. Should be 1 observation per sample in the input data files
    """
    function train_model(input_h5::String, model_location::String; verify::Bool=false, verify_out::String="model_verification.h5", col_subset=:, row_subset=:,
                        n_trees::Int = 21, max_depth::Int=14, class_weights::Vector{Float32} = Vector{Float32}([1.f0,2.f0]))

        ###Load the data
        radar_data = h5open(input_h5)
        printstyled("\nOpening $(radar_data)...\n", color=:blue)
        ###Split into features

        X = read(radar_data["X"])[row_subset , col_subset]
        Y = read(radar_data["Y"])[:][row_subset]
        close(radar_data)

        train_model(X, Y, model_location; verify=verify, verify_out=verify_out,
                    n_trees=n_trees, max_depth=max_depth, class_weights=class_weights)
    end

    function train_model(X::Matrix, Y::Union{Matrix, Vector}, model_location::String;
                        verify::Bool=false, verify_out::String="model_verification.h5",
                        n_trees::Int = 21, max_depth::Int=14, class_weights::Vector{Float32} = Vector{Float32}([1.f0,2.f0]),
                        max_threads::Int = Threads.nthreads())

        Y_vec = reshape(Y, length(Y))

        model = DecisionTree.RandomForestClassifier(n_trees=n_trees, max_depth=max_depth, rng=50)

        if ! (length(Y_vec) == length(class_weights))
            printstyled("WARNING: class_weights of different length than targets.... Continiuing with no class weights...\n", color=:yellow)
            class_weights = ones(length(Y_vec))
        end

        println("FITTING MODEL ($(min(max_threads, Threads.nthreads())) training threads)")
        startTime = time()
        DecisionTree.fit!(model, X, Y_vec, class_weights; max_threads=max_threads)

        println("COMPLETED FITTING MODEL IN $((time() - startTime)) seconds")
        println()

        println("MODEL VERIFICATION:")
        predicted_Y = DecisionTree.predict(model, X)
        accuracy = sum(predicted_Y .== Y_vec) / length(Y_vec)
        println("ACCURACY ON TRAINING SET: $(round(accuracy * 100, sigdigits=3))%")
        println()

        printstyled("SAVING MODEL TO: $(model_location) \n", color=:green)
        save_object(model_location, model)

        if (verify)
            ###NEW: Write out data to HDF5 files for further processing
            println("WRITING VERIFICATION DATA TO $(verify_out)" )
            fid = h5open(verify_out, "w")
            HDF5.write_dataset(fid, "Y_PREDICTED", predicted_Y)
            HDF5.write_dataset(fid, "Y_ACTUAL", Y_vec)
            close(fid)
        end
    end



    ###TODO: Fix arguments etc
    ###Can have one for a single file and one for a directory
    """
    Primary function to apply a trained RF model to certain raw fields of a cfradial scan. Values determined to be
    non-meteorological by the RF model will be replaced with `Missing`

    # Required Arguments
    ```julia
    file_path::String
    ```
    Location of input cfradial or directory of cfradials one wishes to apply QC to

    ```julia
    config_file_path::String
    ```
    Location of config file containing features to calculate as inputs to RF model

    ```julia
    model_path::String
    ```
    Location of trained RF model (in jld2 file format)

    # Optional Arguments
    ```julia
    VARIABLES_TO_QC::Vector{String} = ["ZZ", "VV"]
    ```
    List containing names of raw variables in the CFRadial to apply QC algorithm to.

    ```julia
    QC_suffix::String = "_QC"
    ```
    Used for naming the QC-ed variables in the modified CFRadial file. Field name will be QC_suffix appended to the raw field.
    Example: `DBZ_QC`

    ```julia
    indexer_var::String = "VV"
    ```
    Variable used to determine what gates are considered "missing" in the raw moments. QC will not
    be applied to these gates, they will simply remain missing.

    ```julia
    decision_threshold::Float32 = .5
    ```
    Used to leverage probablistic nature of random forest methodology. When the model has a greater than `decision_threshold`
    level confidence that a gate is meteorological data, it will be assigned as such. Anything at or below this confidence threshold
    will be assigned non-meteorological. At least in the ELDORA case, aggressive thresholds (.8 and above) have been found to maintain
    >92% of the meteorological data while removing >99% of non-meteorological gates.

    ```julia
    output_mask::Bool = true
    ```
    Whether or not to output the QC preditions from the model output. A value of 0 means the model predicted the gate to
    be non-meteorological, 1 corresponds to predicted meteorological data, and -1 denotes data that did not meet minimum
    thresholds

    ```julia
    mask_name::String = "QC_MASK"
    ```
    What to name the output QC predictions.

    ```julia
    verbose::Bool = false
    ````
    Whether to output timing and scan information

    ```julia
    REMOVE_HIGH_PGG::Bool = true
    ```
    Whether or not to remove gates with a specified value of Probability of Ground Gate (PGG) from consideration
    ```julia
    PGG_THRESHOLD::Float32 = 1.f0
    ```
    Threshold at or above to remove data from consideration

    ```julia
    REMOVE_LOW_SIG_QUALITY::Bool = true
    ```
    Whether or not to remove gates with a specified value of signal quality from consideration

    ```julia
    SIG_QUALITY_THRESHOLD::Float32 = .2f0
    ```
    Signal quality threshold to remove data at

    ```julia
    SIG_QUALITY_VAR::String = "NCP"
    ```
    Name of variable representing signal quality

    ```julia
    output_probs::Bool = false
    ```
    Whether or not to output probabilities of meteorological gate from random forest
    ```julia
    prob_varname::String = ""
    ```
    What to name the probability variable in the cfradial file
    """
    ### Currently deprecated
    # function QC_scan(file_path::String, config_file_path::String, model_path::String; VARIABLES_TO_QC::Vector{String}= ["ZZ", "VV"],
    #                  QC_suffix::String = "_QC", indexer_var::String="VV", decision_threshold::Tuple{Float32, Float32} = (.5f0, 1.f0), output_mask::Bool = true,
    #                  mask_name::String = "QC_MASK_2", verbose::Bool=false, REMOVE_HIGH_PGG::Bool = true, PGG_THRESHOLD::Float32 = 1.f0,
    #                  REMOVE_LOW_SIG_QUALITY::Bool = true, SIG_QUALITY_THRESHOLD::Float32=.2f0, SIG_QUALITY_VAR::String="NCP"
    #                  output_probs::Bool = false, prob_varname::String = "")

    #     new_model = load_object(model_path)

    #     paths = Vector{String}()
    #     if isdir(file_path)
    #         paths = parse_directory(file_path)
    #     else
    #         paths = [file_path]
    #     end


    #     for path in paths
    #         ##Open in append mode so output variables can be written
    #         input_cfrad = redirect_stdout(devnull) do
    #             NCDataset(path, "a")
    #         end

    #         cfrad_dims = (input_cfrad.dim["range"], input_cfrad.dim["time"])

    #         ###Will generally NOT return Y, but only (X, indexer)
    #         ###Todo: What do I need to do for parsed args here
    #         starttime=time()
    #         X, Y, indexer = process_single_file(input_cfrad, config_file_path; REMOVE_HIGH_PGG = REMOVE_HIGH_PGG,
    #                                     REMOVE_LOW_NCP = REMOVE_LOW_NCP, remove_variable=indexer_var)
    #         ##Load saved RF model
    #         ##assume that default SYMBOL for saved model is savedmodel
    #         ##For binary classifications, 1 will be at index 2 in the predictions matrix
    #         met_predictions = DecisionTree.predict_proba(new_model, X)[:, 2]
    #         predictions = (met_predictions .> decision_threshold[1]) .& (met_predictions .<= decision_threshold[2])
    #         printstyled("RETAINING GATES BETWEEN $(decision_threshold[1]) and $(decision_threshold[2]) PROBABILITY \n ", color=:yellow)

    #         ##QC each variable in VARIALBES_TO_QC
    #         for var in VARIABLES_TO_QC

    #             ##Create new field to reshape QCed field to
    #             NEW_FIELD = missings(Float32, cfrad_dims)

    #             ##Only modify relevant data based on indexer, everything else should be fill value
    #             QCED_FIELDS = input_cfrad[var][:][indexer]

    #             NEW_FIELD_ATTRS = Dict(
    #                 "units" => input_cfrad[var].attrib["units"],
    #                 "long_name" => "Random Forest Model QC'ed $(var) field",
    #                 "probabilities" => " $(decision_threshold[1]) < p <= $(decision_threshold[2])"
    #             )

    #             ##Set MISSINGS to fill value in current field

    #             initial_count = count(.!map(ismissing, QCED_FIELDS))
    #             ##Apply predictions from model
    #             ##If model predicts 1, this indicates a prediction of meteorological data
    #             QCED_FIELDS = map(x -> Bool(predictions[x[1]]) ? x[2] : missing, enumerate(QCED_FIELDS))
    #             final_count = count(.!map(ismissing, QCED_FIELDS))

    #             ###Need to reconstruct original
    #             NEW_FIELD = NEW_FIELD[:]
    #             NEW_FIELD[indexer] = QCED_FIELDS
    #             NEW_FIELD = reshape(NEW_FIELD, cfrad_dims)


    #             try
    #                 defVar(input_cfrad, var * QC_suffix, NEW_FIELD, ("range", "time"), fillvalue = FILL_VAL; attrib=NEW_FIELD_ATTRS)
    #             catch e
    #                 ###Simply overwrite the variable
    #                 if e.msg == "NetCDF: String match to name in use"
    #                     if verbose
    #                         println("Already exists... overwriting")
    #                     end
    #                     input_cfrad[var*QC_suffix][:,:] = NEW_FIELD
    #                 else
    #                     throw(e)
    #                 end
    #             end

    #             if verbose
    #                 println("\r\nPROCESSING: $(path)")
    #                 println("\r\nCompleted in $(time()-starttime ) seconds")
    #                 println()
    #                 printstyled("REMOVED $(initial_count - final_count) PRESUMED NON-METEORLOGICAL DATAPOINTS\n", color=:green)
    #                 println("FINAL COUNT OF DATAPOINTS IN $(var): $(final_count)")
    #             end

    #         end

    #         if output_mask

    #             MASK = fill(-1, cfrad_dims)[:]
    #             MASK[indexer] = predictions
    #             MASK = reshape(MASK, cfrad_dims)

    #             try
    #                 if verbose
    #                     println("Writing Mask")
    #                 end

    #                 NEW_FIELD_ATTRS = Dict(
    #                 "units" => "Unitless",
    #                 "long_name" => "Ronin Quality Control mask"
    #                 )
    #                 defVar(input_cfrad, mask_name, MASK, ("range", "time"), fillvalue=-1; attrib=NEW_FIELD_ATTRS)
    #             catch e

    #             ###Simply overwrite the variable
    #                 if e.msg == "NetCDF: String match to name in use"
    #                     if verbose
    #                         println("Already exists... overwriting")
    #                     end
    #                     input_cfrad[mask_name][:,:] =  MASK
    #                 else
    #                     throw(e)
    #                 end
    #             end
    #         end

    #         if output_probs

    #             NEW = fill(-1, cfrad_dims)[:]
    #             NEW[indexer] = met_predictions
    #             NEW = reshape(NEW, cfrad_dims)

    #             try
    #                 if verbose
    #                     println("Writing Probabilites to $(prob_varname)")
    #                 end

    #                 NEW_FIELD_ATTRS = Dict(
    #                 "units" => "Unitless",
    #                 "long_name" => "Ronin Decision Tree Probabilities"
    #                 )
    #                 defVar(input_cfrad, prob_varname, MASK, ("range", "time"), fillvalue=-1; attrib=NEW_FIELD_ATTRS)
    #             catch e

    #             ###Simply overwrite the variable
    #                 if e.msg == "NetCDF: String match to name in use"
    #                     if verbose
    #                         println("Already exists... overwriting")
    #                     end
    #                     input_cfrad[prob_varname][:,:] =  MASK
    #                 else
    #                     throw(e)
    #                 end
    #             end

    #         close(input_cfrad)

    #         end

    #     end
    # end



    function split_training_testing_validation!(DIR_PATHS::Vector{String}, TRAINING_PATH::String, TESTING_PATH::String, VALIDATION_PATH::String)

        ###TODO  - make sure to ignore .tmp_hawkedit files OTHERWISE WON'T WORK AS EXPECTED
        TRAINING_FRAC::Float32 = .72f0
        VALIDATION_FRAC::Float32 = .08f0
        TESTING_FRAC:: Float32 = 1.f0 - TRAINING_FRAC - VALIDATION_FRAC

        ###Assume that each directory represents a different case
        NUM_CASES::Int64 = length(DIR_PATHS)

        ###Do a little input sanitaiton
        if TRAINING_PATH[end] != '/'
            TRAINING_PATH = TRAINING_PATH * '/'
        end

        if TESTING_PATH[end] != '/'
            TESTING_PATH = TESTING_PATH * '/'
        end

        for (i, path) in enumerate(DIR_PATHS)
            if path[end] != '/'
                DIR_PATHS[i] = path * '/'
            end
        end

        ###Clean directories and remake them
        rm(TESTING_PATH, force = true, recursive = true)
        rm(TRAINING_PATH, force = true, recursive = true)
        rm(VALIDATION_PATH, force = true, recursive = true)

        mkdir(TESTING_PATH)
        mkdir(TRAINING_PATH)
        mkdir(VALIDATION_PATH)


        TOTAL_SCANS::Int64 = 0

        ###Calculate total number of TDR scans
        for path in DIR_PATHS
            TOTAL_SCANS += length(filter(f -> startswith(f, RADAR_FILE_PREFIX), readdir(path)))
        end

        ###By convention, we will round the number of training scans down
        ###and the number of testing scans up
        TRAINING_SCANS::Int64 = Int(floor(TOTAL_SCANS * (TRAINING_FRAC + VALIDATION_FRAC)))
        VALIDATION_SCANS::Int64 = Int(floor(TRAINING_SCANS * VALIDATION_FRAC))
        TESTING_SCANS::Int64  = Int(ceil(TOTAL_SCANS * (TESTING_FRAC)))

        ###Further by convention, will add the remainder on to the last case
        ###A couple of notes here: Each case must have a minimum of NUM_TESTING_SCANS_PER_CASE
        ###in order to ensure each case is represented preportionally
        ###This will be the number of scans removed, and the rest from the case will be placed into training
        NUM_TRAINING_SCANS_PER_CASE::Int64 = TRAINING_SCANS ÷ NUM_CASES
        TRAINING_REMAINDER::Int64          = TRAINING_SCANS % NUM_CASES

        NUM_TESTING_SCANS_PER_CASE::Int64 = TESTING_SCANS ÷ NUM_CASES
        TESTING_REMAINDER::Int64          = TESTING_SCANS % NUM_CASES

        NUM_VALIDATION_SCANS_PER_CASE::Int64 = VALIDATION_SCANS ÷ NUM_CASES
        VALIDATION_REMAINDER::Int64       = VALIDATION_SCANS % NUM_CASES


        printstyled("\nTOTAL NUMBER OF TDR SCANS ACROSS ALL CASES: $TOTAL_SCANS\n", color=:green)
        printstyled("TESTING SCANS PER CASE $(NUM_TESTING_SCANS_PER_CASE)\n", color=:orange)
        printstyled("VALIDATION SCANS PER CASE $(NUM_VALIDATION_SCANS_PER_CASE)\n", color=:red)

        ###Each sequence of chronological TDR scans will be split as follows
        ###[[T E S T][T   R   A   I   N][T E S T][T   R   A   I   N][T E S T]]
        for path in DIR_PATHS

            contents = filter(f -> startswith(f, RADAR_FILE_PREFIX), readdir(path))
            num_cfrads = length(contents)

            printstyled("NUMBER OF SCANS IN CASE: $(num_cfrads)\n", color=:red)
            ###Take 1/3rd of NUM_TESTING_SCANS_PER_CASE from beginning, 1/3rd from middle, and 1/3rd from end
            ###Need to assume files are ordered chronologically in contents here
            num_scans_for_training = num_cfrads - NUM_TESTING_SCANS_PER_CASE

            ###Need to handle a training group size that is odd
            training_group_size = num_scans_for_training ÷ 2
            training_group_remainder = num_scans_for_training % 2
            printstyled("TRAINING GROUP SIZE: $(training_group_size) + REMAINDER: $(training_group_remainder)\n", color=:red)

            ###If the testing_group_size is not divisible by 3, simply take the remainder from the front end (again, by definiton)
            testing_group_size = NUM_TESTING_SCANS_PER_CASE ÷ 3
            testing_remainder = NUM_TESTING_SCANS_PER_CASE % 3
            printstyled("TESTING GROUP SIZE: $(testing_group_size) + REMAINDER $(testing_remainder)\n", color=:red)

            ###We will construct an indexer to determine which files are testing files and which
            ###files are training files
            testing_indexer = fill(false, num_cfrads)

            ###curr_idx holds the index of the LAST assignment made
            curr_idx = 0

            ###handle first group of testing cases
            testing_indexer[1:testing_group_size + testing_remainder] .= true
            curr_idx = testing_group_size + testing_remainder
            printstyled("\n INDEXES 1 TO $(curr_idx) ASSIGNED TESTING", color=:green)

            ###Add one group of training files
            ###Handle possible remainder here too
            printstyled("\n INDEXES $(curr_idx) ", color=:green)
            curr_idx = curr_idx + training_group_size + training_group_remainder
            printstyled(" TO $(curr_idx) ASSIGNED TRAINING", color=:green)

            ###Next group of testing files
            printstyled("\n INDEXES $(curr_idx + 1)", color=:green)
            testing_indexer[curr_idx + 1: curr_idx + testing_group_size] .= true
            curr_idx = curr_idx + testing_group_size
            printstyled(" TO $(curr_idx) ASSIGNED TESTING", color=:green)

            ###Final group of training files
            printstyled("\n INDEXES $(curr_idx + 1)", color=:green)
            curr_idx = curr_idx + training_group_size
            printstyled(" TO $(curr_idx) ASSIGNED TRAINING", color=:green)

            ###Final group of testing files
            printstyled("\n INDEXES $(curr_idx + 1)", color=:green)
            testing_indexer[curr_idx + 1: curr_idx + testing_group_size] .= true
            curr_idx = curr_idx + testing_group_size
            printstyled(" TO $(curr_idx) ASSIGNED TESTING", color=:green)

            ###Everyting not in testing will be in training
            testing_files = contents[testing_indexer]
            training_files = contents[.!testing_indexer]

            printstyled("\nTotal length of case files: $(num_cfrads)\n", color=:red)
            printstyled("Length of testing files: $(length(testing_files)) - $( (length(testing_files) / (num_cfrads)) ) percent\n" , color=:blue)
            printstyled("Length of training files: $(length(training_files)) - $( (length(training_files) / (num_cfrads)) ) percent\n", color=:blue)

            @assert (length(testing_files) + length(training_files) == num_cfrads)


            ###Grab NUM_VALIDATION_SCANS_PER_CASE random indexes from the training files
            validation_idxes = randperm(length(training_files))[1:NUM_VALIDATION_SCANS_PER_CASE]
            validation_files = training_files[validation_idxes]
            training_files   = deleteat!(training_files, sort(validation_idxes))
            #printstyled("\n SßUM OF TESTING AND TRAINING = $(length(testing_files) + length(training_files))\n",color=:green)
            for file in training_files
                symlink(joinpath(path, file), joinpath(TRAINING_PATH, file))
            end

            for file in testing_files
                symlink(joinpath(path, file), joinpath(TESTING_PATH, file))
            end

            for file in validation_files
                symlink(joinpath(path, file), joinpath(VALIDATION_PATH, file))
            end
        end

    end




    """
    Function to split a given directory or set of directories into training and testing files using the configuration
    described in DesRosiers and Bell 2023. **This function assumes that input directories only contain cfradial files
    that follow standard naming conventions, and are thus implicitly chronologically ordered.** The function operates
    by first dividing file names into training and testing sets following an 80/20 training/testing split, and subsequently
    softlinking each file to the training and testing directories. Attempts to avoid temporal autocorrelation while maximizing
    variance by dividing each case into several different training/testing sections.

    An important note: Always use absolute paths, relative paths will cause issues with the simlinks

    # Required Arguments:

    ```julia
    DIR_PATHS::Vector{String}
    ```
    List of directories containing cfradials to be used for model training/testing. Useful if input data is split
    into several different cases.

    ```julia
    TRAINING_PATH::String
    ```
    Directory to softlink files designated for training into.

    ```julia
    TESTING_PATH::String
    ```
    Directory to softlink files designated for testing into.
    """
    function split_training_testing!(DIR_PATHS::Vector{String}, TRAINING_PATH::String, TESTING_PATH::String)

        ###TODO  - make sure to ignore .tmp_hawkedit files OTHERWISE WON'T WORK AS EXPECTED
        TRAINING_FRAC::Float32 = .72f0
        VALIDATION_FRAC::Float32 = .08f0
        TESTING_FRAC:: Float32 = 1.f0 - TRAINING_FRAC - VALIDATION_FRAC

        ###Assume that each directory represents a different case
        NUM_CASES::Int64 = length(DIR_PATHS)

        ###Do a little input sanitaiton
        if TRAINING_PATH[end] != '/'
            TRAINING_PATH = TRAINING_PATH * '/'
        end

        if TESTING_PATH[end] != '/'
            TESTING_PATH = TESTING_PATH * '/'
        end

        for (i, path) in enumerate(DIR_PATHS)
            if path[end] != '/'
                DIR_PATHS[i] = path * '/'
            end
        end

        ###Clean directories and remake them
        rm(TESTING_PATH, force = true, recursive = true)
        rm(TRAINING_PATH, force = true, recursive = true)

        mkdir(TESTING_PATH)
        mkdir(TRAINING_PATH)


        TOTAL_SCANS::Int64 = 0

        ###Calculate total number of TDR scans
        for path in DIR_PATHS
            TOTAL_SCANS += length(filter(f -> startswith(f, RADAR_FILE_PREFIX), readdir(path)))
        end

        ###By convention, we will round the number of training scans down
        ###and the number of testing scans up
        TRAINING_SCANS::Int64 = Int(floor(TOTAL_SCANS * (TRAINING_FRAC + VALIDATION_FRAC)))
        TESTING_SCANS::Int64  = Int(ceil(TOTAL_SCANS * (TESTING_FRAC)))

        ###Further by convention, will add the remainder on to the last case
        ###A couple of notes here: Each case must have a minimum of NUM_TESTING_SCANS_PER_CASE
        ###in order to ensure each case is represented preportionally
        ###This will be the number of scans removed, and the rest from the case will be placed into training
        NUM_TRAINING_SCANS_PER_CASE::Int64 = TRAINING_SCANS ÷ NUM_CASES
        TRAINING_REMAINDER::Int64          = TRAINING_SCANS % NUM_CASES

        NUM_TESTING_SCANS_PER_CASE::Int64 = TESTING_SCANS ÷ NUM_CASES
        TESTING_REMAINDER::Int64          = TESTING_SCANS % NUM_CASES


        printstyled("\nTOTAL NUMBER OF TDR SCANS ACROSS ALL CASES: $TOTAL_SCANS\n", color=:green)
        printstyled("TESTING SCANS PER CASE $(NUM_TESTING_SCANS_PER_CASE)\n", color=:orange)

        ###Each sequence of chronological TDR scans will be split as follows
        ###[[T E S T][T   R   A   I   N][T E S T][T   R   A   I   N][T E S T]]
        for path in DIR_PATHS

            contents = filter(f -> startswith(f, RADAR_FILE_PREFIX), readdir(path))
            num_cfrads = length(contents)

            printstyled("NUMBER OF SCANS IN CASE: $(num_cfrads)\n", color=:red)
            ###Take 1/3rd of NUM_TESTING_SCANS_PER_CASE from beginning, 1/3rd from middle, and 1/3rd from end
            ###Need to assume files are ordered chronologically in contents here
            num_scans_for_training = num_cfrads - NUM_TESTING_SCANS_PER_CASE

            ###Need to handle a training group size that is odd
            training_group_size = num_scans_for_training ÷ 2
            training_group_remainder = num_scans_for_training % 2
            printstyled("TRAINING GROUP SIZE: $(training_group_size) + REMAINDER: $(training_group_remainder)\n", color=:red)

            ###If the testing_group_size is not divisible by 3, simply take the remainder from the front end (again, by definiton)
            testing_group_size = NUM_TESTING_SCANS_PER_CASE ÷ 3
            testing_remainder = NUM_TESTING_SCANS_PER_CASE % 3
            printstyled("TESTING GROUP SIZE: $(testing_group_size) + REMAINDER $(testing_remainder)\n", color=:red)

            ###We will construct an indexer to determine which files are testing files and which
            ###files are training files
            testing_indexer = fill(false, num_cfrads)

            ###curr_idx holds the index of the LAST assignment made
            curr_idx = 0

            ###handle first group of testing cases
            testing_indexer[1:testing_group_size + testing_remainder] .= true
            curr_idx = testing_group_size + testing_remainder
            printstyled("\n INDEXES 1 TO $(curr_idx) ASSIGNED TESTING", color=:green)

            ###Add one group of training files
            ###Handle possible remainder here too
            printstyled("\n INDEXES $(curr_idx) ", color=:green)
            curr_idx = curr_idx + training_group_size + training_group_remainder
            printstyled(" TO $(curr_idx) ASSIGNED TRAINING", color=:green)

            ###Next group of testing files
            printstyled("\n INDEXES $(curr_idx + 1)", color=:green)
            testing_indexer[curr_idx + 1: curr_idx + testing_group_size] .= true
            curr_idx = curr_idx + testing_group_size
            printstyled(" TO $(curr_idx) ASSIGNED TESTING", color=:green)

            ###Final group of training files
            printstyled("\n INDEXES $(curr_idx + 1)", color=:green)
            curr_idx = curr_idx + training_group_size
            printstyled(" TO $(curr_idx) ASSIGNED TRAINING", color=:green)

            ###Final group of testing files
            printstyled("\n INDEXES $(curr_idx + 1)", color=:green)
            testing_indexer[curr_idx + 1: curr_idx + testing_group_size] .= true
            curr_idx = curr_idx + testing_group_size
            printstyled(" TO $(curr_idx) ASSIGNED TESTING", color=:green)

            ###Everyting not in testing will be in training
            testing_files = contents[testing_indexer]
            training_files = contents[.!testing_indexer]

            printstyled("\nTotal length of case files: $(num_cfrads)\n", color=:red)
            printstyled("Length of testing files: $(length(testing_files)) - $( (length(testing_files) / (num_cfrads)) ) percent\n" , color=:blue)
            printstyled("Length of training files: $(length(training_files)) - $( (length(training_files) / (num_cfrads)) ) percent\n", color=:blue)

            @assert (length(testing_files) + length(training_files) == num_cfrads)

            #printstyled("\n SßUM OF TESTING AND TRAINING = $(length(testing_files) + length(training_files))\n",color=:green)
            for file in training_files
                symlink((path * file), TRAINING_PATH * file)
            end

            for file in testing_files
                symlink((path * file), TESTING_PATH * file)
            end
        end

    end




    function standardize(column)
        col_max = maximum(column)
        col_min = minimum(column)
        return (map(x-> (x - col_min) / (col_max - col_min), column))
    end


    """

    # Uses L1 regression with a variety of λ penalty values to determine the most useful features for
     input to the random forest model.

    ---

    # Required Input

    ---

    ```julia
    input_file_path::String
    ```

    Path to .h5 file containing model training features under `["X"]` parameter, and model targets under `["Y"]` parameter.
     Also expects the h5 file to contain an attribute known as `Parameters` containing abbreviations for the feature types

    ```julia
    λs::Vector{Float32}
    ```

    Vector of values used to vary the strength of the penalty term in the regularization.
    ---

    # Optional Keyword Arguments

    ---

    ```julia
    pred_threshold::Float32
    ```

    Minimum cofidence level for binary classifier when predicting
    ---
    Returns
    ---
    Returns a DataFrame with each row containing info about a regression for a specific λ, the values of the regression coefficients
        for each input feature, and the Root Mean Square Error of the resultant regression.
    """
    function get_feature_importance(input_file_path::String, λs::Vector{Float64}; pred_threshold::Float64 = .5)


        LogisticClassifier = MLJ.@load LogisticClassifier pkg=MLJLinearModels

        training_data = h5open(input_file_path)

        ###Standardize features to expedite regression convergence
        features = mapslices(standardize, training_data["X"][:,:], dims=1)
        ###Flatten targets and convert to categorical datatime
        targets = categorical(training_data["Y"][:,:][:])
        targets_raw = training_data["Y"][:, :][:]
        params = attrs(training_data)["Parameters"]

        close(training_data)

        coef_values = Dict(param => [] for param in params)
        coef_values["λ"] = λs
        rmses = []
        precisions = []
        recalls    = []

        for λ in λs

            mdl = LogisticClassifier(;lambda=λ, penalty=:l1)
            mach = machine(mdl, MLJ.table(features), targets[:])
            fit!(mach)
            coefs = fitted_params(mach).coefs

            y_pred = predict(mach, features)
            results = pdf(y_pred, [0, 1])
            met_predictions = map(x -> x > pred_threshold ? 0 : 1, results[:, 1])

            n_tru_positives = sum(met_predictions[targets_raw .== 1])
            n_fal_positives = sum(met_predictions[targets_raw .== 0])
            n_fal_negatives = sum(met_predictions[targets_raw .== 1] .== 0)

            push!(rmses, MLJ.rmse(met_predictions, targets_raw))
            push!(precisions, (n_tru_positives) / (n_tru_positives + n_fal_positives))
            push!(recalls, (n_tru_positives) / (n_tru_positives + n_fal_negatives))

            for (i, param) in enumerate(params)
                push!(coef_values[param], coefs[i][2])
            end


        end

        coef_values["rmse"] = rmses
        coef_values["precision"] = precisions
        coef_values["recall"]    = recalls
        return(DataFrame(coef_values))

    end




    """
        error_characteristics(file_path::String, config_file_path::String, model_path::String;
        indexer_var::String="VV", QC_variable::String="VG", decision_threshold::Float32 = .5, write_out::Bool=false,
        output_name::String="Model_Error_Characteristics.h5")

    Function to process a set of cfradial files that have already been interactively QC'ed and return information about where errors
    occur in the files relative to model predictions. Requires a pre-trained model and configuration, as well as scans that
    have already been interactively quality controlled.

    #Required Arguments
    ```julia
    file_path::String
    ```
    Path to file or directory of cfradials to be processed

    ```julia
    config_file_path::String
    ```
    Path to configuration file containing parameters to calculate for the cfradials

    ```julia
    model_path::String
    ```
    Path to pre-trained random forest model

    # Optional keyword arguments

    ```julia
    indexer_var::String="VV"
    ```
    Name of a raw variable in input NetCDF files. Used to determine where missing data exists in the input sweeps.
    Data at these locations will be removed from the outputted features.

    ```julia
    QC_variable::String="VG"
    ```
    Name of variable in CFRadial files that has already been interactively QC'ed. Used as the verification data.

    ```julia
    decision_threshold::Float32 = .5
    ```
    Fraction of decision trees in the RF model that must agree for a given gate to be classified as meteorological.
    For example, at .5, >=50% of the trees must predict that a gate is meteorological for it to be classified as such,
    otherwise it is assigned as non-meteorological.

    ```julia
    write_out::Bool=false
    ```
    Whether or not to output the model evaluation data to an HDF5 file

    ```julia
    output_name::String="Model_Error_Characteristics.h5"
    ```
    Name/Path of desired HDF5 output location

    # Returns
    Returns a tuple of (X, Y, indexer, predictions, false_positives, false_negatives)
    Where

    ```julia
    X::Matrix{Float32}
    ```
    Each row in X represents a different radar gate, while each column a different parameter as according to the order
    that they are listed in the config_file_path

    ```julia
    Y::Matrix{Int64}
    ```
    Each row in Y represents a radar gate, and its classification according to the interactive QC applied to it.

    ```julia
    indexer::Matrix{Int64}
    ```
    For all gates in the input directory, contains 1 if the gate passed basic QC thresholds (Low NCP, etc.) and 0 if it did not.
    Useful if one wishes to reconstruct 2D scan from flattened data

    ```julia
    predictions:Matrix{Int32}
    ```
    Trained machine learning model predictions for the classification of a gate - `1` if predicted to be
        meteorological data, `0` otherwise.

    ```julia
    false_postivies::BitMatrix
    ```
    Which gates were misclassified as meteorological data relative to interactive QC

    ```julia
    false_negatives::BitMatrix
    ```
    Which gates were misclassified as non-meteorological data relative to interactive QC
    """
    ## Deprecated?
    # function error_characteristics(file_path::String, config_file_path::String, model_path::String;
    #     indexer_var::String="VV", QC_variable::String="VG", decision_threshold::Float32 = .5f0, write_out::Bool=false,
    #     output_name::String="Model_Error_Characteristics.h5")


    #     ###We can probably refactor this honestly, just do predict with model
    #     ###Do we need to reconstruct the original scans? Probably not.....

    #     new_model = load_object(model_path)


    #     paths = Vector{String}()

    #     if isdir(file_path)
    #         paths = parse_directory(file_path)
    #     else
    #         paths = [file_path]
    #     end

    #     tasks = get_task_params(config_file_path)

    #     X = Matrix{Float32}(undef,0,length(tasks))
    #     Y = Matrix{Int32}(undef,0,1)
    #     indexer = Matrix{Int32}(undef,0,1)
    #     predictions = Matrix{Int32}(undef, 0, 1)

    #     for path in paths

    #         input_cfrad = redirect_stdout(devnull) do
    #            NCDataset(path, "a")
    #         end

    #         cfrad_dims = (input_cfrad.dim["range"], input_cfrad.dim["time"])
    #         ###Todo: What do I need to do for parsed args here
    #         println("\r\nPROCESSING: $(path)")
    #         starttime=time()
    #         try

    #             Xn, Yn, indexern = process_single_file(input_cfrad, config_file_path; REMOVE_HIGH_PGG = true, QC_variable = QC_variable,
    #                                                         REMOVE_LOW_NCP = true, remove_variable=indexer_var, HAS_INTERACTIVE_QC = true)
    #             println("\r\nCompleted in $(time()-starttime ) seconds")

    #                 ##Load saved RF model
    #             ##assume that default SYMBOL for saved model is savedmodel
    #             ##For binary classifications, 1 will be at index 2 in the predictions matrix
    #             met_predictions = DecisionTree.predict_proba(new_model, Xn)[:, 2]
    #             predictionsn = met_predictions .> decision_threshold

    #             ###If we wish to return features for error diagnostics, we simply return X which is the features array,
    #             ###Y which are the correct values, the indexer which shows where data was taken out and where it was not,
    #             ###and the model predictions

    #             X  = vcat(X, Xn)
    #             Y  = vcat(Y, Yn)
    #             indexer = vcat(indexer, indexern)
    #             predictions = vcat(predictions, predictionsn)

    #         catch e
    #             printstyled("POSSIBLE ERROR WITH FILE AT: $(path)...\nCONTINUING\n", color=:red)
    #         end

    #     end

    #     false_positives_idx = (predictions .== 1) .& (Y .== 0)
    #     false_negatives_idx = (predictions .== 0) .& (Y .== 1)


    #     if write_out
    #         println("Writing Data to $(output_name)")

    #         h5open(output_name, "w") do f
    #             f["X"] = X[:,:]
    #             f["Y"] = Y[:]
    #             f["indexer"] = Vector{Int32}(indexer[:])
    #             f["predictions"] = Vector{Int32}(predictions[:])
    #             f["false_positive_index"] = Vector{Int32}(false_positives_idx[:])
    #             f["false_negatives_idx"] = Vector{Int32}(false_negatives_idx[:])
    #             attributes(f)["FEATURE_NAMES"] = tasks
    #         end

    #         printstyled("Successfully Output Model Evaluation Data to $(output_name)\n", color=:green)
    #     end

    #     return (X, Y, indexer, predictions, false_positives_idx, false_negatives_idx)
    # end



    """
        train_multi_model(config::ModelConfig)

    All-in-one function to take in a set of radar data, calculate input features, and train a chain of random forest models
    for meteorological/non-meteorological gate identification.

    #Required arguments
    ```julia
    config::ModelConfig
    ```
    Struct containing configuration info for model training

    #Returns
        -None
    """
    function train_multi_model(config::ModelConfig; pass_config::Dict=Dict())
        ##Quick input sanitation check
        if config.task_mode != "convolution"
            @assert (length(config.model_output_paths) == length(config.feature_output_paths)
                     == length(config.met_probs) == length(config.task_paths) == length(config.task_weights) == length(config.mask_names))
        else
            @assert (length(config.model_output_paths) == length(config.feature_output_paths)
                     == length(config.met_probs) == length(config.mask_names))
        end

        if !(config.HAS_INTERACTIVE_QC)
            throw("ERROR: Input cfradials must have interactive QC present to train a model. Set config HAS_INTERACTIVE_QC flag to true")
        end

        # Save original config values so each pass can override independently
        orig_conv_variables = copy(config.conv_variables)
        orig_selected_features = copy(config.selected_features)
        orig_masked_conv_variables = copy(config.masked_conv_variables)
        orig_masked_conv_kernel_types = copy(config.masked_conv_kernel_types)
        orig_masked_conv_kernel_sizes = copy(config.masked_conv_kernel_sizes)
        orig_masked_conv_threshold = config.masked_conv_threshold
        orig_masked_conv_met_prob_field = config.masked_conv_met_prob_field

        full_start_time = time()
        ###Iteratively train models and apply QC_scan with the specified probabilites to train a multi-pass model
        ###pipeline
        for (i, model_path) in enumerate(config.model_output_paths)

            # Apply per-pass config overrides (conv_variables, selected_features, masked conv)
            if haskey(pass_config, i)
                pc = pass_config[i]
                config.conv_variables = hasproperty(pc, :conv_variables) ? pc.conv_variables : orig_conv_variables
                config.selected_features = hasproperty(pc, :selected_features) ? pc.selected_features : orig_selected_features
                if hasproperty(pc, :masked_conv_variables)
                    config.masked_conv_variables = pc.masked_conv_variables
                    config.masked_conv_kernel_types = pc.masked_conv_kernel_types
                    config.masked_conv_kernel_sizes = pc.masked_conv_kernel_sizes
                    config.masked_conv_threshold = pc.masked_conv_threshold
                    config.masked_conv_met_prob_field = "met_prob_pass_$(i - 1)"
                else
                    config.masked_conv_variables = String[]
                    config.masked_conv_kernel_types = String[]
                    config.masked_conv_kernel_sizes = Int[]
                    config.masked_conv_threshold = orig_masked_conv_threshold
                    config.masked_conv_met_prob_field = ""
                end
            else
                config.conv_variables = orig_conv_variables
                config.selected_features = orig_selected_features
                config.masked_conv_variables = orig_masked_conv_variables
                config.masked_conv_kernel_types = orig_masked_conv_kernel_types
                config.masked_conv_kernel_sizes = orig_masked_conv_kernel_sizes
                config.masked_conv_threshold = orig_masked_conv_threshold
                config.masked_conv_met_prob_field = orig_masked_conv_met_prob_field
            end
            if config.task_mode == "convolution"
                masked_info = isempty(config.masked_conv_variables) ? "" : ", $(length(config.masked_conv_variables)) masked_conv_vars (thresh=$(config.masked_conv_threshold))"
                printstyled("  Pass $(i) config: $(length(config.conv_variables)) conv_variables, " *
                            "$(isempty(config.selected_features) ? "all" : "$(length(config.selected_features))") features$(masked_info)\n",
                            color=:cyan)
            end

            out = config.feature_output_paths[i]

            ##If execution proceeds past the first iteration, a composite model is being created, and
            ##so a further mask will be applied to the features
            if i > 1
                QC_mask = true
            else
                QC_mask = config.QC_mask
            end

            QC_mask ? mask_name = config.mask_names[i] : mask_name = ""

            starttime = time()

            if config.file_preprocessed[i]

                print("Reading input features from file $(out)...\n")
                h5open(out) do f
                    X = f["X"][:,:]
                    Y = f["Y"][:,:]
                end

            elseif config.task_mode == "convolution"
                printstyled("\nCALCULATING CONVOLUTION FEATURES FOR PASS: $(i)\n", color=:green)

                if config.write_out & config.overwrite_output
                    isfile(out) ? rm(out) : ""
                end

                X, Y = calculate_features_conv(config, out;
                                                QC_mask=QC_mask, mask_name=mask_name,
                                                write_out=config.write_out)
                printstyled("FINISHED CALCULATING CONVOLUTION FEATURES FOR PASS $(i) in $(round(time() - starttime, digits=3)) seconds...\n", color=:green)
            else
                currt = config.task_paths[i]
                cw = config.task_weights[i]

                printstyled("\nCALCULATING FEATURES FOR PASS: $(i)\n", color=:green)

                if config.write_out & config.overwrite_output
                    isfile(out) ? rm(out) : ""
                end

                X,Y = calculate_features(config.input_path, currt, out, config.HAS_INTERACTIVE_QC;
                                    verbose = config.verbose,
                                    REMOVE_LOW_SIG_QUALITY = config.REMOVE_LOW_SIG_QUALITY, SIG_QUALITY_THRESHOLD = config.SIG_QUALITY_THRESHOLD, SIG_QUALITY_VAR=config.SIG_QUALITY_VAR,
                                    REMOVE_HIGH_PGG=config.REMOVE_HIGH_PGG, PGG_THRESHOLD = config.PGG_THRESHOLD, QC_variable = config.QC_var,
                                    remove_variable = config.remove_var, replace_missing = config.replace_missing,
                                    write_out = config.write_out, QC_mask = QC_mask, mask_name = mask_name, weight_matrixes=cw)
                printstyled("FINISHED CALCULATING FEATURES FOR PASS $(i) in $(round(time() - starttime, digits = 3)) seconds...\n", color=:green)
            end

            printstyled("\nTRAINING MODEL FOR PASS: $(i)\n", color=:green)
            starttime = time()

            class_weights = Vector{Float32}([0.0,1.0])
            ##Train model based on these features
            if config.class_weights != ""

                if lowercase(config.class_weights) != "balanced"
                    printstyled("ERROR: UNKNOWN CLASS WEIGHT $(config.class_weights)... \nContinuing with no weighting\n", color=:yellow)
                else

                    class_weights = Vector{Float32}(fill(0,length(Y[:,:][:])))
                    weight_dict = compute_balanced_class_weights(Y[:,:][:])
                    for class in keys(weight_dict)
                        class_weights[Y[:,:][:] .== class] .= weight_dict[class]
                    end

                end
            end

            printstyled("\n...TRAINING FOR PASS: $(i) ON $(size(X)[1]) GATES, $(size(X)[2]) FEATURES...\n", color=:green)
            printstyled("  selected_features: $(isempty(config.selected_features) ? "none (all)" : "$(length(config.selected_features)) indices, max=$(maximum(config.selected_features))")\n", color=:cyan)

            printstyled("X TYPE: $(typeof(X))\n", color=:blue)
            train_model(X, Y, model_path, n_trees = config.n_trees, max_depth = config.max_depth, class_weights = class_weights, max_threads = config.max_training_threads)

            # Re-save model with metadata (conv_variables, selected_features)
            if config.task_mode == "convolution"
                curr_model = load_model(model_path, config.task_mode)
                if !isempty(config.selected_features)
                    printstyled("  Model trained on $(length(config.selected_features)) selected features (subset)\n", color=:green)
                end
                JLD2.jldsave(model_path;
                    model=curr_model,
                    selected_features=config.selected_features,
                    conv_variables=config.conv_variables,
                    masked_conv_variables=config.masked_conv_variables,
                    masked_conv_kernel_types=config.masked_conv_kernel_types,
                    masked_conv_kernel_sizes=config.masked_conv_kernel_sizes,
                    masked_conv_threshold=config.masked_conv_threshold,
                    masked_conv_met_prob_field=config.masked_conv_met_prob_field)
            end

            # Feature importance and selection for convolution mode
            if config.task_mode == "convolution" && config.compute_feature_importance
                printstyled("\nCOMPUTING FEATURE IMPORTANCE FOR PASS $(i)...\n", color=:green)
                curr_model = load_model(model_path, config.task_mode)
                importances = compute_rf_feature_importance(curr_model, X, reshape(Y, length(Y));
                    n_repeats=config.n_importance_repeats,
                    subsample_fraction=config.importance_subsample_fraction)

                kernel_bank = build_kernel_bank(config.conv_kernel_sizes)
                masked_conv_kb = if !isempty(config.masked_conv_variables)
                    build_filtered_kernel_bank(config.masked_conv_kernel_types, config.masked_conv_kernel_sizes)
                else
                    ConvolutionKernel[]
                end
                feat_names_full = build_feature_names(config.conv_variables, kernel_bank;
                    masked_conv_variables=config.masked_conv_variables,
                    masked_conv_kernel_bank=masked_conv_kb,
                    masked_conv_threshold=config.masked_conv_threshold)

                # Print importance ranking
                sorted_idx = sortperm(importances, rev=true)
                printstyled("\n  FEATURE IMPORTANCE RANKING (Pass $(i)):\n", color=:cyan)
                for (rank, idx) in enumerate(sorted_idx)
                    name = idx <= length(feat_names_full) ? feat_names_full[idx] : "feature_$(idx)"
                    printstyled("    $(rank). $(name): $(round(importances[idx], digits=6))\n", color=:cyan)
                end

                # Identify recommended features (informational only — model is still trained on all)
                recommended = select_features(importances, config.feature_importance_threshold)
                printstyled("\n  RECOMMENDED $(length(recommended))/$(length(importances)) FEATURES above $(config.feature_importance_threshold * 100)% threshold\n", color=:green)
                printstyled("  selected_features = $(recommended)\n", color=:yellow)
            printstyled("  Copy the line above into your PASS_CONFIG to retrain with this subset.\n", color=:yellow)

                # Save importance metadata alongside model
                # Preserve selected_features so inference knows which columns the model expects
                JLD2.jldsave(model_path;
                    model=curr_model,
                    selected_features=config.selected_features,
                    recommended_features=recommended,
                    feature_names=feat_names_full,
                    importances=importances,
                    conv_variables=config.conv_variables,
                    masked_conv_variables=config.masked_conv_variables,
                    masked_conv_kernel_types=config.masked_conv_kernel_types,
                    masked_conv_kernel_sizes=config.masked_conv_kernel_sizes,
                    masked_conv_threshold=config.masked_conv_threshold,
                    masked_conv_met_prob_field=config.masked_conv_met_prob_field)
                printstyled("  Saved model + importance metadata to $(model_path)\n", color=:green)
            end

            ###If this was the last pass, we don't need to write out a mask, and we're done!
            ###Otherwise, we need to mask out the features we want to apply the model to on the next pass
            if i < config.num_models

                curr_model = load_model(model_path, config.task_mode)
                curr_metprobs = config.met_probs[i]

                paths = Vector{String}()
                file_path = config.input_path

                if isdir(file_path)
                    paths = parse_directory(file_path)
                else
                    paths = [file_path]
                end

                for path in paths

                    dims = Dataset(path) do f
                        (f.dim["range"], f.dim["time"])
                    end

                    if config.task_mode == "convolution"
                        # Load metadata from model JLD2 to match what the model expects
                        md = load_model_with_metadata(model_path, config.task_mode)
                        config_single = deepcopy(config)
                        config_single.input_path = path
                        config_single.write_out = false
                        config_single.selected_features = md.selected_features
                        if !isempty(md.conv_variables)
                            config_single.conv_variables = md.conv_variables
                        end
                        config_single.masked_conv_variables = md.masked_conv_variables
                        config_single.masked_conv_kernel_types = md.masked_conv_kernel_types
                        config_single.masked_conv_kernel_sizes = md.masked_conv_kernel_sizes
                        config_single.masked_conv_threshold = md.masked_conv_threshold
                        config_single.masked_conv_met_prob_field = md.masked_conv_met_prob_field
                        X, Y, idxer_list = calculate_features_conv(config_single, out;
                                                                     QC_mask=QC_mask, mask_name=mask_name,
                                                                     write_out=false, return_idxer=true)
                        idxer = idxer_list
                    else
                        currt = config.task_paths[i]
                        cw = config.task_weights[i]
                        X, Y, idxer = calculate_features(path, currt, out, true;
                                            verbose = config.verbose,
                                            REMOVE_LOW_SIG_QUALITY = config.REMOVE_LOW_SIG_QUALITY, SIG_QUALITY_THRESHOLD = config.SIG_QUALITY_THRESHOLD, SIG_QUALITY_VAR=config.SIG_QUALITY_VAR,
                                            REMOVE_HIGH_PGG=config.REMOVE_HIGH_PGG,PGG_THRESHOLD=config.PGG_THRESHOLD, QC_variable = config.QC_var,
                                            remove_variable = config.remove_var, replace_missing = config.replace_missing, return_idxer=true,
                                            write_out = false, QC_mask = QC_mask, mask_name = mask_name, weight_matrixes=cw)
                    end

                    printstyled("  Prediction: X size=$(size(X)), model_selected_features=$(isempty(md.selected_features) ? "none" : "$(length(md.selected_features)) indices")\n", color=:cyan)
                    met_probs = DecisionTree.predict_proba(curr_model, X)
                    if size(met_probs)[2] < 2
                        throw(DomainError(1, "ERROR: ONLY ONE CLASS IN INPUT DATASET"))
                    end
                    met_probs = met_probs[:, 2]
                    valid_idxs = (met_probs .>= minimum(curr_metprobs)) .& (met_probs .<= maximum(curr_metprobs))
                    print("RESULTANT GATES: $(sum(valid_idxs))")

                    ## Save met_prob predictions for all valid gates so subsequent passes
                    ## can use them as predictors and masks can be regenerated with different
                    ## thresholds without re-running the model
                    met_prob_field = Matrix{Union{Missing, Float32}}(missings(dims))[:]
                    met_prob_idxer = copy(idxer[1][:])
                    met_prob_field[met_prob_idxer] .= Float32.(met_probs)
                    met_prob_field = reshape(met_prob_field, dims)
                    met_prob_name = "met_prob_pass_$(i)"
                    write_field(path, met_prob_name, met_prob_field,
                        attribs=Dict("Units" => "probability",
                                     "Description" => "RF meteorological probability from pass $(i)"))

                    ##Create mask field, fill it, and then write out
                    new_mask = Matrix{Union{Missing, Float32}}(missings(dims))[:]

                    idxer = idxer[1][:]
                    idxer[idxer] .= Vector{Bool}(valid_idxs)
                    new_mask[idxer] .= 1.
                    new_mask = reshape(new_mask, dims)

                    write_field(path, config.mask_names[i+1], new_mask, attribs=Dict("Units" => "Bool", "Description" => "Gates between met prob thresholds"))

                end
            end
        end
        # Restore original config values
        config.conv_variables = orig_conv_variables
        config.selected_features = orig_selected_features
        config.masked_conv_variables = orig_masked_conv_variables
        config.masked_conv_kernel_types = orig_masked_conv_kernel_types
        config.masked_conv_kernel_sizes = orig_masked_conv_kernel_sizes
        config.masked_conv_threshold = orig_masked_conv_threshold
        config.masked_conv_met_prob_field = orig_masked_conv_met_prob_field

        printstyled("\n COMPLETED TRAINING MODEL IN $(round(time() - full_start_time, digits = 3)) seconds...\n", color=:green)
    end

    """
        regenerate_masks(config::ModelConfig, pass::Int, met_probs_threshold::Tuple{Float32, Float32})

    Regenerate the mask for `pass+1` using saved `met_prob_pass_<pass>` fields in CfRadials.
    This avoids re-running the model — it reads the previously-saved met_prob predictions
    and applies the new threshold to create an updated mask.

    The met_prob fields must have been written by a prior `train_multi_model` call.
    """
    function regenerate_masks(config::ModelConfig, pass::Int, met_probs_threshold::Tuple{Float32, Float32})

        paths = isdir(config.input_path) ? parse_directory(config.input_path) : [config.input_path]
        met_prob_name = "met_prob_pass_$(pass)"

        for path in paths
            dims = Dataset(path) do f
                (f.dim["range"], f.dim["time"])
            end

            met_prob_field = Dataset(path) do f
                f[met_prob_name][:, :]
            end

            new_mask = Matrix{Union{Missing, Float32}}(missings(dims))
            for j in 1:dims[2], i in 1:dims[1]
                v = met_prob_field[i, j]
                if !ismissing(v) && v >= met_probs_threshold[1] && v <= met_probs_threshold[2]
                    new_mask[i, j] = 1.0f0
                end
            end

            write_field(path, config.mask_names[pass + 1], new_mask,
                attribs=Dict("Units" => "Bool",
                             "Description" => "Gates between met prob thresholds"),
                verbose=false)
        end
        printstyled("Regenerated masks for pass $(pass+1) with threshold $(met_probs_threshold) ($(length(paths)) files)\n", color=:green)
    end

    """
        met_prob_histogram(config::ModelConfig, pass::Int; data_path::String="",
                           met_probs_threshold::Union{Nothing, Tuple{Float32,Float32}}=nothing)

    Read saved `met_prob_pass_<pass>` from CfRadial files and print a histogram
    of the probability distribution. Uses fine bins near 0 and 1 (where threshold
    choices are most sensitive) and coarser bins in the middle.

    If `met_probs_threshold` is provided, marks the threshold positions and prints
    a summary of how many gates fall into each category. Otherwise shows the raw
    distribution to help choose thresholds.

    Prints cumulative percentages from each tail to help identify where thresholds
    capture the most confident classifications.
    """
    function met_prob_histogram(config::ModelConfig, pass::Int; data_path::String="",
                                 met_probs_threshold::Union{Nothing, Tuple{Float32,Float32}}=nothing)

        target_path = isempty(data_path) ? config.input_path : data_path
        paths = isdir(target_path) ? parse_directory(target_path) : [target_path]
        met_prob_name = "met_prob_pass_$(pass)"

        probs = Float32[]
        n_files = 0
        n_missing = 0
        for path in paths
            NCDataset(path, "r") do f
                if !haskey(f, met_prob_name)
                    n_missing += 1
                    return
                end
                raw = f[met_prob_name][:]
                for v in raw
                    if !ismissing(v) && !isnan(v) && v >= 0.0f0 && v <= 1.0f0
                        push!(probs, Float32(v))
                    end
                end
                n_files += 1
            end
        end

        if n_missing > 0
            printstyled("  WARNING: $(n_missing)/$(length(paths)) files missing $(met_prob_name)\n", color=:yellow)
        end
        if isempty(probs)
            printstyled("  No valid probabilities found for $(met_prob_name)\n", color=:red)
            return
        end

        n = length(probs)

        ## Adaptive bin edges: 0.001 in extreme tails (0–0.01, 0.99–1.0),
        ## 0.01 in near tails (0.01–0.1, 0.9–0.99), coarse in the middle
        bin_edges = Float32[0.000, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.010,
                            0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10,
                            0.20, 0.30, 0.50, 0.70, 0.80, 0.90,
                            0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99,
                            0.991, 0.992, 0.993, 0.994, 0.995, 0.996, 0.997, 0.998, 0.999, 1.001]
        n_bins = length(bin_edges) - 1
        counts = zeros(Int, n_bins)
        for p in probs
            for b in n_bins:-1:1
                if p >= bin_edges[b]
                    counts[b] += 1
                    break
                end
            end
        end
        max_count = maximum(counts)
        bar_width = 50

        header = "Pass $(pass) met_prob distribution ($(n) gates from $(n_files) files)"
        if met_probs_threshold !== nothing
            header *= ", thresholds=$(met_probs_threshold)"
        end

        printstyled("\n  $(header):\n", color=:cyan)
        printstyled("  ", "─"^72, "\n", color=:light_black)

        ## Cumulative from low tail
        cum_low = 0
        ## Cumulative from high tail
        cum_high = 0
        cum_high_counts = zeros(Int, n_bins)
        for b in n_bins:-1:1
            cum_high += counts[b]
            cum_high_counts[b] = cum_high
        end

        cum_low = 0
        for b in 1:n_bins
            lo_edge = bin_edges[b]
            hi_edge = min(bin_edges[b+1], 1.0f0)
            bar_len = max_count > 0 ? round(Int, counts[b] / max_count * bar_width) : 0
            pct = round(counts[b] / n * 100, digits=1)
            cum_low += counts[b]
            cum_low_pct = round(cum_low / n * 100, digits=1)
            cum_hi_pct = round(cum_high_counts[b] / n * 100, digits=1)

            ## Color based on threshold position
            bar_color = :white
            marker = ""
            if met_probs_threshold !== nothing
                lo_t = minimum(met_probs_threshold)
                hi_t = maximum(met_probs_threshold)
                if hi_edge <= lo_t
                    bar_color = :red
                elseif lo_edge >= hi_t
                    bar_color = :green
                else
                    bar_color = :yellow
                end
                if lo_edge < lo_t <= hi_edge
                    marker = " ◀ NMD"
                elseif lo_edge < hi_t <= hi_edge
                    marker = " ◀ MD"
                end
            end

            bar = "█"^bar_len
            label = "$(lpad(string(round(lo_edge, digits=2)), 4))–$(rpad(string(round(hi_edge, digits=2)), 4))"
            printstyled("    $(label) ", color=:light_black)
            printstyled("│$(bar)", color=bar_color)
            printstyled(" $(lpad(string(counts[b]), 7)) ($(lpad(string(pct), 5))%)", color=:light_black)
            ## Show cumulative from the nearer tail
            if b <= n_bins ÷ 2
                printstyled("  cum≤$(round(hi_edge, digits=2)): $(cum_low_pct)%", color=:light_black)
            else
                printstyled("  cum≥$(round(lo_edge, digits=2)): $(cum_hi_pct)%", color=:light_black)
            end
            printstyled("$(marker)\n", color=:cyan)
        end

        printstyled("  ", "─"^72, "\n", color=:light_black)

        ## Summary statistics
        q = sort(probs)
        p5  = q[max(1, round(Int, 0.05 * n))]
        p25 = q[max(1, round(Int, 0.25 * n))]
        p50 = q[max(1, round(Int, 0.50 * n))]
        p75 = q[max(1, round(Int, 0.75 * n))]
        p95 = q[max(1, round(Int, 0.95 * n))]

        printstyled("    Percentiles: 5th=$(round(p5, digits=3))  25th=$(round(p25, digits=3))" *
                    "  50th=$(round(p50, digits=3))  75th=$(round(p75, digits=3))" *
                    "  95th=$(round(p95, digits=3))\n", color=:cyan)

        if met_probs_threshold !== nothing
            lo_t = minimum(met_probs_threshold)
            hi_t = maximum(met_probs_threshold)
            n_nmd = sum(p -> p < lo_t, probs)
            n_md  = sum(p -> p >= hi_t, probs)
            n_unc = n - n_md - n_nmd
            printstyled("    With threshold ($(lo_t), $(hi_t)): " *
                        "$(n_nmd) NMD ($(round(n_nmd/n*100, digits=1))%), " *
                        "$(n_unc) uncertain → pass $(pass+1) ($(round(n_unc/n*100, digits=1))%), " *
                        "$(n_md) MD ($(round(n_md/n*100, digits=1))%)\n", color=:cyan)
        end
        println()
    end

    """
        generate_pass_masks(config::ModelConfig, pass::Int; data_path::String="")

    Run inference with the trained model for `pass` and write `met_prob_pass_<pass>`
    and `mask_pass_<pass+1>` to all CfRadial files. This bridges an existing trained
    pass to the next pass without retraining.

    Use this when you have a trained Pass 1 model and want to prepare data for Pass 2
    training, or any pass N → pass N+1 transition.

    If `data_path` is provided, it overrides `config.input_path` (useful for generating
    masks on both training and testing sets).

    The mask threshold is taken from `config.met_probs[pass]`.
    """
    function generate_pass_masks(config::ModelConfig, pass::Int; data_path::String="")
        model_path = config.model_output_paths[pass]
        if !isfile(model_path)
            error("Model file not found: $(model_path). Train pass $(pass) first.")
        end

        curr_model = load_model(model_path, config.task_mode)
        curr_metprobs = config.met_probs[pass]
        target_path = isempty(data_path) ? config.input_path : data_path

        paths = isdir(target_path) ? parse_directory(target_path) : [target_path]

        QC_mask = pass > 1
        mask_name = QC_mask ? config.mask_names[pass] : ""
        out = config.feature_output_paths[pass]

        # Load model metadata once (not per-file)
        md = config.task_mode == "convolution" ? load_model_with_metadata(model_path, config.task_mode) : nothing

        printstyled("\nGENERATING MASKS FOR PASS $(pass) → $(pass+1)\n", color=:green)
        printstyled("  Model: $(model_path)\n", color=:green)
        printstyled("  met_prob threshold: $(curr_metprobs)\n", color=:green)
        printstyled("  Data: $(target_path) ($(length(paths)) files)\n", color=:green)
        printstyled("  Threads: $(Threads.nthreads())\n", color=:green)

        n_paths = length(paths)
        files_done = Threads.Atomic{Int}(0)
        start_time = time()

        Threads.@threads for fi in 1:n_paths
            path = paths[fi]

            dims = Dataset(path) do f
                (f.dim["range"], f.dim["time"])
            end

            if config.task_mode == "convolution"
                config_single = deepcopy(config)
                config_single.input_path = path
                config_single.write_out = false
                config_single.selected_features = md.selected_features
                config_single.verbose = false
                X_mask, Y_mask, idxer = calculate_features_conv(config_single, out;
                                                                 QC_mask=QC_mask, mask_name=mask_name,
                                                                 write_out=false, return_idxer=true)
            else
                currt = config.task_paths[pass]
                cw = config.task_weights[pass]
                X_mask, Y_mask, idxer = calculate_features(path, currt, out, true;
                                    verbose = false,
                                    REMOVE_LOW_SIG_QUALITY = config.REMOVE_LOW_SIG_QUALITY,
                                    SIG_QUALITY_THRESHOLD = config.SIG_QUALITY_THRESHOLD,
                                    SIG_QUALITY_VAR=config.SIG_QUALITY_VAR,
                                    REMOVE_HIGH_PGG=config.REMOVE_HIGH_PGG,
                                    PGG_THRESHOLD=config.PGG_THRESHOLD, QC_variable = config.QC_var,
                                    remove_variable = config.remove_var,
                                    replace_missing = config.replace_missing, return_idxer=true,
                                    write_out = false, QC_mask = QC_mask, mask_name = mask_name,
                                    weight_matrixes=cw)
            end

            met_probs_pred = DecisionTree.predict_proba(curr_model, X_mask)
            if size(met_probs_pred, 2) < 2
                throw(DomainError(1, "ERROR: ONLY ONE CLASS IN INPUT DATASET for $(path)"))
            end
            met_probs_pred = met_probs_pred[:, 2]
            valid_idxs = (met_probs_pred .>= minimum(curr_metprobs)) .& (met_probs_pred .<= maximum(curr_metprobs))

            ## Save met_prob predictions
            met_prob_field = Matrix{Union{Missing, Float32}}(missings(dims))[:]
            met_prob_idxer = copy(idxer[1][:])
            met_prob_field[met_prob_idxer] .= Float32.(met_probs_pred)
            met_prob_field = reshape(met_prob_field, dims)
            write_field(path, "met_prob_pass_$(pass)", met_prob_field,
                attribs=Dict("Units" => "probability",
                             "Description" => "RF meteorological probability from pass $(pass)"))

            ## Create mask for next pass
            new_mask = Matrix{Union{Missing, Float32}}(missings(dims))[:]
            idxer_flat = idxer[1][:]
            idxer_flat[idxer_flat] .= Vector{Bool}(valid_idxs)
            new_mask[idxer_flat] .= 1.
            new_mask = reshape(new_mask, dims)
            write_field(path, config.mask_names[pass + 1], new_mask,
                attribs=Dict("Units" => "Bool", "Description" => "Gates between met prob thresholds"))

            done = Threads.atomic_add!(files_done, 1)
            if done % 50 == 0 || done == n_paths
                elapsed = round(time() - start_time, digits=1)
                printstyled("  Processed $(done)/$(n_paths) files ($(elapsed)s elapsed)\n", color=:light_black)
            end
        end

        total_time = round(time() - start_time, digits=1)
        printstyled("  Done — wrote met_prob_pass_$(pass) and $(config.mask_names[pass+1]) to $(n_paths) files in $(total_time)s\n", color=:green)
    end

    """
        train_single_pass(config::ModelConfig, pass::Int)

    Train a single pass of the multi-pass cascade. This is the building block for
    pass-by-pass tuning workflows:

    - Pass 1: trains on all data (no mask required)
    - Pass 2+: requires masks from prior passes to exist in the CfRadial files
      (written by `train_multi_model` or `regenerate_masks`)

    After training, if this is not the final pass, saves met_prob predictions to
    CfRadials and generates the mask for the next pass.

    Returns `(X, Y)` — the feature matrix and labels used for training.
    """
    function train_single_pass(config::ModelConfig, pass::Int)
        @assert pass >= 1 && pass <= config.num_models

        out = config.feature_output_paths[pass]
        model_path = config.model_output_paths[pass]

        QC_mask = pass > 1 ? true : config.QC_mask
        mask_name = QC_mask ? config.mask_names[pass] : ""

        starttime = time()

        if config.file_preprocessed[pass]
            print("Reading input features from file $(out)...\n")
            X, Y = h5open(out) do f
                (f["X"][:,:], f["Y"][:,:])
            end
        elseif config.task_mode == "convolution"
            printstyled("\nCALCULATING CONVOLUTION FEATURES FOR PASS: $(pass)\n", color=:green)
            if config.write_out & config.overwrite_output
                isfile(out) && rm(out)
            end
            X, Y = calculate_features_conv(config, out;
                                            QC_mask=QC_mask, mask_name=mask_name,
                                            write_out=config.write_out)
            printstyled("FINISHED CALCULATING CONVOLUTION FEATURES FOR PASS $(pass) in $(round(time() - starttime, digits=3)) seconds...\n", color=:green)
        else
            currt = config.task_paths[pass]
            cw = config.task_weights[pass]
            printstyled("\nCALCULATING FEATURES FOR PASS: $(pass)\n", color=:green)
            if config.write_out & config.overwrite_output
                isfile(out) && rm(out)
            end
            X, Y = calculate_features(config.input_path, currt, out, config.HAS_INTERACTIVE_QC;
                                verbose = config.verbose,
                                REMOVE_LOW_SIG_QUALITY = config.REMOVE_LOW_SIG_QUALITY, SIG_QUALITY_THRESHOLD = config.SIG_QUALITY_THRESHOLD, SIG_QUALITY_VAR=config.SIG_QUALITY_VAR,
                                REMOVE_HIGH_PGG=config.REMOVE_HIGH_PGG, PGG_THRESHOLD = config.PGG_THRESHOLD, QC_variable = config.QC_var,
                                remove_variable = config.remove_var, replace_missing = config.replace_missing,
                                write_out = config.write_out, QC_mask = QC_mask, mask_name = mask_name, weight_matrixes=cw)
            printstyled("FINISHED CALCULATING FEATURES FOR PASS $(pass) in $(round(time() - starttime, digits = 3)) seconds...\n", color=:green)
        end

        ## Train the model
        printstyled("\nTRAINING MODEL FOR PASS: $(pass)\n", color=:green)
        starttime = time()

        class_weights = Vector{Float32}([0.0, 1.0])
        if config.class_weights != ""
            if lowercase(config.class_weights) != "balanced"
                printstyled("ERROR: UNKNOWN CLASS WEIGHT $(config.class_weights)... \nContinuing with no weighting\n", color=:yellow)
            else
                class_weights = Vector{Float32}(fill(0, length(Y[:, :][:])))
                weight_dict = compute_balanced_class_weights(Y[:, :][:])
                for class in keys(weight_dict)
                    class_weights[Y[:, :][:] .== class] .= weight_dict[class]
                end
            end
        end

        printstyled("\n...TRAINING FOR PASS: $(pass) ON $(size(X)[1]) GATES...\n", color=:green)
        train_model(X, Y, model_path, n_trees = config.n_trees, max_depth = config.max_depth, class_weights = class_weights, max_threads = config.max_training_threads)

        # Re-save model with metadata (conv_variables, selected_features)
        if config.task_mode == "convolution"
            curr_model = load_model(model_path, config.task_mode)
            if !isempty(config.selected_features)
                printstyled("  Model trained on $(length(config.selected_features)) selected features (subset)\n", color=:green)
            end
            JLD2.jldsave(model_path;
                model=curr_model,
                selected_features=config.selected_features,
                conv_variables=config.conv_variables,
                masked_conv_variables=config.masked_conv_variables,
                masked_conv_kernel_types=config.masked_conv_kernel_types,
                masked_conv_kernel_sizes=config.masked_conv_kernel_sizes,
                masked_conv_threshold=config.masked_conv_threshold,
                masked_conv_met_prob_field=config.masked_conv_met_prob_field)
        end

        ## Feature importance and selection for convolution mode
        if config.task_mode == "convolution" && config.compute_feature_importance
            printstyled("\nCOMPUTING FEATURE IMPORTANCE FOR PASS $(pass)...\n", color=:green)
            curr_model = load_model(model_path, config.task_mode)
            importances = compute_rf_feature_importance(curr_model, X, reshape(Y, length(Y));
                n_repeats=config.n_importance_repeats,
                subsample_fraction=config.importance_subsample_fraction)

            kernel_bank = build_kernel_bank(config.conv_kernel_sizes)
            masked_conv_kb = if !isempty(config.masked_conv_variables)
                build_filtered_kernel_bank(config.masked_conv_kernel_types, config.masked_conv_kernel_sizes)
            else
                ConvolutionKernel[]
            end
            feat_names_full = build_feature_names(config.conv_variables, kernel_bank;
                masked_conv_variables=config.masked_conv_variables,
                masked_conv_kernel_bank=masked_conv_kb,
                masked_conv_threshold=config.masked_conv_threshold)

            sorted_idx = sortperm(importances, rev=true)
            printstyled("\n  FEATURE IMPORTANCE RANKING (Pass $(pass)):\n", color=:cyan)
            for (rank, idx) in enumerate(sorted_idx)
                name = idx <= length(feat_names_full) ? feat_names_full[idx] : "feature_$(idx)"
                printstyled("    $(rank). $(name): $(round(importances[idx], digits=6))\n", color=:cyan)
            end

            recommended = select_features(importances, config.feature_importance_threshold)
            printstyled("\n  RECOMMENDED $(length(recommended))/$(length(importances)) FEATURES above $(config.feature_importance_threshold * 100)% threshold\n", color=:green)
            printstyled("  selected_features = $(recommended)\n", color=:yellow)
            printstyled("  Copy the line above into your PASS_CONFIG to retrain with this subset.\n", color=:yellow)

            JLD2.jldsave(model_path;
                model=curr_model,
                selected_features=config.selected_features,
                recommended_features=recommended,
                feature_names=feat_names_full,
                importances=importances,
                conv_variables=config.conv_variables,
                masked_conv_variables=config.masked_conv_variables,
                masked_conv_kernel_types=config.masked_conv_kernel_types,
                masked_conv_kernel_sizes=config.masked_conv_kernel_sizes,
                masked_conv_threshold=config.masked_conv_threshold,
                masked_conv_met_prob_field=config.masked_conv_met_prob_field)
            printstyled("  Saved model + importance metadata to $(model_path)\n", color=:green)
        end

        ## Generate masks and save met_prob for next pass
        if pass < config.num_models
            curr_model = load_model(model_path, config.task_mode)
            curr_metprobs = config.met_probs[pass]

            paths = isdir(config.input_path) ? parse_directory(config.input_path) : [config.input_path]

            for path in paths
                dims = Dataset(path) do f
                    (f.dim["range"], f.dim["time"])
                end

                if config.task_mode == "convolution"
                    # Load selected_features from model JLD2 to match what the model expects
                    md = load_model_with_metadata(model_path, config.task_mode)
                    config_single = deepcopy(config)
                    config_single.input_path = path
                    config_single.write_out = false
                    config_single.selected_features = md.selected_features
                    config_single.masked_conv_variables = md.masked_conv_variables
                    config_single.masked_conv_kernel_types = md.masked_conv_kernel_types
                    config_single.masked_conv_kernel_sizes = md.masked_conv_kernel_sizes
                    config_single.masked_conv_threshold = md.masked_conv_threshold
                    config_single.masked_conv_met_prob_field = md.masked_conv_met_prob_field
                    X_mask, Y_mask, idxer = calculate_features_conv(config_single, out;
                                                                     QC_mask=QC_mask, mask_name=mask_name,
                                                                     write_out=false, return_idxer=true)
                else
                    currt = config.task_paths[pass]
                    cw = config.task_weights[pass]
                    X_mask, Y_mask, idxer = calculate_features(path, currt, out, true;
                                        verbose = config.verbose,
                                        REMOVE_LOW_SIG_QUALITY = config.REMOVE_LOW_SIG_QUALITY, SIG_QUALITY_THRESHOLD = config.SIG_QUALITY_THRESHOLD, SIG_QUALITY_VAR=config.SIG_QUALITY_VAR,
                                        REMOVE_HIGH_PGG=config.REMOVE_HIGH_PGG, PGG_THRESHOLD=config.PGG_THRESHOLD, QC_variable = config.QC_var,
                                        remove_variable = config.remove_var, replace_missing = config.replace_missing, return_idxer=true,
                                        write_out = false, QC_mask = QC_mask, mask_name = mask_name, weight_matrixes=cw)
                end

                met_probs_pred = DecisionTree.predict_proba(curr_model, X_mask)
                if size(met_probs_pred)[2] < 2
                    throw(DomainError(1, "ERROR: ONLY ONE CLASS IN INPUT DATASET"))
                end
                met_probs_pred = met_probs_pred[:, 2]
                valid_idxs = (met_probs_pred .>= minimum(curr_metprobs)) .& (met_probs_pred .<= maximum(curr_metprobs))
                print("RESULTANT GATES: $(sum(valid_idxs))")

                ## Save met_prob predictions
                met_prob_field = Matrix{Union{Missing, Float32}}(missings(dims))[:]
                met_prob_idxer = copy(idxer[1][:])
                met_prob_field[met_prob_idxer] .= Float32.(met_probs_pred)
                met_prob_field = reshape(met_prob_field, dims)
                write_field(path, "met_prob_pass_$(pass)", met_prob_field,
                    attribs=Dict("Units" => "probability",
                                 "Description" => "RF meteorological probability from pass $(pass)"))

                ## Create mask for next pass
                new_mask = Matrix{Union{Missing, Float32}}(missings(dims))[:]
                idxer_flat = idxer[1][:]
                idxer_flat[idxer_flat] .= Vector{Bool}(valid_idxs)
                new_mask[idxer_flat] .= 1.
                new_mask = reshape(new_mask, dims)
                write_field(path, config.mask_names[pass + 1], new_mask,
                    attribs=Dict("Units" => "Bool", "Description" => "Gates between met prob thresholds"))
            end
        end

        printstyled("\n COMPLETED TRAINING PASS $(pass) in $(round(time() - starttime, digits = 3)) seconds...\n", color=:green)
        return X, Y
    end

    """
        compute_importance(config::ModelConfig)

    Compute permutation-based feature importance for each pass using existing
    trained models and cached feature files. Does NOT retrain — loads the model
    from the JLD2 file and features from the HDF5 file.

    This is the recommended way to compute importance after initial training
    (Step 2). It avoids redundant retraining and supports configurable
    `n_importance_repeats` and `importance_subsample_fraction` from the config.
    """
    function compute_importance(config::ModelConfig; pass::Int=0)
        passes = pass > 0 ? [pass] : 1:length(config.model_output_paths)
        for i in passes
            model_path = config.model_output_paths[i]
            out = config.feature_output_paths[i]

            if !isfile(model_path)
                @warn "Model file not found: $(model_path). Run training first."
                continue
            end
            if !isfile(out)
                @warn "Feature file not found: $(out). Run training first."
                continue
            end

            # Load this pass's conv_variables from its own model file so we don't
            # accidentally overwrite pass 1 metadata with pass 2 config
            md = load_model_with_metadata(model_path, config.task_mode)
            pass_conv_variables = !isempty(md.conv_variables) ? md.conv_variables : config.conv_variables

            printstyled("\nLOADING FEATURES FROM $(out)...\n", color=:green)
            X, Y = h5open(out) do f
                (f["X"][:,:], f["Y"][:,:])
            end
            Y_vec = reshape(Y, length(Y))

            printstyled("COMPUTING FEATURE IMPORTANCE FOR PASS $(i) ($(size(X,1)) gates, $(size(X,2)) features)...\n", color=:green)
            curr_model = md.model
            importances = compute_rf_feature_importance(curr_model, X, Y_vec;
                n_repeats=config.n_importance_repeats,
                subsample_fraction=config.importance_subsample_fraction)

            kernel_bank = build_kernel_bank(config.conv_kernel_sizes)
            feat_names_full = String[]
            for varname in pass_conv_variables
                for kern in kernel_bank
                    push!(feat_names_full, "$(varname)_$(kern.name)")
                    push!(feat_names_full, "$(varname)_$(kern.name)_vfrac")
                end
            end
            append!(feat_names_full, ["AHT", "ELV", "RNG", "NRG"])

            sorted_idx = sortperm(importances, rev=true)
            printstyled("\n  FEATURE IMPORTANCE RANKING (Pass $(i)):\n", color=:cyan)
            for (rank, idx) in enumerate(sorted_idx)
                name = idx <= length(feat_names_full) ? feat_names_full[idx] : "feature_$(idx)"
                printstyled("    $(rank). $(name): $(round(importances[idx], digits=6))\n", color=:cyan)
            end

            recommended = select_features(importances, config.feature_importance_threshold)
            printstyled("\n  RECOMMENDED $(length(recommended))/$(length(importances)) FEATURES above $(config.feature_importance_threshold * 100)% threshold\n", color=:green)
            printstyled("  selected_features = $(recommended)\n", color=:yellow)
            printstyled("  Copy the line above into your PASS_CONFIG to retrain with this subset.\n", color=:yellow)

            JLD2.jldsave(model_path;
                model=curr_model,
                selected_features=md.selected_features,
                recommended_features=recommended,
                feature_names=feat_names_full,
                importances=importances,
                conv_variables=pass_conv_variables)
            printstyled("  Saved model + importance metadata to $(model_path)\n", color=:green)
        end
    end

    """
        run_evaluation(config::ModelConfig, dataset_name::String, dataset_path::String,
                       met_probs::Vector{Tuple{Float32, Float32}};
                       prediction_outfile::String="", verbose::Bool=true)

    Run `composite_prediction` on a dataset and compute classification metrics.

    Temporarily sets `config.input_path` and `config.met_probs` for the evaluation,
    then restores them. Returns a NamedTuple with all metrics.
    """
    function run_evaluation(config::ModelConfig, dataset_name::String, dataset_path::String,
                            met_probs::Vector{Tuple{Float32, Float32}};
                            prediction_outfile::String = "", verbose::Bool = true,
                            skip_existing_met_probs::Bool = false)

        orig_path = config.input_path
        orig_probs = copy(config.met_probs)

        config.input_path = dataset_path
        config.met_probs  = met_probs[1:config.num_models]

        write_preds = prediction_outfile != ""
        outfile = write_preds ? prediction_outfile : "model_predictions.h5"

        try
            predictions, verification, indexers, _pass_probs = composite_prediction(
                config;
                write_predictions_out = write_preds,
                prediction_outfile    = outfile,
                skip_existing_met_probs = skip_existing_met_probs,
            )

            targets = Vector{Bool}(verification[:])
            preds   = Vector{Bool}(predictions)

            contingency = get_contingency(preds, targets)
            prec, recall, f1, tp, fp, tn, fn, n = evaluate_model(preds, targets)

            md_hit_rate  = Float32(tp / (tp + fn))
            nmd_hit_rate = Float32(tn / (tn + fp))

            expected = ((tp + fn) * (tp + fp) + (tn + fp) * (tn + fn)) / n
            hss = (tp + tn - expected) / (n - expected)

            if verbose
                println("\n", "-"^70)
                println("RESULTS on $(dataset_name): $(dataset_path)")
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
            end

            return (predictions=preds, targets=targets,
                    precision=prec, recall=recall, f1=f1, hss=hss,
                    md_hit_rate=md_hit_rate, nmd_hit_rate=nmd_hit_rate,
                    tp=tp, fp=fp, tn=tn, fn=fn, n=n)
        finally
            config.input_path = orig_path
            config.met_probs  = orig_probs
        end
    end

    """
        sweep_pass2_met_probs(config::ModelConfig, testing_path::String;
                              met_prob_low_grid, met_prob_high_grid,
                              use_met_prob_as_feature::Bool=true,
                              sweep_inference::Bool=true,
                              infer_low_grid, infer_high_grid,
                              nmd_target::Float32=0.99f0,
                              secondary_metric::Symbol=:hss)

    Sweep met_prob thresholds for Pass 2 of a multi-pass cascade.

    Pass 1 must already be trained (its model and `met_prob_pass_1` fields must exist
    in the CfRadial files). For each threshold combination:
      1. Regenerates the Pass 1→2 mask from saved `met_prob_pass_1` fields
      2. Recalculates features on the filtered subset (spatial features change)
      3. Retrains Pass 2 on the "hard" data
      4. Evaluates the full cascade on the testing set

    If `sweep_inference=true`, also sweeps inference thresholds for the best
    training configuration (cheap — no retraining needed).

    Returns a `DataFrame` of results (requires DataFrames to be loaded).
    """
    function sweep_pass2_met_probs(config::ModelConfig,
                                   training_path::String,
                                   testing_path::String;
                                   experiment_name::String = "sweep",
                                   met_prob_low_grid::Vector{Float32} = Float32[0.1, 0.2, 0.3, 0.4],
                                   met_prob_high_grid::Vector{Float32} = Float32[0.6, 0.7, 0.8, 0.9],
                                   use_met_prob_as_feature::Bool = true,
                                   sweep_inference::Bool = true,
                                   infer_low_grid::Vector{Float32} = Float32[0.1, 0.2, 0.3],
                                   infer_high_grid::Vector{Float32} = Float32[0.98, 0.99, 0.999],
                                   nmd_target::Float32 = 0.99f0,
                                   secondary_metric::Symbol = :hss,
                                   skip_existing_sweep::Bool = false)

        @assert config.num_models >= 2 "sweep_pass2_met_probs requires num_models >= 2"
        @assert isfile(config.model_output_paths[1]) "Pass 1 model not found at $(config.model_output_paths[1]). Train Pass 1 first."

        # Save original config state
        orig_conv_variables = copy(config.conv_variables)
        orig_model_paths = copy(config.model_output_paths)
        orig_feature_paths = copy(config.feature_output_paths)
        orig_met_probs = copy(config.met_probs)
        orig_input_path = config.input_path
        orig_masked_conv_variables = copy(config.masked_conv_variables)
        orig_masked_conv_kernel_types = copy(config.masked_conv_kernel_types)
        orig_masked_conv_kernel_sizes = copy(config.masked_conv_kernel_sizes)
        orig_masked_conv_threshold = config.masked_conv_threshold
        orig_masked_conv_met_prob_field = config.masked_conv_met_prob_field

        all_results = NamedTuple[]

        # Summary file — append results as we go so progress is saved
        summary_file = "$(experiment_name)_sweep_summary.txt"
        completed_combos = Set{Tuple{Float32,Float32,Float32}}()
        if skip_existing_sweep && isfile(summary_file)
            for line in eachline(summary_file)
                startswith(line, "#") && continue
                fields = split(strip(line))
                length(fields) >= 3 || continue
                try
                    k = (parse(Float32, fields[1]), parse(Float32, fields[2]), parse(Float32, fields[3]))
                    push!(completed_combos, k)
                catch; end
            end
            if !isempty(completed_combos)
                printstyled("  Loaded $(length(completed_combos)) completed results from $(summary_file)\n", color=:cyan)
            end
        else
            open(summary_file, "w") do io
                println(io, "# RONIN Pass 2 Sweep Summary — $(experiment_name)")
                println(io, "#")
                println(io, "# mask_lo  mask_hi  infer_hi  NMD       MD        HSS       F1        gates     model_path")
            end
        end

        n_combos = length(met_prob_low_grid) * length(met_prob_high_grid)
        n_infer = sweep_inference ? length(infer_high_grid) : 0
        println("\n", "="^70)
        println("PASS 2 MET_PROB SWEEP: $(n_combos) mask thresholds × $(max(1, n_infer)) inference thresholds")
        println("  Pass 1 model: $(config.model_output_paths[1]) (frozen)")
        println("  NMD target: $(nmd_target)")
        println("  Secondary metric: $(secondary_metric)")
        println("  met_prob as Pass 2 feature: $(use_met_prob_as_feature)")
        println("  Summary file: $(summary_file)")
        if sweep_inference
            println("  Inference high grid: $(infer_high_grid)")
        end
        if skip_existing_sweep
            println("  Skip existing: ON (reuses trained models and features if present)")
        end
        println("="^70)

        combo_idx = 0
        for mp_lo in met_prob_low_grid, mp_hi in met_prob_high_grid
            combo_idx += 1
            sweep_threshold = (mp_lo, mp_hi)
            println("\n  [$(combo_idx)/$(n_combos)] mask threshold=$(sweep_threshold)")

            # Configure Pass 2 with unique paths
            sweep_tag = "mp_$(mp_lo)_$(mp_hi)"
            pass2_model_path = "trained_model_$(experiment_name)_pass2_$(sweep_tag).jld2"
            pass2_feature_path = "output_features_$(experiment_name)_pass2_$(sweep_tag).h5"
            config.model_output_paths[2] = pass2_model_path
            config.feature_output_paths[2] = pass2_feature_path
            config.met_probs[1] = sweep_threshold
            config.input_path = training_path

            # Add met_prob as convolution variable if requested
            if use_met_prob_as_feature && config.task_mode == "convolution"
                config.conv_variables = copy(orig_conv_variables)
                if !("met_prob_pass_1" in config.conv_variables)
                    push!(config.conv_variables, "met_prob_pass_1")
                end
            end

            # Skip training if model and features already exist
            if skip_existing_sweep && isfile(pass2_model_path) && isfile(pass2_feature_path)
                printstyled("    Skipping training — model and features exist: $(pass2_model_path)\n", color=:cyan)
                config.file_preprocessed[2] = true
            else
                # Regenerate masks on training and testing data
                config.input_path = training_path
                regenerate_masks(config, 1, sweep_threshold)
                config.input_path = testing_path
                regenerate_masks(config, 1, sweep_threshold)
                config.input_path = training_path

                # Use precomputed features if they exist, otherwise recalculate
                if skip_existing_sweep && isfile(pass2_feature_path)
                    printstyled("    Using precomputed features: $(pass2_feature_path)\n", color=:cyan)
                    config.file_preprocessed[2] = true
                else
                    config.file_preprocessed[2] = false
                end

                println("    Training Pass 2...")
                train_single_pass(config, 2)
            end

            # Run evaluation once to store met_prob_pass_2 in testing CfRadials
            # (pass 1 is frozen so skip_existing_met_probs reads saved pass 1 probs)
            baseline_infer_hi = Float32(maximum(config.met_probs[2]))
            if (mp_lo, mp_hi, baseline_infer_hi) in completed_combos
                printstyled("    Skipping baseline evaluation — already in summary\n", color=:cyan)
            else
                println("    Computing pass 2 predictions on testing set...")
                r = run_evaluation(config, "PASS2_SWEEP", testing_path,
                                   [sweep_threshold, config.met_probs[2]];
                                   skip_existing_met_probs=true, verbose=false)
                println("    NMD=$(round(r.nmd_hit_rate, digits=4))  MD=$(round(r.md_hit_rate, digits=4))  " *
                        "HSS=$(round(r.hss, digits=4))  gates=$(r.n)")
                _append_sweep_result!(all_results, summary_file,
                    mp_lo, mp_hi, baseline_infer_hi, r, pass2_model_path)
            end

            # Sweep inference thresholds (fast — reads stored met_prob_pass_2)
            if sweep_inference
                # Check which inference thresholds still need to run
                remaining = [hi for hi in infer_high_grid if (mp_lo, mp_hi, hi) ∉ completed_combos]
                if isempty(remaining)
                    printstyled("    Skipping inference sweep — all thresholds already in summary\n", color=:cyan)
                else
                    if length(remaining) < length(infer_high_grid)
                        printstyled("    Inference sweep ($(length(remaining))/$(length(infer_high_grid)) remaining): ", color=:cyan)
                    else
                        print("    Inference sweep: ")
                    end
                    for infer_hi in remaining
                        infer_probs = [sweep_threshold, (mp_lo, infer_hi)]
                        r = run_evaluation(config, "INFER_SWEEP", testing_path, infer_probs;
                                           skip_existing_met_probs=true, verbose=false)
                        _append_sweep_result!(all_results, summary_file,
                            mp_lo, mp_hi, infer_hi, r, pass2_model_path)
                        print("hi=$(infer_hi)→HSS=$(round(r.hss, digits=4)) ")
                    end
                    println()
                end
            end

            # Free memory between sweep iterations to reduce GC pressure
            GC.gc()
        end

        # Print sorted results
        println("\n", "="^70)
        println("SWEEP RESULTS (sorted by $(secondary_metric))")
        println("="^70)
        sorted = sort(all_results, by = r -> getfield(r, secondary_metric), rev=true)
        for r in sorted
            println("  mask=($(r.met_prob_low), $(r.met_prob_high))  infer_hi=$(r.infer_high)  " *
                    "NMD=$(round(r.nmd_hit_rate, digits=4))  MD=$(round(r.md_hit_rate, digits=4))  " *
                    "HSS=$(round(r.hss, digits=4))  F1=$(round(r.f1, digits=4))  gates=$(r.pass2_gates)")
        end

        # Find best meeting NMD target
        passing = filter(r -> r.nmd_hit_rate >= nmd_target, sorted)
        best = isempty(passing) ? sorted[1] : passing[1]

        if isempty(passing)
            println("\nWARNING: No config met NMD >= $(nmd_target). Showing best $(secondary_metric).")
        end

        println("\n>> BEST: mask=($(best.met_prob_low), $(best.met_prob_high))  infer_hi=$(best.infer_high)")
        println("   NMD=$(round(best.nmd_hit_rate, digits=4))  MD=$(round(best.md_hit_rate, digits=4))  " *
                "HSS=$(round(best.hss, digits=4))  F1=$(round(best.f1, digits=4))")
        println("   Model: $(best.model_path)")
        println("   met_probs_train = [($(best.met_prob_low)f0, $(best.met_prob_high)f0), ...]")
        println("   met_probs_test  = [($(best.met_prob_low)f0, $(best.met_prob_high)f0), ($(best.met_prob_low)f0, $(best.infer_high)f0)]")
        println("\n   Summary written to: $(summary_file)")

        # Restore original config
        config.conv_variables = orig_conv_variables
        config.model_output_paths = orig_model_paths
        config.feature_output_paths = orig_feature_paths
        config.met_probs = orig_met_probs
        config.input_path = orig_input_path
        config.masked_conv_variables = orig_masked_conv_variables
        config.masked_conv_kernel_types = orig_masked_conv_kernel_types
        config.masked_conv_kernel_sizes = orig_masked_conv_kernel_sizes
        config.masked_conv_threshold = orig_masked_conv_threshold
        config.masked_conv_met_prob_field = orig_masked_conv_met_prob_field

        return (results=all_results, best=best)
    end

    """Helper to record a sweep result to both the results vector and the summary file."""
    function _append_sweep_result!(all_results, summary_file,
                                   mp_lo, mp_hi, infer_hi, r, model_path)
        push!(all_results, (
            met_prob_low  = mp_lo,
            met_prob_high = mp_hi,
            infer_high    = infer_hi,
            pass2_gates   = r.n,
            nmd_hit_rate  = r.nmd_hit_rate,
            md_hit_rate   = r.md_hit_rate,
            precision     = r.precision,
            recall        = r.recall,
            f1            = r.f1,
            hss           = r.hss,
            accuracy      = Float32((r.tp + r.tn) / r.n),
            tp = r.tp, fp = r.fp, tn = r.tn, fn = r.fn,
            model_path    = model_path,
        ))
        open(summary_file, "a") do io
            println(io, "  $(rpad(mp_lo, 8)) $(rpad(mp_hi, 8)) $(rpad(infer_hi, 9)) " *
                        "$(rpad(round(r.nmd_hit_rate, digits=6), 9)) " *
                        "$(rpad(round(r.md_hit_rate, digits=6), 9)) " *
                        "$(rpad(round(r.hss, digits=6), 9)) " *
                        "$(rpad(round(r.f1, digits=6), 9)) " *
                        "$(rpad(r.n, 9)) $(model_path)")
        end
    end

    """
        run_hypertuning(config, pass, training_path, testing_path; ...)

    Sweep RF hyperparameters (n_trees, max_depth) for a single pass, evaluating on the
    testing set with AUC-ROC as the primary metric. Features are computed once, then
    the sweep trains and evaluates in-memory for each combination.

    The caller should set up `config.conv_variables`, `config.selected_features`, and
    masked conv fields for the target pass before calling (e.g., via `configure_pass!`).

    Returns `(results=Vector{NamedTuple}, best=NamedTuple, best_model_path=String)`.
    """
    function run_hypertuning(config::ModelConfig, pass::Int,
                              training_path::String, testing_path::String;
                              experiment_name::String = "hypertune",
                              n_trees_grid::Vector{Int} = [11, 21, 51, 101],
                              max_depth_grid::Vector{Int} = [8, 10, 12, 14, 16],
                              met_threshold::Float32 = 0.5f0,
                              skip_existing::Bool = false,
                              compute_test_importance::Bool = true,
                              n_importance_repeats::Int = config.n_importance_repeats,
                              importance_subsample_fraction::Float64 = config.importance_subsample_fraction)

        @assert config.task_mode == "convolution" "run_hypertuning currently requires convolution mode"
        @assert pass >= 1

        # Save original config state
        orig_input_path = config.input_path
        orig_n_trees = config.n_trees
        orig_max_depth = config.max_depth
        orig_write_out = config.write_out
        orig_overwrite = config.overwrite_output

        try
            QC_mask = pass > 1 ? true : config.QC_mask
            mask_name = QC_mask ? config.mask_names[pass] : ""

            # --- Step 1: Compute training features (once) ---
            train_feature_file = "output_features_$(experiment_name)_hypertune_train_pass$(pass).h5"

            if skip_existing && isfile(train_feature_file)
                printstyled("  Loading precomputed training features: $(train_feature_file)\n", color=:cyan)
                X_train, Y_train = h5open(train_feature_file) do f
                    (f["X"][:,:], f["Y"][:,:])
                end
            else
                printstyled("\nCOMPUTING TRAINING FEATURES...\n", color=:green)
                config.input_path = training_path
                config.write_out = true
                config.overwrite_output = true
                isfile(train_feature_file) && rm(train_feature_file)
                X_train, Y_train = calculate_features_conv(config, train_feature_file;
                                                            QC_mask=QC_mask, mask_name=mask_name,
                                                            write_out=true)
                printstyled("  Training features: $(size(X_train, 1)) gates × $(size(X_train, 2)) features\n", color=:cyan)
            end

            # --- Step 2: Compute testing features (once) ---
            test_feature_file = "output_features_$(experiment_name)_hypertune_test_pass$(pass).h5"

            if skip_existing && isfile(test_feature_file)
                printstyled("  Loading precomputed testing features: $(test_feature_file)\n", color=:cyan)
                X_test, Y_test = h5open(test_feature_file) do f
                    (f["X"][:,:], f["Y"][:,:])
                end
            else
                printstyled("\nCOMPUTING TESTING FEATURES...\n", color=:green)
                config.input_path = testing_path
                config.write_out = true
                config.overwrite_output = true
                isfile(test_feature_file) && rm(test_feature_file)
                X_test, Y_test = calculate_features_conv(config, test_feature_file;
                                                          QC_mask=QC_mask, mask_name=mask_name,
                                                          write_out=true)
                printstyled("  Testing features: $(size(X_test, 1)) gates × $(size(X_test, 2)) features\n", color=:cyan)
            end

            Y_train_vec = reshape(Y_train, length(Y_train))
            Y_test_vec = reshape(Y_test, length(Y_test))

            # --- Step 3: Compute balanced class weights (once) ---
            class_weights = Vector{Float32}(fill(0, length(Y_train_vec)))
            weight_dict = compute_balanced_class_weights(Y_train_vec)
            for class in keys(weight_dict)
                class_weights[Y_train_vec .== class] .= weight_dict[class]
            end

            # --- Step 4: Summary file setup ---
            summary_file = "$(experiment_name)_hypertuning_pass$(pass)_summary.txt"
            completed_combos = Set{Tuple{Int,Int}}()
            if skip_existing && isfile(summary_file)
                for line in eachline(summary_file)
                    startswith(line, "#") && continue
                    fields = split(strip(line))
                    length(fields) >= 2 || continue
                    try
                        k = (parse(Int, fields[1]), parse(Int, fields[2]))
                        push!(completed_combos, k)
                    catch; end
                end
                if !isempty(completed_combos)
                    printstyled("  Loaded $(length(completed_combos)) completed results from $(summary_file)\n", color=:cyan)
                end
            else
                open(summary_file, "w") do io
                    println(io, "# RONIN Hypertuning Summary — $(experiment_name) Pass $(pass)")
                    println(io, "# Train: $(size(X_train, 1)) gates, Test: $(size(X_test, 1)) gates, $(size(X_train, 2)) features")
                    println(io, "#")
                    println(io, "# n_trees  max_depth  AUC_ROC    HSS        F1         NMD_HR     MD_HR      Accuracy   Precision  Recall     gates")
                end
            end

            # --- Step 5: Sweep ---
            all_results = NamedTuple[]
            n_combos = length(n_trees_grid) * length(max_depth_grid)
            combo_idx = 0
            best_auc = -Inf
            best_model = nothing
            best_params = (n_trees=0, max_depth=0)

            printstyled("\nSWEEPING $(n_combos) HYPERPARAMETER COMBINATIONS...\n", color=:green)
            sweep_start = time()

            for nt in n_trees_grid, md in max_depth_grid
                combo_idx += 1

                if (nt, md) in completed_combos
                    printstyled("  [$(combo_idx)/$(n_combos)] n_trees=$(nt), max_depth=$(md) — skipped (exists)\n", color=:light_black)
                    continue
                end

                # Train in-memory
                train_start = time()
                model = DecisionTree.RandomForestClassifier(n_trees=nt, max_depth=md, rng=50)
                DecisionTree.fit!(model, X_train, Y_train_vec, class_weights;
                                  max_threads=config.max_training_threads)
                train_time = time() - train_start

                # Predict on test set
                proba = DecisionTree.predict_proba(model, X_test)
                met_probs_pred = Vector{Float32}(proba[:, 2])

                # AUC-ROC
                auc = compute_auc_roc(met_probs_pred, Y_test_vec)

                # Confusion matrix at met_threshold
                predictions = Vector{Bool}(met_probs_pred .>= met_threshold)
                targets = Vector{Bool}(Y_test_vec .== 1)
                prec, recall, f1, tp, fp, tn, fn, n = evaluate_model(predictions, targets)

                md_hit_rate = Float32(tp / max(tp + fn, 1))
                nmd_hit_rate = Float32(tn / max(tn + fp, 1))
                expected = ((tp + fn) * (tp + fp) + (tn + fp) * (tn + fn)) / max(n, 1)
                hss = Float32((tp + tn - expected) / max(n - expected, 1))
                accuracy = Float32((tp + tn) / max(n, 1))

                result = (n_trees=nt, max_depth=md, auc_roc=auc,
                          precision=prec, recall=recall, f1=f1, hss=hss,
                          md_hit_rate=md_hit_rate, nmd_hit_rate=nmd_hit_rate,
                          accuracy=accuracy, gates=n,
                          tp=tp, fp=fp, tn=tn, fn=fn)
                push!(all_results, result)

                # Append to summary file
                open(summary_file, "a") do io
                    println(io, "  $(rpad(nt, 8)) $(rpad(md, 10)) " *
                                "$(rpad(round(auc, digits=6), 10)) " *
                                "$(rpad(round(hss, digits=6), 10)) " *
                                "$(rpad(round(f1, digits=6), 10)) " *
                                "$(rpad(round(nmd_hit_rate, digits=6), 10)) " *
                                "$(rpad(round(md_hit_rate, digits=6), 10)) " *
                                "$(rpad(round(accuracy, digits=6), 10)) " *
                                "$(rpad(round(prec, digits=6), 10)) " *
                                "$(rpad(round(recall, digits=6), 10)) $(n)")
                end

                # Track best model
                if auc > best_auc
                    best_auc = auc
                    best_model = model
                    best_params = (n_trees=nt, max_depth=md)
                end

                printstyled("  [$(combo_idx)/$(n_combos)] n_trees=$(rpad(nt, 4)) max_depth=$(rpad(md, 3)) " *
                            "AUC=$(round(auc, digits=6))  HSS=$(round(hss, digits=4))  " *
                            "F1=$(round(f1, digits=4))  " *
                            "NMD=$(round(nmd_hit_rate, digits=4))  MD=$(round(md_hit_rate, digits=4))  " *
                            "($(round(train_time, digits=1))s)\n", color=:white)
            end

            sweep_time = time() - sweep_start
            printstyled("\nSweep completed in $(round(sweep_time, digits=1))s\n", color=:green)

            # --- Step 6: Print sorted results ---
            if !isempty(all_results)
                sorted = sort(all_results, by = r -> r.auc_roc, rev=true)

                println("\n", "="^90)
                printstyled("HYPERTUNING RESULTS — Pass $(pass) (sorted by AUC-ROC)\n", color=:cyan)
                println("="^90)
                for (rank, r) in enumerate(sorted)
                    marker = rank == 1 ? " ★" : ""
                    println("  $(rpad(rank, 3)) n_trees=$(rpad(r.n_trees, 4)) max_depth=$(rpad(r.max_depth, 3))  " *
                            "AUC=$(rpad(round(r.auc_roc, digits=6), 9)) " *
                            "HSS=$(rpad(round(r.hss, digits=4), 7)) " *
                            "F1=$(rpad(round(r.f1, digits=4), 7)) " *
                            "NMD=$(rpad(round(r.nmd_hit_rate, digits=4), 7)) " *
                            "MD=$(round(r.md_hit_rate, digits=4))$(marker)")
                end
                println("="^90)

                best_result = sorted[1]
                printstyled("\n>> BEST: n_trees=$(best_result.n_trees), max_depth=$(best_result.max_depth), " *
                            "AUC=$(round(best_result.auc_roc, digits=6)), HSS=$(round(best_result.hss, digits=4))\n", color=:green)
            else
                sorted = NamedTuple[]
                best_result = nothing
            end

            # --- Step 7: Test-set feature importance with best model ---
            feat_names = String[]
            importances = Float64[]
            recommended = Int[]
            if compute_test_importance && best_model !== nothing
                printstyled("\nCOMPUTING TEST-SET FEATURE IMPORTANCE " *
                            "(best: n_trees=$(best_params.n_trees), max_depth=$(best_params.max_depth))...\n", color=:green)

                importances = compute_rf_feature_importance(best_model, X_test, Y_test_vec;
                    n_repeats=n_importance_repeats,
                    subsample_fraction=importance_subsample_fraction)

                kernel_bank = build_kernel_bank(config.conv_kernel_sizes)
                masked_conv_kb = if !isempty(config.masked_conv_variables)
                    build_filtered_kernel_bank(config.masked_conv_kernel_types, config.masked_conv_kernel_sizes)
                else
                    ConvolutionKernel[]
                end
                feat_names = build_feature_names(config.conv_variables, kernel_bank;
                    masked_conv_variables=config.masked_conv_variables,
                    masked_conv_kernel_bank=masked_conv_kb,
                    masked_conv_threshold=config.masked_conv_threshold)

                sorted_idx = sortperm(importances, rev=true)
                printstyled("\n  TEST-SET FEATURE IMPORTANCE RANKING:\n", color=:cyan)
                max_imp = maximum(importances)
                for (rank, idx) in enumerate(sorted_idx)
                    name = idx <= length(feat_names) ? feat_names[idx] : "feature_$(idx)"
                    bar_len = max_imp > 0 ? round(Int, importances[idx] / max_imp * 30) : 0
                    bar = repeat("█", max(bar_len, 0))
                    printstyled("    $(lpad(rank, 4)). $(rpad(name, 35)) $(rpad(round(importances[idx], digits=6), 10)) $(bar)\n", color=:cyan)
                end

                recommended = select_features(importances, config.feature_importance_threshold)
                printstyled("\n  RECOMMENDED $(length(recommended))/$(length(importances)) FEATURES " *
                            "(test-set importance, $(config.feature_importance_threshold * 100)% threshold)\n", color=:green)
                printstyled("  selected_features = $(recommended)\n", color=:yellow)
                printstyled("  Copy the line above into your PASS_CONFIG to retrain with this subset.\n", color=:yellow)
            end

            # --- Step 8: Save best model ---
            best_model_path = "trained_model_$(experiment_name)_best_pass$(pass).jld2"
            if best_model !== nothing
                save_kwargs = Dict{Symbol,Any}(
                    :model => best_model,
                    :selected_features => config.selected_features,
                    :conv_variables => config.conv_variables,
                    :masked_conv_variables => config.masked_conv_variables,
                    :masked_conv_kernel_types => config.masked_conv_kernel_types,
                    :masked_conv_kernel_sizes => config.masked_conv_kernel_sizes,
                    :masked_conv_threshold => config.masked_conv_threshold,
                    :masked_conv_met_prob_field => config.masked_conv_met_prob_field,
                    :n_trees => best_params.n_trees,
                    :max_depth => best_params.max_depth,
                    :auc_roc => best_auc,
                )
                if !isempty(importances)
                    save_kwargs[:importances] = importances
                    save_kwargs[:feature_names] = feat_names
                    save_kwargs[:recommended_features] = recommended
                end
                JLD2.jldsave(best_model_path; save_kwargs...)
                printstyled("\n  Saved best model to $(best_model_path)\n", color=:green)
            end

            return (results=all_results,
                    best=isempty(all_results) ? nothing : sorted[1],
                    best_model_path=best_model_path)

        finally
            config.input_path = orig_input_path
            config.n_trees = orig_n_trees
            config.max_depth = orig_max_depth
            config.write_out = orig_write_out
            config.overwrite_output = orig_overwrite
        end
    end

    """
    `QC_scan(input_cfrad::String, features::Matrix{Float32}, indexer::Vector{Bool}, config::ModelConfig, iter::Int64)`

    """
    function QC_scan(input_cfrad::String, features::Matrix{Float32}, indexer::Vector{Bool}, config::ModelConfig, iter::Int64)

        input_set = redirect_stdout(devnull) do
            NCDataset(input_cfrad, "a")
        end
        new_model = load_model(config.model_output_paths[iter], config.task_mode)
        decision_threshold = config.met_probs[iter]
        met_threshold = maximum(decision_threshold)
        cfrad_dims = (input_set.dim["range"], input_set.dim["time"])

        VARIABLES_TO_QC = config.VARS_TO_QC
        met_predictions = DecisionTree.predict_proba(new_model, features)[:, 2]
        predictions = met_predictions .> met_threshold
        starttime=time()

        ##QC each variable in VARIALBES_TO_QC
        for var in VARIABLES_TO_QC

            ##Create new field to reshape QCed field to
            NEW_FIELD = missings(Float32, cfrad_dims)
            ##Only modify relevant data based on indexer, everything else should be fill value
            QCED_FIELDS = input_set[var][:][indexer]

            NEW_FIELD_ATTRS = Dict(
                "units" => input_set[var].attrib["units"],
                "long_name" => "Random Forest Model QC'ed $(var) field"
            )

            ##Set MISSINGS to fill value in current field

            initial_count = count(.!map(ismissing, QCED_FIELDS))
            ##Apply predictions from model
            ##If model predicts 1, this indicates a prediction of meteorological data
            QCED_FIELDS = map(x -> Bool(predictions[x[1]]) ? x[2] : missing, enumerate(QCED_FIELDS))
            final_count = count(.!map(ismissing, QCED_FIELDS))


            ###Need to reconstruct original
            NEW_FIELD = NEW_FIELD[:]
            NEW_FIELD[indexer] = QCED_FIELDS
            NEW_FIELD = reshape(NEW_FIELD, cfrad_dims)


            try
                defVar(input_set, var * config.QC_SUFFIX, NEW_FIELD, ("range", "time"), fillvalue = FILL_VAL; attrib=NEW_FIELD_ATTRS)
            catch e
                ###Simply overwrite the variable
                if e.msg == "NetCDF: String match to name in use"
                    if config.verbose
                        println("Already exists... overwriting")
                    end
                    input_set[var*config.QC_SUFFIX][:,:] = NEW_FIELD
                else
                    throw(e)
                end
            end
            if config.verbose
                println("\r\nCompleted in $(time()-starttime ) seconds")
                println()
                printstyled("REMOVED $(initial_count - final_count) PRESUMED NON-METEORLOGICAL DATAPOINTS\n", color=:green)
                println("FINAL COUNT OF DATAPOINTS IN $(var): $(final_count)")
            end

        end

        close(input_set)

    end

    """
        QC_scan(config::ModelConfig)

    Applies trained composite model to data within scan or set of scans. Will set gates the
    model deems to be non-meteorological to MISSING, including gates that do not meet
    initial basic quality control thresholds. Wrapper around composite_prediction.

    Returns: None


    """
    function QC_scan(config::ModelConfig)

        composite_prediction(config, write_predictions_out=false, QC_mode = true)

    end



    """
        QC_scan(config::ModelConfig, filepath::String, predictions::Vector{Bool}, init_idxer::Vector{Bool})

        Internal function to apply QC to a scan specified by `filepath` using the predictions/indexer specified
        by `predictions` and `init_idxer`. Generally used in the context of a multi-pass model.

        `config::ModelConfig`
    """
    function QC_scan(config::ModelConfig, filepath::String, predictions::Vector{Bool}, init_idxer::Vector{Bool})

        @assert (length(config.model_output_paths) == length(config.feature_output_paths)
                 == length(config.met_probs) == length(config.task_paths) == length(config.task_weights) == length(config.mask_names))

        starttime = time()

        input_set = redirect_stdout(devnull) do
           NCDataset(filepath, "a")
        end

        sweep_dims = (dimsize(input_set["range"]).range, dimsize(input_set["time"]).time)

        for var in config.VARS_TO_QC
            printstyled("QC-ING $(var) in $(filepath)\n", color=:green)
            ##Create new field to reshape QCed field to
            NEW_FIELD = missings(Float32, sweep_dims)

            if predictions != Vector{Bool}(undef, 0)
                ##Only modify relevant data based on indexer, everything else should be fill value
                QCED_FIELDS = input_set[var][:][init_idxer]

                NEW_FIELD_ATTRS = Dict(
                    "units" => input_set[var].attrib["units"],
                    "long_name" => "Random Forest Model QC'ed $(var) field"
                )

                initial_count = count(.!map(ismissing, QCED_FIELDS))
                print("INITIAL COUNT: $(initial_count)")
                ##Apply predictions from model
                ##If model predicts 1, this indicates a prediction of meteorological data
                QCED_FIELDS = map(x -> Bool(predictions[x[1]]) ? x[2] : missing, enumerate(QCED_FIELDS))
                final_count = count(.!map(ismissing, QCED_FIELDS))

                ###Need to reconstruct original
                NEW_FIELD = NEW_FIELD[:]
                NEW_FIELD[init_idxer] = QCED_FIELDS
                NEW_FIELD = Matrix{Union{Missing, Float32}}(reshape(NEW_FIELD, sweep_dims))
            else
                NEW_FIELD = missings(Float32, sweep_dims)
                NEW_FIELD_ATTRS = Dict(
                    "units" => input_set[var].attrib["units"],
                    "long_name" => "Random Forest Model QC'ed $(var) field"
                )
                initial_count = 0
                final_count = 0
            end

            try
                defVar(input_set, var * config.QC_SUFFIX, NEW_FIELD, ("range", "time"), fillvalue = config.FILL_VAL; attrib=NEW_FIELD_ATTRS)
            catch e
                print(e)
                ###Simply overwrite the variable
                if e.msg == "NetCDF: String match to name in use"
                    if config.verbose
                        println("Already exists... overwriting")
                    end
                    input_set[var*config.QC_SUFFIX][:,:] = NEW_FIELD
                    ###Key assumption here is that we'll always have units and fill val
                    ###Rewrite the fill value and attributes as well
                    input_set[var*config.QC_SUFFIX].attrib["long_name"] = NEW_FIELD_ATTRS["long_name"]
                    input_set[var*config.QC_SUFFIX].attrib["units"] = NEW_FIELD_ATTRS["units"]
                    ##Cannot redefine FILL VALUE
                    #input_set[var*config.QC_SUFFIX].attrib["_FillValue"] = config.FILL_VAL
                else
                    throw(e)
                end
            end

            if config.verbose
                println("\r\nCompleted in $(time()-starttime ) seconds")
                println()
                printstyled("REMOVED $(initial_count - final_count) PRESUMED NON-METEORLOGICAL DATAPOINTS\n", color=:green)
                println("FINAL COUNT OF DATAPOINTS IN $(var): $(final_count)")
            end
        end
        close(input_set)
    end





    """
        composite_prediction(config::ModelConfig; write_features_out::Bool=false, feature_outfile::String="placeholder.h5", return_probs::Bool=false)

    Passes feature data through a model or series of models and returns model classifications. Applies configuration such as
    masking and basic QC (high PGG/low NCP) specified by `config`

    ### Optional keyword arguments
    ```
    write_predictions_out::Bool = false
    ```
    If true, will write the predictions to disk

    ```
    prediction_outfile::String = "model_predictions.h5"
    ```
    Location to write predictions to on disk

    ```
    return_probs::Bool = false
    ```
    If set to true, will return probability of meteorological gate for all gates. More detail below.
    ```

    QC_mode::Bool = false
    ```
    If set to true, the function will instead be used to apply quality control to a (set of) scan(s)

    ### Returns

    * `predictions::Vector{Bool}` Model classifications for gates that passed basic quality control thresholds
    * `values::BitVector` Verification gates correspondant to predictions
    * `init_idxers::Vector{Vector{Float32}}` Information about where original radar data did/did not meet basic quality control thresholds.
                                            Each vector contains a flattened vector describing whether or not a given gate was predicted on.
    * `total_met_probs::Vector{Float32}`If kewyword argument return_probs is set to `true`, then `total_met_probs` will be returned. Each entry
                                        into this vector corresponds to the gate represented by predictions and values, and denotes the fraction of
                                        trees in the random forest that classified the gate as meteorological.

         All values returned will be only those that passed quality control checks in the first pass of the model
        minimum NCP / PGG thresholds. In order to reconstruct a scan, user would need to use the values in the returned indexers.
    """
    function composite_prediction(config::ModelConfig; write_predictions_out::Bool = false, prediction_outfile::String="model_predictions.h5", return_probs::Bool=false, QC_mode::Bool=false, skip_existing_met_probs::Bool=false)

        if config.task_mode != "convolution"
            @assert (length(config.model_output_paths) == length(config.feature_output_paths)
                     == length(config.met_probs) == length(config.task_paths) == length(config.task_weights) == length(config.mask_names))
            printstyled("Inference using hand-tuned predictors....\n", color=:green)
            flush(stdout)
        else
            @assert (length(config.model_output_paths) == length(config.feature_output_paths)
                     == length(config.met_probs) == length(config.mask_names))
            printstyled("Inference using convolution mode....\n", color=:green)
            flush(stdout)
        end

        ###Let's get the files
        if isdir(config.input_path)
            files = parse_directory(config.input_path)
        else
            files = [config.input_path]
        end

        predictions = Vector{Bool}(undef, 0)
        values = BitVector(undef, 0)
        total_met_probs = Vector{Float32}(undef, 0)
        init_idxers = Vector{Vector{Float32}}(undef, 0)
        models = []
        model_selected_features = Vector{Vector{Int}}()

        model_conv_variables = Vector{Vector{String}}()
        model_masked_conv_variables = Vector{Vector{String}}()
        model_masked_conv_kernel_types = Vector{Vector{String}}()
        model_masked_conv_kernel_sizes = Vector{Vector{Int}}()
        model_masked_conv_thresholds = Vector{Float32}()
        model_masked_conv_met_prob_fields = Vector{String}()
        for path in config.model_output_paths
            md = load_model_with_metadata(path, config.task_mode)
            push!(models, md.model)
            push!(model_selected_features, md.selected_features)
            push!(model_conv_variables, md.conv_variables)
            push!(model_masked_conv_variables, md.masked_conv_variables)
            push!(model_masked_conv_kernel_types, md.masked_conv_kernel_types)
            push!(model_masked_conv_kernel_sizes, md.masked_conv_kernel_sizes)
            push!(model_masked_conv_thresholds, md.masked_conv_threshold)
            push!(model_masked_conv_met_prob_fields, md.masked_conv_met_prob_field)
        end

        ## Collect per-pass probabilities for summary histogram
        pass_probs = [Float32[] for _ in 1:config.num_models]

        ###Need to do this file by file so that the spatial context of gates is maintained
        ###Probably can section this off into a different function later since it's also reused in the streaming/realtime version
        for file in files
            curr_starttime = time()
                ###Get dimensions

            scan_dims = redirect_stdout(devnull) do

                scan_dims = NCDataset(file) do f
                    (dimsize(f["range"]).range, dimsize(f["time"]).time)
                end

                scan_dims
            end

            ###init_idxer contains the gates that pass the first-level QC checks (NCP, PGG) + inital mask
            init_idxer = Vector{Bool}(undef, 0)
            ###Keep indexer returned by the last pass of the model. This will describe where predictions
            ###are made on the last set of gates
            final_idxer = Vector{Bool}(undef, 0)

            ###Current verification, final predictions, and probabilites
            curr_Y = Vector{Bool}(undef, 0)
            final_predictions = Vector{Bool}(undef, 0)
            curr_probs = fill(-1.0, scan_dims[:])

            ###For multi-pass models, iteratively construct predictions vector by applying models one at a time
            for (i, model_path) in enumerate(config.model_output_paths)

                met_prob_name = "met_prob_pass_$(i)"
                curr_proba = config.met_probs[i]
                met_threshold = maximum(curr_proba)
                nmd_threshold = minimum(curr_proba)

                ## --- Fast path: read saved probabilities instead of recomputing ---
                ## When skip_existing_met_probs=true and met_prob_pass_<i> already exists,
                ## we only need the cheap QC indexer + verification labels, not the
                ## expensive convolution features or RF prediction.
                can_skip_computation = false
                if skip_existing_met_probs
                    can_skip_computation = NCDataset(file) do ds
                        haskey(ds, met_prob_name)
                    end
                end

                if can_skip_computation
                    ## Read saved probabilities and build indexer cheaply
                    f = redirect_stdout(devnull) do
                        NCDataset(file, "r")
                    end

                    ## Reconstruct the QC indexer (cheap — no convolutions)
                    VT = f[config.remove_var][:]
                    indexer = [!ismissing(x) for x in VT]

                    if i > 1
                        mask_name = config.mask_names[i]
                        feature_mask = Matrix{Bool}(.! map(ismissing, f[mask_name]))
                        indexer = [indexer[j] ? feature_mask[:][j] : false for j in eachindex(indexer)]
                    elseif config.QC_mask
                        mask_name = config.mask_names[i]
                        feature_mask = Matrix{Bool}(.! map(ismissing, f[mask_name]))
                        indexer = [indexer[j] ? feature_mask[:][j] : false for j in eachindex(indexer)]
                    end

                    if config.REMOVE_HIGH_PGG
                        PGG = [ismissing(x) || isnan(x) ? Float32(FILL_VAL) : Float32(x) for x in calc_pgg(f)[:]]
                        indexer[indexer] = [x >= config.PGG_THRESHOLD ? false : true for x in PGG[indexer]]
                    end
                    if config.REMOVE_LOW_SIG_QUALITY
                        SIG = [ismissing(x) || isnan(x) ? Float32(FILL_VAL) : Float32(x) for x in calc_sig(f, config.SIG_QUALITY_VAR)[:]]
                        indexer[indexer] = [x <= config.SIG_QUALITY_THRESHOLD ? false : true for x in SIG[indexer]]
                    end

                    ## Read saved met_probs for this pass
                    saved_probs = f[met_prob_name][:]
                    met_probs = Float32[ismissing(x) ? Float32(-1.0) : Float32(x) for x in saved_probs[:]][indexer]

                    ## Build Y if interactive QC
                    if ((!QC_mode) && config.HAS_INTERACTIVE_QC)
                        VG = f[config.QC_var][:][indexer]
                        VV = f[config.remove_var][:][indexer]
                        Y = reshape([ismissing(x) ? 0 : 1 for x in VG .- VV][:], (:, 1))
                    else
                        Y = false
                    end

                    close(f)

                    final_idxer = indexer
                    curr_probs[indexer] .= met_probs[:]

                    append!(pass_probs[i], met_probs)

                    if i == 1
                        init_idxer = copy(indexer)
                        curr_Y = copy(Y)
                        final_predictions = fill(false, sum(indexer))
                        final_predictions[met_probs .< nmd_threshold] .= false
                        final_predictions[met_probs .> met_threshold] .= true
                    elseif i == config.num_models
                        valid_idxs = indexer[init_idxer]
                        curr_preds = final_predictions[valid_idxs]
                        curr_preds[met_probs .>= met_threshold] .= true
                        curr_preds[met_probs .<  nmd_threshold] .= false
                        final_predictions[valid_idxs] .= curr_preds
                    else
                        valid_idxs = indexer[init_idxer]
                        curr_preds = final_predictions[valid_idxs]
                        curr_preds[met_probs .< nmd_threshold] .= false
                        curr_preds[met_probs .> met_threshold] .= true
                        final_predictions[valid_idxs] .= curr_preds
                    end

                    continue
                end

                ## --- Normal path: compute features and run RF prediction ---

                ###REFACTOR NOTES: I THINK PROCESS_SINGLE_FILE CLOSES THE FILE SO WILL NEED TO CHANGE THAT
                ###TO MOVE OUTSIDE LOOP
                ###We don't need to write these out, just use them briefly
                f = redirect_stdout(devnull) do
                    NCDataset(file, "a")
                end

                if i > 1
                    QC_mask = true
                else
                    QC_mask = config.QC_mask
                end

                QC_mask ? mask_name = config.mask_names[i] : mask_name = ""

                if QC_mask
                    feature_mask = Matrix{Bool}(.! map(ismissing, f[mask_name]))
                else
                    feature_mask = [true true; false false]
                end


                ###If there are zero features of interest because they've all been masked out, we're done. Continue to next model, and eventaully to next file
                if sum(feature_mask) == 0
                    break
                end

                ###Need to actually pass the QC mask
                ###indexer will contain true where gates in the file both were NOT masked out AND met the basic QC thresholds

                if config.task_mode == "convolution"
                    kernel_bank = build_kernel_bank(config.conv_kernel_sizes)
                    config_pred = deepcopy(config)
                    config_pred.HAS_INTERACTIVE_QC = ((!QC_mode) && config.HAS_INTERACTIVE_QC)
                    config_pred.selected_features = model_selected_features[i]
                    if !isempty(model_conv_variables[i])
                        config_pred.conv_variables = model_conv_variables[i]
                    end
                    config_pred.masked_conv_variables = model_masked_conv_variables[i]
                    config_pred.masked_conv_kernel_types = model_masked_conv_kernel_types[i]
                    config_pred.masked_conv_kernel_sizes = model_masked_conv_kernel_sizes[i]
                    config_pred.masked_conv_threshold = model_masked_conv_thresholds[i]
                    config_pred.masked_conv_met_prob_field = model_masked_conv_met_prob_fields[i]
                    result = process_single_file_conv(f, config_pred, kernel_bank;
                                                      feature_mask=feature_mask, mask_features=QC_mask)
                    X, Y, indexer = result[1], result[2], result[3]
                else
                    currt = config.task_paths[i]
                    cw = config.task_weights[i]
                    X, Y, indexer = process_single_file(f, currt, HAS_INTERACTIVE_QC = ((! QC_mode) && config.HAS_INTERACTIVE_QC)
                        , REMOVE_HIGH_PGG = config.REMOVE_HIGH_PGG, PGG_THRESHOLD = config.PGG_THRESHOLD,
                        REMOVE_LOW_SIG_QUALITY = config.REMOVE_LOW_SIG_QUALITY, SIG_QUALITY_THRESHOLD = config.SIG_QUALITY_THRESHOLD, SIG_QUALITY_VAR = config.SIG_QUALITY_VAR,
                        QC_variable = config.QC_var, replace_missing = config.replace_missing, remove_variable = config.remove_var,
                        mask_features = QC_mask, feature_mask = feature_mask, weight_matrixes=cw)
                end
                final_idxer = indexer
                ###If there are no gates that meet the basic QC thresholds now, we're once again done.
                if sum(indexer) != 0

                    curr_model = models[i]
                    ###Here's where we need to modify. The ONLY gates that will go on to the next pass
                    ### will be the ones between the thresholds, (inclusive on both ends)

                    met_probs = DecisionTree.predict_proba(curr_model, X)[:, 2]
                    curr_probs[indexer] .= met_probs[:]
                    append!(pass_probs[i], Float32.(met_probs))

                    if i == 1
                        init_idxer = copy(indexer)
                        curr_Y = copy(Y)
                        ###Instantiate prediction vector - the gates that meet the basic thresholds/masking on pass 1 are the ones we want to predict on
                        final_predictions = fill(false, sum(indexer))
                            ###Set gates below predicted threshold to non-met
                        final_predictions[met_probs .< nmd_threshold] .= false
                        final_predictions[met_probs .> met_threshold] .= true

                    elseif i == config.num_models
                        ###Some weird syntax here because Julia doesn't like double indexing
                        ###Grab spots in the scan where the gates were both passing minimum quality control thresholds
                        ###and also have passed previous passes. Do this to ensure dimensional consistency with the
                        ###final prediction vector.
                        valid_idxs = indexer[init_idxer]
                        ###Grab locations in the prediction vector where this pass is being applied.
                        curr_preds = final_predictions[valid_idxs]
                        ###Final pass: just take the model's (majority vote) predictions for the class of the gates and we're done!
                        curr_preds[met_probs .>= met_threshold] .= true
                        curr_preds[met_probs .<  nmd_threshold] .= false
                        ###Reassign
                        final_predictions[valid_idxs] .= curr_preds
                    else
                        ###Indexer has NOT yet been applied so index in to the existing predictions
                        valid_idxs = indexer[init_idxer]
                        ###Grab locations in the prediction vector where this pass is being applied.
                        curr_preds = final_predictions[valid_idxs]
                        curr_preds[met_probs .< nmd_threshold] .= false
                        curr_preds[met_probs .> met_threshold] .= true

                        final_predictions[valid_idxs] .= curr_preds

                    end
                    close(f)
                    ###If this wasn't the last pass, write met_prob and mask for the next pass
                    if i < config.num_models
                        mask_name_next = config.mask_names[i+1]

                        ## Write met_prob_pass_<i> so the next pass can use it as a feature
                        met_prob_field = Matrix{Union{Missing, Float32}}(missings(scan_dims))[:]
                        met_prob_field[indexer] .= met_probs
                        met_prob_field = reshape(met_prob_field, scan_dims)
                        write_field(file, met_prob_name, met_prob_field,
                            attribs=Dict("Units" => "Probability", "Description" => "Meteorological probability from pass $(i)"),
                            fillval=config.FILL_VAL, verbose=false)

                        ## Write mask for gates between thresholds
                        gates_of_interest = (met_probs .>= nmd_threshold) .& (met_probs .<= met_threshold)
                        new_mask = Matrix{Union{Missing, Float32}}(missings(scan_dims))[:]
                        if sum(gates_of_interest) != 0
                            @assert length(gates_of_interest) == sum(indexer)
                            indexer[indexer] .= gates_of_interest
                            new_mask[indexer] .= 1.
                        end
                        new_mask = reshape(new_mask, scan_dims)
                        write_field(file, mask_name_next, new_mask,  attribs=Dict("Units" => "Bool", "Description" => "Gates between met prob thresholds"), fillval=config.FILL_VAL, verbose=false)
                    end

                    ## Save met_prob for final pass too, so threshold sweeps can skip recomputation
                    if i == config.num_models
                        met_prob_field = Matrix{Union{Missing, Float32}}(missings(scan_dims))[:]
                        met_prob_field[indexer] .= met_probs
                        met_prob_field = reshape(met_prob_field, scan_dims)
                        write_field(file, met_prob_name, met_prob_field,
                            attribs=Dict("Units" => "Probability", "Description" => "Meteorological probability from pass $(i)"),
                            fillval=config.FILL_VAL, verbose=false)
                    end
                else
                    ###If the sum of the indexer is zero, we're done. There's nothing to predict upon.
                    ###This will only happen on the first pass of the model, so we won't have to worry about actually making a prediction
                    break
                end


            end


            ###Probably put the below into a separate function for code clarity
            if QC_mode
                QC_scan(config, file, Vector{Bool}(final_predictions), Vector{Bool}(init_idxer))
                if config.verbose
                    printstyled("COMPLETED FULL QC OF $(file) IN $(round((time() - curr_starttime), digits = 2)) SECONDS\n", color=:green)
                end
            else
                if final_predictions != Vector{Bool}(undef, 0)
                    ##Add indexer to the indexer list
                    push!(init_idxers, init_idxer)
                    ###Add verification to full array
                    values = vcat(values, curr_Y)
                    ##We only care about the probabilities where the indexer is
                    total_met_probs = vcat(total_met_probs, curr_probs[:][init_idxer])
                    ##First need to determine the differenc between the initial indexer and the full scan?

                                # ###init_indexer contains the gates in the scan that did not meet the basic quality control thresholds.
                                # ###A space will be needed in the predictions for each positive value here.
                                # ###Difference of final_indxer and init_index contains gates that were marked as non-meteorological throughout the course
                                # ###of applying the composite model. The final prediction then is ONLY on the gates that are still valid
                                # ###in final_idxer
                                # ###We are interested in returning the predictions and the validation for a set of gates
                                # curr_predictions = fill(false, (sum(init_idxer)))
                                # ###The only gates the final pass of the model applied a prediction to will be those where
                                # ###BOTH the final indexer and the initial indexer flagged as valid. Assign the model predictions to these gates.
                                # pred_idxer = (final_idxer[init_idxer] .== true)
                                # curr_predictions[pred_idxer] = final_predictions

                    ###Add on to final predictions
                    ###Prediction vector has been interatively constructed so will comport with the verification
                    predictions = vcat(predictions, final_predictions)
                end
            end

        end


        if ! QC_mode && predictions == Vector{Bool}(undef, 0)
            throw("ERROR: NO GATES IN INPUT DATASET MET BASIC QC THRESHOLDS")
        end

        ## Print per-pass probability histogram
        if !QC_mode
            for i in 1:config.num_models
                probs = pass_probs[i]
                if isempty(probs)
                    printstyled("\n  Pass $(i): no gates processed\n", color=:yellow)
                    continue
                end
                n = length(probs)
                curr_proba = config.met_probs[i]
                lo = minimum(curr_proba)
                hi = maximum(curr_proba)

                ## 20 bins from 0.0 to 1.0
                n_bins = 20
                bin_edges = range(0.0f0, 1.0f0, length=n_bins+1)
                counts = zeros(Int, n_bins)
                for p in probs
                    bin = clamp(floor(Int, p * n_bins) + 1, 1, n_bins)
                    counts[bin] += 1
                end
                max_count = maximum(counts)
                bar_width = 40

                printstyled("\n  Pass $(i) met_prob distribution ($(n) gates, thresholds=$(lo), $(hi)):\n", color=:cyan)
                for b in 1:n_bins
                    lo_edge = bin_edges[b]
                    hi_edge = bin_edges[b+1]
                    bar_len = max_count > 0 ? round(Int, counts[b] / max_count * bar_width) : 0
                    pct = round(counts[b] / n * 100, digits=1)

                    ## Mark threshold boundaries
                    marker = ""
                    if lo_edge < lo <= hi_edge
                        marker = " ← NMD threshold"
                    elseif lo_edge < hi <= hi_edge
                        marker = " ← MD threshold"
                    end

                    bar = "█"^bar_len
                    printstyled("    $(lpad(string(round(lo_edge, digits=2)), 4))–$(rpad(string(round(hi_edge, digits=2)), 4)) ", color=:light_black)
                    printstyled("│$(bar)", color= lo_edge >= hi ? :green : hi_edge <= lo ? :red : :yellow)
                    printstyled(" $(counts[b]) ($(pct)%)$(marker)\n", color=:light_black)
                end
                n_md  = sum(p -> p >= hi, probs)
                n_nmd = sum(p -> p < lo, probs)
                n_unc = n - n_md - n_nmd
                printstyled("    Summary: $(n_nmd) NMD ($(round(n_nmd/n*100, digits=1))%), " *
                            "$(n_unc) uncertain ($(round(n_unc/n*100, digits=1))%), " *
                            "$(n_md) MD ($(round(n_md/n*100, digits=1))%)\n", color=:cyan)
            end
        end

        if write_predictions_out
            h5open(prediction_outfile, "w") do f
                write_dataset(f, "Predictions", predictions)
                write_dataset(f, "Verification", values)
            end
        end

        if return_probs
            return(predictions, values, init_idxers, total_met_probs, pass_probs)
        elseif QC_mode
            return
        else
            return(predictions, values, init_idxers, pass_probs)
        end

    end


    """
        get_contingency(predictions::Vector{Bool}, verificaiton::Vector{Bool}; normalize::Bool = true)

        Utility to return a DataFrame with the contingency matrix for a binary classificaiton model.
        ## Required Arguments
        * `predictions::Vector{Bool}` Model predicted classes
        * `verification::Vector{Bool}` Ground Truth Classes
        ## Optional Arguments
        * `normalize::Bool = true` Whether or not to return the normalized form of the contingency matrix
        ## Return
        * `DataFrame` containing contingency matrix
    """
    function get_contingency(predictions::Vector{Bool}, verification::Vector{Bool}; normalize::Bool = true)

        tpc = count(verification[predictions .== 1] .== 1)
        tnc = count(verification[predictions .== 0] .== 0)

        fpc = count(verification[predictions .== 1] .== 0)
        fnc = count(verification[predictions .== 0] .== 1)

        row_names = ["Predicted Meteorological", "Predicted Non-Meteorological"]
        col_names = ["", "True Meteorological", "True Non-Meteorological"]

        true_met = [tpc, fnc]
        true_non = [fpc, tnc]

        if normalize
            true_met = [round(x / sum(true_met), digits=3) for x in true_met]
            true_non = [round(x / sum(true_non), digits=3) for x in true_non]
        end

        return(DataFrame(col_names[1] => row_names, col_names[2] => true_met, col_names[3] => true_non))
    end



    function get_idxer(model_config::ModelConfig, low_threshold::Float32, high_threshold::Float32)

        low_predictions, low_verification, low_idxrs, low_probs, _ = composite_prediction(model_config, return_probs=true)
        pred_idxer = Vector{Bool}((low_probs .> low_threshold) .& (low_probs .< high_threshold))
        return(low_predictions, low_verification, low_idxrs, pred_idxer)

    end



    """
        write_field(filepath::String, fieldname::String, NEW_FIELD, overwrite::Bool = true, attribs::Dict = Dict(), dim_names::Tuple = ("range", "time"), verbose::Bool=true)
        Helper function to write/overwrite a 2D field to a netCDF file
        ## Required arguments
        * `filepath::String` Name of netCDF file to write data to
        * `fieldname::String` What to call the data in the netCDF
        * `NEW_FIELD` Data dimensioned by `dim_names` to write to netCDF

    """

    function write_field(filepath::String, fieldname::String, NEW_FIELD; overwrite::Bool = true,
                attribs::Dict = Dict(), dim_names::Tuple=("range", "time"), verbose::Bool=true, fillval::T = FILL_VAL) where T <: Real



        if ! isfile(filepath)
            ds = NCDataset(filepath, "c")
            close(ds)
        end

        input_set = redirect_stdout(devnull) do
            NCDataset(filepath, "a")
        end


        try
            defVar(input_set, fieldname, NEW_FIELD, dim_names, fillvalue = fillval; attrib=attribs)
        catch e
            ###If the variable already exists and overwrite is set, simply overwrite it
            if isa(e, NCDatasets.NetCDFError) && e.msg == "NetCDF: String match to name in use" && overwrite
                if verbose
                    printstyled("$(fieldname) already exists in $(filepath)... overwriting\n", color=:yellow)
                end
                input_set[fieldname][:,:] = NEW_FIELD

                if attribs != Dict("" => "")
                    for key in keys(attribs)
                        if key == "_FillValue"
                            continue
                        end
                        input_set[fieldname].attrib[key] = attribs[key]
                    end
                end
            else
                throw(e)
            end
        finally
            close(input_set)
        end

    end



    """
        `evaluate_model(predictions::Vector{Bool}, targets::Vector{Bool})`

        Given a vector of predictions and targets, calculates various scores and returns them in the order of

        * `prec_score::Float32` -> Precision Score, defined as number of true positives divided by sum of true positives and false positives
        * `recall::Float32` Recall, defined as number of true positives divided by sum of true positives and false negatives
        * `f1::Float32` F1 score
        * `true_positives::Int` Number of true positives
        * `false_positives::Int` Number of false positives
        * `true_negatives::Int` Number of true negatives
        * `false_negatives::Int` Number of false negatives
        * `num_gates::INt` Total number of classifications

    """
    function evaluate_model(predictions::Vector{Bool}, targets::Vector{Bool})

        tp_idx = (predictions .== 1) .& (targets .==1)
        fp_idx = (predictions .== 1) .& (targets .==0)

        tn_idx = (predictions .== 0) .& (targets .==0)
        fn_idx = (predictions .== 0) .& (targets .==1)

        prec = Float32(sum(tp_idx) / (sum(tp_idx) + sum(fp_idx)))
        recall = Float32(sum(tp_idx) / (sum(tp_idx) + sum(fn_idx)))

        f1 = Float32((2 * prec * recall) / (prec + recall))

        return(prec, recall, f1, sum(tp_idx), sum(fp_idx), sum(tn_idx), sum(fn_idx), length(predictions))
    end


    """
        compute_auc_roc(probabilities::Vector{Float32}, labels::Vector{<:Integer})

    Compute Area Under the ROC Curve from predicted probabilities and binary labels.
    Uses the trapezoidal rule on (FPR, TPR) points at each distinct threshold.

    Labels should be 1 (positive/meteorological) or 0 (negative/non-meteorological).
    Returns AUC as Float64 in [0, 1], or NaN for empty input, 0.5 for single-class input.
    """
    function compute_auc_roc(probabilities::Vector{Float32}, labels::Vector{<:Integer})::Float64
        n = length(probabilities)
        n == 0 && return NaN
        n_pos = sum(labels .== 1)
        n_neg = n - n_pos
        (n_pos == 0 || n_neg == 0) && return 0.5

        sorted_idx = sortperm(probabilities, rev=true)

        auc = 0.0
        tp = 0
        fp = 0
        prev_fpr = 0.0
        prev_tpr = 0.0
        prev_prob = Inf32

        for i in 1:n
            idx = sorted_idx[i]
            prob = probabilities[idx]

            if prob != prev_prob && i > 1
                fpr = fp / n_neg
                tpr = tp / n_pos
                auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0
                prev_fpr = fpr
                prev_tpr = tpr
            end
            prev_prob = prob

            if labels[idx] == 1
                tp += 1
            else
                fp += 1
            end
        end

        # Final point
        fpr = fp / n_neg
        tpr = tp / n_pos
        auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0

        return auc
    end

    """
        `evaluate_model(config::ModelConfig)`

        Returns a row of a DataFrame with a variety of metrics about a given model.

        #Arguments

        ```julia
        config::ModelConfig
        ```
        Struct containing information about model training

        ```julia
        models_trained::Bool = false
        ```

    """
    function evaluate_model(config::ModelConfig; models_trained::Bool = false, features_calculated::Bool=false)


        ###This function will not handle the case where the model is trained but the features are not written.
        ###it also implicitly assumes that the features will be written out.

        ##Return dataframe with model configuration charactersitics as well as
        ##Things we want to have here: task paths

        if ! config.write_out
            throw("Error: evaluate_model must write features to disk. Please set config.write_out to true")
        end

        if ! models_trained
            train_multi_model(config)
        elseif ! features_calculated
            for (i, output_path) in enumerate(config.model_output_paths)
                construct_next_pass_features(config, i)
            end
        end

        ###Now, use the calculated features to get the predictions.

        models = [load_model(model, config.task_mode) for model in config.model_output_paths]

        predictions = Vector{Bool}(undef, 0)
        targets = Vector{Bool}(undef, 0)

        ###Eventually want to move this into a function to ensure that the code is exactly the same between the different versions
        ###of the functions used to apply predictions.
        for (i, model) in enumerate(models)

            currf = h5open(config.feature_output_paths[i])
            curr_features = currf["X"][:,:]
            curr_targets = Vector{Bool}(currf["Y"][:,:][:])
            close(currf)

            met_probs = DecisionTree.predict_proba(model, curr_features)[:,2]

            if i == length(models)
                ###If this is the last model in the chain, by convention, gates that are at or above the maximum probability listed
                ###for this pass of the model will be classified as meteorological. Everything else will be classified as
                ###non-meteorological
                thresh = maximum(config.met_probs[i])
                preds = met_probs .>= thresh
                predictions = cat(predictions, preds, dims=1)
                targets = cat(targets, curr_targets, dims=1)
            else
                ###if this isn't the last pass, some indexing needs to be done to ensure that we're looking at the correct gates
                ###and that certain gates are not double counted. The gates that this model will be used upon will be
                ###non-meteorological: < minimum threshold
                ###meteorological: > maximum threshold
                min_t = minimum(config.met_probs[i])
                max_t = maximum(config.met_probs[i])

                idxer = (met_probs .< min_t) .| (met_probs .> max_t)
                preds = met_probs[idxer] .> max_t

                predictions = cat(predictions, preds, dims=1)
                targets = cat(targets, curr_targets[idxer], dims=1)
            end

        end

        ###Returns precision, recall, f1, n true_postives, n false_positives, n true_negatives, n false_negatives
        scores = evaluate_model(Vector{Bool}(predictions), Vector{Bool}(targets))

        retval = DataFrame(
                                met_probs = [config.met_probs],
                                task_paths = [config.task_paths],
                                class_weights = [config.class_weights],
                                n_trees = [config.n_trees],
                                max_depth = [config.max_depth],
                                precision = scores[1],
                                recall = scores[2],
                                f1 = scores[3],
                                true_positives = scores[4],
                                false_positives = scores[5],
                                true_negatives = scores[6],
                                false_negatives = scores[7],
                                MD_retained_frac = scores[4] / (scores[4] + scores[7]),
                                NMD_removed_frac = scores[6] / (scores[6] + scores[5])
        )

        return retval
    end



    """
    `construct_next_pass_features(config::ModelConfig)`
    Function used to iteratively calculate the input features for a multi-pass model. Operates on a sweep-by-sweep basis by taking in
    some set of gates, calculating features on the gates, applying a pre-trained model, and finally determining which gates are between
    the specified thresholds (inclusive on both ends) so that they can be passed on to the next model.

    """
    function construct_next_pass_features(config::ModelConfig, curr_model_num::Int; write_out::Bool=true)

        ##If this was the last pass, we don't need to write out a mask, and we're done!
                ###Otherwise, we need to mask out the features we want to apply the model to on the next pass
        @assert curr_model_num <= config.num_models

        ###If this is the 0th, we're just constructing the features for the first pass and don't need to do much other work
        ###This is basically useful for when we don't have a trained model and want to just get the initial set of features
        if curr_model_num > 0
            curr_model = load_model(config.model_output_paths[curr_model_num], config.task_mode)
            curr_metprobs = config.met_probs[curr_model_num]
            curr_tasks = config.task_paths[curr_model_num]
            curr_weights = config.task_weights[curr_model_num]
            curr_out = config.feature_output_paths[curr_model_num]
            output_cols = get_num_tasks(curr_tasks)
        else
            curr_model = ""
            curr_metprobs = ""
            curr_tasks = config.task_paths[begin]
            curr_weights = config.task_weights[begin]
            curr_out = config.feature_output_paths[begin]
            output_cols = get_num_tasks(curr_tasks)
        end

        paths = Vector{String}()
        file_path = config.input_path

        ##If execution proceeds past the first iteration, a composite model is being created, and
        ##so a further mask will be applied to the features
        if curr_model_num > 1
            QC_mask = true
        else
            QC_mask = config.QC_mask
        end

        QC_mask ? mask_name = config.mask_names[curr_model_num] : mask_name = ""

        if isdir(file_path)
            paths = parse_directory(file_path)
        else
            paths = [file_path]
        end



        newX = X = Matrix{Float32}(undef,0,output_cols)
        newY = Y = Matrix{Int64}(undef, 0,1)
        idxs = Vector{}(undef,0)

        for path in paths

            dims = Dataset(path) do f
                (f.dim["range"], f.dim["time"])
            end

            ###NEED to update this if it's beyond two pass so we can pass it the correct mask
            newX, newY, curr_idx = calculate_features(path, curr_tasks, curr_out, true;
                                verbose = config.verbose,
                                REMOVE_LOW_SIG_QUALITY = config.REMOVE_LOW_SIG_QUALITY, SIG_QUALITY_THRESHOLD = config.SIG_QUALITY_THRESHOLD, SIG_QUALITY_VAR = config.SIG_QUALITY_VAR,
                                REMOVE_HIGH_PGG=config.REMOVE_HIGH_PGG, PGG_THRESHOLD = config.PGG_THRESHOLD, QC_variable = config.QC_var,
                                remove_variable = config.remove_var, replace_missing = config.replace_missing, return_idxer=true,
                                write_out = false, QC_mask = QC_mask, mask_name = mask_name, weight_matrixes=curr_weights)

            if (curr_model_num < config.num_models) && (curr_model_num > 0)

                new_mask = Matrix{Union{Missing, Float32}}(missings(dims))[:]

                if (sum(vec(curr_idx[1])) > 0)
                    met_probs = DecisionTree.predict_proba(curr_model, newX)[:, 2]
                    ###Probabilities inclusive on both ends
                    valid_idxs = (met_probs .>= minimum(curr_metprobs)) .& (met_probs .<= maximum(curr_metprobs))
                    ##Create mask field, fill it, and then write out
                    ##We only care about gates that have met the base QC thresholds, so first index
                    ##by indexer returned from calculate_features, and then set the gates between
                    ##the specified probability levels to valid in the mask. The next model pass will
                    ##thus only be calculated upon these features.
                    idxer = curr_idx[1][:]
                    idxer[idxer] .= Vector{Bool}(valid_idxs)
                    new_mask[idxer] .= 1.
                end
                new_mask = reshape(new_mask, dims)
                write_field(path, config.mask_names[curr_model_num+1], new_mask, attribs=Dict("Units" => "Bool", "Description" => "Gates between met prob thresholds"))
            end

            X = vcat(X, newX)::Matrix{Float32}
            Y = vcat(Y, newY)::Matrix{Int64}

        end

        ##Write broader pass features to disk
        if write_out

            if size(Y)[1] == 0
                throw("Error in concstruct_next_pass_features. No gates met thresholds of current sweep.")
            end

            println("OUTPUTTING DATA IN HDF5 FORMAT TO FILE: $(curr_out)")
            fid = h5open(curr_out, "w")

            ###Add information to output h5 file
            attributes(fid)["Parameters"] = get_task_params(curr_tasks)
            attributes(fid)["MISSING_FILL_VALUE"] = config.FILL_VAL
            println()
            println("WRITING DATA TO FILE OF SHAPE $(size(X))")
            println("X TYPE: $(typeof(X))")

            write_dataset(fid, "X", X)
            write_dataset(fid, "Y", Y)
            close(fid)
        end

    end



    """
    `characterize_misclassified_gates(config::ModelConfig; model_pretrained::Bool = true, features_precalculated::Bool = true)`

    Function used to apply composite model to a set of gates, returning information about gate classifications and their associated input features

    ## Required inputs

    ```julia
        config::ModelConfig
    ```
    Model configuration object containing setup information.

    ## Optional Inputs

    ```julia
    model_pretrained::Bool = true
    ```

    Model training in this function not currently implemented, setting to false with untrained models will result in errors.

    ```julia
    features_precalculated::Bool = true
    ```
    Whether or not the input features for the model have already been written to disk.

    Not currently implemented.

    ##

    ## Returns
    Vector of dataframes (one DataFrame for each model "pass"). DataFrames will only contain information about gates reciving their final classification
    during that pass of the model. That is, if a gate exceeds the `met_probs` thresholds and is not passed on to the next pass, it will be represented in the
    DataFrame corresponding to that present pass of the model.
    """
    function characterize_misclassified_gates(config::ModelConfig; model_pretrained::Bool = true, features_precalculated::Bool = true)
        ###Output features

        ###Issue here is that we will need to feed-forward the predictions to properly calculate features.
        if ! features_precalculated
            for (i, output_path) in enumerate(config.model_output_paths)
                construct_next_pass_features(config, i)
            end
        end



        ###In the simplest case, the model is already pretrained and the features have been calculated. Thus,
        ###predict with the model


        ###Key will be figuring out which gates are predicted on in each pass.
        ###Use these to hold the features and successful or unsuccessful predictions
        accuracy = Vector{Bool}[]
        features = Matrix{Float32}(undef, 0, 0)
        pass_no = Vector{Bool}[]
        ret = Dict{Int, DataFrame}()

        for (i, model) in enumerate(config.model_output_paths)

            if i < config.num_models

                currmodel = load_model(model, config.task_mode)
                ###IMPORTANT: FOR THE INDEXING HERE, WE PROBABLY DON'T EVEN NEED TO DO THE COMPARISON ON THE PREDICTIONS.
                ###NEXT PASS SHOULD ALREADY BE WRITTEN TO A MASK
                input_data = h5open(config.feature_output_paths[i])
                currfeatures = input_data["X"][:,:]
                currtargets  = input_data["Y"][:,:][:]
                curr_thresh = config.met_probs[i]

                println("PASS: $(i), INPUT DATA LOCATED AT : $(input_data), PREDICTING ON $(size(currfeatures))")

                met_probs = DecisionTree.predict_proba(currmodel, currfeatures)[:, 2]

                ###Locations where the probability is greater than max prob (classified as meteorological)
                ###Or less than/equal to minimum probability (classified as non-meteorological)
                curr_idxer = (met_probs .< minimum(curr_thresh) ) .|| (met_probs .> maximum(curr_thresh))


                predictions = met_probs[curr_idxer] .> .5
                verif = predictions .== currtargets[curr_idxer]
                features_of_interest = currfeatures[curr_idxer, :]
                feature_names = attrs(input_data)["Parameters"]

                close(input_data)

                df = DataFrame(features_of_interest, feature_names; makeunique=true)
                df[:, "VERIFICATION"] = verif
                df[:, "MET_PROBS"] = met_probs[curr_idxer]

                ret[i] = df

            else ###last model in the chain so we don't need to do any indexing

                currmodel = load_model(model, config.task_mode)

                input_data = h5open(config.feature_output_paths[i])
                currfeatures = input_data["X"][:,:]
                currtargets  = input_data["Y"][:,:][:]
                curr_thresh = config.met_probs[i]
                println("PASS: $(i), INPUT DATA LOCATED AT : $(input_data), PREDICTING ON $(size(currfeatures))")

                met_probs = DecisionTree.predict_proba(currmodel, currfeatures)[:, 2]
                predictions = met_probs .> .5
                verif = predictions .== currtargets

                feature_names = attrs(input_data)["Parameters"]

                close(input_data)

                df = DataFrame(currfeatures, feature_names; makeunique=true)
                df[:, "VERIFICATION"] = verif
                df[:, "MET_PROBS"] = met_probs

                ret[i] = df
            end

        end

        ret

    end



    """
    Streaming flavor of composite prediction used to
    QC sweeps in a realtime setup.
    """
    function composite_QC(config::ModelConfig,
                                        files::Vector{String}, models::Vector{Ronin.DecisionTree.RandomForestClassifier})

        for file in files

                if isdir(file)
                    continue
                end
                ###Get dimensions
                scan_dims = NCDataset(file) do f
                    (dimsize(f["range"]).range, dimsize(f["time"]).time)
                end

                ###init_idxer contains the gates that pass the first-level QC checks (NCP, PGG) + inital mask
                init_idxer = Vector{Bool}(undef, 0)
                ###Keep indexer returned by the last pass of the model. This will describe where predictions
                ###are made on the last set of gates
                final_idxer = Vector{Bool}(undef, 0)

                ###Current verification, final predictions, and probabilites
                curr_Y = Vector{Bool}(undef, 0)
                final_predictions = Vector{Bool}(undef, 0)
                curr_probs = fill(-1.0, scan_dims[:])

                ###For multi-pass models, iteratively construct predictions vector by applying models one at a time
                # Initialize QC mask as the original mask specified in the config. This will be updated to be the new mask after each pass of the model, but we need to start with the original mask for the first pass.
                QC_mask = config.QC_mask
                for (i, model_path) in enumerate(config.model_output_paths)


                    currt = config.task_paths[i]
                    cw = config.task_weights[i]
                    ###REFACTOR NOTES: I THINK PROCESS_SINGLE_FILE CLOSES THE FILE SO WILL NEED TO CHANGE THAT
                    ###TO MOVE OUTSIDE LOOP
                    ###We don't need to write these out, just use them briefly
                    f = redirect_stdout(devnull) do
                        NCDataset(file, "a")
                    end

                    if i > 1
                        QC_mask = true
                    else
                        QC_mask = config.QC_mask
                    end

                    QC_mask ? mask_name = config.mask_names[i] : mask_name = ""

                    if QC_mask
                        feature_mask = Matrix{Bool}(.! map(ismissing, f[mask_name]))
                    else
                        feature_mask = [true true; false false]
                    end


                    ###If there are zero features of interest because they've all been masked out, we're done. Continue to next model, and eventaully to next file
                    if sum(feature_mask) == 0
                        break
                    end

                    ###Need to actually pass the QC mask
                    ###indexer will contain true where gates in the file both were NOT masked out AND met the basic QC thresholds
                    X, Y, indexer = process_single_file(f, currt, HAS_INTERACTIVE_QC = ((! true) && config.HAS_INTERACTIVE_QC)
                        , REMOVE_HIGH_PGG = config.REMOVE_HIGH_PGG, PGG_THRESHOLD = config.PGG_THRESHOLD,
                        REMOVE_LOW_SIG_QUALITY = config.REMOVE_LOW_SIG_QUALITY, SIG_QUALITY_THRESHOLD = config.SIG_QUALITY_THRESHOLD, SIG_QUALITY_VAR = config.SIG_QUALITY_VAR,
                        QC_variable = config.QC_var, replace_missing = config.replace_missing, remove_variable = config.remove_var,
                        mask_features = QC_mask, feature_mask = feature_mask, weight_matrixes=cw)
                    final_idxer = indexer

                    ###If there are no gates that meet the basic QC thresholds now, we're once again done.
                    if sum(indexer) != 0

                        curr_model = models[i]
                        curr_proba = config.met_probs[i]
                        ###Here's where we need to modify. The ONLY gates that will go on to the next pass
                        ### will be the ones between the thresholds, (inclusive on both ends)

                        met_probs = DecisionTree.predict_proba(curr_model, X)[:, 2]
                        curr_probs[indexer] .= met_probs[:]

                        met_threshold = maximum(curr_proba)
                        nmd_threshold = minimum(curr_proba)

                        if i == 1
                            init_idxer = copy(indexer)
                            curr_Y = copy(Y)
                            ###Instantiate prediction vector - the gates that meet the basic thresholds/masking on pass 1 are the ones we want to predict on
                            final_predictions = fill(false, sum(indexer))
                                ###Set gates below predicted threshold to non-met
                            final_predictions[met_probs .< nmd_threshold] .= false
                            final_predictions[met_probs .> met_threshold] .= true

                        elseif i == config.num_models
                            ###Some weird syntax here because Julia doesn't like double indexing
                            ###Grab spots in the scan where the gates were both passing minimum quality control thresholds
                            ###and also have passed previous passes. Do this to ensure dimensional consistency with the
                            ###final prediction vector.
                            valid_idxs = indexer[init_idxer]
                            ###Grab locations in the prediction vector where this pass is being applied.
                            curr_preds = final_predictions[valid_idxs]
                            ###Final pass: just take the model's (majority vote) predictions for the class of the gates and we're done!
                            curr_preds[met_probs .>= met_threshold] .= true
                            curr_preds[met_probs .<  nmd_threshold] .= false
                            ###Reassign
                            final_predictions[valid_idxs] .= curr_preds
                        else
                            ###Indexer has NOT yet been applied so index in to the existing predictions
                            valid_idxs = indexer[init_idxer]
                            ###Grab locations in the prediction vector where this pass is being applied.
                            curr_preds = final_predictions[valid_idxs]
                            curr_preds[met_probs .< nmd_threshold] .= false
                            curr_preds[met_probs .> met_threshold] .= true

                            final_predictions[valid_idxs] .= curr_preds

                        end
                        close(f)
                        ###Probably need to remove this for speed purposes... keep it in memory,
                        ###clear it for the next scan. Just pass it to QC_mask
                        ###If this wasn't the last pass, need to write a mask for the gates to be predicted upon in the next iteration
                        if i < config.num_models
                            gates_of_interest = (met_probs .>= nmd_threshold) .& (met_probs .<= met_threshold)
                            new_mask = Matrix{Union{Missing, Float32}}(missings(scan_dims))[:]
                            ###If there are no gates of interest, write out the mask as ALL MISSINGS
                            ###Otherwise, fill in the gates of interest with 1's
                            if sum(gates_of_interest) != 0
                                @assert length(gates_of_interest) == sum(indexer)
                                indexer[indexer] .= gates_of_interest
                                new_mask[indexer] .= 1.
                            end
                            new_mask = reshape(new_mask, scan_dims)
                            write_field(file, config.mask_names[i+1], new_mask,  attribs=Dict("Units" => "Bool", "Description" => "Gates between met prob thresholds"), fillval=config.FILL_VAL)
                         end
                    else
                        ###If the sum of the indexer is zero, we're done. There's nothing to predict upon.
                        ###This will only happen on the first pass of the model, so we won't have to worry about actually making a prediction
                        break
                    end


                end


                ###Probably put the below into a separate function for code clarity
                QC_scan(config, file, Vector{Bool}(final_predictions), Vector{Bool}(init_idxer))
        end
    end




end
