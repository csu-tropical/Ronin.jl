module Ronin

    include("./RoninFeatures.jl") 
    include("./Io.jl")
    include("./DecisionTree/DecisionTree.jl")
 

    using NCDatasets
    using ImageFiltering
    using Statistics
    using Images
    using Missings
    using BenchmarkTools
    using HDF5 
    using MLJ, MLJLinearModels, CategoricalArrays
    using DataFrames
    using JLD2
    using DataStructures

    
    export get_NCP, airborne_ht, prob_groundgate
    export calc_avg, calc_std, calc_iso, process_single_file 
    export parse_directory, get_num_tasks, get_task_params, remove_validation 
    export calculate_features
    export split_training_testing! 
    export QC_scan, get_QC_mask
    export evaluate_model, get_feature_importance, error_characteristics
    export train_multi_model, ModelConfig, composite_prediction, get_contingency, compute_balanced_class_weights
    export multipass_uncertain, write_field, characterize_misclassified_gates



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
    REMOVE_LOW_NCP::Bool = true
    ```
    Whether to automatically remove gates that do not meet a basic NCP threshold 

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
    NCP_THRESHOLD::Float32 = .2
    ```
    If REMOVE_LOW_NCP is set to true, threshold at or below which to remove data. 

    ```julia 
    PGG_THRESHOLD::Float32 = 1.
    ```
    If REMOVE_HIGH_PGG is set to true, threshold at or above which to remove data. 

    """
    Base.@kwdef mutable struct ModelConfig

        num_models::Int64
        model_output_paths::Vector{String}
        met_probs::Vector{Tuple{Float32, Float32}} 

        feature_output_paths::Vector{String} 
        
        input_path::String 

        task_mode::String 

        file_preprocessed::Vector{Bool} 

        task_paths::Vector{String} = [""]
        task_list::Vector{String} = [""]
        task_weights::Vector{Vector{Matrix{Union{Float32, Missing}}}} = [[Matrix{Union{Float32, Missing}}(undef, 0,0)]]

        verbose::Bool = true 
        REMOVE_LOW_NCP::Bool = true 
        REMOVE_HIGH_PGG::Bool = true 
        HAS_INTERACTIVE_QC::Bool = false 
        QC_var::String = "VG"
        remove_var::String = "VV" 
        FILL_VAL::Float32 = FILL_VAL 
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

        NCP_THRESHOLD::Float32 = .2f0
        PGG_THRESHOLD::Float32 = 1.f0  
        
    end 

    """ 
    Helper function to compute balanced weights according to the algoirthm described in https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html

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
    REMOVE_LOW_NCP::Bool=false
    ```

    If true, will ignore gates with Normalized Coherent Power/Signal Quality Index below a threshold specified in RQCFeatures.jl

    ```julia 
    NCP_THRESHOLD::Float32 = .2
    ```
    Theshold at or below which to remove data 

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
        verbose::Bool=false, REMOVE_LOW_NCP::Bool = false, NCP_THRESHOLD::Float32 = .2f0, REMOVE_HIGH_PGG::Bool = false, PGG_THRESHOLD::Float32=1.f0,
         QC_variable::String = "VG", remove_variable::String = "VV", 
        replace_missing::Bool = false, write_out::Bool=true, QC_mask::Bool = false, mask_name::String = "", return_idxer::Bool=false, 
        weight_matrixes::Vector{Matrix{Union{Missing, Float32}}}= [Matrix{Union{Missing, Float32}}(undef, 0,0)])

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
            try 
                cfrad = Dataset(path) 
                pathstarttime=time() 
                dims = (cfrad.dim["range"], cfrad.dim["time"])

                if QC_mask
                    ###We wish to calculate features on where the mask is NON MISSING 
                    currmask = Matrix{Bool}(.! map(ismissing, cfrad[mask_name][:,:]))
                    (newX, newY, newIdx) = process_single_file(cfrad, argument_file; 
                                                HAS_INTERACTIVE_QC = HAS_INTERACTIVE_QC, REMOVE_LOW_NCP = REMOVE_LOW_NCP, NCP_THRESHOLD = NCP_THRESHOLD, 
                                                REMOVE_HIGH_PGG = REMOVE_HIGH_PGG, PGG_THRESHOLD=PGG_THRESHOLD, QC_variable = QC_variable, remove_variable = remove_variable, 
                                                replace_missing=replace_missing, feature_mask = currmask, mask_features = true, weight_matrixes=weight_matrixes)
                    
                else 
                    (newX, newY, newIdx) = process_single_file(cfrad, argument_file; 
                                                HAS_INTERACTIVE_QC = HAS_INTERACTIVE_QC, REMOVE_LOW_NCP = REMOVE_LOW_NCP, NCP_THRESHOLD = NCP_THRESHOLD, 
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
                else 
                    printstyled(Base.stderr, "UNRECOVERABLE ERROR\n"; color=:red)
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
    REMOVE_LOW_NCP::Bool = false 
    ```

    If true, will ignore gates with Normalized Coherent Power/Signal Quality Index below a threshold specified in RQCFeatures.jl
    
    ```julia 
    NCP_THRESHOLD::Float32 = .2
    ```
    Theshold at or below which to remove data 

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
    function calculate_features(input_loc::String, tasks::Vector{String}, weight_matrixes::Vector{Matrix{Union{Missing, Float32}}}
        ,output_file::String, HAS_INTERACTIVE_QC::Bool; verbose::Bool=false,
         REMOVE_LOW_NCP = false, NCP_THRESHOLD::Float32 = .2f0, REMOVE_HIGH_PGG = false, PGG_THRESHOLD::Float32=.1f0, QC_variable::String = "VG", remove_variable::String = "VV", 
         replace_missing::Bool=false, write_out::Bool=true, QC_mask::Bool = false, mask_name::String="", return_idxer::Bool =false)

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
            try 
                cfrad = Dataset(path) 
                pathstarttime=time() 
                dims = (cfrad.dim["range"], cfrad.dim["time"])

                if QC_mask

                    currmask = Matrix{Bool}(.! map(ismissing, cfrad[mask_name][:,:]))
                    (newX, newY, indexer) = process_single_file(cfrad, tasks, weight_matrixes; 
                                                HAS_INTERACTIVE_QC = HAS_INTERACTIVE_QC, REMOVE_LOW_NCP = REMOVE_LOW_NCP, NCP_THRESHOLD = NCP_THRESHOLD,
                                                REMOVE_HIGH_PGG = REMOVE_HIGH_PGG, PGG_THRESHOLD = PGG_THRESHOLD, QC_variable = QC_variable, remove_variable = remove_variable, 
                                                replace_missing=replace_missing, feature_mask = currmask, mask_features = true)
                    
                else 
                    (newX, newY, indexer) = process_single_file(cfrad, tasks, weight_matrixes; 
                                                HAS_INTERACTIVE_QC = HAS_INTERACTIVE_QC, REMOVE_LOW_NCP = REMOVE_LOW_NCP, NCP_THRESHOLD=NCP_THRESHOLD,
                                                REMOVE_HIGH_PGG = REMOVE_HIGH_PGG, PGG_THRESHOLD=PGG_THRESHOLD, QC_variable = QC_variable, remove_variable = remove_variable, 
                                                replace_missing=replace_missing)
                end 

                
                close(cfrad)

                if verbose
                    println("Processed $(path) in $(time()-pathstarttime) seconds")
                end 

            catch e
                if isa(e, DimensionMismatch)
                    printstyled(Base.stderr, "POSSIBLE ERRONEOUS CFRAD DIMENSIONS... SKIPPING $(path)\n"; color=:red)
                else 
                    printstyled(Base.stderr, "UNRECOVERABLE ERROR\n"; color=:red)
                    close(fid)
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

        model = DecisionTree.RandomForestClassifier(n_trees=n_trees, max_depth=max_depth, rng=50)
    
        # if balance_weight 
        #     counts = [sum(.! Vector{Bool}(Y[:])), sum(Vector{Bool}(Y[:]))]
        #     dub = 2 * counts
        #     weights = [sum(counts[1] ./ dub), sum(counts[2] ./ dub)]
        #     class_weights = [target ? weights[1] : weights[2] for target in Vector{Bool}(Y[:])]
           
        # else
        #     class_weights = ones(length(Y[:]))
        # end 


        if ! (length(Y) == length(class_weights))
            printstyled("WARNING: class_weights of different length than targets.... Continiuing with no class weights...\n", color=:yellow)
            class_weights = ones(length(Y))
        end 

        println("FITTING MODEL")
        startTime = time() 
        DecisionTree.fit!(model, X, reshape(Y, length(Y),), class_weights)

        println("COMPLETED FITTING MODEL IN $((time() - startTime)) seconds")
        println() 


        println("MODEL VERIFICATION:")
        predicted_Y = DecisionTree.predict(model, X) 
        accuracy = sum(predicted_Y .== Y) / length(Y)
        println("ACCURACY ON TRAINING SET: $(round(accuracy * 100, sigdigits=3))%")
        println()


        printstyled("SAVING MODEL TO: $(model_location) \n", color=:green) 
        save_object(model_location, model)

        if (verify) 
            ###NEW: Write out data to HDF5 files for further processing
            println("WRITING VERIFICATION DATA TO $(verify_out)" )
            fid = h5open(verify_out, "w")
            HDF5.write_dataset(fid, "Y_PREDICTED", predicted_Y)
            HDF5.write_dataset(fid, "Y_ACTUAL", Y)
            close(fid) 
        end 

        close(radar_data) 
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
    output_probs::Bool = false
    ```
    Whether or not to output probabilities of meteorological gate from random forest 
    ```julia
    prob_varname::String = ""
    ```
    What to name the probability variable in the cfradial file 
    """
    function QC_scan(file_path::String, config_file_path::String, model_path::String; VARIABLES_TO_QC::Vector{String}= ["ZZ", "VV"],
                     QC_suffix::String = "_QC", indexer_var::String="VV", decision_threshold::Tuple{Float32, Float32} = (.5f0, 1.f0), output_mask::Bool = true,
                     mask_name::String = "QC_MASK_2", verbose::Bool=false, REMOVE_HIGH_PGG::Bool = true, PGG_THRESHOLD = 1., REMOVE_LOW_NCP::Bool = true, NCP_THRESHOLD=.2,
                     output_probs::Bool = false, prob_varname::String = "")

        new_model = load_object(model_path) 

        paths = Vector{String}() 
        if isdir(file_path) 
            paths = parse_directory(file_path)
        else 
            paths = [file_path]
        end 
        

        for path in paths 
            ##Open in append mode so output variables can be written 
            input_cfrad = redirect_stdout(devnull) do
                NCDataset(path, "a")
            end 

            cfrad_dims = (input_cfrad.dim["range"], input_cfrad.dim["time"])

            ###Will generally NOT return Y, but only (X, indexer)
            ###Todo: What do I need to do for parsed args here 
            starttime=time()
            X, Y, indexer = process_single_file(input_cfrad, config_file_path; REMOVE_HIGH_PGG = REMOVE_HIGH_PGG, REMOVE_LOW_NCP = REMOVE_LOW_NCP, remove_variable=indexer_var)
            ##Load saved RF model 
            ##assume that default SYMBOL for saved model is savedmodel
            ##For binary classifications, 1 will be at index 2 in the predictions matrix 
            met_predictions = DecisionTree.predict_proba(new_model, X)[:, 2]
            predictions = (met_predictions .> decision_threshold[1]) .& (met_predictions .<= decision_threshold[2])
            printstyled("RETAINING GATES BETWEEN $(decision_threshold[1]) and $(decision_threshold[2]) PROBABILITY \n ", color=:yellow)

            ##QC each variable in VARIALBES_TO_QC
            for var in VARIABLES_TO_QC

                ##Create new field to reshape QCed field to 
                NEW_FIELD = missings(Float32, cfrad_dims) 

                ##Only modify relevant data based on indexer, everything else should be fill value 
                QCED_FIELDS = input_cfrad[var][:][indexer]

                NEW_FIELD_ATTRS = Dict(
                    "units" => input_cfrad[var].attrib["units"],
                    "long_name" => "Random Forest Model QC'ed $(var) field",
                    "probabilities" => " $(decision_threshold[1]) < p <= $(decision_threshold[2])"
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
                    defVar(input_cfrad, var * QC_suffix, NEW_FIELD, ("range", "time"), fillvalue = FILL_VAL; attrib=NEW_FIELD_ATTRS)
                catch e
                    ###Simply overwrite the variable 
                    if e.msg == "NetCDF: String match to name in use"
                        if verbose
                            println("Already exists... overwriting") 
                        end 
                        input_cfrad[var*QC_suffix][:,:] = NEW_FIELD 
                    else 
                        throw(e)
                    end 
                end 

                if verbose
                    println("\r\nPROCESSING: $(path)")
                    println("\r\nCompleted in $(time()-starttime ) seconds")
                    println()
                    printstyled("REMOVED $(initial_count - final_count) PRESUMED NON-METEORLOGICAL DATAPOINTS\n", color=:green)
                    println("FINAL COUNT OF DATAPOINTS IN $(var): $(final_count)")
                end 

            end 

            if output_mask

                MASK = fill(-1, cfrad_dims)[:]
                MASK[indexer] = predictions 
                MASK = reshape(MASK, cfrad_dims)

                try
                    if verbose 
                        println("Writing Mask")
                    end 

                    NEW_FIELD_ATTRS = Dict(
                    "units" => "Unitless",
                    "long_name" => "Ronin Quality Control mask"
                    )   
                    defVar(input_cfrad, mask_name, MASK, ("range", "time"), fillvalue=-1; attrib=NEW_FIELD_ATTRS)
                catch e

                ###Simply overwrite the variable 
                    if e.msg == "NetCDF: String match to name in use"
                        if verbose 
                            println("Already exists... overwriting") 
                        end 
                        input_cfrad[mask_name][:,:] =  MASK 
                    else 
                        throw(e)
                    end 
                end
            end 

            if output_probs 

                NEW = fill(-1, cfrad_dims)[:]
                NEW[indexer] = met_predictions
                NEW = reshape(NEW, cfrad_dims)

                try
                    if verbose 
                        println("Writing Probabilites to $(prob_varname)")
                    end 

                    NEW_FIELD_ATTRS = Dict(
                    "units" => "Unitless",
                    "long_name" => "Ronin Decision Tree Probabilities"
                    )   
                    defVar(input_cfrad, prob_varname, MASK, ("range", "time"), fillvalue=-1; attrib=NEW_FIELD_ATTRS)
                catch e

                ###Simply overwrite the variable 
                    if e.msg == "NetCDF: String match to name in use"
                        if verbose 
                            println("Already exists... overwriting") 
                        end 
                        input_cfrad[prob_varname][:,:] =  MASK 
                    else 
                        throw(e)
                    end 
                end
            
            close(input_cfrad)

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
            TOTAL_SCANS += length(readdir(path))
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

            contents = readdir(path)
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
    function get_feature_importance(input_file_path::String, λs::Vector{Float32}; pred_threshold::Float32 = .5f0)


        MLJ.@load LogisticClassifier pkg=MLJLinearModels

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

        for λ in λs

            mdl = LogisticClassifier(;lambda=λ, penalty=:l1)
            mach = machine(mdl, MLJ.table(features), targets[:])
            fit!(mach)
            coefs = fitted_params(mach).coefs 

            y_pred = predict(mach, features)
            results = pdf(y_pred, [0, 1])
            met_predictions = map(x -> x > pred_threshold ? 0 : 1, results[:, 1])

            push!(rmses, MLJ.rmse(met_predictions, targets_raw))

            for (i, param) in enumerate(params)
                push!(coef_values[param], coefs[i][2])
            end 
        
            
        end 

        coef_values["rmse"] = rmses
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
    function error_characteristics(file_path::String, config_file_path::String, model_path::String;
        indexer_var::String="VV", QC_variable::String="VG", decision_threshold::Float32 = .5f0, write_out::Bool=false,
        output_name::String="Model_Error_Characteristics.h5")


        ###We can probably refactor this honestly, just do predict with model 
        ###Do we need to reconstruct the original scans? Probably not..... 
       
        new_model = load_object(model_path) 


        paths = Vector{String}() 
        
        if isdir(file_path) 
            paths = parse_directory(file_path)
        else 
            paths = [file_path]
        end 

        tasks = get_task_params(config_file_path)

        X = Matrix{Float32}(undef,0,length(tasks))
        Y = Matrix{Int32}(undef,0,1) 
        indexer = Matrix{Int32}(undef,0,1)
        predictions = Matrix{Int32}(undef, 0, 1) 

        for path in paths   

            input_cfrad = redirect_stdout(devnull) do
               NCDataset(path, "a")
            end 

            cfrad_dims = (input_cfrad.dim["range"], input_cfrad.dim["time"])
            ###Todo: What do I need to do for parsed args here 
            println("\r\nPROCESSING: $(path)")
            starttime=time()
            try
            
                Xn, Yn, indexern = process_single_file(input_cfrad, config_file_path; REMOVE_HIGH_PGG = true, QC_variable = QC_variable,
                                                            REMOVE_LOW_NCP = true, remove_variable=indexer_var, HAS_INTERACTIVE_QC = true)
                println("\r\nCompleted in $(time()-starttime ) seconds")

                    ##Load saved RF model 
                ##assume that default SYMBOL for saved model is savedmodel
                ##For binary classifications, 1 will be at index 2 in the predictions matrix 
                met_predictions = DecisionTree.predict_proba(new_model, Xn)[:, 2]
                predictionsn = met_predictions .> decision_threshold

                ###If we wish to return features for error diagnostics, we simply return X which is the features array, 
                ###Y which are the correct values, the indexer which shows where data was taken out and where it was not, 
                ###and the model predictions 

                X  = vcat(X, Xn)
                Y  = vcat(Y, Yn)
                indexer = vcat(indexer, indexern)
                predictions = vcat(predictions, predictionsn)
            
            catch e
                printstyled("POSSIBLE ERROR WITH FILE AT: $(path)...\nCONTINUING\n", color=:red)
            end 
            
        end 

        false_positives_idx = (predictions .== 1) .& (Y .== 0)
        false_negatives_idx = (predictions .== 0) .& (Y .== 1)


        if write_out
            println("Writing Data to $(output_name)")
    
            h5open(output_name, "w") do f
                f["X"] = X[:,:]
                f["Y"] = Y[:]
                f["indexer"] = Vector{Int32}(indexer[:])
                f["predictions"] = Vector{Int32}(predictions[:])
                f["false_positive_index"] = Vector{Int32}(false_positives_idx[:])
                f["false_negatives_idx"] = Vector{Int32}(false_negatives_idx[:])
                attributes(f)["FEATURE_NAMES"] = tasks 
            end
    
            printstyled("Successfully Output Model Evaluation Data to $(output_name)\n", color=:green) 
        end 

        return (X, Y, indexer, predictions, false_positives_idx, false_negatives_idx) 
    end 



    """
        train_multi_model(config::ModelConfig)

    All-in-one function to take in a set of radar data, calculate input features, and train a chain of random forest models 
    for meteorological/non-meteorological gate identification. 

    #Required arguments 
    ```julia
    config::ModelConfig
    ```
    Struct containing configuration info for model training 
    """
    function train_multi_model(config::ModelConfig)
        ##Quick input sanitation check 
        @assert (length(config.model_output_paths) == length(config.feature_output_paths)
                 == length(config.met_probs) == length(config.task_paths) == length(config.task_weights) == length(config.mask_names))
        
        if !(config.HAS_INTERACTIVE_QC)
            throw("ERROR: Input cfradials must have interactive QC present to train a model. Set config HAS_INTERACTIVE_QC flag to true") 
        end 
        full_start_time = time() 
        ###Iteratively train models and apply QC_scan with the specified probabilites to train a multi-pass model 
        ###pipeline 
        for (i, model_path) in enumerate(config.model_output_paths)
            
            out = config.feature_output_paths[i] 
            currt = config.task_paths[i]
            cw = config.task_weights[i]

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
    
            else
                printstyled("\nCALCULATING FEATURES FOR PASS: $(i)\n", color=:green)

                ###Check to see if the features file already exists, if so, delete it so 
                ###that it may be overwritten 
                if config.write_out & config.overwrite_output
                    isfile(out) ? rm(out) : ""
                end 

                X,Y = calculate_features(config.input_path, currt, out, config.HAS_INTERACTIVE_QC; 
                                    verbose = config.verbose, REMOVE_LOW_NCP = config.REMOVE_LOW_NCP,NCP_THRESHOLD=config.NCP_THRESHOLD, 
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
            
            printstyled("\n...TRAINING FOR PASS: $(i) ON $(size(X)[1]) GATES...\n", color=:green)
        
            train_model(out, model_path, n_trees = config.n_trees, max_depth = config.max_depth, class_weights = class_weights)
    
            
            ###If this was the last pass, we don't need to write out a mask, and we're done!
            ###Otherwise, we need to mask out the features we want to apply the model to on the next pass 
            if i < config.num_models

                curr_model = load_object(model_path) 
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
                    
                    ###NEED to update this if it's beyond two pass so we can pass it the correct mask
                    X, Y, idxer = calculate_features(path, currt, out, true; 
                                        verbose = config.verbose, REMOVE_LOW_NCP = config.REMOVE_LOW_NCP, NCP_THRESHOLD=config.NCP_THRESHOLD,
                                        REMOVE_HIGH_PGG=config.REMOVE_HIGH_PGG,PGG_THRESHOLD=config.PGG_THRESHOLD, QC_variable = config.QC_var, 
                                        remove_variable = config.remove_var, replace_missing = config.replace_missing, return_idxer=true,
                                        write_out = false, QC_mask = QC_mask, mask_name = mask_name, weight_matrixes=cw)
                    
                    met_probs = DecisionTree.predict_proba(curr_model, X)
                    if size(met_probs)[2] < 2
                        throw(DomainError(1, "ERROR: ONLY ONE CLASS IN INPUT DATASET")) 
                    end 
                    met_probs = met_probs[:, 2]
                    valid_idxs = (met_probs .> minimum(curr_metprobs)) .& (met_probs .<= maximum(curr_metprobs))
                    print("RESULTANT GATES: $(sum(valid_idxs))")
                    ##Create mask field, fill it, and then write out
                    new_mask = Matrix{Union{Missing, Float32}}(missings(dims))[:]
                   
                    ##We only care about gates that have met the base QC thresholds, so first index 
                    ##by indexer returned from calculate_features, and then set the gates between
                    ##the specified probability levels to valid in the mask. The next model pass will 
                    ##thus only be calculated upon these features. 
                    idxer = idxer[1][:]
                    idxer[idxer] .= Vector{Bool}(valid_idxs)
                    new_mask[idxer] .= 1.
                    new_mask = reshape(new_mask, dims)
        
                    write_field(path, config.mask_names[i+1], new_mask, attribs=Dict("Units" => "Bool", "Description" => "Gates between met prob theresholds"))
    
                end 
            end   
        end 
        printstyled("\n COMPLETED TRAINING MODEL IN $(round(time() - full_start_time, digits = 3)) seconds...\n", color=:green)   
    end 
    
    """
    `QC_scan(input_cfrad::String, features::Matrix{Float32}, indexer::Vector{Bool}, config::ModelConfig, iter::Int64)`


    """
    function QC_scan(input_cfrad::String, features::Matrix{Float32}, indexer::Vector{Bool}, config::ModelConfig, iter::Int64)
        
        input_set = redirect_stdout(devnull) do 
            NCDataset(input_cfrad, "a") 
        end 
        new_model = load_object(config.model_output_paths[iter])
        decision_threshold = config.met_probs[iter] 
        cfrad_dims = (input_set.dim["range"], input_set.dim["time"])
        
        VARIABLES_TO_QC = config.VARS_TO_QC
        met_predictions = DecisionTree.predict_proba(new_model, features)[:, 2]
        predictions = met_predictions .> decision_threshold
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
    function composite_prediction(config::ModelConfig; write_predictions_out::Bool = false, prediction_outfile::String="model_predictions.h5", return_probs::Bool=false, QC_mode::Bool=false)

       @assert (length(config.model_output_paths) == length(config.feature_output_paths)
                 == length(config.met_probs) == length(config.task_paths) == length(config.task_weights) == length(config.mask_names))
    
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
    
        printstyled("LOADING MODELS....\n", color=:green)
        flush(stdout)
        models = [] 
    
    
        for path in config.model_output_paths
            push!(models, load_object(path))
        end 
        
        ###Need to do this file by file so that the spatial context of gates is maintained 
        for file in files
            
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
                X, Y, indexer = process_single_file(f, currt, HAS_INTERACTIVE_QC = ((! QC_mode) && config.HAS_INTERACTIVE_QC)
                    , REMOVE_HIGH_PGG = config.REMOVE_HIGH_PGG, PGG_THRESHOLD = config.PGG_THRESHOLD, REMOVE_LOW_NCP = config.REMOVE_LOW_NCP, NCP_THRESHOLD = config.NCP_THRESHOLD, 
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
                        if sum(gates_of_interest) == 0 
                            write_field(file, config.mask_names[i+1], new_mask,  attribs=Dict("Units" => "Bool", "Description" => "Gates between met prob theresholds"), fillval=config.FILL_VAL)
                        else 
                            @assert length(gates_of_interest) == sum(indexer) 
            
                            indexer[indexer] .= gates_of_interest 
                            new_mask[indexer] .= 1. 
                            new_mask = reshape(new_mask, scan_dims)
            
                            write_field(file, config.mask_names[i+1], new_mask,  attribs=Dict("Units" => "Bool", "Description" => "Gates between met prob theresholds"), fillval=config.FILL_VAL)
                        end 
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
        

        if predictions == Vector{Bool}(undef, 0)
            throw("ERROR: NO GATES IN INPUT DATASET MET BASIC QC THRESHOLDS")
        end 


        if write_predictions_out
            h5open(prediction_outfile, "w") do f 
                write_dataset(f, "Predictions", predictions)
                write_dataset(f, "Verification", values)
                #write_dataset(f, "Met_probabilities", met_probs)
                ##Below line is giving me Type Array does not have a definite size errors 
                #write_dataset(f, "Scan_indexers", init_idxers)
            end
        end 
    
        if return_probs
            return(predictions, values, init_idxers, total_met_probs)
        elseif QC_mode 
            return 
        else 
            return(predictions, values, init_idxers)
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

        low_predictions, low_verification, low_idxrs, low_probs = composite_prediction(model_config, return_probs=true)
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
                attribs::Dict = Dict(), dim_names::Tuple=("range", "time"), verbose::Bool=true, fillval::Float32 = FILL_VAL)
        
        if ! isfile(filepath) 
            ds = NCDataset(filepath, "c") 
            close(ds) 
        end 

        Dataset(filepath, "a") do input_set 
            try 
                print(input_set)
                defVar(input_set, fieldname, NEW_FIELD, dim_names, fillvalue = fillval; attrib=attribs)
            catch e
                println(e)
                ###Simply overwrite the variable 
                if e.msg == "NetCDF: String match to name in use" && (overwrite)
                    if verbose
                        println("$(fieldname) Already Exists in $(filepath)... overwriting") 
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
                    ##Copy over new attributes 
                else 
                    close(input_set)
                    throw(e)
                end 
            end 

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
        `evaluate_model(config::ModelConfig)`

        Returns a row of a DataFrame with a variety of metrics about a given model. 
    """
    function evaluate_model(config::ModelConfig; models_trained::Bool = false)


        ###This function will not handle the case where the model is trained but the features are not written. 
        ###it also implicitly assumes that the features will be written out. 

        ##Return dataframe with model configuration charactersitics as well as 
        ##Things we want to have here: task paths 

        if ! config.write_out
            throw("Error: evaluate_model must write features to disk. Please set config.write_out to true")
        end 

        if ! models_trained 
            train_multi_model(config)
        end 

        ###Now, use the calculated features to get the predictions. 

        models = [load_object(model) for model in config.model_output_paths]

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

        curr_model = load_object(config.model_output_paths[curr_model_num]) 
        curr_metprobs = config.met_probs[curr_model_num]
        curr_tasks = config.task_paths[curr_model_num]
        curr_weights = config.task_weights[curr_model_num]
        curr_out = config.feature_output_paths[curr_model_num] 
        output_cols = get_num_tasks(curr_tasks)

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
                                verbose = config.verbose, REMOVE_LOW_NCP = config.REMOVE_LOW_NCP, 
                                REMOVE_HIGH_PGG=config.REMOVE_HIGH_PGG, QC_variable = config.QC_var, 
                                remove_variable = config.remove_var, replace_missing = config.replace_missing, return_idxer=true,
                                write_out = false, QC_mask = QC_mask, mask_name = mask_name, weight_matrixes=curr_weights)
        
            if curr_model_num < config.num_models 
                met_probs = DecisionTree.predict_proba(curr_model, X)[:, 2]
                ###Probabilities inclusive on both ends 
                valid_idxs = (met_probs .>= minimum(curr_metprobs)) .& (met_probs .<= maximum(curr_metprobs))
                print("RESULTANT GATES: $(sum(valid_idxs))")
                ##Create mask field, fill it, and then write out
                new_mask = Matrix{Union{Missing, Float32}}(missings(dims))[:]
                
                ##We only care about gates that have met the base QC thresholds, so first index 
                ##by indexer returned from calculate_features, and then set the gates between
                ##the specified probability levels to valid in the mask. The next model pass will 
                ##thus only be calculated upon these features. 
                idxer = curr_idx[1][:]
                ###Determine where the gates that meet basic QC threshold are between the met thresholds and assign
                idxer[idxer] .= Vector{Bool}(valid_idxs)
                new_mask[idxer] .= 1.
                new_mask = reshape(new_mask, dims)

                write_field(path, config.mask_names[curr_model_num+1], new_mask, attribs=Dict("Units" => "Bool", "Description" => "Gates between met prob theresholds"))
            end 

            X = vcat(X, newX)::Matrix{Float32}
            Y = vcat(Y, newY)::Matrix{Int64}

        end 

        ##Write broader pass features to disk 
        if write_out

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

                currmodel = load_object(model) 



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

                currmodel = load_object(model) 

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



end 
