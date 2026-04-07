##=============================================================================
## RONIN Workflow Runner
##=============================================================================
## Single-script interface to the multi-step RONIN training pipeline.
## Toggle the flags below to control which steps run.
##
## This is a convenience wrapper — each step can also be run independently
## by including the individual script (e.g., include("02_train.jl")).
##
##=============================================================================
## TWO MODES OF OPERATION
##=============================================================================
##
## ---- FULL PIPELINE (recommended for most users) ----
##
##   Phase 1: RUN_FULL_TRAINING = true
##     Trains all passes, evaluates, and computes feature importance.
##     Prints recommended features for each pass, then stops.
##
##   Phase 2: RUN_FULL_RETRAIN = true
##     After reviewing Phase 1 output, update PASS_CONFIG selected_features
##     in 00_config.jl, then run this phase. Retrains all passes with pruned
##     features and evaluates.
##
## ---- INCREMENTAL (fine-grained control) ----
##
##   For advanced use: train/evaluate/retrain individual passes, generate
##   masks between passes, sweep thresholds, or apply QC. All incremental
##   flags default to false.
##
##=============================================================================

##=============================================================================
## FULL PIPELINE FLAGS
##=============================================================================

RUN_SPLIT_DATA           = false   # One-time data split (run before anything else)
RUN_FULL_TRAINING        = false   # Phase 1: train → evaluate → importance (all passes)
RUN_FULL_RETRAIN         = false   # Phase 2: retrain with selected_features → evaluate
USE_PRECOMPUTED_FEATURES = false  # Skip feature calculation (use saved h5, any mode)

## Also evaluate on validation set?
RUN_VALIDATION          = false

##=============================================================================
## INCREMENTAL FLAGS (all default to false)
##=============================================================================

RUN_CALCULATE_FEATURES  = false   # Calculate and save features (no training)
RUN_TRAINING            = false   # Train all passes from scratch
RUN_EVALUATION          = true   # Evaluate on testing set
SKIP_EXISTING_MET_PROBS = false   # Skip re-writing met_prob_pass_N if already in CfRadial files
RUN_IMPORTANCE          = false   # Compute feature importance for TRAIN_PASS
RUN_RETRAIN             = false   # Retrain TRAIN_PASS with pruned features
RUN_RETRAIN_EVALUATION  = false   # Evaluate retrained model
RUN_HISTOGRAM           = false   # Show met_prob distribution for MASK_PASS
RUN_GENERATE_MASKS      = false  # Generate met_prob/masks from MASK_PASS
RUN_TRAIN_NEXT_PASS     = false  # Train TRAIN_PASS using existing prior-pass masks
RUN_PASS2_SWEEP         = false   # Sweep Pass 2 met_prob thresholds
RUN_QC                  = false   # Apply QC to write corrected fields

## For incremental steps: which pass to operate on
MASK_PASS               = 0
TRAIN_PASS              = 1

##=============================================================================
## LOAD SHARED CONFIG
##=============================================================================

# Default is 00_config.jl but you can change that here
include("3pass_config.jl")

## --- Apply precomputed features flag (applies to all modes) ---
if USE_PRECOMPUTED_FEATURES
    for i in 1:config.num_models
        out = config.feature_output_paths[i]
        if !isfile(out)
            @warn "USE_PRECOMPUTED_FEATURES is true but $(out) not found — pass $(i) will recalculate"
            continue
        end

        # Determine what selected_features this pass wants
        pc = get(PASS_CONFIG, i, nothing)
        pass_sf = pc !== nothing && hasproperty(pc, :selected_features) ? pc.selected_features : SELECTED_FEATURES

        # Check what the trained model expects (if model file exists)
        model_path = config.model_output_paths[i]
        model_sf = Int[]
        if task_mode == "convolution" && isfile(model_path)
            try
                md = load_model_with_metadata(model_path, task_mode)
                model_sf = md.selected_features
            catch; end
        end

        # Compare: can we reuse the precomputed features?
        h5_ncols = Ronin.HDF5.h5open(out) do f; size(f["X"], 2); end
        if collect(Int, pass_sf) == collect(Int, model_sf)
            config.file_preprocessed[i] = true
            printstyled("  Pass $(i): using precomputed features from $(out) ($(h5_ncols) columns)\n", color=:cyan)
        else
            config.file_preprocessed[i] = false
            printstyled("  Pass $(i): recalculating features — config has $(isempty(pass_sf) ? "all" : "$(length(pass_sf))") selected features" *
                        ", model has $(isempty(model_sf) ? "all" : "$(length(model_sf))") (h5 has $(h5_ncols) columns)\n", color=:yellow)
        end
    end
end

##=============================================================================
## FULL PIPELINE EXECUTION
##=============================================================================

## --- One-time data split ---
if RUN_SPLIT_DATA
    println("\n", "="^70)
    println("SPLITTING DATA")
    println("="^70)
    Ronin.split_training_testing_validation!(CASE_PATHS, TRAINING_PATH, TESTING_PATH, VALIDATION_PATH)
    println("Data split complete.")
end

## --- Phase 1: Train all passes → Evaluate → Importance ---
if RUN_FULL_TRAINING
    ## Verify PASS_CONFIG selected_features are empty (Phase 1 trains on all features)
    for (p, pc) in sort(collect(PASS_CONFIG))
        sf = hasproperty(pc, :selected_features) ? pc.selected_features : Int[]
        if !isempty(sf)
            @warn "PASS_CONFIG[$(p)] has $(length(sf)) selected_features. " *
                  "Phase 1 trains on ALL features — selected_features will be ignored. " *
                  "To retrain with pruned features, use RUN_FULL_RETRAIN instead."
        end
    end

    ## Build a training-only pass_config (strip selected_features so all features are used)
    train_pass_config = Dict{Int,Any}()
    for (p, pc) in PASS_CONFIG
        if hasproperty(pc, :conv_variables)
            train_pass_config[p] = (conv_variables = pc.conv_variables, selected_features = Int[])
        else
            train_pass_config[p] = (selected_features = Int[],)
        end
    end

    println("\n", "="^70)
    println("PHASE 1: FULL TRAINING — $(EXPERIMENT_NAME)")
    println("="^70)
    println("  num_models = $(num_models), task_mode = $(task_mode)")
    println("  n_trees = $(n_trees), max_depth = $(max_depth), class_weights = $(class_weights)")
    println("  SIG_QUALITY = $(REMOVE_LOW_SIG_QUALITY ? SIG_QUALITY_THRESHOLD : "off")")
    println("  PGG = $(REMOVE_HIGH_PGG ? PGG_THRESHOLD : "off")")
    println("  met_probs_train = $(met_probs_train)")
    if task_mode == "convolution"
        println("  conv_kernel_sizes = $(config.conv_kernel_sizes)")
        for (p, pc) in sort(collect(PASS_CONFIG))
            cv = hasproperty(pc, :conv_variables) ? pc.conv_variables : conv_variables
            println("  Pass $(p): $(length(cv)) conv_variables, all features")
        end
    end
    println("="^70)

    ## Step 1: Train
    printstyled("\n--- Training all passes ---\n", color=:green)
    train_multi_model(config; pass_config=train_pass_config)

    ## Step 2: Evaluate
    printstyled("\n--- Evaluating on TESTING set ---\n", color=:green)
    for (i, path) in enumerate(config.model_output_paths)
        if isfile(path)
            println("\n--- Pass $(i) ---")
            inspect_model_configuration(path)
        end
    end
    test_results = run_evaluation(
        config, "TESTING", TESTING_PATH, met_probs_test;
        prediction_outfile = "predictions_test_$(EXPERIMENT_NAME).h5",
    )
    if RUN_VALIDATION
        println("\nEvaluating on VALIDATION set...")
        val_results = run_evaluation(
            config, "VALIDATION", VALIDATION_PATH, met_probs_test;
            prediction_outfile = "predictions_val_$(EXPERIMENT_NAME).h5",
        )
    end

    ## Step 3: Feature importance
    printstyled("\n--- Computing feature importance ---\n", color=:green)
    println("  n_repeats = $(config.n_importance_repeats)")
    println("  subsample_fraction = $(config.importance_subsample_fraction)")
    println("  threads = $(Threads.nthreads())")
    compute_importance(config)

    ## Print results and next steps
    println("\n", "="^70)
    printstyled("PHASE 1 COMPLETE — RECOMMENDED FEATURES\n", color=:green)
    println("="^70)
    for (i, path) in enumerate(config.model_output_paths)
        if isfile(path)
            md = load_model_with_metadata(path, config.task_mode)
            if !isempty(md.recommended_features)
                println("\n  Pass $(i): $(length(md.recommended_features)) recommended features")
                println("  PASS_CONFIG[$(i)] selected_features = $(md.recommended_features)")
            end
        end
    end
    println("\n", "="^70)
    println("NEXT STEPS:")
    println("  1. Copy recommended indices into PASS_CONFIG in 00_config.jl")
    println("  2. Set RUN_FULL_RETRAIN = true, RUN_FULL_TRAINING = false")
    println("  3. Re-run this script")
    println("="^70)
end

## --- Phase 2: Retrain all passes with pruned features → Evaluate ---
if RUN_FULL_RETRAIN
    ## Verify at least one pass has selected_features set
    local any_selected = false
    for (p, pc) in sort(collect(PASS_CONFIG))
        sf = hasproperty(pc, :selected_features) ? pc.selected_features : Int[]
        if !isempty(sf)
            any_selected = true
        end
    end
    if !any_selected
        error("No passes have selected_features set in PASS_CONFIG.\n" *
              "Run Phase 1 (RUN_FULL_TRAINING) first, then copy recommended\n" *
              "features into PASS_CONFIG in 00_config.jl.")
    end

    println("\n", "="^70)
    println("PHASE 2: FULL RETRAIN WITH PRUNED FEATURES — $(EXPERIMENT_NAME)")
    println("="^70)
    if task_mode == "convolution"
        for (p, pc) in sort(collect(PASS_CONFIG))
            cv = hasproperty(pc, :conv_variables) ? pc.conv_variables : conv_variables
            sf = hasproperty(pc, :selected_features) ? pc.selected_features : Int[]
            println("  Pass $(p): $(length(cv)) conv_variables, $(isempty(sf) ? "all" : "$(length(sf))") features")
        end
    end
    println("="^70)

    ## Step 1: Retrain
    printstyled("\n--- Retraining all passes with pruned features ---\n", color=:green)
    train_multi_model(config; pass_config=PASS_CONFIG)

    ## Step 2: Evaluate
    printstyled("\n--- Evaluating retrained model ---\n", color=:green)
    for (i, path) in enumerate(config.model_output_paths)
        if isfile(path)
            println("\n--- Pass $(i) (retrained) ---")
            inspect_model_configuration(path)
        end
    end

    println("\nEvaluating on TESTING set...")
    test_results = run_evaluation(
        config, "TESTING", TESTING_PATH, met_probs_test;
        prediction_outfile = "predictions_test_$(EXPERIMENT_NAME)_pruned.h5",
    )
    if RUN_VALIDATION
        println("\nEvaluating on VALIDATION set...")
        val_results = run_evaluation(
            config, "VALIDATION", VALIDATION_PATH, met_probs_test;
            prediction_outfile = "predictions_val_$(EXPERIMENT_NAME)_pruned.h5",
        )
    end

    println("\nCompare to Phase 1 results to verify no skill loss.")
end

##=============================================================================
## INCREMENTAL EXECUTION
##=============================================================================

## --- Calculate features (without training) ---
if RUN_CALCULATE_FEATURES
    orig_conv_variables = copy(config.conv_variables)
    orig_selected_features = copy(config.selected_features)

    println("\n", "="^70)
    println("CALCULATING FEATURES — $(EXPERIMENT_NAME)")
    println("="^70)

    for i in 1:config.num_models
        out = config.feature_output_paths[i]

        ## Apply per-pass config
        if haskey(PASS_CONFIG, i)
            pc = PASS_CONFIG[i]
            config.conv_variables = hasproperty(pc, :conv_variables) ? pc.conv_variables : orig_conv_variables
            config.selected_features = hasproperty(pc, :selected_features) ? pc.selected_features : orig_selected_features
        else
            config.conv_variables = orig_conv_variables
            config.selected_features = orig_selected_features
        end

        QC_mask = i > 1 ? true : config.QC_mask
        mask_name = QC_mask ? config.mask_names[i] : ""

        if isfile(out)
            printstyled("  Pass $(i): $(out) already exists, skipping\n", color=:yellow)
            config.file_preprocessed[i] = true
            continue
        end

        printstyled("\n--- Pass $(i): $(length(config.conv_variables)) conv_variables ---\n", color=:green)

        if config.task_mode == "convolution"
            if config.write_out & config.overwrite_output
                isfile(out) && rm(out)
            end
            X, Y = calculate_features_conv(config, out;
                                            QC_mask=QC_mask, mask_name=mask_name,
                                            write_out=true)
            printstyled("  Saved $(size(X,1)) gates × $(size(X,2)) features → $(out)\n", color=:green)
        else
            currt = config.task_paths[i]
            cw = config.task_weights[i]
            if config.write_out & config.overwrite_output
                isfile(out) && rm(out)
            end
            X, Y = calculate_features(config.input_path, currt, out, config.HAS_INTERACTIVE_QC;
                                verbose = config.verbose,
                                REMOVE_LOW_SIG_QUALITY = config.REMOVE_LOW_SIG_QUALITY,
                                SIG_QUALITY_THRESHOLD = config.SIG_QUALITY_THRESHOLD,
                                SIG_QUALITY_VAR=config.SIG_QUALITY_VAR,
                                REMOVE_HIGH_PGG=config.REMOVE_HIGH_PGG,
                                PGG_THRESHOLD = config.PGG_THRESHOLD,
                                QC_variable = config.QC_var,
                                remove_variable = config.remove_var,
                                replace_missing = config.replace_missing,
                                write_out = true, QC_mask = QC_mask, mask_name = mask_name,
                                weight_matrixes=cw)
            printstyled("  Saved $(size(X,1)) gates × $(size(X,2)) features → $(out)\n", color=:green)
        end

        config.file_preprocessed[i] = true
    end

    config.conv_variables = orig_conv_variables
    config.selected_features = orig_selected_features
    println("Feature calculation complete.")
end

## --- Train all passes ---
if RUN_TRAINING
    if TRAIN_PASS > 1
        @warn "RUN_TRAINING trains ALL passes from scratch (starting at pass 1), " *
              "ignoring TRAIN_PASS=$(TRAIN_PASS). To train only pass $(TRAIN_PASS), " *
              "use RUN_TRAIN_NEXT_PASS = true instead."
    end

    println("\n", "="^70)
    println("TRAINING — $(EXPERIMENT_NAME)")
    println("  num_models = $(num_models), task_mode = $(task_mode)")
    println("  n_trees = $(n_trees), max_depth = $(max_depth), class_weights = $(class_weights)")
    println("  SIG_QUALITY = $(REMOVE_LOW_SIG_QUALITY ? SIG_QUALITY_THRESHOLD : "off")")
    println("  PGG = $(REMOVE_HIGH_PGG ? PGG_THRESHOLD : "off")")
    println("  met_probs_train = $(met_probs_train)")
    if task_mode == "convolution"
        println("  conv_kernel_sizes = $(config.conv_kernel_sizes)")
        for (p, pc) in sort(collect(PASS_CONFIG))
            cv = hasproperty(pc, :conv_variables) ? pc.conv_variables : conv_variables
            sf = hasproperty(pc, :selected_features) ? pc.selected_features : Int[]
            println("  Pass $(p): $(length(cv)) conv_variables, $(isempty(sf) ? "all" : "$(length(sf))") features")
        end
    end
    println("="^70)

    train_multi_model(config; pass_config=PASS_CONFIG)
    println("Training complete.")
end

## --- Evaluate ---
if RUN_EVALUATION
    println("\n", "="^70)
    println("EVALUATING — $(EXPERIMENT_NAME)")
    println("="^70)

    for (i, path) in enumerate(config.model_output_paths)
        if isfile(path)
            println("\n--- Pass $(i) ---")
            inspect_model_configuration(path)
        end
    end

    println("\nEvaluating on TESTING set...")
    test_results = run_evaluation(
        config, "TESTING", TESTING_PATH, met_probs_test;
        prediction_outfile = "predictions_test_$(EXPERIMENT_NAME).h5",
        skip_existing_met_probs = SKIP_EXISTING_MET_PROBS,
    )

    if RUN_VALIDATION
        println("\nEvaluating on VALIDATION set...")
        val_results = run_evaluation(
            config, "VALIDATION", VALIDATION_PATH, met_probs_test;
            prediction_outfile = "predictions_val_$(EXPERIMENT_NAME).h5",
            skip_existing_met_probs = SKIP_EXISTING_MET_PROBS,
        )
    end
end

## --- Feature importance ---
if RUN_IMPORTANCE
    configure_pass!(config, TRAIN_PASS)

    if !isempty(config.selected_features)
        @warn "selected_features is non-empty for Pass $(TRAIN_PASS). Feature importance " *
              "should be computed on the full feature set for best results."
    end

    println("\n", "="^70)
    println("COMPUTING FEATURE IMPORTANCE — $(EXPERIMENT_NAME) PASS $(TRAIN_PASS)")
    println("  n_repeats = $(config.n_importance_repeats)")
    println("  subsample_fraction = $(config.importance_subsample_fraction)")
    println("  threads = $(Threads.nthreads())")
    println("="^70)

    compute_importance(config; pass=TRAIN_PASS)

    model_path = config.model_output_paths[TRAIN_PASS]
    if isfile(model_path)
        println("\n--- Pass $(TRAIN_PASS) ---")
        inspect_model_configuration(model_path)

        md = load_model_with_metadata(model_path, config.task_mode)
        if !isempty(md.recommended_features)
            println("\nRecommended features for Pass $(TRAIN_PASS):")
            println("  Add to PASS_CONFIG[$(TRAIN_PASS)] selected_features:")
            println("  selected_features = $(md.recommended_features)")
        end
    end

    println("\n", "="^70)
    println("NEXT STEPS:")
    println("  1. Copy recommended indices to PASS_CONFIG[$(TRAIN_PASS)].selected_features")
    println("  2. Set RUN_RETRAIN = true (or RUN_TRAIN_NEXT_PASS for per-pass retrain)")
    println("  3. Re-run this script")
    println("="^70)
end

## --- Retrain single pass with pruned features ---
if RUN_RETRAIN
    configure_pass!(config, TRAIN_PASS)

    if isempty(config.selected_features)
        error("selected_features is empty for Pass $(TRAIN_PASS) in PASS_CONFIG.\n" *
              "Run feature importance first, then set selected_features " *
              "in PASS_CONFIG[$(TRAIN_PASS)].")
    end

    println("\n", "="^70)
    println("RETRAINING PASS $(TRAIN_PASS) WITH PRUNED FEATURES — $(EXPERIMENT_NAME)")
    println("  Using $(length(config.selected_features)) selected features")
    if task_mode == "convolution"
        println("  conv_variables = $(config.conv_variables)")
    end
    println("="^70)

    train_single_pass(config, TRAIN_PASS)
    println("Retrain complete.")
end

## --- Evaluate retrained model ---
if RUN_RETRAIN_EVALUATION
    println("\n", "="^70)
    println("EVALUATING RETRAINED MODEL — $(EXPERIMENT_NAME)")
    println("="^70)

    for (i, path) in enumerate(config.model_output_paths)
        if isfile(path)
            println("\n--- Pass $(i) (retrained) ---")
            inspect_model_configuration(path)
        end
    end

    println("\nEvaluating pruned model on TESTING set...")
    test_results = run_evaluation(
        config, "TESTING", TESTING_PATH, met_probs_test;
        prediction_outfile = "predictions_test_$(EXPERIMENT_NAME)_pruned.h5",
    )

    if RUN_VALIDATION
        println("\nEvaluating pruned model on VALIDATION set...")
        val_results = run_evaluation(
            config, "VALIDATION", VALIDATION_PATH, met_probs_test;
            prediction_outfile = "predictions_val_$(EXPERIMENT_NAME)_pruned.h5",
        )
    end

    println("\nCompare to initial training results to verify no skill loss.")
end

## --- Show met_prob histogram for a pass ---
if RUN_HISTOGRAM
    println("\n", "="^70)
    println("MET_PROB HISTOGRAM — PASS $(MASK_PASS)")
    println("="^70)

    println("\nTraining set:")
    met_prob_histogram(config, MASK_PASS; data_path=TRAINING_PATH,
                       met_probs_threshold=met_probs_train[MASK_PASS])

    if isdir(TESTING_PATH)
        println("Testing set:")
        met_prob_histogram(config, MASK_PASS; data_path=TESTING_PATH,
                           met_probs_threshold=met_probs_train[MASK_PASS])
    end
end

## --- Generate masks from a trained pass ---
if RUN_GENERATE_MASKS
    if num_models < MASK_PASS + 1
        @warn "num_models=$(num_models) but MASK_PASS=$(MASK_PASS) — setting num_models=$(MASK_PASS + 1) " *
              "so mask/model paths are properly allocated."
        config_kwargs[:compute_feature_importance] = false
        config = make_config(;
            num_models = MASK_PASS + 1,
            input_path = TRAINING_PATH,
            experiment_name = EXPERIMENT_NAME,
            config_kwargs...
        )
    end

    threshold = met_probs_train[MASK_PASS]

    println("\n", "="^70)
    println("REGENERATING MASKS FROM SAVED met_prob_pass_$(MASK_PASS) — $(EXPERIMENT_NAME)")
    println("  Threshold: $(threshold)")
    println("  (reads saved probabilities, no model re-run needed)")
    println("="^70)

    ## Use regenerate_masks: reads saved met_prob_pass_<N> from CfRadials and
    ## rewrites the mask with the new threshold. Much faster than generate_pass_masks
    ## which recomputes features and re-runs the RF model.
    config.input_path = TRAINING_PATH
    regenerate_masks(config, MASK_PASS, threshold)

    if isdir(TESTING_PATH)
        println("Also regenerating masks on TESTING set...")
        config.input_path = TESTING_PATH
        regenerate_masks(config, MASK_PASS, threshold)
    end

    if RUN_VALIDATION && isdir(VALIDATION_PATH)
        println("Also regenerating masks on VALIDATION set...")
        config.input_path = VALIDATION_PATH
        regenerate_masks(config, MASK_PASS, threshold)
    end

    config.input_path = TRAINING_PATH
end

## --- Train next pass using existing prior-pass masks ---
if RUN_TRAIN_NEXT_PASS
    if num_models < TRAIN_PASS
        @warn "num_models=$(num_models) but TRAIN_PASS=$(TRAIN_PASS) — setting num_models=$(TRAIN_PASS) " *
              "so model paths are properly allocated."
        config_kwargs[:compute_feature_importance] = false
        config = make_config(;
            num_models = TRAIN_PASS,
            input_path = TRAINING_PATH,
            experiment_name = EXPERIMENT_NAME,
            config_kwargs...
        )
    end

    configure_pass!(config, TRAIN_PASS)

    println("\n", "="^70)
    println("TRAINING PASS $(TRAIN_PASS) — $(EXPERIMENT_NAME)")
    println("  Requires mask_pass_$(TRAIN_PASS) in CfRadials (from RUN_GENERATE_MASKS)")
    println("  n_trees = $(n_trees), max_depth = $(max_depth)")
    if task_mode == "convolution"
        println("  conv_variables = $(config.conv_variables)")
        println("  selected_features = $(isempty(config.selected_features) ? "all" : config.selected_features)")
    end
    println("="^70)

    train_single_pass(config, TRAIN_PASS)
    println("Pass $(TRAIN_PASS) training complete.")

    model_path = config.model_output_paths[TRAIN_PASS]
    if isfile(model_path)
        println("\n--- Pass $(TRAIN_PASS) model ---")
        inspect_model_configuration(model_path)
    end
end

## --- Pass 2 threshold sweep ---
if RUN_PASS2_SWEEP
    if num_models < 2
        error("Pass 2 sweep requires num_models >= 2 (currently $(num_models))")
    end

    println("\n", "="^70)
    println("PASS 2 THRESHOLD SWEEP — $(EXPERIMENT_NAME)")
    println("  Low grid:  $(SWEEP_MET_PROB_LOW_GRID)")
    println("  High grid: $(SWEEP_MET_PROB_HIGH_GRID)")
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
        skip_existing_sweep    = SKIP_EXISTING_SWEEP,
    )
end

## --- Apply QC ---
if RUN_QC
    println("\n", "="^70)
    println("APPLYING QC — $(EXPERIMENT_NAME)")
    println("  Input: $(QC_PATH)")
    println("  Variables: $(VARS_TO_QC)")
    println("="^70)

    config.input_path = QC_PATH
    config.met_probs  = met_probs_test[1:num_models]
    QC_scan(config)
    println("QC complete. Check output files in $(QC_PATH)")
end

println("\nWorkflow complete.")
