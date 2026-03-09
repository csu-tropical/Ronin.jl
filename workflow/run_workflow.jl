##=============================================================================
## RONIN Workflow Runner
##=============================================================================
## Single-script interface to the multi-step RONIN training pipeline.
## Toggle the flags below to control which steps run.
##
## This is a convenience wrapper — each step can also be run independently
## by including the individual script (e.g., include("02_train.jl")).
##
## WORKFLOW OVERVIEW:
##
##   Step 1:  Split data (one-time)
##   Step 2:  Train multi-pass model on all features
##   Step 2a: Evaluate model on testing set
##   Step 3:  Compute feature importance
##   Step 4:  Retrain with pruned features (requires SELECTED_FEATURES)
##   Step 4a: Evaluate retrained model (same as 2a, different output name)
##   Step 5:  Sweep Pass 2 met_prob thresholds
##   Step 6:  Apply QC to write corrected fields
##
## TYPICAL FIRST RUN:
##   Set RUN_TRAINING = true, RUN_EVALUATION = true, everything else false.
##
## FEATURE OPTIMIZATION:
##   1. Set RUN_IMPORTANCE = true (after initial training exists)
##   2. Review output, set SELECTED_FEATURES in 00_config.jl
##   3. Set RUN_RETRAIN = true, RUN_RETRAIN_EVALUATION = true
##
## THRESHOLD OPTIMIZATION:
##   Set RUN_PASS2_SWEEP = true (after training exists)
##=============================================================================

##=============================================================================
## WORKFLOW FLAGS — toggle which steps to run
##=============================================================================

RUN_SPLIT_DATA          = false   # Step 1:  one-time data split
RUN_TRAINING            = false   # Step 2:  train on all features
RUN_EVALUATION          = false   # Step 2a: evaluate on testing set
RUN_IMPORTANCE          = false    # Step 3:  compute feature importance
RUN_RETRAIN             = false   # Step 4:  retrain with pruned features
RUN_RETRAIN_EVALUATION  = false   # Step 4a: evaluate retrained model
RUN_PASS2_SWEEP         = false   # Step 5:  sweep Pass 2 thresholds
RUN_QC                  = false   # Step 6:  apply QC to data

## Also evaluate on validation set? (applies to both 2a and 4a)
RUN_VALIDATION          = false

##=============================================================================
## LOAD SHARED CONFIG
##=============================================================================

include("00_config.jl")

##=============================================================================
## EXECUTION
##=============================================================================

## --- Step 1: Split data ---
if RUN_SPLIT_DATA
    println("\n", "="^70)
    println("STEP 1: SPLITTING DATA")
    println("="^70)
    Ronin.split_training_testing_validation!(CASE_PATHS, TRAINING_PATH, TESTING_PATH, VALIDATION_PATH)
    println("Data split complete.")
end

## --- Step 2: Train ---
if RUN_TRAINING
    if !isempty(SELECTED_FEATURES)
        @warn "SELECTED_FEATURES is non-empty ($(length(SELECTED_FEATURES)) features). " *
              "Initial training uses all features. " *
              "To retrain with a pruned set, set RUN_RETRAIN = true instead."
    end

    println("\n", "="^70)
    println("STEP 2: TRAINING — $(EXPERIMENT_NAME)")
    println("  num_models = $(num_models), task_mode = $(task_mode)")
    println("  n_trees = $(n_trees), max_depth = $(max_depth), class_weights = $(class_weights)")
    println("  SIG_QUALITY = $(REMOVE_LOW_SIG_QUALITY ? SIG_QUALITY_THRESHOLD : "off")")
    println("  PGG = $(REMOVE_HIGH_PGG ? PGG_THRESHOLD : "off")")
    println("  met_probs_train = $(met_probs_train)")
    if task_mode == "convolution"
        println("  conv_variables = $(config.conv_variables)")
        println("  conv_kernel_sizes = $(config.conv_kernel_sizes)")
    end
    println("="^70)

    train_multi_model(config)
    println("Training complete.")
end

## --- Step 2a: Evaluate ---
if RUN_EVALUATION
    println("\n", "="^70)
    println("STEP 2a: EVALUATING — $(EXPERIMENT_NAME)")
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
    )

    if RUN_VALIDATION
        println("\nEvaluating on VALIDATION set...")
        val_results = run_evaluation(
            config, "VALIDATION", VALIDATION_PATH, met_probs_test;
            prediction_outfile = "predictions_val_$(EXPERIMENT_NAME).h5",
        )
    end
end

## --- Step 3: Feature importance ---
if RUN_IMPORTANCE
    if !isempty(SELECTED_FEATURES)
        @warn "SELECTED_FEATURES is non-empty. Feature importance should be " *
              "computed on the full feature set for best results."
    end

    println("\n", "="^70)
    println("STEP 3: COMPUTING FEATURE IMPORTANCE — $(EXPERIMENT_NAME)")
    println("="^70)

    config.compute_feature_importance = true
    config.file_preprocessed = fill(true, num_models)

    train_multi_model(config)

    for (i, path) in enumerate(config.model_output_paths)
        println("\n--- Pass $(i) ---")
        inspect_model_configuration(path)

        md = load_model_with_metadata(path, config.task_mode)
        if !isempty(md.recommended_features)
            println("\nRecommended features for Pass $(i):")
            println("  SELECTED_FEATURES = $(md.recommended_features)")
        end
    end

    println("\n", "="^70)
    println("NEXT STEPS:")
    println("  1. Copy recommended indices to SELECTED_FEATURES in 00_config.jl")
    println("  2. Set RUN_RETRAIN = true and RUN_RETRAIN_EVALUATION = true")
    println("  3. Re-run this script")
    println("="^70)

    ## Reset so subsequent steps in this run don't re-compute importance
    config.compute_feature_importance = false
end

## --- Step 4: Retrain with pruned features ---
if RUN_RETRAIN
    if isempty(SELECTED_FEATURES)
        error("SELECTED_FEATURES is empty in 00_config.jl.\n" *
              "Run Step 3 (feature importance) first, then set SELECTED_FEATURES " *
              "to the recommended indices.")
    end

    println("\n", "="^70)
    println("STEP 4: RETRAINING WITH PRUNED FEATURES — $(EXPERIMENT_NAME)")
    println("  Using $(length(SELECTED_FEATURES)) selected features")
    println("  Indices: $(SELECTED_FEATURES)")
    println("="^70)

    train_multi_model(config)
    println("Retrain complete.")
end

## --- Step 4a: Evaluate retrained model ---
if RUN_RETRAIN_EVALUATION
    println("\n", "="^70)
    println("STEP 4a: EVALUATING RETRAINED MODEL — $(EXPERIMENT_NAME)")
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

## --- Step 5: Pass 2 sweep ---
if RUN_PASS2_SWEEP
    if num_models < 2
        error("Pass 2 sweep requires num_models >= 2 (currently $(num_models))")
    end

    println("\n", "="^70)
    println("STEP 5: PASS 2 THRESHOLD SWEEP — $(EXPERIMENT_NAME)")
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
    )
end

## --- Step 6: Apply QC ---
if RUN_QC
    println("\n", "="^70)
    println("STEP 6: APPLYING QC — $(EXPERIMENT_NAME)")
    println("  Input: $(QC_PATH)")
    println("  Variables: $(VARS_TO_QC)")
    println("="^70)

    config.input_path = QC_PATH
    config.met_probs  = met_probs_test[1:num_models]
    QC_scan(config)
    println("QC complete. Check output files in $(QC_PATH)")
end

println("\nWorkflow complete.")
