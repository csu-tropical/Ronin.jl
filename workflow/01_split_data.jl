##=============================================================================
## Step 1: Split Data
##=============================================================================
## Run ONCE to split CfRadial files into train/test/val sets.
## After splitting, do not re-run — subsequent scripts expect consistent splits.
##=============================================================================

include("00_config.jl")

println("Splitting data into train/test/val...")
println("  Source: $(CASE_PATHS)")
println("  Train:  $(TRAINING_PATH)")
println("  Test:   $(TESTING_PATH)")
println("  Val:    $(VALIDATION_PATH)")

Ronin.split_training_testing_validation!(CASE_PATHS, TRAINING_PATH, TESTING_PATH, VALIDATION_PATH)

println("Data split complete.")
