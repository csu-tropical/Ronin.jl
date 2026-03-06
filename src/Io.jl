###Function containing capabilites to split input datasets into different training/testing compoments and Otherwise
###Input/output functionality 
export remove_validation 
"""
# Function used to remove a given subset of the rows from a feature set so that they may be used for model validation/tuning. 
Currently configured to utilize the 90/10 split described in DesRosiers and Bell 2023. 
---
# Required arguments 
```julia
input_dataset::String
```
Path to h5 files containing model features 

# Optional keyword arguments 
```julia
training_output::String = "train_no_validation_set.h5"
```
Path to output training features with validation removed to 

```julia
validation_output::String = "validation.h5"
```
Path to output validation features to

```julia
remove_original::Bool = true 
```
Whether or not to remove the original file described by the `input_dataset` path. 
"""
function remove_validation(input_dataset::String; training_output::String="train_no_validation_set.h5", 
                            validation_output::String = "validation.h5", remove_original::Bool=true)
        
    currset = h5open(input_dataset)
    params = attrs(currset)["Parameters"]
    FILL_VAL = attrs(currset)["MISSING_FILL_VALUE"]

    X = currset["X"][:,:]
    Y = currset["Y"][:,:] 

    ###Will include every STEP'th feature in the validation dataset 
    ###A step of 10 will mean that every 10th will feature in the validation set, and everything else in training 
    STEP = 10 

    ###Make every STEPth index false in the testing indexer, and true in the validation indexer 
    test_indexer = [true for i=1:size(X)[1]]
    test_indexer[begin:STEP:end] .= false

    validation_indexer = .!test_indexer 

    validation_out = h5open(validation_output, "w")
    training_out = h5open(training_output, "w")
    
    
    try
        write_dataset(validation_out, "X", X[validation_indexer, :])
        write_dataset(validation_out, "Y", Y[validation_indexer, :])
        attributes(validation_out)["MISSING_FILL_VALUE"] = FILL_VAL
        attributes(validation_out)["Parameters"] = params

        write_dataset(training_out, "X", X[test_indexer, :])
        write_dataset(training_out, "Y", Y[test_indexer, :])
        attributes(training_out)["MISSING_FILL_VALUE"] = FILL_VAL
        attributes(training_out)["Parameters"] = params
    catch e
        printstyled("ERROR in remove_validation: $(e)\n", color=:red)
        rethrow(e)
    finally
        close(currset)
        close(validation_out)
        close(training_out)
    end

    if remove_original
        rm(input_dataset)
    end

end


