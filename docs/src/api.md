# Basic Workflow Walkthrough 
RadarQC is a package that utilizes a methodology developed by [Dr. Alex DesRosiers and Dr. Michael Bell](https://journals.ametsoc.org/view/journals/aies/aop/AIES-D-23-0064.1/AIES-D-23-0064.1.xml) to remove non-meteorological gates from Doppler radar scans by leveraging machine learning techniques. In its current form, it contains functionality to derive a set of features from input radar data, use these features to train a Random Forest classification model, and apply this model to the raw fields contained within the radar scans. It also has some model evaluation ability. The beginning of this guide will walk through a basic workflow to train a model starting from scratch. 



# API / Function References 

## Model Configuration 
```@docs
ModelConfig
```
## Calculating Model Input Features

```@docs
calculate_features(::String, ::String, ::String, ::Bool)
split_training_testing!
train_model(::String, ::String)
remove_validation
get_feature_importance(::String, ::Vector{Float64})
```

## Applying and evaluating a trained model to data

```@docs
QC_scan
characterize_misclassified_gates
```

## Predicting using a composite model

```@docs
train_multi_model
composite_prediction
```

## Non-user facing

```@docs
get_task_params
process_single_file
```
