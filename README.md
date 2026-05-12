# Ronin.jl

Ronin.jl (Random forest Optimized Nonmeteorological IdentificatioN) contains a julia implementation of the algorithm described in [DesRosiers and Bell 2023](https://journals.ametsoc.org/view/journals/aies/aop/AIES-D-23-0064.1/AIES-D-23-0064.1.xml) for removing non-meteoroloigcal gates (Non-Meteorological Data, henceforth NMD) from airborne radar scans while retaining Meteorological Data (henceforth MD). Care has been taken to ensure relative similarity to the form described in the manuscript, but some changes have been made in the interest of computational speed.
  <br>

___
# Installation

Ronin.jl is registered in the Julia General Registry. From the Julia REPL:

```julia
julia> import Pkg; Pkg.add("Ronin")
```

or in package mode (`]`):

```
pkg> add Ronin
```

Supported Julia versions: 1.10 (LTS), 1.11, 1.12 — exercised by CI. See `CHANGELOG.md` for release notes.

___
# Quick start

The recommended starting point is the `workflow/` directory, which provides a
step-by-step pipeline driven by a single shared configuration file:

| Script | Purpose |
|---|---|
| `workflow/00_config.jl` | Shared config (edit experiment name, data paths, hyperparameters) |
| `workflow/01_split_data.jl` | One-time train/test/validation split |
| `workflow/02_train.jl` | Train multi-pass cascade on all features |
| `workflow/02a_evaluate.jl` | Evaluate trained model on the testing set |
| `workflow/03_importance.jl` | Permutation feature importance |
| `workflow/04_retrain.jl` | Retrain with the pruned feature set |
| `workflow/05_sweep_pass2.jl` | Sweep Pass 2 met-probability thresholds |
| `workflow/06_qc.jl` | Apply the final model to write QC'd fields |
| `workflow/run_workflow.jl` | Orchestrator with boolean flags to run any subset |

The default workflow uses **convolution feature mode** (`task_mode = "convolution"`),
which derives features from a kernel bank applied to your chosen radar
variables. The detailed walkthrough further below uses the older hand-crafted
predictor pattern; both still work but convolution mode is the recommended
path for new users.

After a workflow run, the in-memory `ModelConfig` is auto-saved to
`model_config_<EXPERIMENT_NAME>.jld2` so inference / operational scripts can
reload it via `Ronin.load_config(path)` without re-executing the workflow.

___
# Acknowledgments

Much of the data used to train the models in this repository is the product of arduous manual editing of radar scans. ELDORA data is provided by the authors of [Bell, Lee, Wolff, & Cai 2013](https://journals.ametsoc.org/view/journals/apme/52/11/jamc-d-12-0283.1.xml?tab_body=fulltext-display). NOAA P3 TDR Data is courtsey of Dr. Paul Reasor, Dr. John Gamache, and Kelly Neighbour. As mentioned above, the code is adapted from the original work of Dr. Alex DesRosiers.
___
# Developing from source
## Setting up the environment (CSU)
After cloning the repository, start Julia using Ronin as the project directory, either by calling 
```
julia --project=Ronin
```
from the parent directory of `Ronin` or modifying the `JULIA_PROJECT` environment variable. <br>
Then, enter package mode in the REPL by pressing `]`.<br>
<br><br>
Next, run `instantiate` to download the necessary dependencies. This should serve both to download/install dependencies and precompile the Ronin package. Now, exit package mode using the delete key. To ensure that everything was installed properly, run `using Ronin` on the Julia REPL. No errors or information should print out if successful. Run `add IJulia` if you will be viewing the code in a Jupyter notebook and need access to the Jupyter kernel.
> Guide adaped from https://github.com/mmbell/Scythe.jl/tree/main
>
## Setting up the environment (Derecho)
### Getting Julia
export JULIA_DEPOT_PATH=$SCRATCH/julia <br>
curl -fsSL https://install.julialang.org | sh
<br>


Now, exit package mode using the delete key. To ensure that everything was installed properly, run `using Ronin` on the Julia REPL. No errors or information should print out if successful. 
> Guide adapted from https://github.com/mmbell/Scythe.jl/tree/main

## Example notebook 
<br>

If you're looking to jump right in, check out [Ronin Example Notebook](./RONIN_Walkthrough.ipynb) - it contains everything you need to get up and running.
<br><br><br>

___
## Guide: Processing new data, training, and evaluating a new model
___
  <br>
  
The first step in training a new random forest model is determining which portions of the data will be used for training, testing, and validation. A helpful function here is `split_training_testing!` - this can be used to automatically split a collection of scans into a training directory and a testing directory. In order for the script to be configured properly, the variables relating to the different paths must be modified by the user - this is shown in the example notebook. 
<br> <br>The current configuration is consistent with the 80/20 training/testing split described in the manuscript, as well as to have an equal number of scans from each "case" represented in the testing set. It is expected that the script would work for different training/testing splits, but this has not yet been tested. <br><br>

It's also generally beneficial and adherent to ML best practices to remove a validation dataset from the training data to tune model hyperparameters on. This can be done in a variety of different ways and so is left to the user. It is recommended that users adopt a strategy that minimizes temporal autocorrelation between scans - for example, taking every 10th sweep chronologically and placing it in the validation dataset. These approaches help ensure the most generalizable model possible. 


___
## Configuring a new model 
___

<br>
Now that the input data has been split, it's time to configure a model! Ronin is written such that all model hyperparameters and other configuration details are contained within a `ModelConfig` struct. Further detail and setup follows. 

<br> 

At a high level, the first step in the process is calculating a set of input features containing information characterizing each gate in the training radar sweeps.
`config_path` specifies the location of the file containing the information about what features the user wishes to calculate. `input_path` is the path to the file 
or directory where the sweeps are located. 

```
config_path = "./NOAA_all_params.txt"
input_path = TRAINING_PATH
```

Ronin can create a "multi-pass" model, where a model is trained on the full dataset, and successive models are trained on subsets of these data. The motivation for this setup is to leverage the probablistic information provided by the random forest approach. Consider a gate where 90% of the trees in the random forest agree on a certain classification - it's possible that gates such as this may have fundamentally different characteristics than gates where the RF model is more evenly split on a class. It is then natural to expect that training a model specifically on gates of the second type may result in improved classification accuracy. Configuring a multi-pass model involves the specification of the number of models one wishes to use in a composite, as well as a range of probabilities to move on to the next pass. More is explained in the following. 

We'll start with a 2-pass model. Grid search testing on the validation dataset has shown that this number of passes best leverages the desire for performance with the retention of meteorological data. 
```
num_models = 2
```
Now, we'll define which gates are passed on to successive models. 

>`pass_1_probs = (.1,.9)`

This means that gates where between 10-90% of the trees agree (inclusive) will be passed on to the second pass. 
Gates that <10% of the trees classify as meteorological will be assigned a label of non-meteorological and 
gates that >90% of trees classify as meteorological will be assigned a label of meteorological. This can be done for more passes, but we're just doing 2 as a minimal example.

Met probabilities for the final pass of any composite model are interpreted somewhat differently. The maximum of the two probabilites will be taken, and gates where >= max percent of the trees classify a gate as meteorological will be assigned a label of meteorological/MD, with all other gates being assigned a label of non-meteorological/NMD. For example, if one were to set 

`final_met_prob = (.1,.9)`

gates where >=90% of trees agree on a classification of meteorological would be assigned a label of meteorological/MD, with all other gates being assigned a label of non-meteorological/NMD.
```
initial_met_prob = (.1, .9) 
final_met_prob = (.1,.9) 
met_probs = [initial_met_prob, final_met_prob]
```

Another important feature of Ronin is its implementation of spatial features. These calculations take into account not only the gate of interest, but the gates surrounding it as well. The concept can be loosely equated to convolutions in a neural network. As such, it's important to specify weights for each surrounding observation/gate. Ronin provides a series of default weight matrixes that can be used to do so. More detail follows. 

```
###The following are default windows specified in RoninConstants.jl 
###Standard 7x7 window 
sw = Ronin.standard_window 
###7x7 window with only nonzero weights in azimuth dimension 
aw = Ronin.azi_window
###7x7 window with only nonzero weights in range dimension 
rw = Ronin.range_window 
###Placeholder window for tasks that do not require spatial context 
pw = Ronin.placeholder_window 

###Specify a weight matrix for each individual task in the configuration file 
weight_vec = [pw, pw, pw, sw, sw, sw, aw, rw, pw, pw, pw, pw, pw]
###Specify a weight vector for each model pass 
###len(weight_vector) is enforced to be equal to num_models (should have a set of weights for each pass) 
task_weights = [weight_vec, weight_vec] 
```

The model configuration also needs to know where to output trained models and calculated features

```
base_name = "raw_model"
base_name_features = "output_features" 
model_output_paths = [base_name * "_$(i-1).jld2" for i in 1:num_models ]
feature_output_paths = [base_name_features * "_$(i-1).h5" for i in 1:num_models]
```

In order to combat unequal distributions of MD/NMD in the training data (Once basic QC thresholds have been applied, the dataset is heavily weighted toward MD), the user can chose to apply weights to each target in the RF algorithm by setting the `class_weights` variable to `"balanced"` 
```
class_weights = "balanced"
```

We also need to set a couple of different variables to specify what certain variables in the sweeps represent. `QC_var` will contain the name of the variable that has had interactive QC applied to it, and will be used as the targets of the RF model. `remove_var` will contain the name of a raw variable in the sweep, generally raw velocity or raw reflectivity, that will be utilized to mask out values that are already missing. 

```
QC_var = "VE"
remove_var = "VEL"
```

Now that we have set everything up, instantiate a configuration object as follows. 
```
config = ModelConfig(num_models = num_models,model_output_paths =  model_output_paths,met_probs =  met_probs, feature_output_paths = feature_output_paths, input_path = input_path,task_mode="nan",file_preprocessed = file_preprocessed, task_paths = [config_path, config_path], QC_var = QC_var, remove_var = remove_var, QC_mask = false, mask_names = mask_names, VARS_TO_QC = ["VEL"], class_weights = class_weights, HAS_INTERACTIVE_QC=true, task_weights = task_weights)
```

> More detail about the model configuration structures are contained [here](https://csu-tropical.github.io/Ronin.jl/dev/api.html#Ronin.ModelConfig).

___
## Training, evaluating, and applying a model
___
Things get a lot simpler now! We've defined pretty much all the features we need to use in our model, so we may simply inovke 
```
train_multi_model(config) 
```
To train the model accordingly. This can take on the order of a couple hours. 

Then, let's update it to the testing dataset to see how it does! 
```
config.input_path = TESTING_PATH 
```
Call `composite_prediction` to have the model predict on this set of sweeps. 
```
predictions, verification, indexers = composite_prediction(config, write_features_out=true, feature_outfile="NEW_MODEL_PREDICTIONS_OUT.h5")
```


___

## Notes on data conventions
_______
Some important data convetions to make note of: 

* **Meteorological Data is referred to by 1 or `true` or MD**
* **Non-Meteorological Data is referred to by 0 or `false` or NMD**
* **ELDORA** scan variable names: 
    * Raw Velocity: **VV**
    * QC'ed Velocity (Used for ground truth): **VG**
    * Raw Reflectivity: **ZZ**
    * QC'ed Reflectivity (Used for ground truth): **DBZ**
    * Normalized Coherent Power/Signal Quality Index: **NCP**
* **NOAA TDR** scan variable names: 
    * Raw Velocity: **VEL**
    * QC'ed Velocity (Used for ground truth): **VE**
    * Raw Reflectivity: **DBZ**
    * QC'ed Reflectivity (Used for ground truth): **DZ**
    * Normalized Coherent Power/Signal Quality Index: **SQI**


