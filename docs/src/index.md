
Ronin (Random forest Optimized Nonmeteorological IdentificatioN) is a package that utilizes a methodology developed by [Dr. Alex DesRosiers and Dr. Michael Bell](https://journals.ametsoc.org/view/journals/aies/aop/AIES-D-23-0064.1/AIES-D-23-0064.1.xml) to remove non-meteorological gates from Doppler radar scans by leveraging machine learning techniques. In its current form, it contains functionality to derive a set of features from input radar data, use these features to train a Random Forest classification model, and apply this model to the raw fields contained within the radar scans. It also has some model evaluation ability. The beginning of this guide will walk through a basic workflow to train a model starting from scratch.  

# Package Philosophy 
Ronin is written around a single model configuration struct, typed as `ModelConfig`. More documentation can be found in the API. All options necessary to define, train, and use a quality control model are contained wihthin it. 

# Single-model Workflow Walkthrough 

![Roninflowchart](./imgs/Ronin_flowchart.png)

## Preparing input data 

---

The first step in the process is to split our data so that some of it may be utilized for model training and the other portion for model testing. It's important to keep the two sets separate, otherwise the model may overfit. 

The basic requirement here is to have a directory or directories of cfradial scans, and two directories to put training and testing files, respectively. Make sure that no other files are present in these directories. To do this, the [`split_training_testing!`](https://github.com/irslushy/Ronin.jl/tree/main/src/Ronin.jl#943) function will be used. For example, if one had two cases of radar data, located in `./CASE1/` and `./CASE2/` and wanted to split into `./TRAINING` and `./TESTING`, execute the command 

```julia
split_training_testing(["./CASE1", "./CASE2"], "./TRAINING", "./TESTING")
```

More information about this function is contained within the docs. 

--- 

Now that we have split the input files, we can proceed to calculate the features used to train and test our Random Forest model. Further details are contained within the aforementioned manuscript, but it has been shown that the parameters contained [here](https://github.com/irslushy/Ronin.jl/blob/259aa4d306e09fedf9d4208bcc8a584fbabd89a2/MODELS/DesRosiers_Bell_23/config.txt) are most effective for discriminating between meteorological/non-meteorological gates. For this, we will use the [calculate_features](https://github.com/irslushy/Ronin.jl/tree/main/src/Ronin.jl#L265) function. Since we are calculating features to **train** a model at this point, we will assume that intreactive QC has already been applied to them. To get the most skillful model possible, we will want to remove "easy" cases from the training set, so set `REMOVE_LOW_NCP=true` and `REMOVE_HIGH_PGG=true` to ignore data not meeting minimum quality thresholds. It's also important to specify which variable contained with the input scans has already been QC'ed - in the ELDORA scans, this is `VG`. `missing` values must also be removed from the initial training set, so we'll use a raw variable `VV` to determine where these gates are located. With that said, one may now invoke 

```julia
calculate_features("./TRAINING", "./config.txt", "TRAINING_FEATURES.h5", true;
                    verbose=true, REMOVE_LOW_NCP = true, REMOVE_HIGH_PGG=true,
                    QC_variable="VG", remove_variable="VV")
```

To use the config file located at `./config.txt` to output the training features to `TRAINING_FEATURES.h5`. 

For clarity's sake, for the testing features this step would be exectued again but changing the input directory and output location. 

This is a somewhat computationally expensive process for large datasets >1000 scans, and so can take over 15 minutes. 

---

## Training a Random Forest Model 
---

Now that the somewhat arduous process of calculating input features has completed, it's time to train our model! We'll use the **training** set for this, which we have previously defined to be located at `./TRAINING_FEATURES.h5`. 

First, in order to combat the class imbalance problem, we must calculate weights for the non-meteorological and meteorological gates. This can be achieved as follows.

```julia
weights = Vector{Float64}(undef, 0)
class_weights = h5open("TRAINING_FEATURES.h5") do f
    samples = f["Y"][:,:][:]
    class_weights = Vector{Float32}(fill(0,length(samples)))
    weight_dict = compute_balanced_class_weights(samples)

    for class in keys(weight_dict) 
        class_weights[samples .== class] .= weight_dict[class]
    end 
    return(class_weights)
end 
```

Now we can train the model! 

```julia
train_model("./TRAINING_FEATURES.h5", "TRAINED_MODEL.joblib"; class_weights = Vector{Float32}(class_weights) , verify=true, verify_out="TRAINING_SET_VERIFICATION.h5")
```

This will train a model based off the information contained within `TRAINING_FEATURES.h5`, saving it for further use at `./TRAINED_MODEL.joblib`. The `verify` keyword arguments means that, once trained, the model will be automatically applied to the training dataset and the predictions will be output, along with the ground truth/correct answers, to the h5 file at `TRAINING_SET_VERIFICATION.h5`.   


---
## Applying the trained model to a radar scan   
---
  


It's finally time to begin cleaning radar data! We'll use the `QC_scan` function to apply our model to raw moments. Let's imagine we want to clean the scan located at `./cfrad_example_scan` using the model we previously trained at `TRAINED_MODEL.joblib`. By default, this will clean the variables with the names `ZZ` and `VV`, though this can be changed by modifying the `VARIABLES_TO_QC` argument. They will be added to the cfradial file with the names `ZZ_QC` and `VV_QC`, respectively, though this suffix can be changed though keyword arguments. Execute as  

```
QC_scan("./cfrad_example_scan", "./config.txt", "./TRAINED_MODEL.joblib")
```

---
## Spatial Predictors Reference 
---

A key portion of this methodology is deriving "predictors" from raw input radar moments. Raw moments include quantities such as Doppler velocity and reflectivity, while derived variables include things such as the standard deviation of a raw moment across a set of azimuths and ranges in a radar scan. Calculating these features allows the addition of spatial context to the classification problem even when only classifying a single gate. 

Each of the spatial predictors (Currently STD, ISO, and AVG) have predefined "windows" that specify the area they calculate. These windows are specified as matrixes at the top of [RoninFeatures.jl](https://github.com/irslushy/Ronin.jl/blob/main/src/RoninFeatures.jl). They can also be user specified when using [calculate_features](https://irslushy.github.io/Ronin.jl/dev/api.html#Ronin.calculate_features-Tuple{String,%20Vector{String},%20Vector{Matrix{Union{Missing,%20Float64}}},%20String,%20Bool}). &nbsp;


### Currently Implemented Functions:   

#### **STD(VAR)**
--- 
Calculates the standard deviation of each gate of the variable with name VAR in the given radar sweep. By default, gates that contain `missing` values are ignored in this calculation. Further by default 

---

#### **ISO(VAR)**
--- 
Calculates the "isolation" of each gate of the variable with name VAR in the given radar sweep. This calculation sums the number of adjacent gates in both range and aziumth that contain `missing` values. 

---
#### **AVG(VAR)**
---
Calculates the average of each gate of the variable with name VAR in the given radar sweep. By default, gates that contain `missing` values are ignored in this calculation. 

---
#### **RNG/NRG**
---
Calculates the range of all radar gates (RNG) from the airborne platform, or normalized by altitude (NRG). 

---
#### **PGG**
**P**robability of **G**round **G**ate - a geometric calculation that gives the probability that a given radar gate is a result of reflection from the ground. 
___


#### **AHT**
**A**ircraft **H**eigh**T** - calculates platform height while factoring in Earth curvature. 
______


### **Implementing a new parameter**

The code is written in such a way that it would hopefully be relatively easy for a user to add a function of their own to apply to radar data. One could go about this using the following as a guide.    


There are two types of functions in Ronin: Those that act on radar variables (STD, AVG) and those that operate relatively independent of them (RNG, PGG). Functions that act on variables must be defined using a **3 letter abbreviation** and begin with `calc_`. Furthermore, the function should take **1 positional** and **2 keyword** arguments. The positional argument should be the variable in matrix form to operate upon. The keyword arguments should be `weights` and `window`, where both have the same dimensions. `window` specifies the area for the spatial parameter to take for each gate, and `weights` specifies how much weight to give each neighboring gate. For example, if we wanted to define a function that gave us the logartihm of a variable, we could name it `LOG`, with the function defined in code as

```
function calc_LOG(var::Matrix{Union{Missing, Float64}}; weights=default_weights, window=default_window)
```

The function should return an array of the same size as `var`. 

Finally, add the three letter abbreviation `LOG` to the `valid_funcs` array at the top of `RoninFeatures.jl`. 

Congratulations! You've added a new function! 

---
## Data Conventions/Glossary 
---
Some important data convetions to make note of: 

* **Meteorological Data is referred to by 1 or `true`**
* **Non-Meteorological Data is referred to by 0 or `false`**
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
