# Legacy Hand-Tuned Mode

Before the convolution feature mode, Ronin used a hand-crafted predictor file
and a single Random Forest. This mode is **fully preserved** but is no longer
the recommended path — new work should use the
[Convolution Feature Mode](convolution.md). Select it by leaving
`task_mode = ""` (any value other than `"convolution"`) and supplying
`task_paths` / `task_weights` (or, in the single-model API, a `config.txt`
predictor file).

## Legacy single-model walkthrough

This is the original v1.1.0 flow, condensed.

### 1. Split the data

You need one or more directories of cfradial scans and separate output
directories for training and testing files. [`split_training_testing!`](@ref)
softlinks files into them:

```julia
split_training_testing!(["./CASE1", "./CASE2"], "./TRAINING", "./TESTING")
```

(Use [`split_training_testing_validation!`](@ref) for a three-way split with a
held-out validation set.)

### 2. Calculate features

Use [`calculate_features`](@ref) with a predictor `config.txt`. Because we are
computing features to **train** a model, interactive QC is assumed already
applied; the QC'd variable is the ground truth (ELDORA: `VG`). To get the most
skillful model, drop "easy" cases with `REMOVE_LOW_SIG_QUALITY = true` and
`REMOVE_HIGH_PGG = true`, and remove `missing` gates using a raw variable:

```julia
calculate_features("./TRAINING", "./config.txt", "TRAINING_FEATURES.h5", true;
                    verbose=true, REMOVE_LOW_SIG_QUALITY = true, REMOVE_HIGH_PGG = true,
                    QC_variable="VG", remove_variable="VV")
```

!!! note "Deprecated keyword"

    The v1.1.0 keyword `REMOVE_LOW_NCP` still works (in `calculate_features`,
    `process_single_file`, and `make_config`, plus the
    `ModelConfig.REMOVE_LOW_NCP` field) but emits a deprecation warning and
    forwards to `REMOVE_LOW_SIG_QUALITY`. Use the new name.

Repeat for the testing set with the testing directory and a different output
path. For large datasets (>1000 scans) this can take 15+ minutes.

### 3. Train the model

Combat class imbalance by computing per-sample weights, then train:

```julia
class_weights = h5open("TRAINING_FEATURES.h5") do f
    samples = f["Y"][:,:][:]
    cw = Vector{Float32}(fill(0, length(samples)))
    weight_dict = compute_balanced_class_weights(samples)
    for class in keys(weight_dict)
        cw[samples .== class] .= weight_dict[class]
    end
    return cw
end

train_model("./TRAINING_FEATURES.h5", "TRAINED_MODEL.joblib";
            class_weights = Vector{Float32}(class_weights),
            verify=true, verify_out="TRAINING_SET_VERIFICATION.h5")
```

`verify=true` applies the freshly trained model back to the training set and
writes predictions plus ground truth to the verification HDF5.

### 4. Apply the model

```julia
QC_scan("./cfrad_example_scan", "./config.txt", "./TRAINED_MODEL.joblib")
```

By default this cleans `ZZ` and `VV`, writing `ZZ_QC` / `VV_QC` (configurable
via the variables-to-QC argument and the QC suffix).

## Spatial Predictors Reference

!!! note

    These predictors are the feature set for the **legacy hand-tuned mode**
    (`task_mode = ""`). Convolution mode derives its own features from a kernel
    bank — see the [Convolution Feature Mode](convolution.md) page.

A key part of this methodology is deriving "predictors" from raw moments. Raw
moments are quantities like Doppler velocity and reflectivity; derived
variables add spatial context (e.g. the standard deviation of a moment over a
window of azimuths and ranges), letting the classifier reason spatially even
when labelling a single gate.

Each spatial predictor (STD, ISO, AVG) has a predefined **window** specifying
the area it covers, declared as a matrix at the top of `RoninFeatures.jl`. The
window/weights can also be user-specified through `calculate_features`.

### Currently implemented functions

**`STD(VAR)`** — standard deviation of the named variable at each gate. Gates
containing `missing` are ignored by default.

**`ISO(VAR)`** — "isolation": sums the number of adjacent gates in range and
azimuth that contain `missing`.

**`AVG(VAR)`** — average of the named variable at each gate; `missing` gates
ignored by default.

**`RNG` / `NRG`** — range of all gates from the airborne platform (`RNG`), or
that range normalized by altitude (`NRG`).

**`PGG`** — **P**robability of **G**round **G**ate; a geometric calculation of
the probability that a gate is a reflection from the ground.

**`AHT`** — **A**ircraft **H**eigh**T**; platform height accounting for Earth
curvature.

### Implementing a new predictor

The code is structured so adding your own predictor is straightforward. There
are two kinds of functions: those that act on a radar variable (STD, AVG) and
those that operate independently of one (RNG, PGG).

A variable function must use a **3-letter abbreviation**, be named `calc_<abbr>`
(lowercase), and take **1 positional** and **2 keyword** arguments: the
positional argument is the variable matrix to operate on; the keywords are
`weights` and `window`, both the same dimensions. `window` is the per-gate
footprint; `weights` is the per-neighbour weight. For example, a log predictor
named `LOG`:

```julia
function calc_LOG(var::Matrix{Union{Missing, Float64}};
                  weights=default_weights, window=default_window)
    # ... return an array the same size as `var`
end
```

Finally, add the 3-letter abbreviation (`LOG`) to the `valid_funcs` array at
the top of `RoninFeatures.jl`. The new predictor is now usable in a
`config.txt`.
