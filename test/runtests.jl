###Rudimentary test suite to ensure that updates do not break the code 
using Ronin
using Missings 
using HDF5 
using NCDatasets
using BenchmarkTools 
using StatsBase
using Scratch


global scratchspace = @get_scratch!("ronin_testing")


###Will undergo a basic training/QC pipeline. Model is not meant to output 
###Correct results, but rather simply show that it can produce data, train a model, 
###and correctly apply the model to a scan. 

###The below will be testing for a single-pass model

TRAINING_PATH = "../BENCHMARKING/benchmark_cfrads/"
config_file_path = "../BENCHMARKING/benchmark_setup/config.txt"
sample_model = "../BENCHMARKING/benchmark_setup/benchmark_model.joblib"
NOAA_PATH =  "../BENCHMARKING/benchmark_NOAA_cfrads"
###Read in tasks

###NEED TO ALLOW THIS TO IGNORE COMMENTS 
tasks = Ronin.get_task_params(config_file_path)

placeholder_matrix = allowmissing(ones(3,3))
center_weight::Float64 = 0

###Weight matrixes for calculating spatial parameters 
iso_weights::Matrix{Union{Missing, Float64}} = allowmissing(ones(7,7))
iso_weights[4,4] = center_weight 
iso_window::Tuple{Int64, Int64} = (7,7)

avg_weights::Matrix{Union{Missing, Float64}} = allowmissing(ones(5,5))
avg_weights[3,3] = center_weight 
avg_window::Tuple{Int64, Int64} = (5,5)

std_weights::Matrix{Union{Missing, Float64}} = allowmissing(ones(5,5))
std_weights[3,3] = center_weight 
std_window::Tuple{Int64, Int64} = (5,5)


weight_matrixes = [placeholder_matrix, placeholder_matrix, std_weights, placeholder_matrix, placeholder_matrix, iso_weights]

path1 = joinpath(scratchspace, "_1.h5")
path2 = joinpath(scratchspace, "_2.h5")

X, Y, idxers_1 = calculate_features(TRAINING_PATH, tasks, weight_matrixes, 
                    path1, true; verbose=true, 
                    REMOVE_LOW_NCP = true, REMOVE_HIGH_PGG = true, 
                    QC_variable="VG", remove_variable="VV", return_idxer = true)


###Ensure multiple dispatch is functioning properly and giving us the same results 
X, Y, idxers_2 = calculate_features(TRAINING_PATH, config_file_path, path2, true,
                    verbose=true, REMOVE_LOW_NCP = true, 
                    REMOVE_HIGH_PGG = true, remove_variable = "VV", QC_variable="VG", return_idxer = true)

@assert idxers_1 == idxers_2 

h5open(path1) do f
    h5open(path2) do f2
        print(f["X"][begin:5,:])
        print(f2["X"][begin:5,:])
        @assert map(round, f["X"][begin:10, :]) == map(round, f2["X"][begin:10, :])
    end
end 



###Also ensure that both are equal to a pre-set version of the benchmark 

print("OK")
h5open(path1) do f 
    h5open("../BENCHMARKING/standard_bench_file.h5") do f2
        @assert map(round, f["X"][:,:]) == map(round, f2["X"][:,:])
    end 
end 

print("OK2")
X = Matrix{Float64}(undef, 0, 6)
Y = Matrix{Float64}(undef, 0, 1)

y_list = []
indexer_list = []

###Get verification for each of the cfradials 
for cfrad_name in readdir(TRAINING_PATH)

    if cfrad_name == ".tmp_hawkedit"
        rm(TRAINING_PATH * cfrad_name, force=true, recursive = true) 
        continue
    end 

    Dataset(TRAINING_PATH * cfrad_name) do f
        X_new, Y_new, indexer = Ronin.process_single_file(f, config_file_path; HAS_INTERACTIVE_QC=true, REMOVE_LOW_NCP=true, REMOVE_HIGH_PGG=true, QC_variable = "VG", 
                                                    remove_variable = "VV", replace_missing = false) 
        global X
        global Y
        X = vcat(X, X_new)
        Y = vcat(Y, Y_new)

        push!(indexer_list, indexer) 
        push!(y_list, Y)
    end 
end 

min_bad_threshold = .9 
min_retain_threshold = .9 

###Now, let's check that we can QC a scan. We'll use the trained ELDORA model here. 
for (i, cfrad_name) in enumerate(readdir(TRAINING_PATH))

    if cfrad_name[1] == '.'
        continue
    end 

    QC_scan(TRAINING_PATH * cfrad_name, config_file_path, sample_model)

    NCDataset(TRAINING_PATH * cfrad_name) do currset
        ###Ensure that the QC is removing at least 90% of bad data and retaining at least 90% of the good data 
        ZZ_QC = currset["ZZ_QC"][:,:][:][indexer_list[i]]
        ZZ_raw = currset["ZZ"][:, :][:][indexer_list[i]]

        ZZ_manual = currset["DBZ"][:, :][:][indexer_list[i]] 
        ###If they are equal, model predicted meteorological 
        model_predictions = [ismissing(x) ? false : true for x in (ZZ_raw .== ZZ_QC)]
        ground_truth = [ismissing(x) ? false : true for x in (ZZ_raw .== ZZ_manual)] 

        true_positives = count(ground_truth[model_predictions .== true] .== true) 
        num_total_positives = count(ground_truth) 

        true_negatives = count(ground_truth[model_predictions .== false] .== false)
        num_total_negatives = count(.!ground_truth) 

        pos_retained_frac = true_positives / num_total_positives
        neg_removed_frac = true_negatives / num_total_negatives
    


        printstyled("Positive Retained Fraction: " * string(pos_retained_frac) * "\n", color = :green) 
        printstyled("Negative Removed Fraction: " * string(neg_removed_frac) * "\n", color=:green)

        @assert pos_retained_frac > min_retain_threshold
        @assert neg_removed_frac > min_bad_threshold

        
        
    end 
end 

###New test in order to test threaded QC_scan feature 
times = []
for i in 1:10
    starttime = time()
    QC_scan(TRAINING_PATH, config_file_path, sample_model)
    push!(times, round(time() - starttime, sigdigits=4))
end 

printstyled("QC_SCAN AVERAGE TIME FOR 10 SCANS: $(mean(times)) seonds \n", color=:green)

################################################################################################################################
###Define some default configurations 
###Begin testing the trainin of multi-model configurations 
TRAINING_PATH = "../BENCHMARKING/benchmark_cfrads/"
config_file_path = "../BENCHMARKING/benchmark_setup/config.txt"
sample_model = "../BENCHMARKING/benchmark_setup/benchmark_model.joblib"
###NEED TO ALLOW THIS TO IGNORE COMMENTS 
tasks = Ronin.get_task_params(config_file_path)

placeholder_matrix = allowmissing(ones(3,3))
center_weight::Float64 = 0

###Weight matrixes for calculating spatial parameters 
iso_weights::Matrix{Union{Missing, Float64}} = allowmissing(ones(7,7))
iso_weights[4,4] = center_weight 
iso_window::Tuple{Int64, Int64} = (7,7)

avg_weights::Matrix{Union{Missing, Float64}} = allowmissing(ones(5,5))
avg_weights[3,3] = center_weight 
avg_window::Tuple{Int64, Int64} = (5,5)

std_weights::Matrix{Union{Missing, Float64}} = allowmissing(ones(5,5))
std_weights[3,3] = center_weight 
std_window::Tuple{Int64, Int64} = (5,5)


weight_matrixes = [placeholder_matrix, placeholder_matrix, std_weights, placeholder_matrix, placeholder_matrix, iso_weights]

path1 = joinpath(scratchspace, "_1.h5")
path2 = joinpath(scratchspace, "_2.h5")
################################################################################################################################
################################################################################################################################
###Function to return a new model configuration object 
function clean_config() 

    task_path = "./tasks.txt"


    task_paths = [task_path, task_path] 
    input_path = ds_path
    num_models = 2
    initial_met_prob = (.1, .9) 
    final_met_prob = (.1,.9)
    
    ###Combine into vector for model configuration object 
    ###It's important to note that len(met_probs) is enforced to be equal to num_models 
    met_probs = [initial_met_prob, final_met_prob]
    
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
    weight_vec = [pw, rw]
    ###Specify a weight vector for each model pass 
    ###len(weight_vector) is enforced to be equal to num_models (should have a set of weights for each pass) 
    task_weights = [weight_vec, weight_vec] 
    
    base_name = joinpath(scratchspace, "raw_model")
    base_name_features = joinpath(scratchspace, "output_features")
    ###List of paths to output trained models to. Enforced to be same size as num_models 
    model_output_paths = [base_name * "_$(i-1).jld2" for i in 1:num_models ]
    ###List of paths to output calculated features to. Enforced to be same size as num_models 
    feature_output_paths = [base_name_features * "_$(i-1).h5" for i in 1:num_models]
    
    
    ###Options are "balanced" or "". If "balanced", the decision trees will be trained 
    ###on a weighted version of the existing classes in order to combat class imbalance 
    class_weights = "balanced"
    
    ###Name of variable in cfradials that has already had interactive QC applied 
    QC_var = "VG"
    
    ###Name of a variable in cfradials that will be used to mask what gates are predicted upon.
    ###Missing values in this variable mean that gates will be removed
    remove_var = "VV"
    ###Name of a variable in input cfradials that has not had postprocessing applied. 
    ###This variable is used to determine where MISSING gates exist in the scan 
    remove_var = "VEL"
    
    ###Whether or not the input features for the model have already been calculated 
    file_preprocessed = [false, false]
    
    ###Where to write out the masks to in cfradial file. 
    mask_names = ["PASS_1_MASK", "PASS_2_MASK"]
    
    
    ###Create model config object
    config = ModelConfig(num_models = num_models,model_output_paths =  model_output_paths,met_probs =  met_probs, 
                        feature_output_paths = feature_output_paths, input_path = input_path,task_mode="nan",file_preprocessed = [false, false],
                         task_paths = task_paths, QC_var = QC_var, remove_var = remove_var, QC_mask = false, mask_names = mask_names,
                         VARS_TO_QC = ["VEL"], class_weights = class_weights, HAS_INTERACTIVE_QC=true, task_weights = task_weights,
                         REMOVE_HIGH_PGG=false, REMOVE_LOW_NCP=false)

end
################################################################################################################################


################################################################################################################################
##Create a toy sized cfradial file that we know the exact values of calculations for 
ds_path = joinpath(scratchspace, "sample_set.nc")
isfile(ds_path) && rm(ds_path)
ds = NCDataset(ds_path, "c")
# import Base.size
# import Base.length 
# function size(v::Matrix{Union{Missing, Float32}}, dim::Int64)
    
#     return 5
# end 

# function length(v::Nothing)
#     return 5
# end 
####start with 5x5 
range_dim = 5
time_dim  = 5

times = collect(1:1:5)
ranges = collect(1:1:5)
alt_step = 5
alts = collect(2000:5:2000 + (length(ranges) - 1) * alt_step)

sample_DBZ = Matrix{Union{Missing, Float32}}(reshape(sample(1:65, range_dim*time_dim),(range_dim, time_dim)))
sample_VEL = Matrix{Float32}(reshape(sample(-20:20, range_dim*time_dim), (range_dim, time_dim)))
sample_NCP = fill(1., (range_dim, time_dim))
sample_NCP[:,1] .= .1

sample_PGG = fill(.1, (range_dim, time_dim))
sample_PGG[1,:] .= 1

sample_VG = Matrix{Union{Missing, Float32}}(sample_VEL)
###Collocate with the low values of NCP 
sample_VG[:,1] .= missing
###Add center ro 
sample_VG[3,:] .= missing 

defDim(ds, "range", range_dim)
defDim(ds, "time", time_dim)

tv = defVar(ds, "time", Float32, ("time",), attrib=Dict("units" => "s"))
tv[:] = times

rv = defVar(ds, "range", Float32, ("range",), attrib=Dict("units" => "m"))
rv[:] = ranges

altitude = defVar(ds, "altitude", Float32, ("time", ), attrib=Dict("units" => "m"))
altitude[:] = alts 

NCP = defVar(ds, "NCP", Float32, ("range", "time"), attrib=Dict("units" => "NCP units"))
NCP[:,:] = sample_NCP 

VEL = defVar(ds, "VEL", Float32, ("range", "time"), attrib=Dict("units" => "m/s"))
VEL[:,:] = sample_VEL

defVar(ds, "DBZ", sample_DBZ, ("range", "time"), attrib=Dict("units" => "log"))
defVar(ds, "PGG", sample_PGG, ("range", "time"), attrib=Dict("units" => "percent"))
defVar(ds, "VG", sample_VG, ("range", "time"), attrib=Dict("units" => "m/s"))

close(ds)
################################################################################################################################

################################################################################################################################
###Tests to ensure that removal based on NCP works as expected 
config = clean_config() 
config.REMOVE_HIGH_PGG = false  
config.REMOVE_LOW_NCP = false  

valid_NCP_gates = sum(sample_NCP .> config.NCP_THRESHOLD)
total_gates = length(sample_DBZ)

config.REMOVE_LOW_NCP = true 

try 
    train_multi_model(config)
catch 

end 

NCDataset(config.feature_output_paths[1]) do f
    @assert size(f["X"][:,:])[1] == valid_NCP_gates
end 

config.REMOVE_LOW_NCP = false 
try 
    train_multi_model(config)
catch 

end 

NCDataset(config.feature_output_paths[1]) do f
    @assert size(f["X"][:,:])[1] == total_gates
end 
################################################################################################################################



################################################################################################################################
###Test to ensure that QC_var is properly passed to the functions 
###Does so by ensuring that the returned target Y array is the same as the 
###specified QC_variable 
config = clean_config() 
VG_map = map(! ismissing, sample_VG)
DBZ_map = map( ! ismissing, sample_DBZ)
@assert DBZ_map != VG_map 
config.QC_var = "VG"
X,Y = calculate_features(config.input_path, config.task_paths[1], config.feature_output_paths[1], config.HAS_INTERACTIVE_QC; 
                                    verbose = config.verbose, REMOVE_LOW_NCP = config.REMOVE_LOW_NCP,NCP_THRESHOLD=config.NCP_THRESHOLD, 
                                    REMOVE_HIGH_PGG=config.REMOVE_HIGH_PGG, PGG_THRESHOLD = config.PGG_THRESHOLD, QC_variable = config.QC_var, 
                                    remove_variable = config.remove_var, replace_missing = config.replace_missing,
                                    write_out = config.write_out)

@assert reshape(Y, (range_dim, time_dim)) == VG_map     

config.QC_var = "DBZ"
X,Y = calculate_features(config.input_path, config.task_paths[1], config.feature_output_paths[1], config.HAS_INTERACTIVE_QC; 
                                    verbose = config.verbose, REMOVE_LOW_NCP = config.REMOVE_LOW_NCP,NCP_THRESHOLD=config.NCP_THRESHOLD, 
                                    REMOVE_HIGH_PGG=config.REMOVE_HIGH_PGG, PGG_THRESHOLD = config.PGG_THRESHOLD, QC_variable = config.QC_var, 
                                    remove_variable = config.remove_var, replace_missing = config.replace_missing,
                                    write_out = config.write_out)

@assert reshape(Y, (range_dim, time_dim)) == DBZ_map
################################################################################################################################

################################################################################################################################
### Test to ensure remove_var is properly passed to calculate_features by checking
### That the shape of the feature array changes associated with the variable passed to it
config = clean_config() 
config.remove_var = "VG"
X,Y = calculate_features(config.input_path, config.task_paths[1], config.feature_output_paths[1], config.HAS_INTERACTIVE_QC; 
                                    verbose = config.verbose, REMOVE_LOW_NCP = config.REMOVE_LOW_NCP,NCP_THRESHOLD=config.NCP_THRESHOLD, 
                                    REMOVE_HIGH_PGG=config.REMOVE_HIGH_PGG, PGG_THRESHOLD = config.PGG_THRESHOLD, QC_variable = config.QC_var, 
                                    remove_variable = config.remove_var, replace_missing = config.replace_missing,
                                    write_out = config.write_out)
@assert size(X)[1] == sum(VG_map)
config.remove_var = "DBZ"
X,Y = calculate_features(config.input_path, config.task_paths[1], config.feature_output_paths[1], config.HAS_INTERACTIVE_QC; 
                                    verbose = config.verbose, REMOVE_LOW_NCP = config.REMOVE_LOW_NCP,NCP_THRESHOLD=config.NCP_THRESHOLD, 
                                    REMOVE_HIGH_PGG=config.REMOVE_HIGH_PGG, PGG_THRESHOLD = config.PGG_THRESHOLD, QC_variable = config.QC_var, 
                                    remove_variable = config.remove_var, replace_missing = config.replace_missing,
                                    write_out = config.write_out)
###make sure we have the same number of features as valid spotsx in the DBZ dataset 
@assert size(X)[1] == sum(DBZ_map)
################################################################################################################################


################################################################################################################################
config = clean_config() 
###Ensure that we have the full number of gates 
@assert sum(DBZ_map) == (range_dim * time_dim)
###Then try and mask something out... first we need to write it to file though 
###Can just used the QC'ed stuff 

config.QC_mask = true 
###Setting the QC mask to VG means that we should ONLY have met_gates in the dataset since we are removing all the 
###non-meteorological ones 
config.mask_names = ["VG", "OK"]

try 
    train_multi_model(config)
catch DomainError 
    println("OK")
    NCDataset(config.feature_output_paths[1]) do f1 
        @assert size(f1["X"][:,:])[1] == sum(.! map(ismissing, sample_VG)) 
    end 
    ###We should get a domain variable because we are removing the non-met 
    ###gates in the first pass 
else 
    @assert false
end 

###Now let's try it without the mask 
config.QC_mask = false 

try 
    train_multi_model(config)
catch DomainError 
    ###possible we're just getting 100% accuracy 
    println("DOMAIN ERROR") 
    NCDataset(config.feature_output_paths[1]) do f1 
        println(size(f1["X"][:,:])[1] )
        @assert size(f1["X"][:,:])[1] == length(sample_DBZ)
    end 
else 
    @assert true
end 


###Also ensure that the function returns an error if the mask name is invalid 
config.mask_names = ["OK"]
try
    train_multi_model(config)
catch AssertionError

else 
    @assert false 
end 

try 
    composite_prediction(config) 
catch AssertionError 
else 
    @assert false 
end 

config.mask_names = ["OK", "MASK_2"]
config.QC_mask = false 

################################################################################################################################


################################################################################################################################
####Test tree depth, n trees, etc. 
config = clean_config() 
config.HAS_INTERACTIVE_QC = true
config.QC_var = "VG"
config.n_trees = 40 
config.max_depth = 20 
train_multi_model(config)
classifier = load_object(config.model_output_paths[1])
@assert classifier.n_trees == config.n_trees 
@assert classifier.max_depth == config.max_depth 
################################################################################################################################

################################################################################################################################
###Create model config object
###Ensure that the file preprocessed flag works correctly by not 
###Modifying the existing features if it's already been processed 
config = ModelConfig(num_models = num_models,model_output_paths =  model_output_paths,met_probs =  met_probs, 
                    feature_output_paths = feature_output_paths, input_path = input_path,task_mode="nan",file_preprocessed = [false, false],
                     task_paths = task_paths, QC_var = QC_var, remove_var = remove_var, QC_mask = false, mask_names = mask_names,
                     VARS_TO_QC = ["VEL"], class_weights = class_weights, HAS_INTERACTIVE_QC=true, task_weights = task_weights,
                     REMOVE_HIGH_PGG=false, REMOVE_LOW_NCP=false)
sleep(1) 

config.file_preprocessed = [false, false] 
train_multi_model(config)
@assert (Base.time() - mtime(config.feature_output_paths[1])) < 1 
sleep(2) 
config.file_preprocessed = [true, true] 
train_multi_model(config) 
@assert (Base.time() - mtime(config.feature_output_paths[1])) > 2

################################################################################################################################



################################################################################################################################
###Ensure that a model cannot be trained on empty data
isfile(config.feature_output_paths[1]) ? rm(config.feature_output_paths[1]) : ""

config = ModelConfig(num_models = num_models,model_output_paths =  model_output_paths,met_probs =  met_probs, 
                    feature_output_paths = feature_output_paths, input_path = input_path,task_mode="nan",file_preprocessed = [false, false],
                     task_paths = task_paths, QC_var = QC_var, remove_var = remove_var, QC_mask = false, mask_names = mask_names,
                     VARS_TO_QC = ["VEL"], class_weights = class_weights, HAS_INTERACTIVE_QC=true, task_weights = task_weights,
                     REMOVE_HIGH_PGG=false, REMOVE_LOW_NCP=false)

train_multi_model(config)


try
    config = ModelConfig(num_models = num_models,model_output_paths =  model_output_paths,met_probs =  met_probs, 
                    feature_output_paths = feature_output_paths, input_path = input_path,task_mode="nan",file_preprocessed = [false, false],
                     task_paths = task_paths, QC_var = QC_var, remove_var = remove_var, QC_mask = false, mask_names = mask_names,
                     VARS_TO_QC = ["VEL"], class_weights = class_weights, HAS_INTERACTIVE_QC=false, task_weights = task_weights,
                     REMOVE_HIGH_PGG=false, REMOVE_LOW_NCP=false)

    train_multi_model(config)
catch Exception 
    println("GOOD!")
else 
    @assert false 
end 
################################################################################################################################

################################################################################################################################
###Test to ensure that Y values are NOT returned if HAS_INTERACTIVE_QC flag is set to false 
config = clean_config()
i = 1
out = config.feature_output_paths[i] 
currt = config.task_paths[i]
cw = config.task_weights[i]
config.write_out = false
config.HAS_INTERACTIVE_QC = false
config.REMOVE_LOW_NCP = true
##If execution proceeds past the first iteration, a composite model is being created, and 
##so a further mask will be applied to the features 
if i > 1
    QC_mask = true 
else 
    QC_mask = config.QC_mask 
end 

QC_mask ? mask_name = config.mask_names[i] : mask_name = ""
    
X,Y,idxer = calculate_features(config.input_path, currt, out, config.HAS_INTERACTIVE_QC; 
                                    verbose = config.verbose, REMOVE_LOW_NCP = config.REMOVE_LOW_NCP,NCP_THRESHOLD=config.NCP_THRESHOLD, 
                                    REMOVE_HIGH_PGG=config.REMOVE_HIGH_PGG, PGG_THRESHOLD = config.PGG_THRESHOLD, QC_variable = config.QC_var, 
                                    remove_variable = config.remove_var, replace_missing = config.replace_missing,
                                    write_out = config.write_out, QC_mask = QC_mask, mask_name = mask_name, weight_matrixes=cw, return_idxer = true)
@assert Y == [0;;]
@assert sum(idxer[1][:]) == sum(sample_NCP .> config.NCP_THRESHOLD)
################################################################################################################################


################################################################################################################################
###Test the evaluate model with a vector of predictions and targets so we ensure that it returns the correct values
###1 true positive, 1 false positive, 1 true negative, 1 false negative. 
###Should have Precision of 1/2 and Recall of 1/2 
sample_predictions = Vector{Bool}([1,1,0,0])
sample_targets     = Vector{Bool}([1,0,1,0])
prec, recall, f1, tp, tn, fp, fn, n = evaluate_model(sample_predictions, sample_targets)

@assert prec == .5 
@assert recall == .5
@assert f1 == (2/((1/prec) + (1/recall)))
@assert tp == tn == fp == fn == 1
@assert n == length(sample_predictions) == length(sample_targets)
################################################################################################################################



################################################################################################################################
###Test evaluate_model functionality 
##Test pretrained versus non-pretrained models 
config = clean_config() 
config.input_path = NOAA_PATH

for model_path in config.model_output_paths
    isfile(model_path) ? rm(model_path) : ""
end 

model_stats = evaluate_model(config) 
## Ensure model files exist 
for model_path in config.model_output_paths 
    @assert isfile(model_path) 
end 
sleep(5) 
## Run it again, ensure these files are not overwritten 
model_stats_two = evaluate_model(config; models_trained=true)
for model_path in config.model_output_paths 
    @assert (Base.time() - mtime(model_path)) > 5 
end 

@assert model_stats == model_stats_two 

###Apply interacive version of the quality control and ensure that results are consistent 
predictions = [] 
targets = [] 
###Test this by interactively going through the models
###start by opening the first model 
predictions = [] 
targets = [] 
###Test this by interactively going through the models
###start by opening the first model 
for (i, model) in enumerate(config.model_output_paths)
    currm = load_object(model) 

    currh5 = h5open(config.feature_output_paths[i])
    currfeatures = currh5["X"][:,:]
    currtargets = currh5["Y"][:,:][:]
    close(currh5) 

    push!(predictions, DecisionTree.predict_proba(currm, currfeatures))
    push!(targets, currtargets)

    ###Construct prediction vector  
end 


total_predictions = fill(-1, length(predictions[1][:,2]))
total_targets =  fill(-1, length(predictions[1][:,2]))
idxer = fill(true, length(predictions[1][:,2]))
still_to_predict = fill(true, length(predictions[1][:,2]))

###iteratively construct predictions vector 
###THIS IS ALSO TECHNICALLY NOT TESTING THIS ON A NEW SET OF FEATURES, BUT RATHER ONES THAT 
###HAVE ALREADY BEEN CALCULATED 
for (i, prediction_vec) in enumerate(predictions) 

    ###All the gates are still to be predicted upon 
    cp_mps = predictions[i][:,2]
    cp_metprobs = config.met_probs[i]
    ###Subset of gates from current pass to be predicted upon 
    curr_idx = (cp_mps .< cp_metprobs[1]) .| (cp_mps .> cp_metprobs[2])
    println(sum(curr_idx))
    curr_predictions = cp_mps .> fp_metprobs[2]
    ###We predict where both still_to_predict is 1, and curr_idx is 1 
    ###Still_to_predict will have a value of 1s at all locations 
    ###Just overwrite the next set of predictions too 

    ###At the valid locations in the current idxer, we will write gates 
    total_predictions[still_to_predict] .= curr_predictions 
    still_to_predict[still_to_predict] .= .! curr_idx

end 
_prec, _rec, _f1, _tp, _fp, _tn, _fn, _tot = evaluate_model(Vector{Bool}(total_predictions), Vector{Bool}(targets[1])) 
@assert _prec == model_stats[1, "precision"] 
@assert _rec  == model_stats[1, "recall"]
@assert _f1   == model_stats[1, "f1"]
@assert _fp   == model_stats[1, "false_positives"]
################################################################################################################################


###############################################Test `Write_field`###############################################################
###Start by making it create a new NetCDF file 
nc_out_path = joinpath(scratchspace, "sample.nc") 
###Write out sample_DBZ 
write_field(nc_out_path, "SAMPLE_DBZ", sample_DBZ, true) 
###Test that it was written correctly 
NCDataset(nc_out_path) do f 
    @assert f["SAMPLE_DBZ"][:,:] == sample_DBZ
end 
###Test overwriting 
new_DBZ = fill(0.f0, size(sample_DBZ))
write_field(nc_out_path, "SAMPLE_DBZ", new_DBZ, true) 
NCDataset(nc_out_path) do f 
    @assert f["SAMPLE_DBZ"][:,:] == sample_DBZ 
end 
###Make sure it doesn't overwrite if overwrite is set to false 
try     
    write_field(nc_out_path, "SAMPLE_DBZ", sample_DBZ; overwrite=false, attribs=Dict())
catch Exception 
     
else 
    @assert false 
end 

test_attrs = Dict("TEST_ATTR_ONE" => 3.14159265, "TEST_ATTR_TWO" => "Pi")
write_field(nc_out_path, "SAMPLE_DBZ"; overwrite=true, attribs=test_attrs)
read_attrs = NCDataset(nc_out_path, "r") do currf
    Dict(currf["SAMPLE_DBZ"].attrib)
end 
@assert test_attrs["TEST_ATTR_ONE"] == read_attrs["TEST_ATTR_ONE"] 

################################################################################################################################



################################################################################################################################
###Test that missing_std and missing_avg have expected behavior 
@assert Ronin.missing_std([missing, missing]) == Ronin.FILL_VAL
@assert Ronin.missing_std([missing, 3, 4])    == std([3, 4]) 
@assert Ronin.missing_avg([missing, missing]) == Ronin.FILL_VAL
@assert Ronin.missing_avg([missing, 3, 4])    == avg([3,4])
################################################################################################################################

################################################################################################################################
###Test that calc_ncp and calc_rng function as expected 

alt_map = repeat(transpose(alts),length(ranges), 1)
rng_map = repeat(ranges, 1, length(times))

NCDataset(ds_path) do ds 
    @assert Ronin.calc_ncp(ds) == Vector{Float32}(sample_NCP[:])
    @assert Ronin.calc_rng(ds) == rng_map
    @assert Ronin.calc_nrg(ds) == Matrix{Float32}(rng_map ./ alt_map)
end 
################################################################################################################################

################################################################################################################################
###Test that all three multiple dispatch versions of _weighted_func work as expected 
@assert Ronin._weighted_func([1 2; 1 2], [3 4; 3 4], std) == std([1*3 2*4; 1*3 2*4])

v1 = AbstractMatrix{Union{Float32, Missing}}([1 missing; 1 2])
v3 = Matrix{Float32}([1 2; 1 2])
v2 = AbstractMatrix{Union{Missing, Float32}}([3 4; 3 4])
@assert Ronin._weighted_func(v1, v2, Ronin.missing_std) == Float32(std([3,3,8]))

@assert Ronin._weighted_func(v3, v2, Ronin.missing_std) == Float32(std([3 8; 3 8]))
################################################################################################################################

################################################################################################################################
##Test isolation calculation functions 
sample_iso_array = [1.f0 missing 1.f0; 1.f0 1.f0 missing; missing missing missing]
expected_iso = [1.0f0 2.0f0 2.0f0; 3.0f0 5.0f0 5.0f0; 2.0f0 4.0f0 3.0f0]
calced_iso = Ronin.calc_iso(sample_iso_array; weights = fill(1.f0, (3,3)), window=(3,3))
@assert calced_iso == expected_iso 

###Test multiple dispatch version with weights maybe 

sample_iso_array = Matrix{Union{Float32, Missing}}(fill(1.0f0, (3,3)))
sample_iso_weights = Matrix{Union{Missing, Float32}}(fill(missing, (3,3)))
@assert map(ismissing, Ronin.calc_iso(sample_iso_array; weights=sample_iso_weights, window=(3,3))) == fill(true, (3,3))
################################################################################################################################


################################################################################################################################
###Test get_num_tasks 



################################################################################################################################