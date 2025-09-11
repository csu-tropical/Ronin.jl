using Pkg 

ronin_dir = "../"
println("")
println("")
println("###############---------------###############")
println("##      RONIN REALTIME PROCESSING SCRIPT   ##")
println("##                 V1.0.0                  ##")
println("###############---------------###############")
println("")
println("")

println("CURRENT RONIN DIRECTORY: $(abspath(ronin_dir))")
println("IS THIS CORRECT? (Y/N)") 
valid = lowercase(readline())

while valid == "n" 
    println("PLEASE ENTER PATH TO RONIN DIRECTORY") 
    
    global ronin_dir = readline() 
    println("CURRENT RONIN DIRECTORY: $(abspath(ronin_dir))")
    println("IS THIS CORRECT? (Y/N)") 
    global valid = readline()
end 

Pkg.activate(ronin_dir) 
using NCDatasets, JLD2, Ronin 

println("Please enter model configuration path: ")
config_path = readline()
println("")
println("") 

println("Please enter path to streaming directory (where cfradial files are written)") 
inpath = readline() 
println("") 
println("") 

println("Please enter path to output directory (where to move cfradials after they are QC'ed)") 
outpath = readline() 
println("")
println("")

println("BEGINNING PROCESSING FOR $(inpath)")
println("") 
printstyled("LOADING MODEL CONFIGURATION... \n", color=:green)
println("")
curr_config = load_object(config_path) 

printstyled("LOADING MODELS... \n", color=:green)
models = [load_object(model) for model in curr_config.model_output_paths]
printstyled("\n BEGINNING PROCESSING\n", color=:green) 
while true 
    fnames = readdir(inpath)
    curr_fs = [joinpath(inpath, currn) for currn in fnames] 
    for (fpath, fname) in zip(curr_fs, fnames)
        composite_QC(curr_config, [fpath], models) 
        mv(fpath, joinpath(outpath, fname))
    end 
    sleep(.1)  
end 


