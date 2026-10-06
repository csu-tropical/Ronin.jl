using NCDatasets
using Dates
using Printf

# ============================================================
# DBZ offset preprocessing
#
# This script is included after 00_config.jl and before the
# data split. It does not overwrite CASE_PATHS.
#
# It:
#   1. Reads original CASE_PATHS from 00_config.jl
#   2. Copies files from DATA to DATA_DBZOffsetCorrected
#   3. Applies the aircraft-specific DBZ offset to copied files only
#   4. Creates DBZ_OFFSET_CASE_PATHS for downstream splitting
#
# CASE_PATHS remains original.
# DBZ_OFFSET_CASE_PATHS points to corrected data.
# ============================================================

@isdefined(CASE_PATHS) || error("CASE_PATHS must be defined in 00_config.jl")

if !@isdefined(DBZ_OFFSET_FOLDER)
    DBZ_OFFSET_FOLDER = "DATA_DBZOffsetCorrected"
end

INNER_KM = 5.6

function root_from_case_path(case_path::String)
    marker = "/DATA/"
    r = findfirst(marker, case_path)

    if r === nothing
        error("Could not find /DATA/ in CASE_PATH: $case_path")
    end

    return case_path[1:first(r)-1]
end

ROOT = root_from_case_path(CASE_PATHS[1])
SRC_ROOT = joinpath(ROOT, "DATA")
DST_ROOT = joinpath(ROOT, DBZ_OFFSET_FOLDER)

function nc_files(path::String)
    files = String[]
    for (root, _, fs) in walkdir(path)
        for f in fs
            if endswith(f, ".nc") && startswith(f, "cfrad.")
                push!(files, joinpath(root, f))
            end
        end
    end
    return sort(files)
end

function range_to_km(r)
    r_float = Float64.(r)

    # Most CfRadial range variables are in meters.
    # If values are already small, assume they are km.
    if maximum(skipmissing(r_float)) > 100.0
        return r_float ./ 1000.0
    else
        return r_float
    end
end

function offset_for_case_path(case_path::String)
    if occursin("/N42/", case_path) || occursin("N42", case_path)
        return 9.0f0
    elseif occursin("/N43/", case_path) || occursin("N43", case_path)
        return 5.0f0
    else
        error("Could not determine DBZ offset for CASE_PATH: $case_path")
    end
end

function corrected_case_path(case_path::String)
    relpath_from_data = relpath(case_path, SRC_ROOT)

    if startswith(relpath_from_data, "..")
        error("CASE_PATH is not inside DATA root: $case_path")
    end

    return joinpath(DST_ROOT, relpath_from_data)
end

function correct_file!(file::String, offset_db::Float32)
    NCDataset(file, "a") do ds
        if !haskey(ds, "DBZ")
            @warn "No DBZ field found; skipping" file
            return false
        end

        if !haskey(ds, "range")
            @warn "No range variable found; skipping" file
            return false
        end

        dbz = ds["DBZ"]
        range_vals = ds["range"][:]
        range_km = range_to_km(range_vals)
        inner_mask = range_km .<= INNER_KM

        # Keep DBZ as a 2D array.
        data = dbz[:, :]

        if ndims(data) != 2
            @warn "DBZ is not 2D; skipping" file size=size(data)
            return false
        end

        # Apply offset only inside 5.6 km, matching whichever DBZ dimension is range.
        if size(data, 1) == length(inner_mask)
            data[inner_mask, :] .= data[inner_mask, :] .+ offset_db
        elseif size(data, 2) == length(inner_mask)
            data[:, inner_mask] .= data[:, inner_mask] .+ offset_db
        else
            @warn "Could not match range dimension to DBZ; skipping" file dbz_size=size(data) range_length=length(inner_mask)
            return false
        end

        # Write DBZ back as 2D.
        dbz[:, :] = data

        ds.attrib["ronin_dbz_offset_corrected"] = "true"
        ds.attrib["ronin_dbz_offset_db"] = string(offset_db)
        ds.attrib["ronin_dbz_offset_inner_km"] = string(INNER_KM)
        ds.attrib["ronin_dbz_offset_timestamp_utc"] = string(now(UTC))

        return true
    end
end

println()
println("============================================================")
println("Running DBZ offset correction preprocessing")
println("Source root:      ", SRC_ROOT)
println("Destination root: ", DST_ROOT)
println("Inner range:      ", INNER_KM, " km")
println("============================================================")

DBZ_OFFSET_CASE_PATHS = String[]

grand_total = 0
grand_corrected = 0

for case_path in CASE_PATHS
    global grand_total, grand_corrected

    offset_db = offset_for_case_path(case_path)

    src_dir = case_path
    dst_dir = corrected_case_path(case_path)

    push!(DBZ_OFFSET_CASE_PATHS, dst_dir)

    println()
    println("------------------------------------------------------------")
    println("Original CASE_PATH:  ", src_dir)
    println("Corrected CASE_PATH: ", dst_dir)
    println("Offset:              +", offset_db, " dB inside ", INNER_KM, " km")
    println("------------------------------------------------------------")

    files = nc_files(src_dir)
    println("Found ", length(files), " source cfrad files")

    mkpath(dst_dir)

    corrected = 0

    for (i, src_file) in enumerate(files)
        dst_file = joinpath(dst_dir, basename(src_file))

        # Always copy from original DATA so rerunning does not double-apply offset.
        cp(realpath(src_file), dst_file; force=true)

        ok = correct_file!(dst_file, offset_db)
        corrected += ok ? 1 : 0

        if i % 25 == 0 || i == length(files)
            @printf("  %d/%d files copied/corrected\n", i, length(files))
            flush(stdout)
        end
    end

    println("Corrected ", corrected, " of ", length(files), " files")

    grand_total += length(files)
    grand_corrected += corrected
end

println()
println("DONE DBZ OFFSET PREPROCESSING")
println("Total files found:     ", grand_total)
println("Total files corrected: ", grand_corrected)
println("Corrected data root:   ", DST_ROOT)

println()
println("Original CASE_PATHS remain unchanged:")
for p in CASE_PATHS
    println("  ", p)
end

println()
println("DBZ_OFFSET_CASE_PATHS created for downstream split:")
for p in DBZ_OFFSET_CASE_PATHS
    println("  ", p)
end
println()
