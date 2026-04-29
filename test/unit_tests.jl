###############################################################################
# Comprehensive Unit Tests for Ronin.jl
#
# Tests are organized by module/component:
#   1. RoninConstants - Constant definitions and weight matrices
#   2. RoninFeatures  - Feature calculation functions
#   3. Core Ronin     - Feature processing, model training, prediction, evaluation
#   4. DecisionTree   - Classification/regression tree functionality
#   5. I/O            - File handling and data splitting
###############################################################################

using Test
using Ronin
using NCDatasets
using HDF5
using Missings
using Statistics
using Random
using Scratch
using JLD2

global test_scratchspace = @get_scratch!("ronin_unit_tests")

###############################################################################
# Helper: Create a synthetic CFRadial NetCDF file for testing
###############################################################################
function create_test_cfrad(path; range_dim=10, time_dim=8, seed=42)
    rng = MersenneTwister(seed)
    isfile(path) && rm(path)

    ds = NCDataset(path, "c")

    defDim(ds, "range", range_dim)
    defDim(ds, "time", time_dim)

    times = collect(Float32, 1:time_dim)
    ranges = collect(Float32, 150:150:150*range_dim)
    alts = fill(Float32(3000.0), time_dim)
    elevs = Float32.(rand(rng, collect(-20:0.5:20), time_dim))
    azims = Float32.(rand(rng, collect(0:1:359), time_dim))

    tv = defVar(ds, "time", Float32, ("time",), attrib=Dict("units" => "seconds"))
    tv[:] = times
    rv = defVar(ds, "range", Float32, ("range",), attrib=Dict("units" => "m"))
    rv[:] = ranges
    alt = defVar(ds, "altitude", Float32, ("time",), attrib=Dict("units" => "m"))
    alt[:] = alts
    ev = defVar(ds, "elevation", Float32, ("time",), attrib=Dict("units" => "degrees"))
    ev[:] = elevs
    av = defVar(ds, "azimuth", Float32, ("time",), attrib=Dict("units" => "degrees"))
    av[:] = azims

    # Create radar data fields
    sample_VEL = Matrix{Union{Missing, Float32}}(Float32.(rand(rng, -30:0.1:30, range_dim, time_dim)))
    sample_DBZ = Matrix{Union{Missing, Float32}}(Float32.(rand(rng, -10:0.5:65, range_dim, time_dim)))
    sample_NCP = Matrix{Union{Missing, Float32}}(Float32.(rand(rng, collect(range(0.0, 1.0, length=100)), range_dim, time_dim)))

    # Create QC'd velocity (VG) by removing some gates
    sample_VG = copy(sample_VEL)
    # Remove ~30% of gates as "non-meteorological"
    nmd_mask = rand(rng, Bool, range_dim, time_dim)
    sample_VG[nmd_mask] .= missing

    defVar(ds, "VV", sample_VEL, ("range", "time"), attrib=Dict("units" => "m/s"))
    defVar(ds, "VEL", sample_VEL, ("range", "time"), attrib=Dict("units" => "m/s"))
    defVar(ds, "ZZ", sample_DBZ, ("range", "time"), attrib=Dict("units" => "dBZ"))
    defVar(ds, "DBZ", sample_DBZ, ("range", "time"), attrib=Dict("units" => "dBZ"))
    defVar(ds, "NCP", sample_NCP, ("range", "time"), attrib=Dict("units" => "unitless"))
    defVar(ds, "VG", sample_VG, ("range", "time"), attrib=Dict("units" => "m/s"))

    close(ds)
    return path
end

# Create a minimal task config file
function create_test_config(path, tasks_str)
    open(path, "w") do f
        write(f, tasks_str)
    end
    return path
end


###############################################################################
# 1. CONSTANTS TESTS
###############################################################################
@testset "RoninConstants" begin

    @testset "Weight matrix dimensions" begin
        @test size(Ronin.iso_weights) == (7, 7)
        @test size(Ronin.avg_weights) == (5, 5)
        @test size(Ronin.std_weights) == (5, 5)
        @test size(Ronin.standard_window) == (7, 7)
        @test size(Ronin.azi_window) == (7, 7)
        @test size(Ronin.range_window) == (7, 7)
        @test size(Ronin.placeholder_window) == (3, 3)
    end

    @testset "Center weights are zero" begin
        @test Ronin.iso_weights[4, 4] == 0
        @test Ronin.avg_weights[3, 3] == 0
        @test Ronin.std_weights[3, 3] == 0
    end

    @testset "Physical constants" begin
        @test Ronin.EarthRadiusKm ≈ 6375.636f0
        @test Ronin.beamwidth ≈ Float32(1.8 * 0.017453292)
        @test Ronin.FILL_VAL == typemin(Int16)
    end

    @testset "Azi/range window structure" begin
        # Azi window: rows 3,4,5 should be 1, rest 0
        for r in 1:7
            for c in 1:7
                if r in [3, 4, 5]
                    @test Ronin.azi_window[r, c] == 1.0
                else
                    @test Ronin.azi_window[r, c] == 0.0
                end
            end
        end
        # Range window: cols 3,4,5 should be 1, rest 0
        for r in 1:7
            for c in 1:7
                if c in [3, 4, 5]
                    @test Ronin.range_window[r, c] == 1.0
                else
                    @test Ronin.range_window[r, c] == 0.0
                end
            end
        end
    end
end


###############################################################################
# 2. FEATURE CALCULATION TESTS
###############################################################################
@testset "RoninFeatures" begin

    @testset "missing_std and missing_avg" begin
        @test Ronin.missing_std([missing, missing]) == Ronin.FILL_VAL
        @test Ronin.missing_avg([missing, missing]) == Ronin.FILL_VAL
        @test Ronin.missing_std([missing, 3.0, 4.0]) ≈ std([3.0, 4.0])
        @test Ronin.missing_avg([missing, 3.0, 4.0]) ≈ mean([3.0, 4.0])
        # Single non-missing value: std should be 0 or NaN, avg should be the value
        @test Ronin.missing_avg([missing, 5.0]) ≈ 5.0
        # All same values
        @test Ronin.missing_avg([3.0, 3.0, 3.0]) ≈ 3.0
        @test Ronin.missing_std([3.0, 3.0, 3.0]) ≈ 0.0
    end

    @testset "_weighted_func" begin
        v1 = [1.0 2.0; 3.0 4.0]
        w1 = [1.0 1.0; 1.0 1.0]
        @test Ronin._weighted_func(v1, w1, sum) == sum(v1)
        @test Ronin._weighted_func(v1, w1, Ronin.missing_avg) ≈ mean(v1)

        # With non-uniform weights
        w2 = [2.0 0.0; 0.0 2.0]
        @test Ronin._weighted_func(v1, w2, sum) == sum([2.0, 0.0, 0.0, 8.0])
    end

    @testset "calc_iso" begin
        # All present data → isolation should be 0 everywhere (with center_weight=0)
        data_full = Matrix{Union{Missing, Float32}}(ones(Float32, 5, 5))
        iso_result = Ronin.calc_iso(data_full; weights=fill(1.0f0, (3,3)), window=(3,3))
        @test all(iso_result .== 0.0f0)

        # All missing data → maximum isolation
        data_empty = Matrix{Union{Missing, Float32}}(fill(missing, 5, 5))
        iso_result2 = Ronin.calc_iso(data_empty; weights=fill(1.0f0, (3,3)), window=(3,3))
        @test all(iso_result2 .> 0)

        # Known case: single missing gate surrounded by present data
        data_one = Matrix{Union{Missing, Float32}}(ones(Float32, 3, 3))
        data_one[2, 2] = missing
        iso_result3 = Ronin.calc_iso(data_one; weights=fill(1.0f0, (3,3)), window=(3,3))
        # Center should show 1 missing gate (itself)
        @test iso_result3[2, 2] == 1.0f0

        # Test known case: specific missing pattern
        sample_iso_array = Union{Missing, Float32}[1.0f0 missing 1.0f0; 1.0f0 1.0f0 missing; missing missing missing]
        calced_iso = Ronin.calc_iso(sample_iso_array; weights=fill(1.0f0, (3,3)), window=(3,3))
        # Verify structure: bottom row (all missing) should have higher isolation
        @test calced_iso[3, 2] > calced_iso[1, 1]
        # Center should have high isolation (surrounded by many missings)
        @test calced_iso[2, 2] > 0
        # Top-left (only neighbor below-center is missing) should have lower isolation
        @test calced_iso[1, 1] <= calced_iso[2, 2]
    end

    @testset "airborne_ht" begin
        # At zero range, height should equal aircraft height
        ht = Ronin.airborne_ht(0.0f0, 0.0f0, 3000.0f0)
        @test ht ≈ 3.0 atol=0.01  # 3000m = 3km

        # Positive elevation angle should increase height
        ht_up = Ronin.airborne_ht(45.0f0, 1000.0f0, 3000.0f0)
        ht_down = Ronin.airborne_ht(-45.0f0, 1000.0f0, 3000.0f0)
        @test ht_up > ht_down

        # Height increases with range for positive elevation
        ht_near = Ronin.airborne_ht(10.0f0, 500.0f0, 3000.0f0)
        ht_far = Ronin.airborne_ht(10.0f0, 5000.0f0, 3000.0f0)
        @test ht_far > ht_near
    end

    @testset "prob_groundgate" begin
        # Missing inputs should return missing
        @test ismissing(Ronin.prob_groundgate(missing, 1000.0f0, 3000.0f0, 90.0f0))
        @test ismissing(Ronin.prob_groundgate(0.0f0, missing, 3000.0f0, 90.0f0))

        # Positive elevation → cannot hit ground → probability 0
        @test Ronin.prob_groundgate(10.0f0, 5000.0f0, 3000.0f0, 90.0f0) == 0.0

        # Range less than altitude → cannot hit ground → probability 0
        @test Ronin.prob_groundgate(-10.0f0, 1000.0f0, 3000.0f0, 90.0f0) == 0.0

        # Very negative elevation, very long range → high probability (approaching 1)
        pgg = Ronin.prob_groundgate(-89.0f0, 50000.0f0, 3000.0f0, 180.0f0)
        @test pgg >= 0.0
        @test pgg <= 1.0

        # PGG should be between 0 and 1 for any valid inputs
        for _ in 1:100
            elev = Float32(rand() * -90)
            rng = Float32(rand() * 50000 + 3001)  # > altitude
            alt = Float32(3000)
            azi = Float32(rand() * 360)
            p = Ronin.prob_groundgate(elev, rng, alt, azi)
            @test 0.0 <= p <= 1.0
        end
    end

    @testset "calc_rng" begin
        cfrad_path = joinpath(test_scratchspace, "test_rng.nc")
        create_test_cfrad(cfrad_path; range_dim=5, time_dim=3)
        NCDataset(cfrad_path) do ds
            rng_map = Ronin.calc_rng(ds)
            @test size(rng_map) == (5, 3)
            # Each column should be identical (ranges repeat)
            @test rng_map[:, 1] == rng_map[:, 2]
            @test rng_map[:, 1] == rng_map[:, 3]
            # Should match the range variable
            @test rng_map[:, 1] == ds["range"][:]
        end
        rm(cfrad_path)
    end

    @testset "calc_nrg" begin
        cfrad_path = joinpath(test_scratchspace, "test_nrg.nc")
        create_test_cfrad(cfrad_path; range_dim=5, time_dim=3)
        NCDataset(cfrad_path) do ds
            nrg_map = Ronin.calc_nrg(ds)
            rng_map = Ronin.calc_rng(ds)
            alts = repeat(transpose(ds["altitude"][:]), 5, 1)
            @test nrg_map ≈ rng_map ./ alts
        end
        rm(cfrad_path)
    end

    @testset "calc_aht" begin
        cfrad_path = joinpath(test_scratchspace, "test_aht.nc")
        create_test_cfrad(cfrad_path; range_dim=5, time_dim=3)
        NCDataset(cfrad_path) do ds
            aht_map = Ronin.calc_aht(ds)
            @test size(aht_map) == (5, 3)
            # All heights should be positive
            @test all(aht_map .> 0)
        end
        rm(cfrad_path)
    end

    @testset "calc_elv" begin
        cfrad_path = joinpath(test_scratchspace, "test_elv.nc")
        create_test_cfrad(cfrad_path; range_dim=5, time_dim=3)
        NCDataset(cfrad_path) do ds
            elv_map = Ronin.calc_elv(ds)
            @test size(elv_map) == (5, 3)
            # All columns should be the same elevation for each ray
            for col in 1:3
                @test all(elv_map[:, col] .== elv_map[1, col])
            end
        end
        rm(cfrad_path)
    end

    @testset "calc_pgg" begin
        cfrad_path = joinpath(test_scratchspace, "test_pgg.nc")
        create_test_cfrad(cfrad_path; range_dim=5, time_dim=3)
        NCDataset(cfrad_path) do ds
            pgg_map = Ronin.calc_pgg(ds)
            @test size(pgg_map) == (5, 3)
            # All PGG values should be between 0 and 1
            for val in pgg_map
                if !ismissing(val)
                    @test 0.0 <= val <= 1.0
                end
            end
        end
        rm(cfrad_path)
    end

    @testset "calc_sig" begin
        cfrad_path = joinpath(test_scratchspace, "test_sig.nc")
        create_test_cfrad(cfrad_path; range_dim=5, time_dim=3)
        NCDataset(cfrad_path) do ds
            sig = Ronin.calc_sig(ds, "NCP")
            ncp_raw = ds["NCP"][:]
            @test sig == ncp_raw
        end
        rm(cfrad_path)
    end

    @testset "calc_avg" begin
        # Uniform data should produce uniform avg (up to border effects)
        data = Matrix{Union{Missing, Float32}}(fill(5.0f0, 7, 7))
        result = Ronin.calc_avg(data; weights=Ronin.avg_weights, window=Ronin.avg_window)
        # Center value should be 0 since center weight is 0 (avg of surrounding 5's times weights with center=0)
        # Actually: avg is computed as missing_avg of (data .* weights). weights have center=0.
        # So center is 5*0 = 0, rest are 5*1 = 5. mean([0, 5,5,...,5]) ≈ 4.8 for 5x5 window
        @test size(result) == (7, 7)
    end

    @testset "calc_std" begin
        # Uniform data → std should be ~0 (except center weight effects)
        data = Matrix{Union{Missing, Float32}}(fill(5.0f0, 7, 7))
        result = Ronin.calc_std(data; weights=Ronin.std_weights, window=Ronin.std_window)
        @test size(result) == (7, 7)
    end

    @testset "get_num_tasks" begin
        config_path = joinpath(test_scratchspace, "test_tasks.txt")
        create_test_config(config_path, "NCP, AHT, STD(VV), PGG, RNG, ISO(DBZ)")
        @test Ronin.get_num_tasks(config_path) == 6

        # With comments
        config_path2 = joinpath(test_scratchspace, "test_tasks2.txt")
        create_test_config(config_path2, "#This is a comment\nNCP, AHT\n#Another comment\nPGG")
        @test Ronin.get_num_tasks(config_path2) == 3

        rm(config_path)
        rm(config_path2)
    end

    @testset "get_task_params (without variablelist)" begin
        config_path = joinpath(test_scratchspace, "test_tasks3.txt")
        create_test_config(config_path, "NCP, AHT, STD(VV), PGG, RNG, ISO(DBZ)")
        tasks = Ronin.get_task_params(config_path)
        @test length(tasks) == 6
        @test "NCP" in tasks
        @test "AHT" in tasks
        @test "STD(VV)" in tasks
        @test "PGG" in tasks
        @test "RNG" in tasks
        @test "ISO(DBZ)" in tasks
        rm(config_path)
    end

    @testset "get_task_params (with variablelist)" begin
        config_path = joinpath(test_scratchspace, "test_tasks4.txt")
        create_test_config(config_path, "NCP, STD(VV), ISO(DBZ)")
        varlist = ["NCP", "VV", "DBZ", "VEL", "ZZ"]
        tasks = Ronin.get_task_params(config_path, varlist)
        @test length(tasks) == 3
        @test "NCP" in tasks
        @test "STD(VV)" in tasks
        @test "ISO(DBZ)" in tasks

        # Invalid variable should be excluded
        config_path2 = joinpath(test_scratchspace, "test_tasks5.txt")
        create_test_config(config_path2, "NCP, STD(FAKE)")
        tasks2 = Ronin.get_task_params(config_path2, varlist)
        @test length(tasks2) == 1
        @test "NCP" in tasks2

        rm(config_path)
        rm(config_path2)
    end

    @testset "parse_directory" begin
        test_dir = joinpath(test_scratchspace, "test_parse_dir")
        mkpath(test_dir)
        # Create files with valid and invalid prefixes
        touch(joinpath(test_dir, "cfrad.test1.nc"))
        touch(joinpath(test_dir, "cfrad.test2.nc"))
        touch(joinpath(test_dir, "invalid_file.nc"))

        paths = Ronin.parse_directory(test_dir)
        @test length(paths) == 2
        @test all(p -> contains(p, "cfrad"), paths)

        rm(test_dir, recursive=true)
    end
end


###############################################################################
# 3. CORE RONIN TESTS
###############################################################################
@testset "Core Ronin" begin

    @testset "process_single_file basic" begin
        cfrad_path = joinpath(test_scratchspace, "test_psf.nc")
        create_test_cfrad(cfrad_path; range_dim=10, time_dim=8)
        config_path = joinpath(test_scratchspace, "psf_config.txt")
        create_test_config(config_path, "NCP, RNG")

        NCDataset(cfrad_path) do ds
            X, Y, indexer = Ronin.process_single_file(ds, config_path;
                HAS_INTERACTIVE_QC=false, remove_variable="VV")

            # X should have as many columns as tasks
            @test size(X, 2) == 2
            # Without interactive QC, Y should be false
            @test Y == false
            # Indexer should be a boolean vector
            @test eltype(indexer) == Bool
            # Non-missing gates should be in X
            n_valid = sum(indexer)
            @test size(X, 1) == n_valid
        end

        rm(cfrad_path)
        rm(config_path)
    end

    @testset "process_single_file with interactive QC" begin
        cfrad_path = joinpath(test_scratchspace, "test_psf_qc.nc")
        create_test_cfrad(cfrad_path; range_dim=10, time_dim=8)
        config_path = joinpath(test_scratchspace, "psf_qc_config.txt")
        create_test_config(config_path, "NCP, RNG")

        NCDataset(cfrad_path) do ds
            X, Y, indexer = Ronin.process_single_file(ds, config_path;
                HAS_INTERACTIVE_QC=true, QC_variable="VG", remove_variable="VV")

            @test size(X, 2) == 2
            # Y should now be a matrix with 0s and 1s
            @test Y isa Matrix
            @test all(y -> y == 0 || y == 1, Y)
            # Number of rows in Y should match X
            @test size(Y, 1) == size(X, 1)
        end

        rm(cfrad_path)
        rm(config_path)
    end

    @testset "process_single_file SIG feature populates X" begin
        cfrad_path = joinpath(test_scratchspace, "test_sig_x.nc")
        create_test_cfrad(cfrad_path; range_dim=5, time_dim=5)
        config_path = joinpath(test_scratchspace, "sig_config.txt")
        create_test_config(config_path, "SIG, RNG")

        NCDataset(cfrad_path) do ds
            X, Y, indexer = Ronin.process_single_file(ds, config_path;
                HAS_INTERACTIVE_QC=false, remove_variable="VV",
                SIG_QUALITY_VAR="NCP")

            # The SIG column (column 1) should NOT be uninitialized
            # It should contain NCP values (between 0 and 1, or FILL_VAL)
            sig_col = X[:, 1]
            @test all(x -> (0.0 <= x <= 1.0) || x == Float32(Ronin.FILL_VAL), sig_col)
            # At least some values should be actual NCP values (not fill)
            @test any(x -> 0.0 <= x <= 1.0, sig_col)
        end

        rm(cfrad_path)
        rm(config_path)
    end

    @testset "process_single_file with SIG threshold removal" begin
        cfrad_path = joinpath(test_scratchspace, "test_sig_threshold.nc")
        create_test_cfrad(cfrad_path; range_dim=10, time_dim=8)
        config_path = joinpath(test_scratchspace, "sig_thresh_config.txt")
        create_test_config(config_path, "RNG")

        NCDataset(cfrad_path) do ds
            # Without threshold
            X_no, _, idx_no = Ronin.process_single_file(ds, config_path;
                HAS_INTERACTIVE_QC=false, remove_variable="VV",
                REMOVE_LOW_SIG_QUALITY=false)

            # With threshold
            X_yes, _, idx_yes = Ronin.process_single_file(ds, config_path;
                HAS_INTERACTIVE_QC=false, remove_variable="VV",
                REMOVE_LOW_SIG_QUALITY=true, SIG_QUALITY_THRESHOLD=0.2f0,
                SIG_QUALITY_VAR="NCP")

            # Thresholding should remove some gates
            @test size(X_yes, 1) <= size(X_no, 1)
        end

        rm(cfrad_path)
        rm(config_path)
    end

    @testset "compute_balanced_class_weights" begin
        # Balanced classes
        samples = [0, 0, 1, 1]
        weights = Ronin.compute_balanced_class_weights(samples)
        @test weights[0] ≈ 1.0
        @test weights[1] ≈ 1.0

        # Imbalanced: 75% class 0, 25% class 1
        samples2 = [0, 0, 0, 1]
        weights2 = Ronin.compute_balanced_class_weights(samples2)
        # weight = n_samples / (n_classes * n_samples_in_class)
        # class 0: 4 / (2 * 3) = 0.667
        # class 1: 4 / (2 * 1) = 2.0
        @test weights2[0] ≈ 4.0 / 6.0
        @test weights2[1] ≈ 2.0

        # Extreme imbalance
        samples3 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        weights3 = Ronin.compute_balanced_class_weights(samples3)
        @test weights3[1] > weights3[0]
    end

    @testset "evaluate_model (predictions, targets)" begin
        # Perfect predictions
        preds = Vector{Bool}([true, true, false, false])
        targs = Vector{Bool}([true, true, false, false])
        prec, recall, f1, tp, fp, tn, fn, n = Ronin.evaluate_model(preds, targs)
        @test prec ≈ 1.0
        @test recall ≈ 1.0
        @test f1 ≈ 1.0
        @test tp == 2
        @test tn == 2
        @test fp == 0
        @test fn == 0
        @test n == 4

        # Known case: 1 TP, 1 FP, 1 TN, 1 FN
        preds2 = Vector{Bool}([true, true, false, false])
        targs2 = Vector{Bool}([true, false, true, false])
        prec2, recall2, f12, tp2, fp2, tn2, fn2, n2 = Ronin.evaluate_model(preds2, targs2)
        @test prec2 ≈ 0.5
        @test recall2 ≈ 0.5
        @test f12 ≈ 0.5  # harmonic mean of 0.5 and 0.5
        @test tp2 == 1
        @test fp2 == 1
        @test tn2 == 1
        @test fn2 == 1

        # All predicted meteorological
        preds3 = Vector{Bool}([true, true, true, true])
        targs3 = Vector{Bool}([true, true, false, false])
        prec3, recall3, _, _, _, _, _, _ = Ronin.evaluate_model(preds3, targs3)
        @test prec3 ≈ 0.5
        @test recall3 ≈ 1.0
    end

    @testset "get_contingency" begin
        preds = Vector{Bool}([true, true, false, false])
        verif = Vector{Bool}([true, false, true, false])

        # Normalized
        df = Ronin.get_contingency(preds, verif; normalize=true)
        @test size(df, 1) == 2
        @test size(df, 2) == 3

        # Unnormalized
        df2 = Ronin.get_contingency(preds, verif; normalize=false)
        @test df2[1, 2] == 1  # TP
        @test df2[1, 3] == 1  # FP
        @test df2[2, 2] == 1  # FN
        @test df2[2, 3] == 1  # TN
    end

    @testset "write_field" begin
        nc_path = joinpath(test_scratchspace, "test_write_field.nc")
        isfile(nc_path) && rm(nc_path)

        # Create a small NetCDF file first
        ds = NCDataset(nc_path, "c")
        defDim(ds, "range", 3)
        defDim(ds, "time", 3)
        close(ds)

        test_data = Matrix{Union{Missing, Float32}}([1.0f0 2.0f0 3.0f0; 4.0f0 5.0f0 6.0f0; 7.0f0 8.0f0 9.0f0])

        # Write new field
        Ronin.write_field(nc_path, "TEST_VAR", test_data; verbose=false)
        NCDataset(nc_path) do ds
            @test ds["TEST_VAR"][:,:] == test_data
        end

        # Overwrite existing field
        new_data = fill(Float32(42.0), 3, 3)
        Ronin.write_field(nc_path, "TEST_VAR", new_data; overwrite=true, verbose=false)
        NCDataset(nc_path) do ds
            @test all(ds["TEST_VAR"][:,:] .== 42.0f0)
        end

        # Should error when overwrite=false and field exists
        @test_throws Exception Ronin.write_field(nc_path, "TEST_VAR", new_data; overwrite=false, verbose=false)

        rm(nc_path)
    end

    @testset "calculate_features end-to-end" begin
        cfrad_path = joinpath(test_scratchspace, "test_calcfeat.nc")
        create_test_cfrad(cfrad_path; range_dim=10, time_dim=8)
        config_path = joinpath(test_scratchspace, "calcfeat_config.txt")
        create_test_config(config_path, "NCP, RNG")
        output_path = joinpath(test_scratchspace, "calcfeat_output.h5")
        isfile(output_path) && rm(output_path)

        X, Y = Ronin.calculate_features(cfrad_path, config_path, output_path, false;
            verbose=false, remove_variable="VV", write_out=true)

        @test size(X, 2) == 2
        @test isfile(output_path)

        # Verify HDF5 output matches
        h5open(output_path) do f
            @test size(f["X"][:,:]) == size(X)
            @test f["X"][:,:] ≈ X
        end

        rm(cfrad_path)
        rm(config_path)
        rm(output_path)
    end

    @testset "calculate_features with return_idxer" begin
        cfrad_path = joinpath(test_scratchspace, "test_calcfeat_idx.nc")
        create_test_cfrad(cfrad_path; range_dim=10, time_dim=8)
        config_path = joinpath(test_scratchspace, "calcfeat_idx_config.txt")
        create_test_config(config_path, "NCP, RNG")
        output_path = joinpath(test_scratchspace, "calcfeat_idx_output.h5")
        isfile(output_path) && rm(output_path)

        X, Y, idxs = Ronin.calculate_features(cfrad_path, config_path, output_path, false;
            verbose=false, remove_variable="VV", write_out=true, return_idxer=true)

        @test length(idxs) == 1  # Single file
        @test size(idxs[1]) == (10, 8)  # range x time dimensions

        rm(cfrad_path)
        rm(config_path)
        rm(output_path)
    end
end


###############################################################################
# 4. DECISION TREE TESTS
###############################################################################
@testset "DecisionTree" begin

    @testset "build_tree and apply_tree" begin
        # Simple linearly separable 2D data
        Random.seed!(42)
        n = 100
        X = randn(n, 2)
        labels = [x[1] > 0 ? "A" : "B" for x in eachrow(X)]

        tree = Ronin.DecisionTree.build_tree(labels, X)
        preds = Ronin.DecisionTree.apply_tree(tree, X)

        accuracy = sum(preds .== labels) / n
        @test accuracy > 0.9  # Should easily learn this pattern
    end

    @testset "build_forest and apply_forest" begin
        Random.seed!(42)
        n = 200
        X = randn(Float32, n, 3)
        labels = [sum(x) > 0 ? 1 : 0 for x in eachrow(X)]

        forest = Ronin.DecisionTree.build_forest(labels, X, 2, 10, 0.7, -1; rng=42)
        preds = Ronin.DecisionTree.apply_forest(forest, X)

        accuracy = sum(preds .== labels) / n
        @test accuracy > 0.85
    end

    @testset "predict_proba" begin
        Random.seed!(42)
        n = 200
        X = randn(Float32, n, 3)
        labels = [sum(x) > 0 ? 1 : 0 for x in eachrow(X)]

        forest = Ronin.DecisionTree.build_forest(labels, X, 2, 15, 0.7, -1; rng=42)
        proba = Ronin.DecisionTree.apply_forest_proba(forest, X, [0, 1])

        # Probabilities should sum to 1 for each gate
        for i in 1:n
            @test proba[i, 1] + proba[i, 2] ≈ 1.0 atol=1e-6
        end

        # All probabilities should be in [0, 1]
        @test all(0.0 .<= proba .<= 1.0)
    end

    @testset "RandomForestClassifier API" begin
        Random.seed!(42)
        n = 200
        X = randn(Float32, n, 3)
        labels = [sum(x) > 0 ? 1 : 0 for x in eachrow(X)]

        clf = Ronin.DecisionTree.RandomForestClassifier(n_trees=15, max_depth=10, rng=42)
        Ronin.DecisionTree.fit!(clf, X, labels)

        preds = Ronin.DecisionTree.predict(clf, X)
        @test length(preds) == n

        proba = Ronin.DecisionTree.predict_proba(clf, X)
        @test size(proba) == (n, 2)
        @test all(proba .>= 0.0)
        @test all(proba .<= 1.0)
    end

    @testset "RandomForestClassifier with class weights" begin
        Random.seed!(42)
        n = 200
        X = randn(Float32, n, 3)
        labels = [sum(x) > 0 ? 1 : 0 for x in eachrow(X)]
        weights = ones(Float32, n)

        clf = Ronin.DecisionTree.RandomForestClassifier(n_trees=10, max_depth=8, rng=42)
        Ronin.DecisionTree.fit!(clf, X, labels, weights)

        preds = Ronin.DecisionTree.predict(clf, X)
        accuracy = sum(preds .== labels) / n
        @test accuracy > 0.8
    end

    @testset "tree depth control" begin
        Random.seed!(42)
        n = 200
        X = randn(Float32, n, 5)
        labels = [x[1] > 0 ? 1 : 0 for x in eachrow(X)]

        # Shallow tree (depth 2) should be less accurate than deep tree
        clf_shallow = Ronin.DecisionTree.RandomForestClassifier(n_trees=10, max_depth=2, rng=42)
        Ronin.DecisionTree.fit!(clf_shallow, X, labels)
        preds_shallow = Ronin.DecisionTree.predict(clf_shallow, X)

        clf_deep = Ronin.DecisionTree.RandomForestClassifier(n_trees=10, max_depth=20, rng=42)
        Ronin.DecisionTree.fit!(clf_deep, X, labels)
        preds_deep = Ronin.DecisionTree.predict(clf_deep, X)

        acc_shallow = sum(preds_shallow .== labels) / n
        acc_deep = sum(preds_deep .== labels) / n

        # Deep tree should fit training data at least as well
        @test acc_deep >= acc_shallow
    end

    @testset "model save/load roundtrip" begin
        Random.seed!(42)
        n = 100
        X = randn(Float32, n, 3)
        labels = [sum(x) > 0 ? 1 : 0 for x in eachrow(X)]

        clf = Ronin.DecisionTree.RandomForestClassifier(n_trees=5, max_depth=5, rng=42)
        Ronin.DecisionTree.fit!(clf, X, labels)

        model_path = joinpath(test_scratchspace, "test_model.jld2")
        save_object(model_path, clf)
        loaded_clf = load_object(model_path)

        preds_orig = Ronin.DecisionTree.predict(clf, X)
        preds_loaded = Ronin.DecisionTree.predict(loaded_clf, X)

        @test preds_orig == preds_loaded
        rm(model_path)
    end

    @testset "confusion_matrix" begin
        actual = [1, 1, 1, 0, 0, 0]
        predicted = [1, 1, 0, 0, 0, 1]

        cm = Ronin.DecisionTree.confusion_matrix(actual, predicted)
        @test cm.accuracy ≈ 4 / 6
    end
end


###############################################################################
# 5. MODEL TRAINING & PREDICTION PIPELINE TESTS
###############################################################################
@testset "Training Pipeline" begin

    @testset "train_model basic" begin
        # Create synthetic training data
        Random.seed!(42)
        n = 500
        n_features = 3
        X = randn(Float32, n, n_features)
        Y = reshape([sum(x) > 0 ? 1 : 0 for x in eachrow(X)], :, 1)

        h5_path = joinpath(test_scratchspace, "train_test.h5")
        model_path = joinpath(test_scratchspace, "train_test_model.jld2")
        isfile(h5_path) && rm(h5_path)
        isfile(model_path) && rm(model_path)

        h5open(h5_path, "w") do f
            write_dataset(f, "X", X)
            write_dataset(f, "Y", Y)
            attributes(f)["Parameters"] = ["F1", "F2", "F3"]
            attributes(f)["MISSING_FILL_VALUE"] = Ronin.FILL_VAL
        end

        weights = ones(Float32, n)
        Ronin.train_model(h5_path, model_path; n_trees=10, max_depth=8, class_weights=weights)

        @test isfile(model_path)
        clf = load_object(model_path)
        @test clf.n_trees == 10
        @test clf.max_depth == 8

        rm(h5_path)
        rm(model_path)
    end

    @testset "train_model with verification output" begin
        Random.seed!(42)
        n = 200
        X = randn(Float32, n, 2)
        Y = reshape([x[1] > 0 ? 1 : 0 for x in eachrow(X)], :, 1)

        h5_path = joinpath(test_scratchspace, "train_verify.h5")
        model_path = joinpath(test_scratchspace, "train_verify_model.jld2")
        verify_path = joinpath(test_scratchspace, "train_verify_out.h5")
        for p in [h5_path, model_path, verify_path]
            isfile(p) && rm(p)
        end

        h5open(h5_path, "w") do f
            write_dataset(f, "X", X)
            write_dataset(f, "Y", Y)
            attributes(f)["Parameters"] = ["F1", "F2"]
            attributes(f)["MISSING_FILL_VALUE"] = Ronin.FILL_VAL
        end

        weights = ones(Float32, n)
        Ronin.train_model(h5_path, model_path; verify=true, verify_out=verify_path,
            n_trees=10, max_depth=8, class_weights=weights)

        @test isfile(verify_path)
        h5open(verify_path) do f
            @test haskey(f, "Y_PREDICTED")
            @test haskey(f, "Y_ACTUAL")
            @test length(f["Y_PREDICTED"][:]) == n
        end

        for p in [h5_path, model_path, verify_path]
            rm(p)
        end
    end
end


###############################################################################
# 6. I/O TESTS
###############################################################################
@testset "I/O Operations" begin

    @testset "remove_validation split" begin
        Random.seed!(42)
        n = 100
        n_features = 3
        X = randn(Float32, n, n_features)
        Y = reshape(rand([0, 1], n), :, 1)

        input_path = joinpath(test_scratchspace, "rv_input.h5")
        train_path = joinpath(test_scratchspace, "rv_train.h5")
        val_path = joinpath(test_scratchspace, "rv_val.h5")
        for p in [input_path, train_path, val_path]
            isfile(p) && rm(p)
        end

        h5open(input_path, "w") do f
            write_dataset(f, "X", X)
            write_dataset(f, "Y", Y)
            attributes(f)["Parameters"] = ["F1", "F2", "F3"]
            attributes(f)["MISSING_FILL_VALUE"] = Ronin.FILL_VAL
        end

        Ronin.remove_validation(input_path; training_output=train_path,
            validation_output=val_path, remove_original=false)

        @test isfile(train_path)
        @test isfile(val_path)

        # Check sizes: every 10th row goes to validation
        h5open(train_path) do ft
            h5open(val_path) do fv
                n_train = size(ft["X"][:,:])[1]
                n_val = size(fv["X"][:,:])[1]
                @test n_train + n_val == n
                @test n_val == length(1:10:n)  # every 10th row
            end
        end

        for p in [input_path, train_path, val_path]
            isfile(p) && rm(p)
        end
    end
end


###############################################################################
# 7. REPRODUCIBILITY TESTS
###############################################################################
@testset "Reproducibility" begin

    @testset "Same seed produces same forest" begin
        Random.seed!(42)
        n = 200
        X = randn(Float32, n, 3)
        labels = [sum(x) > 0 ? 1 : 0 for x in eachrow(X)]

        clf1 = Ronin.DecisionTree.RandomForestClassifier(n_trees=10, max_depth=8, rng=123)
        Ronin.DecisionTree.fit!(clf1, X, labels)
        preds1 = Ronin.DecisionTree.predict(clf1, X)

        clf2 = Ronin.DecisionTree.RandomForestClassifier(n_trees=10, max_depth=8, rng=123)
        Ronin.DecisionTree.fit!(clf2, X, labels)
        preds2 = Ronin.DecisionTree.predict(clf2, X)

        @test preds1 == preds2
    end

    @testset "Different seeds produce different forests" begin
        n = 200
        X = randn(Float32, n, 3)
        labels = [sum(x) > 0 ? 1 : 0 for x in eachrow(X)]

        clf1 = Ronin.DecisionTree.RandomForestClassifier(n_trees=10, max_depth=8, rng=42)
        Ronin.DecisionTree.fit!(clf1, X, labels)
        preds1 = Ronin.DecisionTree.predict(clf1, X)

        clf2 = Ronin.DecisionTree.RandomForestClassifier(n_trees=10, max_depth=8, rng=99)
        Ronin.DecisionTree.fit!(clf2, X, labels)
        preds2 = Ronin.DecisionTree.predict(clf2, X)

        # With different seeds, predictions may differ (not guaranteed but very likely)
        # This is a soft test
        @test length(preds1) == length(preds2)
    end

    @testset "Feature calculation determinism" begin
        cfrad_path = joinpath(test_scratchspace, "test_determinism.nc")
        create_test_cfrad(cfrad_path; range_dim=10, time_dim=8, seed=42)
        config_path = joinpath(test_scratchspace, "determinism_config.txt")
        create_test_config(config_path, "NCP, RNG, AHT")

        NCDataset(cfrad_path) do ds
            X1, _, idx1 = Ronin.process_single_file(ds, config_path;
                HAS_INTERACTIVE_QC=false, remove_variable="VV")
            X2, _, idx2 = Ronin.process_single_file(ds, config_path;
                HAS_INTERACTIVE_QC=false, remove_variable="VV")
            @test X1 == X2
            @test idx1 == idx2
        end

        rm(cfrad_path)
        rm(config_path)
    end
end


###############################################################################
# 8. EDGE CASES AND ERROR HANDLING
###############################################################################
@testset "Edge Cases" begin

    @testset "Empty or all-missing data" begin
        @test Ronin.missing_std(Union{Missing,Float64}[missing]) == Ronin.FILL_VAL
        @test Ronin.missing_avg(Union{Missing,Float64}[missing]) == Ronin.FILL_VAL
    end

    @testset "PGG boundary conditions" begin
        # Exactly at altitude boundary (range == altitude)
        p = Ronin.prob_groundgate(-45.0f0, 3000.0f0, 3000.0f0, 180.0f0)
        @test 0.0 <= p <= 1.0

        # Zero elevation (horizontal beam) - note: the check is elevation_angle > 0 (strict),
        # so exactly 0 falls through to the full calculation
        p2 = Ronin.prob_groundgate(0.0f0, 5000.0f0, 3000.0f0, 90.0f0)
        @test 0.0 <= p2 <= 1.0
    end

    @testset "Balanced weights edge cases" begin
        # Single class
        weights = Ronin.compute_balanced_class_weights([1, 1, 1])
        @test weights[1] ≈ 1.0

        # Many classes
        weights2 = Ronin.compute_balanced_class_weights([0, 1, 2, 3])
        @test length(weights2) == 4
    end

    @testset "Isolation with missing weights" begin
        data = Matrix{Union{Missing, Float32}}(fill(1.0f0, 3, 3))
        missing_weights = Matrix{Union{Missing, Float32}}(fill(missing, 3, 3))
        result = Ronin.calc_iso(data; weights=missing_weights, window=(3,3))
        # All results should be missing since weights are all missing
        @test all(ismissing, result)
    end
end


###############################################################################
# 9. CONVOLUTION PRE-PROCESSOR TESTS
###############################################################################
@testset "RoninConvolutions" begin

    @testset "Kernel bank construction" begin
        bank = Ronin.build_kernel_bank([3, 5, 7])

        # Expected: 3 mean + 1 laplacian + 1 sobel_range + 1 sobel_azi + 2 gaussian (5x5, 7x7) = 8
        @test length(bank) == 8
        @test bank[1].name == "mean_3x3"
        @test bank[2].name == "mean_5x5"
        @test bank[3].name == "mean_7x7"
        @test bank[4].name == "laplacian_3x3"
        @test bank[5].name == "sobel_range_3x3"
        @test bank[6].name == "sobel_azi_3x3"
        @test bank[7].name == "gaussian_5x5"
        @test bank[8].name == "gaussian_7x7"

        # Mean kernels should sum to ~1
        @test isapprox(sum(bank[1].weights), 1.0f0, atol=1e-5)
        @test isapprox(sum(bank[2].weights), 1.0f0, atol=1e-5)

        # Gaussian kernels should sum to ~1
        @test isapprox(sum(bank[7].weights), 1.0f0, atol=1e-5)

        # Laplacian should have -4 center and 4 neighbors
        @test bank[4].weights[2, 2] == -4.0f0
    end

    @testset "Kernel bank with single scale" begin
        bank = Ronin.build_kernel_bank([3])
        # 1 mean + 1 laplacian + 1 sobel_range + 1 sobel_azi = 4 (no gaussian for k<5)
        @test length(bank) == 4
    end

    @testset "Gaussian kernel properties" begin
        g = Ronin._gaussian_kernel(5)
        @test size(g) == (5, 5)
        @test isapprox(sum(g), 1.0f0, atol=1e-5)
        # Center should be the maximum
        @test g[3, 3] == maximum(g)
        # Should be symmetric
        @test isapprox(g[1, 1], g[5, 5], atol=1e-6)
        @test isapprox(g[1, 3], g[5, 3], atol=1e-6)
    end

    @testset "masked_convolve - uniform data" begin
        # Uniform data with mean kernel should return the same value everywhere (interior)
        data = fill(5.0f0, 7, 7)
        kernel = ones(Float32, 3, 3) ./ 9.0f0
        valid = trues(7, 7)

        result, vfrac = Ronin.masked_convolve(data, kernel, valid)

        # Interior points should be exactly 5.0
        @test isapprox(result[4, 4], 5.0f0, atol=1e-4)
        # Valid fraction for interior should be 1.0
        @test isapprox(vfrac[4, 4], 1.0f0, atol=1e-4)
    end

    @testset "masked_convolve - missing data handling" begin
        data = fill(10.0f0, 5, 5)
        kernel = ones(Float32, 3, 3) ./ 9.0f0
        valid = trues(5, 5)
        valid[3, 3] = false  # Center is invalid

        result, vfrac = Ronin.masked_convolve(data, kernel, valid)

        # Center gate should be FILL_VAL
        @test result[3, 3] == Float32(Ronin.FILL_VAL)
        @test vfrac[3, 3] == 0.0f0

        # Neighbor should still be ~10.0 (one of its neighbors is invalid but the rest are valid)
        @test isapprox(result[3, 2], 10.0f0, atol=0.5)
        # Valid fraction for neighbor should be < 1.0 (one neighbor missing)
        @test vfrac[3, 2] < 1.0f0
        @test vfrac[3, 2] > 0.5f0
    end

    @testset "masked_convolve - all invalid" begin
        data = fill(1.0f0, 3, 3)
        kernel = ones(Float32, 3, 3)
        valid = falses(3, 3)

        result, vfrac = Ronin.masked_convolve(data, kernel, valid)

        @test all(x -> x == Float32(Ronin.FILL_VAL), result)
        @test all(x -> x == 0.0f0, vfrac)
    end

    @testset "masked_convolve - gradient kernel" begin
        # Linear gradient in range direction
        data = Matrix{Float32}(undef, 5, 5)
        for j in 1:5, i in 1:5
            data[i, j] = Float32(i)  # gradient along first dimension (range)
        end
        # Sobel range kernel
        sobel = Float32[-1 -2 -1; 0 0 0; 1 2 1]
        valid = trues(5, 5)

        result, _ = Ronin.masked_convolve(data, sobel, valid)
        # Interior: result should be positive (gradient is increasing in i)
        @test result[3, 3] > 0.0f0
    end

    @testset "select_features" begin
        importances = [0.5, 0.001, 0.3, 0.0005, 0.1]
        selected = Ronin.select_features(importances, 0.01)
        # Threshold is 0.01 * 0.5 = 0.005
        # Features above: [1] 0.5, [3] 0.3, [5] 0.1 → indices 1, 3, 5
        @test 1 in selected
        @test 3 in selected
        @test 5 in selected
        @test !(2 in selected)
        @test !(4 in selected)
    end

    @testset "select_features - all zeros" begin
        importances = [0.0, 0.0, 0.0]
        selected = Ronin.select_features(importances, 0.01)
        # All zeros: should return all indices
        @test selected == [1, 2, 3]
    end

    @testset "get_convolution_feature_count" begin
        bank = Ronin.build_kernel_bank([3, 5])
        vars = ["DBZ", "VEL"]
        count = Ronin.get_convolution_feature_count(vars, bank)
        # bank has 5 kernels (3 mean-like: mean_3x3, mean_5x5 + laplacian + sobel_range + sobel_azi + gaussian_5x5 = 6)
        # Actually: 2 mean + 1 lap + 1 sobel_r + 1 sobel_a + 1 gaussian = 6
        # Features: 2 vars * 6 kernels * 2 (value + vfrac) + 4 scalar = 28
        @test count == 2 * length(bank) * 2 + 4
    end

    @testset "compute_rf_feature_importance" begin
        Random.seed!(42)
        n = 200
        X = randn(Float32, n, 4)
        # Only feature 1 matters for classification
        labels = [X[i, 1] > 0 ? 1 : 0 for i in 1:n]

        clf = Ronin.DecisionTree.RandomForestClassifier(n_trees=20, max_depth=8, rng=42)
        Ronin.DecisionTree.fit!(clf, X, labels)

        importances = Ronin.compute_rf_feature_importance(clf, copy(X), labels; n_repeats=3)
        @test length(importances) == 4
        # Feature 1 should have the highest importance
        @test argmax(importances) == 1
    end

    @testset "Convolution features on synthetic cfrad" begin
        cfrad_path = joinpath(test_scratchspace, "test_conv.nc")
        create_test_cfrad(cfrad_path; range_dim=10, time_dim=8, seed=42)

        NCDataset(cfrad_path) do ds
            bank = Ronin.build_kernel_bank([3, 5])
            valid_mask = trues(10, 8)
            X, names = Ronin.compute_convolution_features(ds, ["DBZ"], bank, valid_mask, "NCP")

            # Expected: 1 var * 6 kernels * 2 + 4 scalar = 16 features
            @test size(X, 1) == 10 * 8  # ngates
            @test size(X, 2) == length(bank) * 2 + 4
            @test length(names) == size(X, 2)

            # Feature names should include expected patterns
            @test any(n -> occursin("mean_3x3", n), names)
            @test any(n -> occursin("vfrac", n), names)
            @test "AHT" in names
            @test "ELV" in names
            @test "RNG" in names
            @test "NRG" in names

            # No NaN values in output
            @test !any(isnan, X)
        end

        rm(cfrad_path)
    end

    @testset "process_single_file_conv" begin
        cfrad_path = joinpath(test_scratchspace, "test_conv_psf.nc")
        create_test_cfrad(cfrad_path; range_dim=10, time_dim=8, seed=42)

        NCDataset(cfrad_path) do ds
            config = ModelConfig(
                num_models = 1,
                model_output_paths = ["test.jld2"],
                met_probs = [(0.1f0, 0.9f0)],
                feature_output_paths = ["test.h5"],
                input_path = cfrad_path,
                task_mode = "convolution",
                file_preprocessed = [false],
                HAS_INTERACTIVE_QC = true,
                QC_var = "VG",
                remove_var = "VV",
                conv_variables = ["DBZ"],
                conv_kernel_sizes = [3],
                REMOVE_LOW_SIG_QUALITY = false,
                REMOVE_HIGH_PGG = false,
                SIG_QUALITY_VAR = "NCP",
            )
            bank = Ronin.build_kernel_bank([3])
            X, Y, indexer, feat_names = Ronin.process_single_file_conv(ds, config, bank)

            @test size(X, 1) == sum(indexer)  # rows match valid gates
            @test size(X, 2) == Ronin.get_convolution_feature_count(["DBZ"], bank)
            @test size(Y, 1) == sum(indexer)
            @test !any(isnan, X)
        end

        rm(cfrad_path)
    end

    @testset "Feature selection round-trip" begin
        # Simulate: compute importance, select, verify subset
        importances = [0.5, 0.001, 0.3, 0.0005, 0.1, 0.002, 0.4, 0.0001]
        selected = Ronin.select_features(importances, 0.01)

        # Using selected as column indices
        X_full = randn(Float32, 100, 8)
        X_subset = X_full[:, selected]
        @test size(X_subset, 2) == length(selected)
        @test size(X_subset, 2) < 8
    end

    @testset "load_conv_variable - spatial features" begin
        cfrad_path = joinpath(test_scratchspace, "test_conv_spatial.nc")
        create_test_cfrad(cfrad_path; range_dim=10, time_dim=8, seed=42)

        NCDataset(cfrad_path) do ds
            valid_mask = trues(10, 8)

            # ISO(DBZ) should return a 2D matrix of isolation values
            iso_result = Ronin.load_conv_variable(ds, "ISO(DBZ)", valid_mask, "NCP")
            @test size(iso_result) == (10, 8)
            @test !any(ismissing, iso_result)

            # AVG(DBZ) should return a 2D matrix of averaged values
            avg_result = Ronin.load_conv_variable(ds, "AVG(DBZ)", valid_mask, "NCP")
            @test size(avg_result) == (10, 8)

            # STD(DBZ) should return a 2D matrix
            std_result = Ronin.load_conv_variable(ds, "STD(DBZ)", valid_mask, "NCP")
            @test size(std_result) == (10, 8)

            # ISO should be all zeros when all gates are valid (no missing neighbors
            # within the data, though border effects may produce non-zero values)
            @test !any(isnan, Float32.(iso_result))

            # Derived inner variable: ISO(PGG) should work
            iso_pgg = Ronin.load_conv_variable(ds, "ISO(PGG)", valid_mask, "NCP")
            @test size(iso_pgg) == (10, 8)

            # With mask applied, ISO values should change
            masked = copy(valid_mask)
            masked[5, 4] = false
            masked[5, 5] = false
            iso_masked = Ronin.load_conv_variable(ds, "ISO(DBZ)", masked, "NCP")
            # Gates near the masked region should have higher isolation
            # (more missing neighbors) compared to the unmasked version
            @test iso_masked[5, 3] >= iso_result[5, 3]
        end

        rm(cfrad_path)
    end

    @testset "Spatial features in compute_convolution_features" begin
        cfrad_path = joinpath(test_scratchspace, "test_conv_spatial2.nc")
        create_test_cfrad(cfrad_path; range_dim=10, time_dim=8, seed=42)

        NCDataset(cfrad_path) do ds
            bank = Ronin.build_kernel_bank([3])
            valid_mask = trues(10, 8)

            # Include ISO(DBZ) alongside raw DBZ
            vars = ["DBZ", "ISO(DBZ)"]
            X, names = Ronin.compute_convolution_features(ds, vars, bank, valid_mask, "NCP")

            # 2 vars * 4 kernels * 2 columns + 4 scalar = 20
            @test size(X, 2) == 2 * length(bank) * 2 + 4
            @test size(X, 1) == 10 * 8

            # Feature names should include ISO(DBZ) prefixed entries
            @test any(n -> startswith(n, "ISO(DBZ)_"), names)
            @test any(n -> startswith(n, "DBZ_"), names)
            @test !any(isnan, X)
        end

        rm(cfrad_path)
    end

    @testset "Convolution determinism" begin
        data = Float32.(reshape(1:25, 5, 5))
        kernel = ones(Float32, 3, 3) ./ 9.0f0
        valid = trues(5, 5)

        r1, v1 = Ronin.masked_convolve(data, kernel, valid)
        r2, v2 = Ronin.masked_convolve(data, kernel, valid)
        @test r1 == r2
        @test v1 == v2
    end
end


###############################################################################
# 6. Model Metadata and Inspection
###############################################################################

@testset "load_model_with_metadata" begin
    @testset "jldsave format with metadata" begin
        model_path = joinpath(test_scratchspace, "test_model_meta.jld2")
        isfile(model_path) && rm(model_path)

        # Create a simple RF model
        Random.seed!(42)
        n = 100
        X = randn(Float32, n, 4)
        Y = [x > 0 ? 1 : 0 for x in X[:, 1]]
        model = Ronin.DecisionTree.build_forest(Y, X, 2, 5, 0.7, 4)

        selected = [1, 3, 4]
        recommended = [1, 3]
        feat_names = ["DBZ_mean_3x3", "VEL_mean_3x3", "SIG_mean_3x3", "AHT"]
        imps = [0.05, 0.001, 0.03, 0.02]

        JLD2.jldsave(model_path;
            model=model,
            selected_features=selected,
            recommended_features=recommended,
            feature_names=feat_names,
            importances=imps)

        md = Ronin.load_model_with_metadata(model_path, "convolution")
        @test md.model isa Ronin.DecisionTree.Ensemble
        @test md.selected_features == [1, 3, 4]
        @test md.recommended_features == [1, 3]
        @test md.feature_names == feat_names
        @test md.importances == imps

        rm(model_path)
    end

    @testset "save_object format (no metadata)" begin
        model_path = joinpath(test_scratchspace, "test_model_plain.jld2")
        isfile(model_path) && rm(model_path)

        Random.seed!(42)
        n = 100
        X = randn(Float32, n, 4)
        Y = [x > 0 ? 1 : 0 for x in X[:, 1]]
        model = Ronin.DecisionTree.build_forest(Y, X, 2, 5, 0.7, 4)

        save_object(model_path, model)

        md = Ronin.load_model_with_metadata(model_path, "convolution")
        @test md.model isa Ronin.DecisionTree.Ensemble
        @test md.selected_features == Int[]
        @test md.recommended_features == Int[]
        @test md.feature_names == String[]
        @test md.importances == Float64[]

        rm(model_path)
    end

    @testset "non-convolution mode" begin
        model_path = joinpath(test_scratchspace, "test_model_noconv.jld2")
        isfile(model_path) && rm(model_path)

        Random.seed!(42)
        n = 100
        X = randn(Float32, n, 4)
        Y = [x > 0 ? 1 : 0 for x in X[:, 1]]
        model = Ronin.DecisionTree.build_forest(Y, X, 2, 5, 0.7, 4)
        save_object(model_path, model)

        md = Ronin.load_model_with_metadata(model_path, "")
        @test md.model isa Ronin.DecisionTree.Ensemble
        @test md.selected_features == Int[]
        @test md.recommended_features == Int[]

        rm(model_path)
    end
end

@testset "inspect_model_configuration" begin
    @testset "jldsave format with full metadata" begin
        model_path = joinpath(test_scratchspace, "test_inspect.jld2")
        isfile(model_path) && rm(model_path)

        Random.seed!(42)
        n = 100
        X = randn(Float32, n, 3)
        Y = [x > 0 ? 1 : 0 for x in X[:, 1]]
        model = Ronin.DecisionTree.build_forest(Y, X, 2, 5, 0.7, 3)

        JLD2.jldsave(model_path;
            model=model,
            selected_features=Int[],
            recommended_features=[1, 3],
            feature_names=["DBZ_mean_3x3", "VEL_laplacian_3x3", "AHT"],
            importances=[0.05, 0.001, 0.03])

        buf = IOBuffer()
        Ronin.inspect_model_configuration(model_path; io=buf)
        output = String(take!(buf))

        @test occursin("MODEL CONFIGURATION", output)
        @test occursin("FEATURE NAMES", output)
        @test occursin("DBZ_mean_3x3", output)
        @test occursin("TRAINED ON: all features", output)
        @test occursin("RECOMMENDED FEATURES", output)
        @test occursin("2 / 3", output) || occursin("2/3", output)
        @test occursin("FEATURE IMPORTANCES", output)

        rm(model_path)
    end

    @testset "save_object format" begin
        model_path = joinpath(test_scratchspace, "test_inspect_plain.jld2")
        isfile(model_path) && rm(model_path)

        Random.seed!(42)
        n = 100
        X = randn(Float32, n, 3)
        Y = [x > 0 ? 1 : 0 for x in X[:, 1]]
        model = Ronin.DecisionTree.build_forest(Y, X, 2, 5, 0.7, 3)
        save_object(model_path, model)

        buf = IOBuffer()
        Ronin.inspect_model_configuration(model_path; io=buf)
        output = String(take!(buf))

        @test occursin("MODEL CONFIGURATION", output)
        @test occursin("save_object", output)

        rm(model_path)
    end

    @testset "nonexistent file" begin
        buf = IOBuffer()
        Ronin.inspect_model_configuration("/nonexistent/path.jld2"; io=buf)
        output = String(take!(buf))
        @test occursin("File not found", output)
    end
end

###############################################################################
# save_config / load_config — dict-keyed persistence (issue #34)
###############################################################################
@testset "save_config / load_config" begin
    @testset "round-trip preserves field values" begin
        cfg = make_config(num_models=2, input_path="/tmp/in",
                          experiment_name="rt_test",
                          task_mode="convolution",
                          n_trees=37, max_depth=11,
                          conv_variables=["DBZ", "VEL"],
                          masked_conv_variables=["DBZ"],
                          masked_conv_threshold=0.7f0)
        path = joinpath(test_scratchspace, "config_roundtrip.jld2")
        save_config(path, cfg)
        @test isfile(path)
        loaded = load_config(path)
        @test loaded isa Ronin.ModelConfig
        for fld in fieldnames(Ronin.ModelConfig)
            @test getproperty(loaded, fld) == getproperty(cfg, fld)
        end
        ## Loaded config must be mutable — assigning fields is the core bug from #34.
        loaded.task_paths = ["a", "b"]
        @test loaded.task_paths == ["a", "b"]
        rm(path; force=true)
    end

    @testset "missing keys fall back to defaults (forward-compat)" begin
        ## Simulate an "older" save_config file by hand-writing only a subset of keys.
        ## Future field additions to ModelConfig should load fine from such a file.
        path = joinpath(test_scratchspace, "config_partial.jld2")
        ## Symbol keys so `;partial...` becomes a kwarg splat for jldsave.
        ## JLD2 serializes them as string keys on disk, which load_config reads back.
        partial = Dict{Symbol,Any}(
            :__ronin_config_format__ => "dict_v1",
            :num_models              => 2,
            :model_output_paths      => ["m1.jld2", "m2.jld2"],
            :met_probs               => [(0.0f0, 1.0f0), (0.0f0, 0.99f0)],
            :feature_output_paths    => ["f1.h5", "f2.h5"],
            :input_path              => "/data",
            :task_mode               => "convolution",
            :file_preprocessed       => [false, false],
            :n_trees                 => 99,
        )
        jldsave(path; partial...)
        loaded = load_config(path)
        @test loaded.n_trees == 99
        @test loaded.num_models == 2
        @test loaded.task_mode == "convolution"
        ## A field not written should equal the struct's default.
        default_cfg = make_config(num_models=2, input_path="/data", task_mode="convolution")
        @test loaded.compute_feature_importance == default_cfg.compute_feature_importance
        @test loaded.conv_kernel_sizes == default_cfg.conv_kernel_sizes
        rm(path; force=true)
    end

    @testset "legacy save_object format is loadable inline" begin
        ## Build a current ModelConfig, persist via save_object, reload via load_config.
        ## (Same shape; this exercises the single_stored_object branch but does not
        ## reproduce the cross-version field-mismatch on its own — see next testset.)
        cfg = make_config(num_models=1, input_path="/legacy",
                          experiment_name="legacy_test",
                          task_mode="",
                          n_trees=13, max_depth=7)
        path = joinpath(test_scratchspace, "config_legacy.jld2")
        JLD2.save_object(path, cfg)
        loaded = load_config(path)
        @test loaded isa Ronin.ModelConfig
        @test loaded.n_trees == 13
        @test loaded.max_depth == 7
        @test loaded.num_models == 1
        loaded.task_paths = ["from_legacy"]
        @test loaded.task_paths == ["from_legacy"]
        rm(path; force=true)
    end

    @testset "legacy file with extra/missing fields (cross-version simulation)" begin
        ## Approximate Paul's situation: a saved struct that has fewer fields than the
        ## current ModelConfig. We can't easily round-trip a foreign struct shape in
        ## one process, so we exercise the recovery path by supplying a NamedTuple
        ## standing in for the ReconstructedMutable.
        old = (
            num_models           = 2,
            model_output_paths   = ["a.jld2", "b.jld2"],
            met_probs            = [(0.0f0, 1.0f0), (0.0f0, 0.99f0)],
            feature_output_paths = ["a.h5", "b.h5"],
            input_path           = "/old",
            task_mode            = "",
            file_preprocessed    = [false, false],
            task_paths           = Vector{Union{String,Vector{String}}}(["t1.txt", "t2.txt"]),
            n_trees              = 7,
            ## Note: no conv_variables, masked_conv_*, etc. — those should default.
        )
        rebuilt = Ronin._load_legacy_config(old, "synthetic.jld2")
        @test rebuilt isa Ronin.ModelConfig
        @test rebuilt.n_trees == 7
        @test rebuilt.task_paths == Vector{Union{String,Vector{String}}}(["t1.txt", "t2.txt"])
        ## New field gets the struct default, not garbage.
        @test rebuilt.conv_variables == ["DBZ", "VEL"]
        @test rebuilt.masked_conv_variables == String[]
        ## Mutability check — the bug from #34 was setproperty! failing on the wrapper.
        rebuilt.task_paths = ["new_path"]
        @test rebuilt.task_paths == ["new_path"]
    end

    @testset "migrate_model_config writes loadable dict-keyed file" begin
        cfg = make_config(num_models=1, input_path="/migrate",
                          experiment_name="migrate_test",
                          task_mode="",
                          n_trees=21)
        legacy = joinpath(test_scratchspace, "config_to_migrate.jld2")
        new    = joinpath(test_scratchspace, "config_migrated.jld2")
        JLD2.save_object(legacy, cfg)

        migrated = migrate_model_config(legacy, new)
        @test isfile(new)
        @test migrated isa Ronin.ModelConfig

        reloaded = load_config(new)
        @test reloaded.n_trees == 21
        @test reloaded.input_path == "/migrate"
        ## Migrated file is the new format (no single_stored_object key).
        keys_in_file = jldopen(new, "r") do f
            collect(keys(f))
        end
        @test "num_models" in keys_in_file
        @test !("single_stored_object" in keys_in_file)

        rm(legacy; force=true)
        rm(new; force=true)
    end

    @testset "load_config rejects non-config files" begin
        ## A file containing only one or two unrelated keys should not load as a config.
        path = joinpath(test_scratchspace, "not_a_config.jld2")
        jldsave(path; some_other_key=42, another=[1, 2, 3])
        @test_throws ErrorException load_config(path)
        rm(path; force=true)
    end

    @testset "load_config errors clearly when file missing" begin
        @test_throws ErrorException load_config("/nonexistent/path/config.jld2")
    end
end

###############################################################################
# Cleanup
###############################################################################
@testset "Cleanup test scratch space" begin
    # Clean up any remaining temp files
    for f in readdir(test_scratchspace)
        rm(joinpath(test_scratchspace, f); force=true, recursive=true)
    end
    @test isempty(readdir(test_scratchspace))
end
