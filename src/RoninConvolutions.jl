using Statistics
using Random

"""
    ConvolutionKernel

Holds a named convolution kernel matrix for spatial feature computation.
"""
struct ConvolutionKernel
    name::String
    weights::Matrix{Float32}
end

"""
    build_kernel_bank(kernel_sizes::Vector{Int})

Build a bank of convolution kernels at the specified scales.

Returns a `Vector{ConvolutionKernel}` containing:
- Mean kernels at each scale (uniform weights, normalized)
- Laplacian 3x3 (edge/texture detection)
- Sobel-like range gradient 3x3
- Sobel-like azimuth gradient 3x3
- Gaussian kernels at scales >= 5
"""
function build_kernel_bank(kernel_sizes::Vector{Int})
    bank = ConvolutionKernel[]

    for k in kernel_sizes
        # Mean kernel (uniform, normalized to sum=1)
        w = ones(Float32, k, k) ./ Float32(k * k)
        push!(bank, ConvolutionKernel("mean_$(k)x$(k)", w))
    end

    # Laplacian 3x3 (edge detector, sum=0 for flat fields)
    lap = Float32[0 1 0; 1 -4 1; 0 1 0]
    push!(bank, ConvolutionKernel("laplacian_3x3", lap))

    # Sobel-like range gradient (vertical = range dimension)
    sobel_range = Float32[-1 -2 -1; 0 0 0; 1 2 1]
    push!(bank, ConvolutionKernel("sobel_range_3x3", sobel_range))

    # Sobel-like azimuth gradient (horizontal = azimuth dimension)
    sobel_azi = Float32[-1 0 1; -2 0 2; -1 0 1]
    push!(bank, ConvolutionKernel("sobel_azi_3x3", sobel_azi))

    for k in kernel_sizes
        if k >= 5
            # Gaussian kernel
            g = _gaussian_kernel(k)
            push!(bank, ConvolutionKernel("gaussian_$(k)x$(k)", g))
        end
    end

    return bank
end

"""
    build_filtered_kernel_bank(kernel_types::Vector{String}, kernel_sizes::Vector{Int})

Build a kernel bank from the Cartesian product of `kernel_types` and `kernel_sizes`.

Some kernel types are size-restricted **by design** — `laplacian`,
`sobel_range`, and `sobel_azi` are only the canonical 3x3 stencils, and
`gaussian` is only built at sizes >= 5 (a 3x3 Gaussian with the default sigma
is degenerate and redundant with `mean_3x3`). Those Cartesian-product
combinations are skipped **silently** here: callers pass arrays of sizes and
expect to get the valid subset, so a per-call warning would be log spam
(this runs once per scan per pass). The skipped set is reported once at the
start of training via [`masked_conv_skipped_combos`]; it is deterministic
from config, never data-dependent. An unknown kernel type is still a hard
error.

Supported kernel types: "mean", "gaussian", "laplacian", "sobel_range", "sobel_azi".
"""
function build_filtered_kernel_bank(kernel_types::Vector{String}, kernel_sizes::Vector{Int})
    bank = ConvolutionKernel[]
    for ktype in kernel_types
        for k in kernel_sizes
            if ktype == "mean"
                w = ones(Float32, k, k) ./ Float32(k * k)
                push!(bank, ConvolutionKernel("mean_$(k)x$(k)", w))
            elseif ktype == "laplacian"
                if k == 3
                    push!(bank, ConvolutionKernel("laplacian_3x3", Float32[0 1 0; 1 -4 1; 0 1 0]))
                end
                # else: size-restricted by design — see masked_conv_skipped_combos
            elseif ktype == "sobel_range"
                if k == 3
                    push!(bank, ConvolutionKernel("sobel_range_3x3", Float32[-1 -2 -1; 0 0 0; 1 2 1]))
                end
                # else: size-restricted by design — see masked_conv_skipped_combos
            elseif ktype == "sobel_azi"
                if k == 3
                    push!(bank, ConvolutionKernel("sobel_azi_3x3", Float32[-1 0 1; -2 0 2; -1 0 1]))
                end
                # else: size-restricted by design — see masked_conv_skipped_combos
            elseif ktype == "gaussian"
                if k >= 5
                    push!(bank, ConvolutionKernel("gaussian_$(k)x$(k)", _gaussian_kernel(k)))
                end
                # else: size-restricted by design — see masked_conv_skipped_combos
            else
                error("Unknown kernel type: $(ktype). Supported: mean, gaussian, laplacian, sobel_range, sobel_azi")
            end
        end
    end
    return bank
end

"""
    masked_conv_skipped_combos(kernel_types, kernel_sizes) -> Vector{Tuple{String,Int}}

Return the `(kernel_type, size)` pairs that `build_filtered_kernel_bank` skips
**by design** because that kernel type is size-restricted:

- `laplacian`, `sobel_range`, `sobel_azi` exist only at 3x3
- `gaussian` exists only at sizes >= 5

This is purely a reporting helper so training can surface the skipped set once
(it is fully determined by config, not by data). The size rules below MUST be
kept in sync with `build_filtered_kernel_bank` above. Unknown kernel types are
intentionally not flagged here — `build_filtered_kernel_bank` errors on them.
"""
function masked_conv_skipped_combos(kernel_types::Vector{String}, kernel_sizes::Vector{Int})
    skipped = Tuple{String, Int}[]
    for ktype in kernel_types
        for k in kernel_sizes
            if (ktype in ("laplacian", "sobel_range", "sobel_azi") && k != 3) ||
               (ktype == "gaussian" && k < 5)
                push!(skipped, (ktype, k))
            end
        end
    end
    return skipped
end

"""
    _gaussian_kernel(k::Int; sigma=nothing)

Generate a k x k Gaussian kernel. If sigma is not given, defaults to (k-1)/4.
"""
function _gaussian_kernel(k::Int; sigma=nothing)
    if sigma === nothing
        sigma = (k - 1) / 4.0
    end
    center = (k + 1) / 2.0
    w = Matrix{Float32}(undef, k, k)
    for j in 1:k, i in 1:k
        dx = i - center
        dy = j - center
        w[i, j] = Float32(exp(-(dx^2 + dy^2) / (2 * sigma^2)))
    end
    w ./= sum(w)
    return w
end

"""
    masked_convolve(data, kernel, valid; neighbor_mask=valid)

Compute a masked convolution over `data` using `kernel`.

- `valid` controls which center gates receive a computed value (others get FILL_VAL).
- `neighbor_mask` controls which neighbors contribute to the convolution (default: same as `valid`).

Using a separate `neighbor_mask` enables met_prob-masked convolutions: compute features
for all valid gates, but only let high-confidence neighbors contribute to the spatial statistics.

Returns `(result, valid_fraction)` where:
- `result[i,j]` = weighted sum of valid neighbors / sum of weights at valid neighbors
- `valid_fraction[i,j]` = fraction of kernel footprint that had valid data (ISO equivalent)

Missing/invalid gates contribute nothing. Edges are zero-padded (treated as invalid).
"""
function masked_convolve(data::Matrix{Float32}, kernel::Matrix{Float32}, valid::AbstractMatrix{Bool};
                         neighbor_mask::AbstractMatrix{Bool}=valid)
    nrows, ncols = size(data)
    kr, kc = size(kernel)
    hr = kr ÷ 2
    hc = kc ÷ 2

    result = Matrix{Float32}(undef, nrows, ncols)
    valid_frac = Matrix{Float32}(undef, nrows, ncols)

    abs_kernel = abs.(kernel)

    @inbounds for j in 1:ncols, i in 1:nrows
        if !valid[i, j]
            result[i, j] = Float32(FILL_VAL)
            valid_frac[i, j] = 0.0f0
            continue
        end

        weighted_sum = 0.0f0
        weight_sum = 0.0f0
        total_abs_weight = 0.0f0

        for dj in -hc:hc, di in -hr:hr
            ni = i + di
            nj = j + dj
            ki = di + hr + 1
            kj = dj + hc + 1
            kw = kernel[ki, kj]
            akw = abs_kernel[ki, kj]

            total_abs_weight += akw

            if ni >= 1 && ni <= nrows && nj >= 1 && nj <= ncols && neighbor_mask[ni, nj]
                weighted_sum += kw * data[ni, nj]
                weight_sum += akw
            end
        end

        if weight_sum > 0.0f0
            result[i, j] = weighted_sum / weight_sum
            valid_frac[i, j] = total_abs_weight > 0 ? weight_sum / total_abs_weight : 0.0f0
        else
            result[i, j] = Float32(FILL_VAL)
            valid_frac[i, j] = 0.0f0
        end
    end

    return result, valid_frac
end

"""
    load_conv_variable(cfrad, varname::String, valid_mask::AbstractMatrix{Bool},
                       SIG_QUALITY_VAR::String)

Load or compute a 2D variable for convolution. Supports:
  - Raw CfRadial variables: "DBZ", "VEL", etc.
  - Derived variables: "PGG", "SIG"
  - Prior-pass met probability: "met_prob_pass_1", etc.
  - Spatial features: "AVG(var)", "ISO(var)", "STD(var)"
    These are computed with the valid_mask applied so that masked gates
    are treated as missing — spatial statistics reflect the filtered data.

Returns a 2D matrix (nrows × ncols) with missing values where data is unavailable.
"""
function load_conv_variable(cfrad, varname::AbstractString, valid_mask::AbstractMatrix{Bool},
                            SIG_QUALITY_VAR::AbstractString)
    nrows, ncols = size(valid_mask)

    # Check for spatial function syntax: AVG(var), ISO(var), STD(var)
    spatial_match = match(r"^(AVG|ISO|STD)\((\w+)\)$", varname)
    if spatial_match !== nothing
        func_name = spatial_match.captures[1]
        inner_var = spatial_match.captures[2]

        # Recursively load the inner variable (supports e.g. ISO(PGG), AVG(SIG))
        inner_data = load_conv_variable(cfrad, inner_var, valid_mask, SIG_QUALITY_VAR)

        # Apply mask: set invalid gates to missing so spatial calculations
        # reflect the filtered data landscape (neighbors that were removed
        # become missing, changing AVG/STD/ISO values for remaining gates)
        masked_data = Matrix{Union{Missing, Float32}}(undef, nrows, ncols)
        for j in 1:ncols, i in 1:nrows
            v = inner_data[i, j]
            if !valid_mask[i, j] || ismissing(v) || (v isa AbstractFloat && isnan(v))
                masked_data[i, j] = missing
            else
                masked_data[i, j] = Float32(v)
            end
        end

        if func_name == "AVG"
            return calc_avg(masked_data)
        elseif func_name == "ISO"
            return calc_iso(masked_data)
        elseif func_name == "STD"
            return calc_std(masked_data)
        end
    end

    # Direct variable lookup
    if varname in keys(cfrad)
        return cfrad[varname][:, :]
    elseif varname == "PGG"
        return calc_pgg(cfrad)
    elseif varname == "SIG"
        return reshape(calc_sig(cfrad, SIG_QUALITY_VAR), nrows, ncols)
    elseif startswith(varname, "met_prob_pass_")
        if varname in keys(cfrad)
            return cfrad[varname][:, :]
        else
            @warn "$(varname) not found in CfRadial — filling with zeros"
            return zeros(Union{Missing, Float32}, nrows, ncols)
        end
    else
        error("Unknown convolution variable: $(varname). " *
              "Supported: CfRadial field names, PGG, SIG, met_prob_pass_<N>, " *
              "AVG(<var>), ISO(<var>), STD(<var>)")
    end
end

"""
    compute_convolution_features(cfrad, conv_variables::Vector{String},
                                  kernel_bank::Vector{ConvolutionKernel},
                                  valid_mask::AbstractMatrix{Bool},
                                  SIG_QUALITY_VAR::String)

Compute the full convolution feature matrix for a single sweep.

For each variable in `conv_variables` x each kernel in `kernel_bank`, produces 2 columns:
  1. The convolved value
  2. The valid fraction (ISO equivalent)

Additionally appends scalar (non-convolved) physical features: AHT, ELV, RNG, NRG.

Returns `(X::Matrix{Float32}, feature_names::Vector{String})` where X is
(num_range * num_time) x num_features, with FILL_VAL for invalid gates.
"""
function compute_convolution_features(cfrad, conv_variables::Vector{String},
                                       kernel_bank::Vector{ConvolutionKernel},
                                       valid_mask::AbstractMatrix{Bool},
                                       SIG_QUALITY_VAR::String;
                                       masked_conv_variables::Vector{String} = String[],
                                       masked_conv_kernel_bank::Vector{ConvolutionKernel} = ConvolutionKernel[],
                                       masked_conv_threshold::Float32 = 0.1f0,
                                       met_prob_mask::Union{Nothing, AbstractMatrix{Bool}} = nothing)

    nrows, ncols = size(valid_mask)
    ngates = nrows * ncols

    # Count features: normal conv + masked conv + scalar
    n_conv_features = length(conv_variables) * length(kernel_bank) * 2
    n_masked_conv_features = (met_prob_mask !== nothing) ? length(masked_conv_variables) * length(masked_conv_kernel_bank) * 2 : 0
    n_scalar_features = 4  # AHT, ELV, RNG, NRG
    n_total = n_conv_features + n_masked_conv_features + n_scalar_features

    X = Matrix{Float32}(undef, ngates, n_total)
    feature_names = Vector{String}(undef, n_total)

    col = 1

    for varname in conv_variables
        # Load or compute the variable data
        raw_data = load_conv_variable(cfrad, varname, valid_mask, SIG_QUALITY_VAR)

        # Convert to Float32 matrix, replacing missing/NaN with 0 (masked out by valid_mask)
        data_f32 = Matrix{Float32}(undef, nrows, ncols)
        for j in 1:ncols, i in 1:nrows
            v = raw_data[i, j]
            data_f32[i, j] = (ismissing(v) || (v isa AbstractFloat && isnan(v))) ? 0.0f0 : Float32(v)
        end

        for kern in kernel_bank
            conv_result, vfrac = masked_convolve(data_f32, kern.weights, valid_mask)

            X[:, col] = conv_result[:]
            feature_names[col] = "$(varname)_$(kern.name)"
            col += 1

            X[:, col] = vfrac[:]
            feature_names[col] = "$(varname)_$(kern.name)_vfrac"
            col += 1
        end
    end

    # Masked convolution features: convolve with neighbor_mask derived from met_prob
    if met_prob_mask !== nothing && !isempty(masked_conv_variables)
        masked_valid = valid_mask .&& met_prob_mask
        thresh_str = string(masked_conv_threshold)

        for varname in masked_conv_variables
            raw_data = load_conv_variable(cfrad, varname, valid_mask, SIG_QUALITY_VAR)

            data_f32 = Matrix{Float32}(undef, nrows, ncols)
            for j in 1:ncols, i in 1:nrows
                v = raw_data[i, j]
                data_f32[i, j] = (ismissing(v) || (v isa AbstractFloat && isnan(v))) ? 0.0f0 : Float32(v)
            end

            for kern in masked_conv_kernel_bank
                conv_result, vfrac = masked_convolve(data_f32, kern.weights, valid_mask;
                                                      neighbor_mask=masked_valid)

                X[:, col] = conv_result[:]
                feature_names[col] = "$(varname)_masked_$(thresh_str)_$(kern.name)"
                col += 1

                X[:, col] = vfrac[:]
                feature_names[col] = "$(varname)_masked_$(thresh_str)_$(kern.name)_vfrac"
                col += 1
            end
        end
    end

    # Scalar physical features
    aht = calc_aht(cfrad)
    X[:, col] = [ismissing(x) || isnan(x) ? Float32(FILL_VAL) : Float32(x) for x in aht[:]]
    feature_names[col] = "AHT"
    col += 1

    elv = calc_elv(cfrad)
    X[:, col] = [ismissing(x) || isnan(x) ? Float32(FILL_VAL) : Float32(x) for x in elv[:]]
    feature_names[col] = "ELV"
    col += 1

    rng = calc_rng(cfrad)
    X[:, col] = [ismissing(x) || isnan(x) ? Float32(FILL_VAL) : Float32(x) for x in rng[:]]
    feature_names[col] = "RNG"
    col += 1

    nrg = calc_nrg(cfrad)
    X[:, col] = [ismissing(x) || isnan(x) ? Float32(FILL_VAL) : Float32(x) for x in nrg[:]]
    feature_names[col] = "NRG"
    col += 1

    return X, feature_names
end

"""
    select_features(importance_scores::Vector{Float64}, threshold_fraction::Float64)

Given RF feature importance scores, return indices of features whose importance
is above `threshold_fraction` of the maximum importance.

Used during training to prune low-value features. The returned indices are saved
with the model for use at inference time.
"""
function select_features(importance_scores::Vector{Float64}, threshold_fraction::Float64=0.01)
    max_imp = maximum(importance_scores)
    if max_imp <= 0.0
        return collect(1:length(importance_scores))
    end
    threshold = threshold_fraction * max_imp
    return findall(x -> x >= threshold, importance_scores)
end

"""
    compute_rf_feature_importance(model, X::Matrix{Float32}, Y::Vector;
                                   n_repeats::Int=3, subsample_fraction::Float64=1.0)

Compute permutation-based feature importance for a random forest ensemble.
For each feature, randomly shuffle that column and measure the drop in accuracy.
Higher drop = more important feature.

Performance optimizations:
- Features are evaluated in parallel using `Threads.@threads`
- `subsample_fraction` < 1.0 evaluates on a random subset (e.g., 0.5 = 50% of samples)
  This is statistically stable for large datasets and provides significant speedup.
- Each thread works on its own copy of the subsampled data to avoid race conditions.

Returns a Vector{Float64} of importance scores (accuracy drop), one per feature.
"""
function compute_rf_feature_importance(model, X::Matrix{Float32}, Y::Vector;
                                        n_repeats::Int=3, subsample_fraction::Float64=1.0)
    n_samples, n_features = size(X)

    # Subsample if requested
    if subsample_fraction < 1.0
        n_sub = max(1, round(Int, n_samples * subsample_fraction))
        sub_idx = randperm(n_samples)[1:n_sub]
        X_eval = X[sub_idx, :]
        Y_eval = Y[sub_idx]
        printstyled("  Subsampling $(n_sub)/$(n_samples) gates ($(round(subsample_fraction * 100, digits=1))%) for importance evaluation\n", color=:cyan)
    else
        X_eval = copy(X)
        Y_eval = Y
        n_sub = n_samples
    end

    # Baseline accuracy on the evaluation subset
    baseline_preds = DecisionTree.predict(model, X_eval)
    baseline_acc = sum(baseline_preds .== Y_eval) / n_sub

    importances = Vector{Float64}(undef, n_features)
    start_time = time()

    max_tid = Threads.maxthreadid()
    mem_per_copy = sizeof(X_eval) / 1024^2
    printstyled("  Using $(Threads.nthreads()) thread(s) (max tid=$(max_tid)), $(n_repeats) repeats per feature, $(n_features) features\n", color=:cyan)
    printstyled("  Memory: $(round(mem_per_copy, digits=1)) MB per thread × $(max_tid) slots = $(round(mem_per_copy * max_tid, digits=1)) MB\n", color=:cyan)

    # Pre-allocate one matrix copy per thread (not per feature) to bound memory
    # Use maxthreadid() since threadid() can exceed nthreads() with interactive threads
    thread_X = [copy(X_eval) for _ in 1:max_tid]

    Threads.@threads for f in 1:n_features
        tid = Threads.threadid()
        X_local = thread_X[tid]
        original_col = copy(X_local[:, f])
        acc_drops = 0.0

        for _ in 1:n_repeats
            shuffled = original_col[randperm(n_sub)]
            X_local[:, f] = shuffled

            perm_preds = DecisionTree.predict(model, X_local)
            perm_acc = sum(perm_preds .== Y_eval) / n_sub
            acc_drops += (baseline_acc - perm_acc)
        end

        # Restore original column so the thread's copy is clean for the next feature
        X_local[:, f] = original_col
        importances[f] = acc_drops / n_repeats

        elapsed = time() - start_time
        printstyled("  Feature $(f)/$(n_features) done (elapsed: $(round(elapsed, digits=1))s)\n", color=:light_black)
    end

    total_time = time() - start_time
    printstyled("  Feature importance completed in $(round(total_time, digits=1))s\n", color=:green)

    return importances
end

"""
    build_feature_names(conv_variables, kernel_bank; masked_conv_variables, masked_conv_kernel_bank, masked_conv_threshold)

Build the complete ordered list of feature names matching the column layout of
`compute_convolution_features`: [normal_conv] [masked_conv] [AHT, ELV, RNG, NRG].
"""
function build_feature_names(conv_variables::Vector{String}, kernel_bank::Vector{ConvolutionKernel};
                              masked_conv_variables::Vector{String} = String[],
                              masked_conv_kernel_bank::Vector{ConvolutionKernel} = ConvolutionKernel[],
                              masked_conv_threshold::Float32 = 0.1f0)
    names = String[]
    for varname in conv_variables
        for kern in kernel_bank
            push!(names, "$(varname)_$(kern.name)")
            push!(names, "$(varname)_$(kern.name)_vfrac")
        end
    end
    if !isempty(masked_conv_variables) && !isempty(masked_conv_kernel_bank)
        thresh_str = string(masked_conv_threshold)
        for varname in masked_conv_variables
            for kern in masked_conv_kernel_bank
                push!(names, "$(varname)_masked_$(thresh_str)_$(kern.name)")
                push!(names, "$(varname)_masked_$(thresh_str)_$(kern.name)_vfrac")
            end
        end
    end
    append!(names, ["AHT", "ELV", "RNG", "NRG"])
    return names
end

"""
    get_convolution_feature_count(conv_variables, kernel_bank; masked_conv_variables, masked_conv_kernel_bank)

Return the total number of features that will be produced:
(n_variables * n_kernels * 2) + (n_masked_variables * n_masked_kernels * 2) + 4 scalar features.
"""
function get_convolution_feature_count(conv_variables::Vector{String}, kernel_bank::Vector{ConvolutionKernel};
                                        masked_conv_variables::Vector{String} = String[],
                                        masked_conv_kernel_bank::Vector{ConvolutionKernel} = ConvolutionKernel[])
    n_normal = length(conv_variables) * length(kernel_bank) * 2
    n_masked = length(masked_conv_variables) * length(masked_conv_kernel_bank) * 2
    return n_normal + n_masked + 4
end
