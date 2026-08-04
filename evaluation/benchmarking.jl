using BEAST
using CompScienceMeshes
using ParallelKMeans
using H2Trees
using AdaptiveCrossApproximation
using ButterflyFactorizations
using LinearAlgebra
using BenchmarkTools
using Printf
using PlotlyJS
using Random
using SparseArrays
using StaticArrays
using OhMyThreads
using Logging

# Define a custom logger to count max-rank warnings globally
struct WarningCounterLogger <: AbstractLogger
    parent::AbstractLogger
    count::Base.RefValue{Int}
end

Logging.min_enabled_level(l::WarningCounterLogger) = Logging.min_enabled_level(l.parent)
Logging.shouldlog(l::WarningCounterLogger, level, _module, group, id) =
    Logging.shouldlog(l.parent, level, _module, group, id)
Logging.catch_exceptions(l::WarningCounterLogger) = Logging.catch_exceptions(l.parent)

function Logging.handle_message(
    l::WarningCounterLogger, level, message, _module, group, id, file, line; kwargs...
)
    if level == Logging.Warn
        msg_str = lowercase(string(message))
        # Catch standard ACA max rank warnings
        if occursin("rank", msg_str) || occursin("max", msg_str)
            l.count[] += 1
        end
    end
    # Pass the message through so it still prints to the console
    return Logging.handle_message(
        l.parent, level, message, _module, group, id, file, line; kwargs...
    )
end

# --- Fitting Helper Functions ---
f_nlogn(N) = N * log2(N)
f_nlog2n(N) = N * (log2(N))^2
f_n43logn(N) = (N^(4/3)) * log2(N) # 🚀 NEW: Standard ACA Complexity

function fit_scaling_factor(
    N_vec::Vector{Int}, Y_vec::Vector{Float64}, scaling_func::Function
)
    valid_idx = .!isnan.(Y_vec)
    if sum(valid_idx) == 0
        return NaN
    end
    N_valid = N_vec[valid_idx]
    Y_valid = Y_vec[valid_idx]

    f_vals = scaling_func.(N_valid)
    return sum(Y_valid .* f_vals) / sum(f_vals .^ 2)
end

# 🚀 NEW: Log-Log Linear Regression to find empirical O(N^p) scaling
function estimate_empirical_power(N_vec::Vector{Int}, Y_vec::Vector{Float64})
    valid_idx = .!isnan.(Y_vec) .& (Y_vec .> 0)
    if sum(valid_idx) < 2
        return NaN, NaN
    end
    x = log10.(N_vec[valid_idx])
    y = log10.(Y_vec[valid_idx])

    n = length(x)
    sum_x = sum(x)
    sum_y = sum(y)
    sum_xy = sum(x .* y)
    sum_xx = sum(x .^ 2)

    m = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x^2)
    c = (sum_y - m * sum_x) / n
    return m, 10^c # returns exponent (p) and coefficient (c)
end

function extract_ranks_per_level(Bfmat)
    level_max_ranks = Dict{Int,Int}()

    for bf in Bfmat.BFs
        q_rank = maximum(
            [size(b.data, 1) for b in bf.Q if b.data isa AbstractMatrix]; init=0
        )
        level_max_ranks[1] = max(get(level_max_ranks, 1, 0), q_rank)

        for (l, level) in enumerate(bf.R)
            r_rank = maximum(
                [size(b.data, 1) for b in level.blocks if b.data isa AbstractMatrix]; init=0
            )
            level_max_ranks[l + 1] = max(get(level_max_ranks, l+1, 0), r_rank)
        end
    end

    return level_max_ranks
end

function bf_matrix_memory(Bfmat)
    num_elements = 0
    for bf in Bfmat.BFs
        for block in bf.Q
            if block.data isa AbstractMatrix
                num_elements += length(block.data)
            end
        end
        for block in bf.P
            if block.data isa AbstractMatrix
                num_elements += length(block.data)
            end
        end
        for level in bf.R
            for block in level.blocks
                if block.data isa AbstractMatrix
                    num_elements += length(block.data)
                end
            end
        end
    end

    num_elements += SparseArrays.nnz(Bfmat.nearinteractions)
    mem_bytes = num_elements * sizeof(ComplexF64)
    return mem_bytes / 1024^2
end

function extract_memory_per_depth(Bfmat)
    mem_dict = Dict{Int,Float64}()

    for bf in Bfmat.BFs
        depth = length(bf.R) + 2
        num_elements = 0
        for block in bf.Q
            if block.data isa AbstractMatrix
                num_elements += length(block.data)
            end
        end
        for block in bf.P
            if block.data isa AbstractMatrix
                num_elements += length(block.data)
            end
        end
        for level in bf.R
            for block in level.blocks
                if block.data isa AbstractMatrix
                    num_elements += length(block.data)
                end
            end
        end

        mem_mb = (num_elements * sizeof(ComplexF64)) / 1024^2
        mem_dict[depth] = get(mem_dict, depth, 0.0) + mem_mb
    end

    return mem_dict
end

function build_benchmark_mesh(shape, h::Float64)
    if shape isa Function
        return shape(h)
    elseif shape == :sphere
        return meshsphere(1.0, h)
    elseif shape == :cube || shape == :box
        return meshcuboid(1.0, 1.0, 1.0, h)
    else
        error(
            "Unsupported shape: $shape. Options: :sphere, :cube, :cylinder, or a custom function h -> mesh",
        )
    end
end

function build_beast_operator(ie_type::Symbol, k::Float64)
    if ie_type == :EFIE
        return Maxwell3D.singlelayer(; wavenumber=k)
    elseif ie_type == :MFIE
        return Maxwell3D.doublelayer(; wavenumber=k)
    else
        error("Unsupported ie_type: $ie_type. Options: :EFIE, :MFIE")
    end
end

function run_benchmarks(
    h_values;
    shape::Union{Symbol,Function}=:sphere,
    treekind::Symbol=:KMeansTree,
    ie_type::Symbol=:EFIE,
    bf_tol::Float64=1e-3,
    minbflvl::Int=3,
    maxpointsbisection=80,
    adaptive=false,
    admissibility_spec::Symbol=:isFarFunctor,
    checkfarfieldaccuracy::Bool=false,
    acarefmat::Bool=true,
    acacomparison::Bool=false,
    disjointgeom::Bool=true,
    unbalancedints::Bool=false,
    highfscaling::Bool=true,
    leafcompression::Bool=false,
    csv_file::String="benchmark_results.csv",
    log_file_path::String="benchmark_log.txt",
    scheduler=OhMyThreads.DynamicScheduler(),
    acamaxrank::Int=60,
    refacamaxrank::Int=100,
)
    BLAS.set_num_threads(1)

    # Storage arrays
    N_vals = Int[]
    t_aca_vals = Float64[]
    t_bf_vals = Float64[]
    mem_aca_vals = Float64[]
    mem_bf_total_vals = Float64[]
    mem_ButterflyFactorization_Mat_vals = Float64[]
    err_bf_vals = Float64[]
    t_mv_aca_vals = Float64[]
    t_mv_bf_vals = Float64[]
    k_vals = Float64[]
    max_R_ranks = Int[]

    mem_by_depth_history = Dict{Int,Vector{Float64}}()
    p_level_ranks_all = PlotlyJS.SyncPlot[]

    println("==========================================================")
    println(" Starting Benchmarks | Shape: $shape | IE Type: $ie_type")
    println(" ACA Comparison      : $acacomparison")
    println(" Wavenumber Scaling  : $(highfscaling ? "Dynamic" : "Fixed (High-F)")")
    println("==========================================================\n")

    csv_stream = open(csv_file, "w")
    write(
        csv_stream,
        "h,N,shape,ie_type,time_aca_s,time_bf_s,ref_bf_time_nlogn,ref_bf_time_nlog2n,mem_aca_mb,mem_bf_total_mb,mem_bf_entries_mb,ref_bf_mem_nlogn,ref_bf_mem_nlog2n,rel_err_bf_far,mv_aca_s,mv_bf_s,ref_bf_mv_nlogn\n",
    )
    flush(csv_stream)

    log_stream = open(log_file_path, "w")
    write(log_stream, "Starting benchmarks for Shape: $shape, IE: $ie_type\n")
    flush(log_stream)

    i = 1
    for h in h_values
        round_str = @sprintf(
            "==================================================== Round %-1d (h = %.3f) ==========================================================",
            i,
            h
        )
        println(round_str)
        write(log_stream, round_str * "\n")
        flush(log_stream)

        # 🚀 FIX: Lock to the smallest h if highfscaling is false
        if highfscaling
            lambda = 10 * h
        else
            lambda = 10 * minimum(h_values)
        end
        k = 2 * pi / lambda

        op = build_beast_operator(ie_type, k)
        m = build_benchmark_mesh(shape, h)
        X = raviartthomas(m)
        N = length(X)

        if disjointgeom
            y = translate(build_benchmark_mesh(shape, h), SVector(3.0, 0.0, 0.0))
            Y = raviartthomas(y)
            if treekind == :KMeansTree
                Stree = H2Trees.KMeansTree(X.pos, 2; minvalues=50)
                Otree = H2Trees.KMeansTree(Y.pos, 2; minvalues=50)
            elseif treekind == :BisectionTree
                Stree = ButterflyFactorizations.build_bisection_tree(
                    X.pos; max_points=maxpointsbisection
                )
                Otree = ButterflyFactorizations.build_bisection_tree(
                    Y.pos; max_points=maxpointsbisection
                )
            else
                Stree = H2Trees.TwoNTree(X, h)
                Otree = H2Trees.TwoNTree(Y, h)
            end

            blktree = H2Trees.BlockTree(Otree, Stree)
            ACAstree = H2Trees.KMeansTree(X.pos, 2; minvalues=100)
            ACAotree = H2Trees.KMeansTree(Y.pos, 2; minvalues=100)
            ACAblktree = H2Trees.BlockTree(ACAstree, ACAotree)
        else
            if treekind == :KMeansTree
                tree = H2Trees.KMeansTree(X.pos, 2; minvalues=30)
            elseif treekind == :BisectionTree
                tree = ButterflyFactorizations.build_bisection_tree(
                    X.pos; max_points=maxpointsbisection
                )
            else
                tree = H2Trees.TwoNTree(X, h)
            end

            blktree = H2Trees.BlockTree(tree, tree)
            ACAtree = H2Trees.KMeansTree(X.pos, 2; minvalues=100)
            ACAblktree = H2Trees.BlockTree(ACAtree, ACAtree)
        end

        if admissibility_spec == :CenterDistanceAdmissibility
            admissibility = ButterflyFactorizations.CenterDistanceAdmissibility(
                ButterflyFactorizations.tree_parameters(
                    blktree, ButterflyFactorizations.CenterDistanceAdmissibility
                ).β,
            )
        elseif admissibility_spec == :isFarFunctor
            admissibility = ButterflyFactorizations.isFarFunctor(
                ButterflyFactorizations.tree_parameters(
                    blktree, ButterflyFactorizations.isFarFunctor
                ).α,
            )
        else
            error("Unsupported admissibility: $admissibility_spec")
        end

        local_isnear(ta, tb, na, nb) = !admissibility(ta, tb, na, nb)

        println("\nStarting ButterflyFactorization ($ie_type)...")
        farints, nearints = ButterflyFactorizations.nearandfar(
            blktree,
            admissibility;
            unbalancedints=unbalancedints,
            leafcomp=leafcompression,
            leafimbalance=(!checkfarfieldaccuracy),
            minbflvl=minbflvl,
        )
        ButterflyFactorizations.compute_interaction_percentages(
            nearints,
            farints,
            ButterflyFactorizations.cluster_testtree(blktree),
            ButterflyFactorizations.cluster_trialtree(blktree),
        )

        if disjointgeom
            t_bf = @elapsed begin
                Bfmat = ButterflyFactorizations.PetrovGalerkinBF(
                    op,
                    Y,
                    X,
                    blktree,
                    k;
                    compressor=ButterflyFactorizations.PartialQR(),
                    scheduler=scheduler,
                    tol=bf_tol,
                    unbalancedints=unbalancedints,
                    leafimbalance=(!checkfarfieldaccuracy),
                    leafcomp=leafcompression,
                    admissibility=admissibility,
                    minbflvl=minbflvl,
                    adaptive=adaptive,
                )
            end
        else
            t_bf = @elapsed begin
                Bfmat = ButterflyFactorizations.PetrovGalerkinBF(
                    op,
                    X,
                    X,
                    blktree,
                    k;
                    compressor=ButterflyFactorizations.PartialQR(),
                    scheduler=scheduler,
                    tol=bf_tol,
                    unbalancedints=unbalancedints,
                    leafimbalance=(!checkfarfieldaccuracy),
                    leafcomp=leafcompression,
                    admissibility=admissibility,
                    minbflvl=minbflvl,
                    adaptive=adaptive,
                )
            end
        end

        ranks_dict = extract_ranks_per_level(Bfmat)
        current_mem_by_depth = extract_memory_per_depth(Bfmat)

        for d in keys(current_mem_by_depth)
            if !haskey(mem_by_depth_history, d)
                mem_by_depth_history[d] = fill(NaN, i - 1)
            end
        end
        for d in keys(mem_by_depth_history)
            push!(mem_by_depth_history[d], get(current_mem_by_depth, d, NaN))
        end

        println("--- Maximum Ranks per BF Level ---")
        for l in sort(collect(keys(ranks_dict)))
            println("Level $l: ", ranks_dict[l])
        end

        println("--- Memory Breakdown by BF Depth ---")
        for d in sort(collect(keys(current_mem_by_depth)))
            @printf("Depth %d: %.2f MB\n", d, current_mem_by_depth[d])
            write(
                log_stream,
                @sprintf("Depth %d Memory: %.2f MB\n", d, current_mem_by_depth[d])
            )
        end
        println("----------------------------------")

        tree_height = length(blktree.testcluster.nodesatlevel)
        println("Total Tree Height: $tree_height")

        sorted_levels = sort(collect(keys(ranks_dict)))
        ranks_at_levels = [ranks_dict[l] for l in sorted_levels]

        p_level_ranks = plot(
            [
                scatter(;
                    x=sorted_levels,
                    y=ranks_at_levels,
                    mode="lines+markers",
                    line=attr(; color="darkorange", width=3),
                    marker=attr(; size=8),
                ),
            ],
            Layout(;
                title="BF Rank vs BF Level (h = $h, k = $(round(k, digits=2)))<br><sup>Total Tree Height = $tree_height</sup>",
                xaxis_title="Butterfly Factorization Level (1 = Leaves)",
                yaxis_title="Maximum Rank",
                template="plotly_white",
                yaxis=attr(; rangemode="tozero"),
            ),
        )

        savefig(p_level_ranks, "plot_ranks_vs_level_h_$(h).html")
        push!(p_level_ranks_all, p_level_ranks)

        r_ranks = [ranks_dict[l] for l in sorted_levels if l > 1]
        max_r_rank = isempty(r_ranks) ? 0 : maximum(r_ranks)

        push!(k_vals, k)
        push!(max_R_ranks, max_r_rank)

        time_bf_str = @sprintf("Time for ButterflyFactorization: %.3f seconds", t_bf)
        println(time_bf_str)
        write(log_stream, time_bf_str * "\n")

        mem_bf_total = Base.summarysize(Bfmat) / 1024^2
        mem_ButterflyFactorization_Mat = bf_matrix_memory(Bfmat)
        mem_bf_str = @sprintf(
            "Total memory for ButterflyFactorization: %.2f MB | Entries only: %.2f MB",
            mem_bf_total,
            mem_ButterflyFactorization_Mat
        )
        println(mem_bf_str)
        write(log_stream, mem_bf_str * "\n")

        xtest = randn(ComplexF64, size(Bfmat, 2))

        println("Benchmarking MV product for ButterflyFactorization...")
        t_mv_bf = @belapsed $Bfmat * $xtest
        x_mv_bf = Bfmat * xtest
        mv_bf_str = @sprintf("Time for Butterfly mat-vec: %.5f seconds", t_mv_bf)
        println(mv_bf_str)
        write(log_stream, mv_bf_str * "\n")

        err_bf = NaN
        ref_aca_warnings = 0
        if checkfarfieldaccuracy
            if acarefmat
                println(
                    "\nComputing reference ACA Far-Matrix for accuracy check (tol = $(bf_tol * 1e-2))...",
                )
                ref_aca_logger = WarningCounterLogger(current_logger(), Ref(0))
                refmat = with_logger(ref_aca_logger) do
                    if disjointgeom
                        HMatrix(
                            op,
                            Y,
                            X,
                            blktree;
                            tol=bf_tol * 1e-2,
                            spaceordering=AdaptiveCrossApproximation.PreserveSpaceOrder(),
                            scheduler=scheduler,
                            maxrank=refacamaxrank,
                            isnear=local_isnear,
                        )
                    else
                        HMatrix(
                            op,
                            X,
                            X,
                            blktree;
                            tol=bf_tol * 1e-2,
                            spaceordering=AdaptiveCrossApproximation.PreserveSpaceOrder(),
                            scheduler=scheduler,
                            maxrank=refacamaxrank,
                            isnear=local_isnear,
                        )
                    end
                end
                ref_aca_warnings = ref_aca_logger.count[]
                ref_far = AdaptiveCrossApproximation.farmatrix(refmat)
                bf_far = ButterflyFactorizations.farmatrix(Bfmat)

                y_exact_far = ref_far * xtest
                y_bf_far = bf_far * xtest

                err_bf = norm(y_exact_far - y_bf_far) / norm(y_exact_far)

                err_str = @sprintf(
                    "Relative error of Far-field mat-vec: %.2e | Ref ACA MaxRank Warnings: %d",
                    err_bf,
                    ref_aca_warnings
                )
                println(err_str)
                write(log_stream, err_str * "\n")
            else
                println(
                    "\nComputing reference Butterfly Far-Matrix for accuracy check (tol = $(bf_tol * 1e-2))...",
                )
                refmat = if disjointgeom
                    ButterflyFactorizations.PetrovGalerkinBF(
                        op,
                        Y,
                        X,
                        blktree,
                        k;
                        compressor=ButterflyFactorizations.PartialQR(),
                        scheduler=scheduler,
                        tol=bf_tol*1e-2,
                        unbalancedints=unbalancedints,
                        leafimbalance=(!checkfarfieldaccuracy),
                        leafcomp=leafcompression,
                        admissibility=admissibility,
                        minbflvl=minbflvl,
                        adaptive=adaptive,
                        farfieldonly=true,
                    )
                else
                    ButterflyFactorizations.PetrovGalerkinBF(
                        op,
                        X,
                        X,
                        blktree,
                        k;
                        compressor=ButterflyFactorizations.PartialQR(),
                        scheduler=scheduler,
                        tol=bf_tol*1e-2,
                        unbalancedints=unbalancedints,
                        leafimbalance=(!checkfarfieldaccuracy),
                        leafcomp=leafcompression,
                        admissibility=admissibility,
                        minbflvl=minbflvl,
                        adaptive=adaptive,
                        farfieldonly=true,
                    )
                end

                ref_far = ButterflyFactorizations.farmatrix(refmat)
                bf_far = ButterflyFactorizations.farmatrix(Bfmat)

                y_exact_far = ref_far * xtest
                y_bf_far = bf_far * xtest

                err_bf = norm(y_exact_far - y_bf_far) / norm(y_exact_far)

                err_str = @sprintf("Relative error of Far-field mat-vec: %.2e", err_bf)
                println(err_str)
                write(log_stream, err_str * "\n")
            end
        end

        t_aca = NaN
        mem_aca = NaN
        t_mv_aca = NaN
        aca_warnings = 0

        if acacomparison
            println("\nStarting Standard ACA ($ie_type)...")
            aca_logger = WarningCounterLogger(current_logger(), Ref(0))
            t_aca = @elapsed begin
                hmat = with_logger(aca_logger) do
                    if disjointgeom
                        HMatrix(
                            op,
                            Y,
                            X,
                            ACAblktree;
                            tol=bf_tol,
                            spaceordering=AdaptiveCrossApproximation.PreserveSpaceOrder(),
                            scheduler=scheduler,
                            maxrank=acamaxrank,
                        )
                    else
                        HMatrix(
                            op,
                            X,
                            X,
                            ACAblktree;
                            tol=bf_tol,
                            spaceordering=AdaptiveCrossApproximation.PreserveSpaceOrder(),
                            scheduler=scheduler,
                            maxrank=acamaxrank,
                        )
                    end
                end
            end
            aca_warnings = aca_logger.count[]

            time_aca_str = @sprintf(
                "Time for HMatrix (ACA): %.3f seconds | ACA MaxRank Warnings: %d",
                t_aca,
                aca_warnings
            )
            println(time_aca_str)
            write(log_stream, time_aca_str * "\n")

            mem_aca = Base.summarysize(hmat) / 1024^2
            mem_aca_str = @sprintf("Total memory for HMatrix (ACA): %.2f MB", mem_aca)
            println(mem_aca_str)
            write(log_stream, mem_aca_str * "\n")

            println("Benchmarking MV product for ACA...")
            t_mv_aca = @belapsed $hmat * $xtest
            mv_aca_str = @sprintf("Time for ACA mat-vec: %.5f seconds", t_mv_aca)
            x_mv_aca = hmat * xtest
            relerror = norm(x_mv_aca - x_mv_bf) / norm(x_mv_aca)
            relerror_str = @sprintf(
                "Relative error between ACA and BF mat-vec: %.2e", relerror
            )
            println(relerror_str)
            write(log_stream, relerror_str * "\n")
            println(mv_aca_str)
            write(log_stream, mv_aca_str * "\n")
        end

        push!(N_vals, N)
        push!(t_aca_vals, t_aca)
        push!(t_bf_vals, t_bf)
        push!(mem_aca_vals, mem_aca)
        push!(mem_bf_total_vals, mem_bf_total)
        push!(mem_ButterflyFactorization_Mat_vals, mem_ButterflyFactorization_Mat)
        push!(err_bf_vals, err_bf)
        push!(t_mv_aca_vals, t_mv_aca)
        push!(t_mv_bf_vals, t_mv_bf)

        # 🚀 Theoretical Fits
        c_time_nlogn = fit_scaling_factor(N_vals, t_bf_vals, f_nlogn)
        c_time_nlog2n = fit_scaling_factor(N_vals, t_bf_vals, f_nlog2n)
        c_mem_nlogn = fit_scaling_factor(N_vals, mem_bf_total_vals, f_nlogn)
        c_mem_nlog2n = fit_scaling_factor(N_vals, mem_bf_total_vals, f_nlog2n)
        c_mv_nlogn = fit_scaling_factor(N_vals, t_mv_bf_vals, f_nlogn)
        c_mv_nlog2n = fit_scaling_factor(N_vals, t_mv_bf_vals, f_nlog2n)

        # ACA Fits
        c_time_aca = fit_scaling_factor(N_vals, t_aca_vals, f_n43logn)
        c_mem_aca = fit_scaling_factor(N_vals, mem_aca_vals, f_n43logn)
        c_mv_aca = fit_scaling_factor(N_vals, t_mv_aca_vals, f_n43logn)

        # 🚀 Empirical Actual Scalings (Log-Log Regression)
        p_time_emp, coef_time_emp = estimate_empirical_power(N_vals, t_bf_vals)
        p_mem_emp, coef_mem_emp = estimate_empirical_power(N_vals, mem_bf_total_vals)
        p_mv_emp, coef_mv_emp = estimate_empirical_power(N_vals, t_mv_bf_vals)

        shape_str = shape isa Symbol ? string(shape) : "custom"
        csv_row = @sprintf(
            "%.4f,%d,%s,%s,%.5f,%.5f,%.5f,%.5f,%.2f,%.2f,%.2f,%.2f,%.2f,%.3e,%.5f,%.5f,%.5f\n",
            h,
            N,
            shape_str,
            string(ie_type),
            t_aca,
            t_bf,
            c_time_nlogn * f_nlogn(N),
            c_time_nlog2n * f_nlog2n(N),
            mem_aca,
            mem_bf_total,
            mem_ButterflyFactorization_Mat,
            c_mem_nlogn * f_nlogn(N),
            c_mem_nlog2n * f_nlog2n(N),
            err_bf,
            t_mv_aca,
            t_mv_bf,
            c_mv_nlogn * f_nlogn(N)
        )
        write(csv_stream, csv_row)
        flush(csv_stream)

        # --- INCREMENTAL PLOTTING ---
        ref_time_nlogn = c_time_nlogn .* f_nlogn.(N_vals)
        ref_time_nlog2n = c_time_nlog2n .* f_nlog2n.(N_vals)
        ref_mem_nlogn = c_mem_nlogn .* f_nlogn.(N_vals)
        ref_mem_nlog2n = c_mem_nlog2n .* f_nlog2n.(N_vals)
        ref_mv_nlogn = c_mv_nlogn .* f_nlogn.(N_vals)
        ref_mv_nlog2n = c_mv_nlog2n .* f_nlog2n.(N_vals)

        time_traces = [
            scatter(;
                x=N_vals,
                y=t_bf_vals,
                name="Butterfly",
                mode="lines+markers",
                line=attr(; color="royalblue"),
            ),
            scatter(;
                x=N_vals,
                y=ref_time_nlogn,
                name="O(N log N) Fit",
                mode="lines",
                line=attr(; color="gray", dash="dash"),
            ),
            scatter(;
                x=N_vals,
                y=ref_time_nlog2n,
                name="O(N log² N) Fit",
                mode="lines",
                line=attr(; color="black", dash="dot"),
            ),
        ]
        if !isnan(p_time_emp)
            emp_label = @sprintf("Empirical BF O(N^%.2f)", p_time_emp)
            push!(
                time_traces,
                scatter(;
                    x=N_vals,
                    y=coef_time_emp .* (N_vals .^ p_time_emp),
                    name=emp_label,
                    mode="lines",
                    line=attr(; color="blue", dash="dot"),
                ),
            )
        end

        mem_traces = [
            scatter(;
                x=N_vals,
                y=mem_bf_total_vals,
                name="Butterfly Total",
                mode="lines+markers",
                line=attr(; color="royalblue", width=4),
            ),
            scatter(;
                x=N_vals,
                y=mem_ButterflyFactorization_Mat_vals,
                name="Butterfly Matrix Entries",
                mode="lines+markers",
                line=attr(; color="skyblue", width=3, dash="dash"),
            ),
            scatter(;
                x=N_vals,
                y=ref_mem_nlog2n,
                name="Total O(N log² N) Fit",
                mode="lines",
                line=attr(; color="black", dash="dot"),
            ),
        ]
        if !isnan(p_mem_emp)
            emp_label = @sprintf("Empirical BF O(N^%.2f)", p_mem_emp)
            push!(
                mem_traces,
                scatter(;
                    x=N_vals,
                    y=coef_mem_emp .* (N_vals .^ p_mem_emp),
                    name=emp_label,
                    mode="lines",
                    line=attr(; color="blue", dash="dot"),
                ),
            )
        end

        for d in sort(collect(keys(mem_by_depth_history)))
            vals = mem_by_depth_history[d]
            c_fit = fit_scaling_factor(N_vals, vals, f_nlog2n)
            ref_fit = c_fit .* f_nlog2n.(N_vals)

            push!(
                mem_traces,
                scatter(;
                    x=N_vals,
                    y=vals,
                    name="Depth $d BFs",
                    mode="lines+markers",
                    marker=attr(; size=6),
                ),
            )
            push!(
                mem_traces,
                scatter(;
                    x=N_vals,
                    y=ref_fit,
                    name="Depth $d Fit (O(N log² N))",
                    mode="lines",
                    line=attr(; dash="dot"),
                ),
            )
        end

        mv_traces = [
            scatter(;
                x=N_vals,
                y=t_mv_bf_vals,
                name="Butterfly",
                mode="lines+markers",
                line=attr(; color="royalblue"),
            ),
            scatter(;
                x=N_vals,
                y=ref_mv_nlogn,
                name="O(N log N) Fit",
                mode="lines",
                line=attr(; color="gray", dash="dash"),
            ),
            scatter(;
                x=N_vals,
                y=ref_mv_nlog2n,
                name="O(N log² N) Fit",
                mode="lines",
                line=attr(; color="black", dash="dot"),
            ),
        ]
        if !isnan(p_mv_emp)
            emp_label = @sprintf("Empirical BF O(N^%.2f)", p_mv_emp)
            push!(
                mv_traces,
                scatter(;
                    x=N_vals,
                    y=coef_mv_emp .* (N_vals .^ p_mv_emp),
                    name=emp_label,
                    mode="lines",
                    line=attr(; color="blue", dash="dot"),
                ),
            )
        end

        if acacomparison
            push!(
                time_traces,
                scatter(;
                    x=N_vals,
                    y=t_aca_vals,
                    name="ACA",
                    mode="lines+markers",
                    line=attr(; color="firebrick"),
                ),
            )
            if !isnan(c_time_aca)
                push!(
                    time_traces,
                    scatter(;
                        x=N_vals,
                        y=c_time_aca .* f_n43logn.(N_vals),
                        name="ACA Fit O(N^(4/3) log N)",
                        mode="lines",
                        line=attr(; color="red", dash="dash"),
                    ),
                )
            end

            push!(
                mem_traces,
                scatter(;
                    x=N_vals,
                    y=mem_aca_vals,
                    name="ACA Total",
                    mode="lines+markers",
                    line=attr(; color="firebrick"),
                ),
            )
            if !isnan(c_mem_aca)
                push!(
                    mem_traces,
                    scatter(;
                        x=N_vals,
                        y=c_mem_aca .* f_n43logn.(N_vals),
                        name="ACA Fit O(N^(4/3) log N)",
                        mode="lines",
                        line=attr(; color="red", dash="dash"),
                    ),
                )
            end

            push!(
                mv_traces,
                scatter(;
                    x=N_vals,
                    y=t_mv_aca_vals,
                    name="ACA",
                    mode="lines+markers",
                    line=attr(; color="firebrick"),
                ),
            )
            if !isnan(c_mv_aca)
                push!(
                    mv_traces,
                    scatter(;
                        x=N_vals,
                        y=c_mv_aca .* f_n43logn.(N_vals),
                        name="ACA Fit O(N^(4/3) log N)",
                        mode="lines",
                        line=attr(; color="red", dash="dash"),
                    ),
                )
            end
        end

        p_time = plot(
            time_traces,
            Layout(;
                title="Build Time vs N ($shape_str, $ie_type)",
                xaxis_title="N (DOFs)",
                yaxis_title="Time (s)",
                yaxis_type="log",
                xaxis_type="log",
                template="plotly_white",
            ),
        )
        savefig(p_time, "plot_build_time.html")

        p_mem = plot(
            mem_traces,
            Layout(;
                title="Memory Usage vs N Grouped by Depth ($shape_str, $ie_type)",
                xaxis_title="N (DOFs)",
                yaxis_title="Memory (MB)",
                yaxis_type="log",
                xaxis_type="log",
                template="plotly_white",
            ),
        )
        savefig(p_mem, "plot_memory_usage.html")

        p_mv = plot(
            mv_traces,
            Layout(;
                title="Mat-Vec Time vs N ($shape_str, $ie_type)",
                xaxis_title="N (DOFs)",
                yaxis_title="Time (s)",
                yaxis_type="log",
                xaxis_type="log",
                template="plotly_white",
            ),
        )
        savefig(p_mv, "plot_mat_vec_time.html")

        p_err = plot(
            [
                scatter(;
                    x=N_vals,
                    y=err_bf_vals,
                    name="Butterfly Far-Field Rel Error",
                    mode="lines+markers",
                    line=attr(; color="seagreen"),
                ),
            ],
            Layout(;
                title="Accuracy vs N ($shape_str, $ie_type)",
                xaxis_title="N (DOFs)",
                yaxis_title="Relative Error",
                yaxis_type="log",
                xaxis_type="log",
                template="plotly_white",
            ),
        )
        savefig(p_err, "plot_accuracy.html")

        p_rank_vs_k = plot(
            [
                scatter(;
                    x=k_vals,
                    y=max_R_ranks,
                    mode="lines+markers",
                    line=attr(; color="purple", width=3),
                    marker=attr(; size=8),
                ),
            ],
            Layout(;
                title="Max R-Block Rank vs. Wavenumber (k) | $shape_str, $ie_type",
                xaxis_title="Wavenumber (k)",
                yaxis_title="Maximum Rank in R Factors",
                xaxis_type="log",
                yaxis_type="linear",
                template="plotly_white",
                yaxis=attr(; rangemode="tozero"),
            ),
        )
        savefig(p_rank_vs_k, "plot_max_R_rank_vs_k.html")

        end_str = @sprintf(
            "================================================ End of Round %-1d =======================================================\n",
            i
        )
        println(end_str)
        write(log_stream, end_str * "\n")
        flush(log_stream)
        i += 1
    end

    close(csv_stream)

    header1 = "=========================================================================================================================="
    header2 = @sprintf(
        "%-6s | %-8s | %-12s | %-12s | %-12s | %-12s | %-12s | %-10s | %-10s",
        "h",
        "N",
        "Time ACA (s)",
        "Time BF (s)",
        "Mem ACA (MB)",
        "Mem BF (MB)",
        "Err BF Far",
        "MV ACA (s)",
        "MV BF (s)"
    )

    println(header1)
    println(header2)
    println(header1)
    write(
        log_stream,
        "\n\nFINAL SUMMARY TABLE:\n" * header1 * "\n" * header2 * "\n" * header1 * "\n",
    )

    for j in 1:length(h_values)
        row_str = @sprintf(
            "%-6.2f | %-8d | %-12.3f | %-12.3f | %-12.2f | %-12.2f | %-12.2e | %-10.5f | %-10.5f",
            h_values[j],
            N_vals[j],
            t_aca_vals[j],
            t_bf_vals[j],
            mem_aca_vals[j],
            mem_bf_total_vals[j],
            err_bf_vals[j],
            t_mv_aca_vals[j],
            t_mv_bf_vals[j]
        )
        println(row_str)
        write(log_stream, row_str * "\n")
    end
    println(header1)
    write(log_stream, header1 * "\n")
    close(log_stream)

    println("\nAll benchmarks complete! Data saved to '$csv_file'.")

    return p_time, p_mem, p_mv, p_err, p_rank_vs_k, p_level_ranks_all
end
# --- Execution ---
h_values = [0.03, 0.02, 0.015, 0.01, 0.0075, 0.005]
# [0.03, 0.02, 0.015, 0.01, 0.0075, 0.005] --> N = [45k, 90k, 180k, 360k, 720k, 1.440M] for sphere
p_time, p_mem, p_mv, p_err, p_rank_vs_k, p_level_ranks_all = run_benchmarks(
    h_values;

    # -------------------------------------------------------------------------
    # Physical Setup & Geometry
    # -------------------------------------------------------------------------
    shape=:sphere,              # Geometry of the scatterer (Options: :sphere, :cube, or a custom h -> mesh function)
    ie_type=:EFIE,              # Integral equation formulation (Options: :EFIE, :MFIE)
    highfscaling=false,         # false: Locks wavenumber k to the finest mesh (isolates O(N log^2 N) scaling). true: Scales k with h (tracks physical radar cross-section growth).
    disjointgeom=false,         # false: Standard self-interaction (dense diagonal). true: Translates target mesh away to evaluate pure far-field transmission.

    # -------------------------------------------------------------------------
    # Butterfly Tree & Admissibility Configuration
    # -------------------------------------------------------------------------
    treekind=:KMeansTree,                   # Clustering strategy (Options: :KMeansTree, :BisectionTree, :TwoNTree)
    admissibility_spec=:isFarFunctor,       # Near/Far separation criteria (Options: :CenterDistanceAdmissibility, :isFarFunctor)
    maxpointsbisection=100,                 # Maximum allowed degrees of freedom in a leaf node (specifically for BisectionTrees)
    minbflvl=3,                             # Tree depth where compression begins (ignored if leafcompression is true)
    unbalancedints=false,                   # Allows butterfly interactions between source and observer clusters situated at different tree depths

    # -------------------------------------------------------------------------
    # Compression & Accuracy Targets
    # -------------------------------------------------------------------------
    bf_tol=1e-3,                # Target relative tolerance for the PartialQR compressor
    adaptive=false,             # true: Dynamically estimates rank bounds during compression. false: Uses fixed rank limits.
    leafcompression=true,       # true: Compresses interactions all the way down to leaf nodes (Crucial for 1:1 error comparison with ACA).

    # -------------------------------------------------------------------------
    # Benchmarking Flags (ACA vs. Butterfly)
    # -------------------------------------------------------------------------
    checkfarfieldaccuracy=true, # explicitly computes the relative error of the far-field Mat-Vec product against a highly accurate reference matrix
    acarefmat=false,            # true: Uses a tightly toleranced ACA matrix as the exact truth. false: Uses a highly toleranced Butterfly matrix as truth.
    acacomparison=true,         # Builds a standard ACA HMatrix alongside the Butterfly matrix to compare build time, Mat-Vec time, and memory footprint
    refacamaxrank=100,          # Hard limit for the maximum rank allowed in the ACA accuracy reference matrix
    acamaxrank=60,              # Hard limit for the maximum rank allowed in the standard ACA comparison matrix

    # -------------------------------------------------------------------------
    # Logging & Schedulers
    # -------------------------------------------------------------------------
    scheduler=OhMyThreads.DynamicScheduler(), # Threading strategy for matrix assembly and compression
    csv_file="benchmark_results.csv",
    log_file_path="benchmark_log.txt",
);

display(p_time)
display(p_mem)
display(p_mv)
display(p_err)
display(p_rank_vs_k)

for p in p_level_ranks_all
    display(p)
end
