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

# --- Geometry Helper ---
function compute_hmax(mesh)
    max_edge = 0.0
    pts = vertices(mesh)
    for cell in cells(mesh)
        for i in cell.indices[1:length(cell.indices)]
            for j in cell.indices[(i+1):length(cell.indices)]
                max_edge = max(max_edge, norm(pts[i] - pts[j]))
            end
        end
    end
    return max_edge
end

# --- Fitting Helper Functions ---
f_nlogn(N) = N * log2(N)
f_nlog2n(N) = N * (log2(N))^2
f_n43logn(N) = (N^(4/3)) * log2(N)

# Log-space fitting to give equal weight to all N points
function fit_scaling_factor(
    N_vec::Vector{Int}, Y_vec::Vector{Float64}, scaling_func::Function
)
    valid_idx = .!isnan.(Y_vec) .& (Y_vec .> 0.0)
    if sum(valid_idx) == 0
        return NaN
    end
    N_valid = N_vec[valid_idx]
    Y_valid = Y_vec[valid_idx]

    # Minimize sum((log(Y) - log(c * f(N)))^2)
    # log(c) = mean(log(Y) - log(f(N)))
    log_c = sum(log.(Y_valid) .- log.(scaling_func.(N_valid))) / length(N_valid)
    return exp(log_c)
end

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
    return m, 10^c
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
            level_max_ranks[l+1] = max(get(level_max_ranks, l+1, 0), r_rank)
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

function build_beast_operator(ie_type::Symbol, k::Float64)
    if ie_type == :EFIE
        return Maxwell3D.singlelayer(; wavenumber=k)
    elseif ie_type == :MFIE
        return Maxwell3D.doublelayer(; wavenumber=k)
    else
        error("Unsupported ie_type: $ie_type. Options: :EFIE, :MFIE")
    end
end


function estimate_norm(mat; tol=1e-4, itmax=100)
    v = rand(ComplexF64, size(mat, 2))
    v ./= norm(v)
    itermin = 3
    i = 1
    σold = 1.0
    σnew = 1.0

    while (norm(sqrt(σold) - sqrt(σnew)) / norm(sqrt(σold)) > tol || i < itermin) &&
        i < itmax
        σold = σnew
        w = mat * v
        x = adjoint(mat) * w
        σnew = norm(x)
        v = x ./ σnew
        i += 1
    end
    return sqrt(σnew)
end

function estimate_reldifference(hmat::H, refmat; tol=1e-4, itmax=100) where {H}
    @assert size(hmat) == size(refmat) "Dimensions of matrices do not match"

    v = rand(ComplexF64, size(hmat, 2))
    v ./= norm(v)
    itermin = 3
    i = 1
    σold = 1.0
    σnew = 1.0

    while (norm(sqrt(σold) - sqrt(σnew)) / norm(sqrt(σold)) > tol || i < itermin) &&
        i < itmax
        σold = σnew
        w = (hmat * v) .- (refmat * v)
        x = (adjoint(hmat) * w) .- (adjoint(refmat) * w)
        σnew = norm(x)

        if σnew < 1e-14
            v .= zero(eltype(v))
            break
        end

        v = x ./ σnew
        i += 1
    end

    norm_refmat = estimate_norm(refmat; tol=tol)
    return sqrt(σnew) / norm_refmat
end

function run_benchmarks(
    mesh_files::Vector{String};
    separation_distance::Float64=150.0,
    treekind::Symbol=:KMeansTree,
    ie_type::Symbol=:EFIE,
    bf_tol::Float64=1e-3,
    minbflvl::Int=3,
    maxpointsbisection=80,
    adaptive=false,
    admissibility_spec::Symbol=:isFarFunctor,
    checkaccuracy::Bool=true,
    acacomparison::Bool=false,
    disjointgeom::Bool=true,
    unbalancedints::Bool=false,
    highfscaling::Bool=true,
    leafcompression::Bool=false,
    rankestimator_type::Symbol=:Geometric,
    csv_file::String="benchmark_results.csv",
    log_file_path::String="benchmark_log.txt",
    scheduler=OhMyThreads.DynamicScheduler(),
    acamaxrank::Int=60,
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
    err_aca_vals = Float64[]
    t_mv_aca_vals = Float64[]
    t_mv_bf_vals = Float64[]
    k_vals = Float64[]
    max_R_ranks = Int[]

    mem_by_depth_history = Dict{Int,Vector{Float64}}()

    # Plot histories to prevent memory garbage collection on crash
    p_time_history = PlotlyJS.SyncPlot[]
    p_mem_history = PlotlyJS.SyncPlot[]
    p_mv_history = PlotlyJS.SyncPlot[]
    p_err_history = PlotlyJS.SyncPlot[]
    p_rank_history = PlotlyJS.SyncPlot[]
    p_level_ranks_all = PlotlyJS.SyncPlot[]

    println("==========================================================")
    println(" Starting Benchmarks | IE Type: $ie_type")
    println(" ACA Comparison      : $acacomparison")
    println(" Accuracy Check      : $checkaccuracy")
    println(" Wavenumber Scaling  : $(highfscaling ? "Dynamic" : "Fixed (High-F)")")
    println(" Rank Estimator      : $rankestimator_type")
    println("==========================================================\n")

    # Pre-parse meshes and compute global minimum h if highfscaling is false
    loaded_meshes = []
    h_max_values = Float64[]
    mesh_names = String[]
    for fn in mesh_files
        println("Loading mesh: $(basename(fn))")
        m = CompScienceMeshes.read_gmsh_mesh(fn)
        h = compute_hmax(m)
        push!(loaded_meshes, m)
        push!(h_max_values, h)
        push!(mesh_names, basename(fn))
        @printf("  -> Discretization max step (h_max): %.4f\n", h)
    end
    min_h_global = minimum(h_max_values)

    csv_stream = open(csv_file, "w")
    write(
        csv_stream,
        "mesh_name,h,N,ie_type,time_aca_s,time_bf_s,ref_bf_time_nlogn,ref_bf_time_nlog2n,mem_aca_mb,mem_bf_total_mb,mem_bf_entries_mb,ref_bf_mem_nlogn,ref_bf_mem_nlog2n,rel_err_bf,rel_err_aca,mv_aca_s,mv_bf_s,ref_bf_mv_nlogn\n",
    )
    flush(csv_stream)

    log_stream = open(log_file_path, "w")
    write(log_stream, "Starting real geometry benchmarks for IE: $ie_type\n")
    flush(log_stream)

    try
        for i in 1:length(loaded_meshes)
            m = loaded_meshes[i]
            h = h_max_values[i]
            m_name = mesh_names[i]

            round_str = @sprintf(
                "==================================================== Round %-1d (%s, h = %.3f) ==========================================================",
                i,
                m_name,
                h
            )
            println(round_str)
            write(log_stream, round_str * "\n")
            flush(log_stream)

            if highfscaling
                lambda = 10 * h
            else
                lambda = 10 * min_h_global
            end
            k = 2 * pi / lambda

            op = build_beast_operator(ie_type, k)
            X = raviartthomas(m)
            N = length(X)

            if disjointgeom
                y = translate(m, SVector(separation_distance, 0.0, 0.0))
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
                    ButterflyFactorizations.tree_parameters(blktree).β,
                )
            elseif admissibility_spec == :isFarFunctor
                admissibility = ButterflyFactorizations.isFarFunctor(
                    ButterflyFactorizations.tree_parameters(blktree).α,
                )
            else
                error("Unsupported admissibility: $admissibility_spec")
            end

            local_isnear(ta, tb, na, nb) = !admissibility(ta, tb, na, nb)

            tree_params = ButterflyFactorizations.tree_parameters(blktree, admissibility)
            active_estimator = if rankestimator_type == :Butterfly
                ButterflyFactorizations.ButterflyRankEstimator(tree_params.Cε)
            else
                ButterflyFactorizations.GeometricRankEstimator(tree_params.C, tree_params.Cε)
            end

            is_adaptive = (rankestimator_type == :Butterfly) || adaptive

            xtest = randn(ComplexF64, length(X))
            refmat = nothing

            if checkaccuracy
                println("\nComputing highly accurate reference Butterfly matrix (tol = $(bf_tol * 1e-2))...")
                refmat = ButterflyFactorizations.PetrovGalerkinBF(
                    op,
                    disjointgeom ? Y : X,
                    X,
                    blktree,
                    k;
                    compressor=ButterflyFactorizations.PartialQR(),
                    scheduler=scheduler,
                    tol=bf_tol * 1e-2,
                    unbalancedints=unbalancedints,
                    leafimbalance=true,
                    leafcomp=leafcompression,
                    admissibility=admissibility,
                    rankestimator=active_estimator,
                    minbflvl=minbflvl,
                    adaptive=true, # Always true for reference accuracy
                )
                println("Reference matrix computed. (Kept in memory for rigorous error estimation)")
            end

            println("\nStarting ButterflyFactorization ($ie_type)...")
            farints, nearints = ButterflyFactorizations.nearandfar(
                blktree,
                admissibility;
                unbalancedints=unbalancedints,
                leafcomp=leafcompression,
                leafimbalance=true,
                minbflvl=minbflvl,
            )
            ButterflyFactorizations.compute_interaction_percentages(
                nearints,
                farints,
                ButterflyFactorizations.cluster_testtree(blktree),
                ButterflyFactorizations.cluster_trialtree(blktree),
            )

            t_bf = @elapsed begin
                Bfmat = ButterflyFactorizations.PetrovGalerkinBF(
                    op,
                    disjointgeom ? Y : X,
                    X,
                    blktree,
                    k;
                    compressor=ButterflyFactorizations.PartialQR(),
                    scheduler=scheduler,
                    tol=bf_tol,
                    unbalancedints=unbalancedints,
                    leafimbalance=true,
                    leafcomp=leafcompression,
                    admissibility=admissibility,
                    rankestimator=active_estimator,
                    minbflvl=minbflvl,
                    adaptive=is_adaptive,
                )
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
                    title="BF Rank vs BF Level ($m_name, k = $(round(k, digits=2)))<br><sup>Total Tree Height = $tree_height</sup>",
                    xaxis_title="Butterfly Factorization Level (1 = Leaves)",
                    yaxis_title="Maximum Rank",
                    template="plotly_white",
                    yaxis=attr(; rangemode="tozero"),
                ),
            )

            savefig(p_level_ranks, "plot_ranks_vs_level_mesh_$(i).html")
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

            println("Benchmarking MV product for ButterflyFactorization...")
            t_mv_bf = @belapsed $Bfmat * $xtest
            mv_bf_str = @sprintf("Time for Butterfly mat-vec: %.5f seconds", t_mv_bf)
            println(mv_bf_str)
            write(log_stream, mv_bf_str * "\n")

            err_bf = NaN
            if checkaccuracy && refmat !== nothing
                println("Estimating rigorous relative error for Butterfly matrix...")
                err_bf = estimate_reldifference(Bfmat, refmat; tol=1e-4, itmax=150)
                err_str = @sprintf("Rigorous relative error of BF (Tol %g): %.2e", bf_tol, err_bf)
                println(err_str)
                write(log_stream, err_str * "\n")
            end

            t_aca = NaN
            mem_aca = NaN
            t_mv_aca = NaN
            err_aca = NaN
            aca_warnings = 0

            if acacomparison
                println("\nStarting Standard ACA ($ie_type)...")
                aca_logger = WarningCounterLogger(current_logger(), Ref(0))
                t_aca = @elapsed begin
                    hmat = with_logger(aca_logger) do
                        HMatrix(
                            op, disjointgeom ? Y : X, X, ACAblktree;
                            tol=bf_tol,
                            spaceordering=AdaptiveCrossApproximation.PreserveSpaceOrder(),
                            scheduler=scheduler,
                            maxrank=acamaxrank,
                        )
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

                if checkaccuracy && refmat !== nothing
                    println("Estimating rigorous relative error for ACA matrix...")
                    err_aca = estimate_reldifference(hmat, refmat; tol=1e-4, itmax=150)
                    err_str = @sprintf("Rigorous relative error of ACA (Tol %g): %.2e", bf_tol, err_aca)
                    println(err_str)
                    write(log_stream, err_str * "\n")
                end

                println(mv_aca_str)
                write(log_stream, mv_aca_str * "\n")
            end

            # Explicitly free reference matrix memory
            refmat = nothing
            GC.gc()

            push!(N_vals, N)
            push!(t_aca_vals, t_aca)
            push!(t_bf_vals, t_bf)
            push!(mem_aca_vals, mem_aca)
            push!(mem_bf_total_vals, mem_bf_total)
            push!(mem_ButterflyFactorization_Mat_vals, mem_ButterflyFactorization_Mat)
            push!(err_bf_vals, err_bf)
            push!(err_aca_vals, err_aca)
            push!(t_mv_aca_vals, t_mv_aca)
            push!(t_mv_bf_vals, t_mv_bf)

            c_time_nlogn = fit_scaling_factor(N_vals, t_bf_vals, f_nlogn)
            c_time_nlog2n = fit_scaling_factor(N_vals, t_bf_vals, f_nlog2n)
            c_mem_nlogn = fit_scaling_factor(N_vals, mem_bf_total_vals, f_nlogn)
            c_mem_nlog2n = fit_scaling_factor(N_vals, mem_bf_total_vals, f_nlog2n)
            c_mv_nlogn = fit_scaling_factor(N_vals, t_mv_bf_vals, f_nlogn)
            c_mv_nlog2n = fit_scaling_factor(N_vals, t_mv_bf_vals, f_nlog2n)

            c_time_aca = fit_scaling_factor(N_vals, t_aca_vals, f_n43logn)
            c_mem_aca = fit_scaling_factor(N_vals, mem_aca_vals, f_n43logn)
            c_mv_aca = fit_scaling_factor(N_vals, t_mv_aca_vals, f_n43logn)

            p_time_emp, coef_time_emp = estimate_empirical_power(N_vals, t_bf_vals)
            p_mem_emp, coef_mem_emp = estimate_empirical_power(N_vals, mem_bf_total_vals)
            p_mv_emp, coef_mv_emp = estimate_empirical_power(N_vals, t_mv_bf_vals)

            csv_row = @sprintf(
                "%s,%.4f,%d,%s,%.5f,%.5f,%.5f,%.5f,%.2f,%.2f,%.2f,%.2f,%.2f,%.3e,%.3e,%.5f,%.5f,%.5f\n",
                m_name,
                h,
                N,
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
                err_aca,
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

            err_traces = GenericTrace[]
            if checkaccuracy
                push!(
                    err_traces,
                    scatter(;
                        x=N_vals,
                        y=err_bf_vals,
                        name="BF Rel Error (Tol: $bf_tol)",
                        mode="lines+markers",
                        line=attr(; color="seagreen", width=3),
                    )
                )
                if acacomparison
                    push!(
                        err_traces,
                        scatter(;
                            x=N_vals,
                            y=err_aca_vals,
                            name="ACA Rel Error (Tol: $bf_tol)",
                            mode="lines+markers",
                            line=attr(; color="firebrick", dash="dot", width=3),
                        )
                    )
                end
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
                    title="Build Time vs N (Real Shuttle, $ie_type)",
                    xaxis_title="N (DOFs)",
                    yaxis_title="Time (s)",
                    yaxis_type="log",
                    xaxis_type="log",
                    template="plotly_white",
                ),
            )
            savefig(p_time, "plot_build_time.html")
            push!(p_time_history, p_time)

            p_mem = plot(
                mem_traces,
                Layout(;
                    title="Memory Usage vs N Grouped by Depth (Real Shuttle, $ie_type)",
                    xaxis_title="N (DOFs)",
                    yaxis_title="Memory (MB)",
                    yaxis_type="log",
                    xaxis_type="log",
                    template="plotly_white",
                ),
            )
            savefig(p_mem, "plot_memory_usage.html")
            push!(p_mem_history, p_mem)

            p_mv = plot(
                mv_traces,
                Layout(;
                    title="Mat-Vec Time vs N (Real Shuttle, $ie_type)",
                    xaxis_title="N (DOFs)",
                    yaxis_title="Time (s)",
                    yaxis_type="log",
                    xaxis_type="log",
                    template="plotly_white",
                ),
            )
            savefig(p_mv, "plot_mat_vec_time.html")
            push!(p_mv_history, p_mv)

            p_err = plot(
                err_traces,
                Layout(;
                    title="Accuracy vs N (Real Shuttle, $ie_type)",
                    xaxis_title="N (DOFs)",
                    yaxis_title="Relative Error",
                    yaxis_type="log",
                    xaxis_type="log",
                    template="plotly_white",
                ),
            )
            savefig(p_err, "plot_accuracy.html")
            push!(p_err_history, p_err)

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
                    title="Max R-Block Rank vs. Wavenumber (k) | Real Shuttle, $ie_type",
                    xaxis_title="Wavenumber (k)",
                    yaxis_title="Maximum Rank in R Factors",
                    xaxis_type="log",
                    yaxis_type="linear",
                    template="plotly_white",
                    yaxis=attr(; rangemode="tozero"),
                ),
            )
            savefig(p_rank_vs_k, "plot_max_R_rank_vs_k.html")
            push!(p_rank_history, p_rank_vs_k)

            end_str = @sprintf(
                "================================================ End of Round %-1d =======================================================\n",
                i
            )
            println(end_str)
            write(log_stream, end_str * "\n")
            flush(log_stream)
        end

    catch e
        if e isa InterruptException
            println("\n\n⚠️  Benchmark manually aborted (Ctrl+C)! Salvaging data and plots up to step $(length(N_vals))...")
        else
            println("\n\n❌  An unexpected error occurred! Salvaging data and plots up to step $(length(N_vals))...")
            Base.showerror(stdout, e)
            println()
        end
    finally
        if length(N_vals) > 0
            header1 = "===================================================================================================================================="
            header2 = @sprintf(
                "%-6s | %-8s | %-12s | %-12s | %-12s | %-12s | %-12s | %-12s | %-10s | %-10s",
                "h", "N", "Time ACA (s)", "Time BF (s)", "Mem ACA (MB)", "Mem BF (MB)", "Err BF", "Err ACA", "MV ACA (s)", "MV BF (s)"
            )

            println(header1)
            println(header2)
            println(header1)

            if isopen(log_stream)
                write(log_stream, "\n\nFINAL SUMMARY TABLE:\n" * header1 * "\n" * header2 * "\n" * header1 * "\n")
            end

            for j in 1:length(N_vals)
                row_str = @sprintf(
                    "%-6.2f | %-8d | %-12.3f | %-12.3f | %-12.2f | %-12.2f | %-12.2e | %-12.2e | %-10.5f | %-10.5f",
                    h_max_values[j], N_vals[j], t_aca_vals[j], t_bf_vals[j], mem_aca_vals[j],
                    mem_bf_total_vals[j], err_bf_vals[j], err_aca_vals[j], t_mv_aca_vals[j], t_mv_bf_vals[j]
                )
                println(row_str)
                if isopen(log_stream)
                    write(log_stream, row_str * "\n")
                end
            end

            println(header1)
            if isopen(log_stream)
                write(log_stream, header1 * "\n")
            end
        end

        if isopen(csv_stream)
            close(csv_stream)
            println("\n✅ Data safely flushed to '$csv_file'.")
        end
        if isopen(log_stream)
            close(log_stream)
        end
    end

    final_p_time = isempty(p_time_history) ? plot() : p_time_history[end]
    final_p_mem = isempty(p_mem_history) ? plot() : p_mem_history[end]
    final_p_mv = isempty(p_mv_history) ? plot() : p_mv_history[end]
    final_p_err = isempty(p_err_history) ? plot() : p_err_history[end]
    final_p_rank = isempty(p_rank_history) ? plot() : p_rank_history[end]

    return final_p_time, final_p_mem, final_p_mv, final_p_err, final_p_rank, p_level_ranks_all
end

# --- Execution ---
mesh_files = [
    joinpath(dirname(@__FILE__), "shuttle_gmsh.msh"),
    joinpath(dirname(@__FILE__), "shuttle_gmsh_refined.msh"),
    #joinpath(dirname(@__FILE__), "shuttle_gmsh_refinedx2.msh"),
    #joinpath(dirname(@__FILE__), "shuttle_gmsh_refinedx3.msh")
]

p_time, p_mem, p_mv, p_err, p_rank_vs_k, p_level_ranks_all = run_benchmarks(
    mesh_files;

    # -------------------------------------------------------------------------
    # Physical Setup & Geometry
    # -------------------------------------------------------------------------
    ie_type=:EFIE,              # Integral equation formulation (Options: :EFIE, :MFIE)
    highfscaling=true,         # false: Locks wavenumber k to the finest mesh. true: Scales k with h.
    disjointgeom=true,          # Translates target mesh away to evaluate pure far-field transmission.
    separation_distance=150.0,  # Ensure complete far-field separation based on Shuttle bounds.

    # -------------------------------------------------------------------------
    # Butterfly Tree & Admissibility Configuration
    # -------------------------------------------------------------------------
    treekind=:BisectionTree,                   # Clustering strategy (Options: :KMeansTree, :BisectionTree, :TwoNTree)
    admissibility_spec=:CenterDistanceAdmissibility,       # Near/Far separation criteria (Options: :CenterDistanceAdmissibility, :isFarFunctor)
    maxpointsbisection=100,                 # Maximum allowed degrees of freedom in a leaf node (specifically for BisectionTrees)
    leafcompression=true,       # true: Compresses interactions all the way down to leaf nodes.
    minbflvl=3,                             # Tree depth where compression begins (ignored if leafcompression is true)
    unbalancedints=false,                   # Allows butterfly interactions between source and observer clusters situated at different tree depths

    # -------------------------------------------------------------------------
    # Compression & Accuracy Targets
    # -------------------------------------------------------------------------
    bf_tol=1e-3,                # Target relative tolerance for the PartialQR compressor
    adaptive=true,             # true: Dynamically estimates rank bounds during compression.

    # -------------------------------------------------------------------------
    # Benchmarking Flags (ACA vs. Butterfly)
    # -------------------------------------------------------------------------
    checkaccuracy=true,         # Explicitly compute standard Butterfly and standard ACA against highly accurate Butterfly matrix.
    acacomparison=true,         # Builds a standard ACA HMatrix alongside the Butterfly matrix.
    acamaxrank=100,             # Hard limit for the maximum rank allowed in the standard ACA comparison matrix

    # -------------------------------------------------------------------------
    # Logging & Schedulers
    # -------------------------------------------------------------------------
    rankestimator_type=:Butterfly,            # Choose :Butterfly or :Geometric
    scheduler=OhMyThreads.DynamicScheduler(), # Threading strategy for matrix assembly and compression
    csv_file="benchmark_results_shuttle.csv",
    log_file_path="benchmark_log_shuttle.txt",
);

display(p_time)
display(p_mem)
display(p_mv)
display(p_err)
display(p_rank_vs_k)

for p in p_level_ranks_all
    display(p)
end
