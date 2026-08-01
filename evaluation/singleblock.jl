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

# --- Fitting Helper Functions ---
f_nlogn(N) = N * log2(N)
f_nlog2n(N) = N * (log2(N))^2

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

function bf_matrix_memory(bf)
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
    mem_bytes = num_elements * sizeof(ComplexF64)
    return mem_bytes / 1024^2
end

# 🚀 NEW: Adjusted to read directly from a single BF block
function extract_ranks_per_level_single(bf)
    level_max_ranks = Dict{Int,Int}()

    # Q blocks (Leaves / BF Level 1)
    q_rank = maximum([size(b.data, 2) for b in bf.Q if b.data isa AbstractMatrix]; init=0)
    level_max_ranks[1] = max(get(level_max_ranks, 1, 0), q_rank)

    # R blocks (Intermediate BF levels)
    for (l, level) in enumerate(bf.R)
        r_rank = maximum(
            [size(b.data, 2) for b in level.blocks if b.data isa AbstractMatrix]; init=0
        )
        level_max_ranks[l + 1] = max(get(level_max_ranks, l+1, 0), r_rank)
    end

    return level_max_ranks
end

function run_single_farfield_benchmark(
    h_values;
    highf::Bool=true,
    separation_distance::Float64=3.0,
    bf_tol::Float64=1e-3,
    maxpointsbisection::Int=50,
    csv_file::String="single_farfield_results.csv",
    scheduler=OhMyThreads.DynamicScheduler(),
)
    BLAS.set_num_threads(1)

    N_vals = Int[]
    t_bf_vals = Float64[]
    mem_bf_entries_vals = Float64[]
    mem_bf_total_vals = Float64[]
    t_mv_bf_vals = Float64[]

    # 🚀 NEW: Arrays for Error and Rank tracking
    err_bf_vals = Float64[]
    k_vals = Float64[]
    max_R_ranks = Int[]
    p_level_ranks_all = PlotlyJS.SyncPlot[]

    println("==========================================================")
    println(" Starting Single Far-Field Interaction Benchmark")
    println(" Separation Distance : $separation_distance")
    println(" Target Tolerance    : $bf_tol")
    println(" Output CSV          : $csv_file")
    println("==========================================================\n")

    csv_stream = open(csv_file, "w")
    write(
        csv_stream,
        "h,N,time_bf_s,ref_time_nlogn,mem_entries_mb,ref_mem_nlogn,mv_bf_s,ref_mv_nlogn,err_bf,max_R_rank\n",
    )
    flush(csv_stream)

    for (i, h) in enumerate(h_values)
        println("--- Round $i (h = $h) ---")
        if highf
            lambda = 10 * h
            k = 2 * pi / lambda
        else
            lambda = 1.0
            k = 2 * pi / lambda
        end

        op = Maxwell3D.singlelayer(; wavenumber=k)

        # 1. Geometry: Target sphere translated far away along X-axis
        m_src = meshsphere(1.0, h)
        m_tgt = translate(meshsphere(1.0, h), SVector(separation_distance, 0.0, 0.0))

        X = raviartthomas(m_src)
        Y = raviartthomas(m_tgt)
        N = length(X)

        nearmatrix_far = ButterflyFactorizations.AbstractKernelMatrix(op, Y, X; type=:far)

        # 2. Build Trees for Butterfly
        Stree = H2Trees.BisectionTree(X.pos; max_points=maxpointsbisection)
        Otree = H2Trees.BisectionTree(Y.pos; max_points=maxpointsbisection)
        blktree = H2Trees.BlockTree(Otree, Stree)

        # 3. Assemble Single Far-Field Butterfly Factorization
        t_bf = @elapsed begin
            Bfmat = ButterflyFactorizations.assemble_BF(
                nearmatrix_far,
                blktree,
                1,
                1,
                k,
                bf_tol;
                compressor=ButterflyFactorizations.PartialQR(),
                scheduler=scheduler,
            )
        end

        mem_total = Base.summarysize(Bfmat) / 1024^2
        mem_entries = bf_matrix_memory(Bfmat)

        # 4. Mat-Vec Benchmark
        xtest = randn(ComplexF64, length(X))
        t_mv_bf = @belapsed $Bfmat * $xtest
        x_mv_bf = Bfmat * xtest

        # 🚀 NEW: Reference ACA Computation for Accuracy Validation
        println("Computing reference ACA Matrix (tol = $(bf_tol * 1e-2))...")
        ACAstree = H2Trees.KMeansTree(X.pos, 2; minvalues=100)
        ACAotree = H2Trees.KMeansTree(Y.pos, 2; minvalues=100)
        ACAblktree = H2Trees.BlockTree(ACAotree, ACAstree)

        # Default isnear is fine, it will safely detect everything as farfield
        hmat = HMatrix(
            op,
            Y,
            X,
            ACAblktree;
            tol=bf_tol * 1e-2,
            spaceordering=AdaptiveCrossApproximation.PreserveSpaceOrder(),
            scheduler=scheduler,
            maxrank=150,
        )

        x_mv_aca = hmat * xtest
        err_bf = norm(x_mv_aca - x_mv_bf) / norm(x_mv_aca)
        err_str = @sprintf("Relative Error vs ACA: %.2e", err_bf)
        println(err_str)

        # 🚀 NEW: Extract Ranks and Plot per Level
        ranks_dict = extract_ranks_per_level_single(Bfmat)
        tree_height = length(blktree.testcluster.nodesatlevel)
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
                title="Single BF Rank vs Level (h = $h, k = $(round(k, digits=2)))<br><sup>Total Tree Height = $tree_height</sup>",
                xaxis_title="Butterfly Factorization Level (1 = Leaves)",
                yaxis_title="Maximum Rank",
                template="plotly_white",
                yaxis=attr(; rangemode="tozero"),
            ),
        )
        savefig(p_level_ranks, "plot_single_ranks_vs_level_h_$(h).html")
        push!(p_level_ranks_all, p_level_ranks)

        r_ranks = [ranks_dict[l] for l in sorted_levels if l > 1]
        max_r_rank = isempty(r_ranks) ? 0 : maximum(r_ranks)

        # Push metrics to arrays
        push!(N_vals, N)
        push!(t_bf_vals, t_bf)
        push!(mem_bf_total_vals, mem_total)
        push!(mem_bf_entries_vals, mem_entries)
        push!(t_mv_bf_vals, t_mv_bf)
        push!(err_bf_vals, err_bf)
        push!(k_vals, k)
        push!(max_R_ranks, max_r_rank)

        num_bfs = 1
        @printf(
            "N: %6d | BF Count: %d | Tree Depth: %d | Build: %.4fs | Mem Entries: %.2fMB | MV: %.5fs\n",
            N,
            num_bfs,
            tree_height,
            t_bf,
            mem_entries,
            t_mv_bf
        )

        # Dynamic Reference Curves Fitting
        c_time_nlogn = fit_scaling_factor(N_vals, t_bf_vals, f_nlogn)
        c_mem_nlogn = fit_scaling_factor(N_vals, mem_bf_entries_vals, f_nlogn)
        c_mv_nlogn = fit_scaling_factor(N_vals, t_mv_bf_vals, f_nlogn)

        csv_row = @sprintf(
            "%.4f,%d,%.5f,%.5f,%.2f,%.2f,%.5f,%.5f,%.3e,%d\n",
            h,
            N,
            t_bf,
            c_time_nlogn * f_nlogn(N),
            mem_entries,
            c_mem_nlogn * f_nlogn(N),
            t_mv_bf,
            c_mv_nlogn * f_nlogn(N),
            err_bf,
            max_r_rank
        )
        write(csv_stream, csv_row)
        flush(csv_stream)
    end

    close(csv_stream)

    # 5. Final Incremental Plot Generation
    c_time_nlogn = fit_scaling_factor(N_vals, t_bf_vals, f_nlogn)
    c_mem_nlogn = fit_scaling_factor(N_vals, mem_bf_entries_vals, f_nlogn)
    c_mv_nlogn = fit_scaling_factor(N_vals, t_mv_bf_vals, f_nlogn)

    ref_time = c_time_nlogn .* f_nlogn.(N_vals)
    ref_mem = c_mem_nlogn .* f_nlogn.(N_vals)
    ref_mv = c_mv_nlogn .* f_nlogn.(N_vals)

    p_time = plot(
        [
            scatter(;
                x=N_vals,
                y=t_bf_vals,
                name="Single BF Build Time",
                mode="lines+markers",
                line=attr(; color="royalblue", width=2),
            ),
            scatter(;
                x=N_vals,
                y=ref_time,
                name="O(N log N) Fit",
                mode="lines",
                line=attr(; color="gray", dash="dash"),
            ),
        ],
        Layout(;
            title="Single Far-Field BF Assembly Time vs N",
            xaxis_title="N (DOFs)",
            yaxis_title="Time (s)",
            yaxis_type="log",
            xaxis_type="log",
            template="plotly_white",
        ),
    )

    p_mem = plot(
        [
            scatter(;
                x=N_vals,
                y=mem_bf_entries_vals,
                name="Single BF Matrix Entries",
                mode="lines+markers",
                line=attr(; color="royalblue", width=2),
            ),
            scatter(;
                x=N_vals,
                y=ref_mem,
                name="O(N log N) Fit",
                mode="lines",
                line=attr(; color="gray", dash="dash"),
            ),
        ],
        Layout(;
            title="Single Far-Field BF Memory Footprint vs N",
            xaxis_title="N (DOFs)",
            yaxis_title="Memory (MB)",
            yaxis_type="log",
            xaxis_type="log",
            template="plotly_white",
        ),
    )

    p_mv = plot(
        [
            scatter(;
                x=N_vals,
                y=t_mv_bf_vals,
                name="Single BF Mat-Vec Time",
                mode="lines+markers",
                line=attr(; color="royalblue", width=2),
            ),
            scatter(;
                x=N_vals,
                y=ref_mv,
                name="O(N log N) Fit",
                mode="lines",
                line=attr(; color="gray", dash="dash"),
            ),
        ],
        Layout(;
            title="Single Far-Field BF Mat-Vec Time vs N",
            xaxis_title="N (DOFs)",
            yaxis_title="Time (s)",
            yaxis_type="log",
            xaxis_type="log",
            template="plotly_white",
        ),
    )

    # 🚀 NEW: Accuracy and Rank plots
    p_err = plot(
        [
            scatter(;
                x=N_vals,
                y=err_bf_vals,
                name="Single BF Relative Error",
                mode="lines+markers",
                line=attr(; color="seagreen", width=2),
            ),
        ],
        Layout(;
            title="Single BF Accuracy vs N",
            xaxis_title="N (DOFs)",
            yaxis_title="Relative Error",
            yaxis_type="log",
            xaxis_type="log",
            template="plotly_white",
        ),
    )

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
            title="Single BF Max R-Block Rank vs. Wavenumber (k)",
            xaxis_title="Wavenumber (k)",
            yaxis_title="Maximum Rank in R Factors",
            xaxis_type="log",
            yaxis_type="linear",
            template="plotly_white",
            yaxis=attr(; rangemode="tozero"),
        ),
    )

    savefig(p_time, "plot_single_bf_build_time.html")
    savefig(p_mem, "plot_single_bf_memory.html")
    savefig(p_mv, "plot_single_bf_matvec.html")
    savefig(p_err, "plot_single_bf_accuracy.html")
    savefig(p_rank_vs_k, "plot_single_bf_max_R_rank_vs_k.html")

    println("\nSingle far-field benchmark complete! Data saved to '$csv_file'.")
    return p_time, p_mem, p_mv, p_err, p_rank_vs_k, p_level_ranks_all
end

# --- Execution ---
h_values = [0.1, 0.05, 0.025, 0.0125, 0.00625]

p_time, p_mem, p_mv, p_err, p_rank_vs_k, p_level_ranks_all = run_single_farfield_benchmark(
    h_values;
    highf=true,
    separation_distance=4.0,
    bf_tol=1e-3,
    maxpointsbisection=50,
    csv_file="single_farfield_results.csv",
);

display(p_time)
display(p_mem)
display(p_mv)
display(p_err)
display(p_rank_vs_k)

for p in p_level_ranks_all
    display(p)
end
