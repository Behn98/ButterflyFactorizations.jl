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
using LinearMaps

# --- Fitting Helper Functions ---
f_nlogn(N) = N * log2(N)

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

function extract_ranks_per_level_single(bf)
    level_max_ranks = Dict{Int,Int}()
    q_rank = maximum([size(b.data, 1) for b in bf.Q if b.data isa AbstractMatrix]; init=0)
    level_max_ranks[1] = max(get(level_max_ranks, 1, 0), q_rank)

    for (l, level) in enumerate(bf.R)
        r_rank = maximum(
            [size(b.data, 1) for b in level.blocks if b.data isa AbstractMatrix]; init=0
        )
        level_max_ranks[l + 1] = max(get(level_max_ranks, l+1, 0), r_rank)
    end
    return level_max_ranks
end

# --- Benchmark Execution ---
function run_single_farfield_benchmark(
    h_values;
    highf::Bool=true,
    separation_distance::Float64=3.0,
    tolvalues::Vector{Float64}=[1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10],
    maxpointsbisection::Int=50,
    csv_file::String="single_farfield_results.csv",
    rectangulargeom::Bool=true,
    geomfaceoff::Bool=true,
    treekind::Symbol=:KMeansTree,
    denseassemble::Bool=false,
    scheduler=OhMyThreads.DynamicScheduler(),
)
    BLAS.set_num_threads(1)

    # Data collection arrays
    data_N = Int[]
    data_tol = Float64[]
    data_err = Float64[]
    data_time = Float64[]
    data_mem = Float64[]
    data_mv = Float64[]
    data_max_rank = Int[]
    p_level_ranks_all = PlotlyJS.SyncPlot[]
    dist = 2 * sqrt(2)

    if rectangulargeom
        println("==========================================================")
        println(" Starting Single Far-Field Interaction Benchmark")
        println(" Separation Distance : $dist")
        println(" Output CSV          : $csv_file")
        println("==========================================================\n")
    else
        println("==========================================================")
        println(" Starting Single Far-Field Interaction Benchmark")
        println(" Separation Distance : $separation_distance")
        println(" Output CSV          : $csv_file")
        println("==========================================================\n")
    end

    csv_stream = open(csv_file, "w")
    write(csv_stream, "h,N,target_tol,actual_err,time_s,mem_mb,mv_s,max_R_rank\n")
    flush(csv_stream)

    for (i, h) in enumerate(h_values)
        println("--- Round $i (h = $h) ---")
        lambda = highf ? 10 * h : 1.0
        k = 2 * pi / lambda
        op = Maxwell3D.singlelayer(; wavenumber=k)

        if rectangulargeom
            m_src = meshrectangle(1.0, 1.0, h)
            m_tgt = translate(
                meshrectangle(1.0, 1.0, h),
                geomfaceoff ? SVector(0.0, 0.0, dist) : SVector(dist, 0.0, 0.0),
            )
        else
            m_src = meshsphere(1.0, h)
            m_tgt = translate(meshsphere(1.0, h), SVector(separation_distance, 0.0, 0.0))
        end

        X = raviartthomas(m_src)
        Y = raviartthomas(m_tgt)
        N = length(X)
        if denseassemble
            println("Computing Dense Reference Matrix A for N = $N...")
            A = assemble(op, Y, X)
        end
        nearmatrix_far = ButterflyFactorizations.AbstractKernelMatrix(op, Y, X; type=:far)

        if treekind == :KMeansTree
            Stree = KMeansTree(X.pos, 2; minvalues=100)
            Otree = KMeansTree(Y.pos, 2; minvalues=100)
        elseif treekind == :BisectionTree
            Stree = ButterflyFactorizations.build_bisection_tree(
                X.pos; max_points=maxpointsbisection
            )
            Otree = ButterflyFactorizations.build_bisection_tree(
                Y.pos; max_points=maxpointsbisection
            )
        end
        blktree = H2Trees.BlockTree(Otree, Stree)
        tree_height = length(blktree.testcluster.nodesatlevel)

        traces_level_ranks = GenericTrace[]

        for bf_tol in tolvalues
            @printf("  -> Testing tol = %.1e ... \n", bf_tol)
            @printf(
                "Computing Butterfly Factorization (N = %d, h = %.4f, tol = %.1e)\n",
                N,
                h,
                bf_tol
            )

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

            mem_entries = bf_matrix_memory(Bfmat)

            xtest = randn(ComplexF64, length(X))
            t_mv_bf = @belapsed $Bfmat * $xtest
            err_bf = NaN
            if denseassemble
                # Using a tight tol=1e-5 for the power iteration to ensure the error measurement is accurate
                err_bf = estimate_reldifference(Bfmat, A; tol=1e-5, itmax=150)
            end
            ranks_dict = extract_ranks_per_level_single(Bfmat)
            sorted_levels = sort(collect(keys(ranks_dict)))
            ranks_at_levels = [ranks_dict[l] for l in sorted_levels]

            r_ranks = [ranks_dict[l] for l in sorted_levels if l > 1]
            max_r_rank = isempty(r_ranks) ? 0 : maximum(r_ranks)

            # Store Data
            push!(data_N, N)
            push!(data_tol, bf_tol)
            push!(data_err, err_bf)
            push!(data_time, t_bf)
            push!(data_mem, mem_entries)
            push!(data_mv, t_mv_bf)
            push!(data_max_rank, max_r_rank)

            push!(
                traces_level_ranks,
                scatter(;
                    x=sorted_levels,
                    y=ranks_at_levels,
                    name="Tol = $(bf_tol)",
                    mode="lines+markers",
                ),
            )

            @printf("Err: %.2e | Time: %.3fs | Mem: %.1fMB\n", err_bf, t_bf, mem_entries)

            # Write immediately to CSV
            csv_row = @sprintf(
                "%.4f,%d,%.1e,%.3e,%.5f,%.2f,%.5f,%d\n",
                h,
                N,
                bf_tol,
                err_bf,
                t_bf,
                mem_entries,
                t_mv_bf,
                max_r_rank
            )
            write(csv_stream, csv_row)
            flush(csv_stream)
        end

        # Plot Ranks per level for this specific geometry (N)
        p_lvl = plot(
            traces_level_ranks,
            Layout(;
                title="BF Rank per Level (N = $N, h = $h)",
                xaxis_title="Butterfly Factorization Level (1 = Leaves)",
                yaxis_title="Maximum Rank",
                template="plotly_white",
            ),
        )
        savefig(p_lvl, "plot_single_bf_ranks_N$(N)_h$(h).html")
        push!(p_level_ranks_all, p_lvl)
    end
    close(csv_stream)

    # --- Generate Plots grouped by N (Trade-offs) ---
    unique_Ns = unique(data_N)
    traces_acc = GenericTrace[]
    traces_mem = GenericTrace[]
    traces_rank = GenericTrace[]

    push!(
        traces_acc,
        scatter(;
            x=tolvalues,
            y=tolvalues,
            name="Ideal (Err = Tol)",
            mode="lines",
            line=attr(; color="black", dash="dash"),
        ),
    )

    for n in unique_Ns
        idx = findall(x -> x == n, data_N)
        sort_idx = sortperm(data_tol[idx])
        idx = idx[sort_idx]

        x_tol = data_tol[idx]
        y_err = data_err[idx]
        y_mem_tradeoff = data_mem[idx]
        y_rank = data_max_rank[idx]

        push!(traces_acc, scatter(; x=x_tol, y=y_err, name="N = $n", mode="lines+markers"))
        push!(
            traces_mem,
            scatter(; x=y_err, y=y_mem_tradeoff, name="N = $n", mode="lines+markers"),
        )
        push!(
            traces_rank, scatter(; x=x_tol, y=y_rank, name="N = $n", mode="lines+markers")
        )
    end

    # --- Generate Plots grouped by Tolerance (Scaling over N) ---
    unique_tols = unique(data_tol)
    traces_time_N = GenericTrace[]
    traces_mv_N = GenericTrace[]
    traces_mem_N = GenericTrace[] # 🚀 NEW: Memory vs N

    # We use a color palette sequence so the N log N fit matches the data line color
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    for (idx_tol, tol) in enumerate(unique_tols)
        idx = findall(x -> x == tol, data_tol)
        sort_idx = sortperm(data_N[idx])
        idx = idx[sort_idx]

        x_N = data_N[idx]
        y_time = data_time[idx]
        y_mv = data_mv[idx]
        y_mem = data_mem[idx] # 🚀 NEW

        c = colors[((idx_tol - 1) % length(colors)) + 1]

        # Actual Data
        push!(
            traces_time_N,
            scatter(;
                x=x_N,
                y=y_time,
                name="Tol = $tol",
                mode="lines+markers",
                line=attr(; color=c),
            ),
        )
        push!(
            traces_mv_N,
            scatter(;
                x=x_N, y=y_mv, name="Tol = $tol", mode="lines+markers", line=attr(; color=c)
            ),
        )
        push!(
            traces_mem_N,
            scatter(;
                x=x_N,
                y=y_mem,
                name="Tol = $tol",
                mode="lines+markers",
                line=attr(; color=c),
            ),
        ) # 🚀 NEW

        # N log N Fit (only if we have more than 1 point to fit)
        if length(x_N) > 1
            c_time = fit_scaling_factor(x_N, y_time, f_nlogn)
            c_mv = fit_scaling_factor(x_N, y_mv, f_nlogn)
            c_mem = fit_scaling_factor(x_N, y_mem, f_nlogn) # 🚀 NEW

            push!(
                traces_time_N,
                scatter(;
                    x=x_N,
                    y=c_time .* f_nlogn.(x_N),
                    name="O(N log N)",
                    mode="lines",
                    showlegend=false,
                    line=attr(; color=c, dash="dash", width=1),
                ),
            )
            push!(
                traces_mv_N,
                scatter(;
                    x=x_N,
                    y=c_mv .* f_nlogn.(x_N),
                    name="O(N log N)",
                    mode="lines",
                    showlegend=false,
                    line=attr(; color=c, dash="dash", width=1),
                ),
            )
            push!(
                traces_mem_N,
                scatter(;
                    x=x_N,
                    y=c_mem .* f_nlogn.(x_N),
                    name="O(N log N)",
                    mode="lines",
                    showlegend=false,
                    line=attr(; color=c, dash="dash", width=1),
                ),
            ) # 🚀 NEW
        end
    end

    # Build Layouts
    p_acc = plot(
        traces_acc,
        Layout(;
            title="Measured Error vs. Target Tolerance",
            xaxis_title="Target Tolerance",
            yaxis_title="Actual Relative Error",
            xaxis_type="log",
            yaxis_type="log",
            template="plotly_white",
        ),
    )
    p_mem = plot(
        traces_mem,
        Layout(;
            title="Memory Footprint vs. Measured Error",
            xaxis_title="Actual Relative Error",
            yaxis_title="Memory (MB)",
            xaxis_type="log",
            yaxis_type="log",
            template="plotly_white",
        ),
    )
    p_rank = plot(
        traces_rank,
        Layout(;
            title="Maximum Rank vs. Target Tolerance",
            xaxis_title="Target Tolerance",
            yaxis_title="Max R-Block Rank",
            xaxis_type="log",
            template="plotly_white",
        ),
    )

    p_time_N = plot(
        traces_time_N,
        Layout(;
            title="Assembly Time vs N (Dashed = O(N log N))",
            xaxis_title="N (DOFs)",
            yaxis_title="Assembly Time (s)",
            xaxis_type="log",
            yaxis_type="log",
            template="plotly_white",
        ),
    )
    p_mv_N = plot(
        traces_mv_N,
        Layout(;
            title="Mat-Vec Time vs N (Dashed = O(N log N))",
            xaxis_title="N (DOFs)",
            yaxis_title="Mat-Vec Time (s)",
            xaxis_type="log",
            yaxis_type="log",
            template="plotly_white",
        ),
    )
    p_mem_N = plot(
        traces_mem_N,
        Layout(;
            title="Memory vs N (Dashed = O(N log N))",
            xaxis_title="N (DOFs)",
            yaxis_title="Memory (MB)",
            xaxis_type="log",
            yaxis_type="log",
            template="plotly_white",
        ),
    ) # 🚀 NEW

    # Save to HTML
    savefig(p_acc, "plot_single_bf_accuracy.html")
    savefig(p_mem, "plot_single_bf_memory_tradeoff.html")
    savefig(p_rank, "plot_single_bf_max_rank.html")
    savefig(p_time_N, "plot_single_bf_time_vs_N.html")
    savefig(p_mv_N, "plot_single_bf_mv_vs_N.html")
    savefig(p_mem_N, "plot_single_bf_memory_vs_N.html") # 🚀 NEW

    println("\nBenchmark complete! All data saved to '$csv_file'.")
    return p_acc, p_time_N, p_mem, p_mem_N, p_mv_N, p_rank, p_level_ranks_all
end

# --- Execution ---
h_values = [0.0125, 0.008, 0.006, 0.004, 0.003]
#[0.0125, 0.008, 0.006, 0.004, 0.003] N = [19040, 46625, 83333, 187000, 332001]
#[0.0055] N = [99008]
p_acc, p_time_N, p_mem, p_mem_N, p_mv_N, p_rank, p_level_ranks_all = run_single_farfield_benchmark(
    h_values;
    highf=true,
    separation_distance=4.0,
    maxpointsbisection=100,
    treekind=:BisectionTree,
    rectangulargeom=true,
    geomfaceoff=false,
    denseassemble=false,
    tolvalues=[1e-3], #∈ [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
    scheduler=OhMyThreads.DynamicScheduler(),
    csv_file="single_farfield_results.csv",
);

display(p_acc)
display(p_time_N)
display(p_mem)
display(p_mem_N)
display(p_mv_N)
display(p_rank)

for p in p_level_ranks_all
    display(p)
end
