using BEAST
using CompScienceMeshes
using ParallelKMeans
using H2Trees
using AdaptiveCrossApproximation
using OhMyThreads
using ButterflyFactorizations
using LinearAlgebra
using BenchmarkTools
using Printf
using PlotlyJS
using Random
using SparseArrays
using Statistics
using BlockSparseMatrices

function benchmark_and_validate_alpha(
    operator,
    testspace,
    trialspace,
    tree::H2Trees.BlockTree,
    k::Float64;
    compressor=ButterflyFactorizations.PartialQR(),
    tol=1e-3,
    alpha_range=1.0:0.1:3.0,
    n_error_samples=50,
    scheduler=OhMyThreads.DynamicScheduler(),
    farfield_only::Bool=false,
)
    benchmark_results = NamedTuple{
        (
            :alpha,
            :time_near,
            :time_far,
            :time_total,
            :num_near,
            :num_far,
            :avg_err,
            :max_err,
        ),
        Tuple{Float64,Float64,Float64,Float64,Int,Int,Float64,Float64},
    }[]

    println("==================================================")
    println(" Starting Alpha Benchmark & Accuracy Validation")
    println(" Target Tolerance (τ): $tol")
    println(" Mode: $(farfield_only ? "Far-Field ONLY" : "Full Assembly")")
    println("==================================================")

    for α in alpha_range
        @printf("\nTesting α = %.2f ... \n", α)

        # Always traverse the tree to get interaction lists (this is fast)
        farints, nearints = ButterflyFactorizations.nearandfar(
            tree, α; unbalancedints=true, leafcom=true
        )

        # --- 1. Timing Near-Field Assembly ---
        time_near = 0.0
        if !farfield_only
            time_near = @elapsed begin
                nearmatrix_near = ButterflyFactorizations.AbstractKernelMatrix(
                    operator, testspace, trialspace; type=:near
                )

                blocks = Vector{Matrix{ComplexF64}}(undef, length(nearints))
                test_indices = Vector{Vector{Int64}}(undef, length(nearints))
                trial_indices = Vector{Vector{Int64}}(undef, length(nearints))

                let nearmatrix_near = nearmatrix_near
                    @tasks for i in eachindex(nearints)
                        @set scheduler = scheduler
                        (node_o, node_s) = nearints[i]
                        test_indices[i] = H2Trees.values(tree.testcluster, node_o)
                        trial_indices[i] = H2Trees.values(tree.trialcluster, node_s)

                        blk = zeros(
                            ComplexF64, length(test_indices[i]), length(trial_indices[i])
                        )
                        nearmatrix_near(blk, test_indices[i], trial_indices[i])
                        blocks[i] = blk
                    end
                end
                # Combine into BlockSparseMatrix (omitted variable assignment for brevity)
            end
        end

        # --- 2. Timing Far-Field Butterfly Assembly ---
        nearmatrix_far = ButterflyFactorizations.AbstractKernelMatrix(
            operator, testspace, trialspace; type=:far
        )
        fly = Vector{
            ButterflyFactorizations.ButterflyFactorization{ComplexF64,typeof(tree)}
        }(
            undef, length(farints)
        )

        time_far = @elapsed begin
            let nearmatrix_far = nearmatrix_far
                @tasks for i in eachindex(farints)
                    @set scheduler = scheduler
                    (NO, NS) = farints[i]
                    fly[i] = ButterflyFactorizations.assemble_BF(
                        nearmatrix_far,
                        tree,
                        NO,
                        NS,
                        k,
                        tol;
                        compressor=compressor,
                        scheduler=OhMyThreads.SerialScheduler(),
                    )
                end
            end
        end

        time_total = time_near + time_far

        # --- 3. Validate Accuracy (Not Timed) ---
        max_err = 0.0
        avg_err = 0.0

        if !isempty(farints)
            n_samples = min(n_error_samples, length(farints))
            sample_indices = shuffle(1:length(farints))[1:n_samples]

            for i in sample_indices
                (node_o, node_s) = farints[i]
                test_idx = H2Trees.values(tree.testcluster, node_o)
                trial_idx = H2Trees.values(tree.trialcluster, node_s)

                # Exact BEAST dense block
                Z_exact = zeros(ComplexF64, length(test_idx), length(trial_idx))
                nearmatrix_far(Z_exact, test_idx, trial_idx)

                # Random vector test
                x = randn(ComplexF64, length(trial_idx))
                y_exact = Z_exact * x

                # Zero-padded global vector for multiplication
                x_tmp = zeros(ComplexF64, length(trialspace))
                x_tmp[trial_idx] .= x

                y_bf = zeros(ComplexF64, length(testspace))
                mul!(y_bf, fly[i], x_tmp)

                rel_err = norm(y_exact - y_bf[test_idx]) / norm(y_exact)
                max_err = max(max_err, rel_err)
                avg_err += rel_err
            end
            avg_err /= n_samples
        end

        if farfield_only
            @printf(" -> Time: %.3fs (Far Only)\n", time_far)
        else
            @printf(
                " -> Time: %.3fs (Near) + %.3fs (Far) = %.3fs\n",
                time_near,
                time_far,
                time_total
            )
        end
        @printf(" -> Error: Avg = %.2e | Max = %.2e\n", avg_err, max_err)

        push!(
            benchmark_results,
            (
                alpha=α,
                time_near=time_near,
                time_far=time_far,
                time_total=time_total,
                num_near=length(nearints),
                num_far=length(farints),
                avg_err=avg_err,
                max_err=max_err,
            ),
        )
    end

    return benchmark_results
end

function plot_alpha_performance_and_accuracy(results, target_tol::Float64)
    alphas = [r.alpha for r in results]
    t_near = [r.time_near for r in results]
    t_far = [r.time_far for r in results]
    t_total = [r.time_total for r in results]

    avg_errs = [max(r.avg_err, 1e-16) for r in results] # Prevent log(0)
    max_errs = [max(r.max_err, 1e-16) for r in results]

    # Find optimal time
    min_idx = argmin(t_total)
    opt_alpha = alphas[min_idx]
    opt_time = t_total[min_idx]

    # 🚀 Use reshape to explicitly create a 2x1 Matrix of strings
    fig = make_subplots(;
        rows=2,
        cols=1,
        subplot_titles=reshape(
            [
                "1. Assembly Wall-Clock Time vs. Admissibility (α)",
                "2. Block-Level Relative Error vs. Admissibility (α)",
            ],
            2,
            1,
        ),
        vertical_spacing=0.15,
    )

    # --- TOP PANEL: TIME ---
    trace_near = scatter(;
        x=alphas,
        y=t_near,
        mode="lines+markers",
        name="Near Time",
        line=attr(; color="firebrick"),
    )
    trace_far = scatter(;
        x=alphas,
        y=t_far,
        mode="lines+markers",
        name="Far Time",
        line=attr(; color="royalblue"),
    )
    trace_total = scatter(;
        x=alphas,
        y=t_total,
        mode="lines+markers",
        name="Total Time",
        line=attr(; color="black", width=3),
    )

    trace_opt = scatter(;
        x=[opt_alpha],
        y=[opt_time],
        mode="markers",
        showlegend=false,
        marker=attr(;
            symbol="star", size=14, color="gold", line=attr(; color="black", width=1)
        ),
    )

    add_trace!(fig, trace_near; row=1, col=1)
    add_trace!(fig, trace_far; row=1, col=1)
    add_trace!(fig, trace_total; row=1, col=1)
    add_trace!(fig, trace_opt; row=1, col=1)

    # --- BOTTOM PANEL: ERROR ---
    trace_avg = scatter(;
        x=alphas,
        y=avg_errs,
        mode="lines+markers",
        name="Average Error",
        line=attr(; color="seagreen", dash="dash"),
    )
    trace_max = scatter(;
        x=alphas,
        y=max_errs,
        mode="lines+markers",
        name="Max Error",
        line=attr(; color="darkgreen", width=2),
    )

    # Target Tolerance Line
    trace_tol = scatter(;
        x=[minimum(alphas), maximum(alphas)],
        y=[target_tol, target_tol],
        mode="lines",
        name="Target Tolerance (τ)",
        line=attr(; color="red", dash="dot"),
    )

    add_trace!(fig, trace_avg; row=2, col=1)
    add_trace!(fig, trace_max; row=2, col=1)
    add_trace!(fig, trace_tol; row=2, col=1)

    # --- LAYOUT UPDATE ---
    relayout!(
        fig;
        title_text="Butterfly Factorization: Runtime & Accuracy Profiling",
        height=800,
        width=900,
        template="plotly_white",
        hovermode="x unified",
        xaxis_title="",
        yaxis_title="Execution Time (Seconds)",
        xaxis2_title="α (Separation Parameter)",
        yaxis2_title="Relative Error (Log Scale)",
        yaxis2_type="log", # 🚀 Essential for viewing errors spanning 1e-3 to 1e-12
        yaxis2_exponentformat="e",
    )

    return fig
end

h = 0.1
lambda = 10 * h
k = 2 * pi / lambda
op = Maxwell3D.singlelayer(; wavenumber=k)
m = meshsphere(1.0, h)
X = raviartthomas(m)
N = length(X)

# Bygg träd
#tree = H2Trees.KMeansTree(X.pos, 2; minvalues=100)
#tree = H2Trees.TwoNTree(X, h;)#minvalues=100
tree = H2Trees.BisectionTree(X.pos; max_points=50)
blktree = H2Trees.BlockTree(tree, tree)
#As = assemble(op, X, X)

BLAS.set_num_threads(1) # Avoid nested threading issues
tol = 1e-3
results = benchmark_and_validate_alpha(
    op,
    X,
    X,
    blktree,
    k;
    compressor=ButterflyFactorizations.PartialQR(),
    tol=tol,
    alpha_range=1.0:0.1:3.0,
    n_error_samples=100,
    scheduler=OhMyThreads.DynamicScheduler(),
    farfield_only=false,
)

fig = plot_alpha_performance_and_accuracy(results, tol)
