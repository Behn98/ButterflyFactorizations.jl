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
    criterion::Symbol=:isFarFunctor,
    rankestimator_type::Symbol=:Geometric, # 🚀 NEW: Estimator toggle
    farfield_only::Bool=false,
    scheduler=OhMyThreads.DynamicScheduler(),
    acctype=ComplexF64,
)
    benchmark_results = NamedTuple{
        (
            :alpha,
            :time_near,
            :time_far,
            :time_total,
            :num_near,
            :num_far,
            :farfielderr,
            :bf_counts,
        ),
        Tuple{Float64,Float64,Float64,Float64,Int,Int,Float64,Dict{Int,Int}},
    }[]

    println("==================================================")
    println(" Starting Alpha Benchmark & Accuracy Validation")
    println(" Target Tolerance (τ): $tol")
    println(" Mode: $(farfield_only ? "Far-Field ONLY" : "Full Assembly")")
    println(" Admissibility Criterion: $criterion")
    println(" Rank Estimator: $rankestimator_type")
    println("==================================================")

    for α in alpha_range
        @printf("\nTesting α = %.2f ... \n", α)
        if criterion == :isFarFunctor
            admissible = ButterflyFactorizations.isFarFunctor(α)
        elseif criterion == :CenterDistanceAdmissibility
            admissible = ButterflyFactorizations.CenterDistanceAdmissibility(α)
        end
        local_isnear(ta, tb, na, nb) = !admissible(ta, tb, na, nb)

        farints, nearints = ButterflyFactorizations.nearandfar(
            tree,
            admissible;
            unbalancedints=false,
            leafcomp=true,
            minbflvl=0,
            leafimbalance=false,
        )

        # 🚀 NEW: Dynamically fetch calibrated constants and instantiate the requested estimator
        tree_params = ButterflyFactorizations.tree_parameters(tree, admissible)

        active_estimator = if rankestimator_type == :Butterfly
            ButterflyFactorizations.ButterflyRankEstimator(tree_params.Cτ)
        else
            ButterflyFactorizations.GeometricRankEstimator(tree_params.C, tree_params.Cε)
        end

        # 🚀 NEW: The Butterfly estimator REQUIRES adaptivity to guarantee precision
        # because its initial sample size guess is very small (O(1)).
        is_adaptive = (rankestimator_type == :Butterfly)

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
            end
            if length(nearints) > 0
                nears = BlockSparseMatrix(
                    blocks,
                    test_indices,
                    trial_indices,
                    size(nearmatrix_near);
                    scheduler=scheduler,
                )
            else
                nears = BlockSparseMatrix(
                    Matrix{acctype}[],
                    Int[],
                    Int[],
                    size(nearmatrix_near);
                    scheduler=SerialScheduler(),
                )
            end
        else
            nears = BlockSparseMatrix(
                Matrix{acctype}[],
                Int[],
                Int[],
                size(nearmatrix_near);
                scheduler=SerialScheduler(),
            )
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
                        rankestimator=active_estimator, # 🚀 Injected Estimator
                        adaptive=is_adaptive,           # 🚀 Injected Adaptivity logic
                        scheduler=OhMyThreads.SerialScheduler(),
                    )
                end
            end
        end

        time_total = time_near + time_far

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

        bf_counts = Dict{Int,Int}()
        for bf in fly
            r_levels = length(bf.R) # 0 means just Q and P factors (shallowest BF)
            bf_counts[r_levels] = get(bf_counts, r_levels, 0) + 1
        end

        # --- 3. Validate Accuracy (Not Timed) ---
        Bfmat = PetrovGalerkinBF{ComplexF64}(
            nears,
            tree,
            fly,
            size(nearmatrix_far),
            sparse(Matrix{Int64}(undef, 0, 0)),
            sparse(Matrix{Int64}(undef, 0, 0)),
        )
        err_bf = 0.0
        if !isempty(farints)
            println(
                "\nComputing reference ACA Far-Matrix for accuracy check (tol = $(tol * 1e-2))...",
            )

            refmat = HMatrix(
                operator,
                testspace,
                trialspace,
                tree;
                tol=tol * 1e-2,
                spaceordering=AdaptiveCrossApproximation.PreserveSpaceOrder(),
                scheduler=scheduler,
                maxrank=100,
                isnear=local_isnear,
            )

            ref_far = AdaptiveCrossApproximation.farmatrix(refmat)
            bf_far = ButterflyFactorizations.farmatrix(Bfmat)
            xtest = randn(ComplexF64, size(ref_far, 2))
            y_exact_far = ref_far * xtest
            y_bf_far = bf_far * xtest

            err_bf = norm(y_exact_far - y_bf_far) / norm(y_exact_far)
            err_str = @sprintf("Relative error of Far-field mat-vec: %.2e", err_bf)
            println(err_str)
        end

        push!(
            benchmark_results,
            (
                alpha=α,
                time_near=time_near,
                time_far=time_far,
                time_total=time_total,
                num_near=length(nearints),
                num_far=length(farints),
                farfielderr=err_bf,
                bf_counts=bf_counts,
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

    farfielderror = [max(r.farfielderr, 1e-16) for r in results]

    min_idx = argmin(t_total)
    opt_alpha = alphas[min_idx]
    opt_time = t_total[min_idx]

    fig = make_subplots(;
        rows=3,
        cols=1,
        subplot_titles=reshape(
            [
                "1. Assembly Wall-Clock Time vs. Admissibility (α)",
                "2. Block-Level Relative Error vs. Admissibility (α)",
                "3. Number of BFs per Level (Intermediate R-Factors)",
            ],
            3,
            1,
        ),
        vertical_spacing=0.1,
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

    # --- MIDDLE PANEL: ERROR ---
    trace_farfielderror = scatter(;
        x=alphas,
        y=farfielderror,
        mode="lines+markers",
        name="Average Error",
        line=attr(; color="seagreen", dash="dash"),
    )
    trace_tol = scatter(;
        x=[minimum(alphas), maximum(alphas)],
        y=[target_tol, target_tol],
        mode="lines",
        name="Target Tolerance (τ)",
        line=attr(; color="red", dash="dot"),
    )

    add_trace!(fig, trace_farfielderror; row=2, col=1)
    add_trace!(fig, trace_tol; row=2, col=1)

    # --- BOTTOM PANEL: BF LEVEL COUNTS ---
    all_r_levels = sort(unique(vcat([collect(keys(r.bf_counts)) for r in results]...)))

    for r_lvl in all_r_levels
        counts = [get(r.bf_counts, r_lvl, 0) for r in results]
        trace_cnt = scatter(;
            x=alphas, y=counts, mode="lines+markers", name="Level $r_lvl BFs (R-factors)"
        )
        add_trace!(fig, trace_cnt; row=3, col=1)
    end

    # --- LAYOUT UPDATE ---
    relayout!(
        fig;
        title_text="Butterfly Factorization: Runtime & Accuracy Profiling",
        height=1100,
        width=900,
        template="plotly_white",
        hovermode="x unified",
        xaxis_title="",
        xaxis2_title="",
        xaxis3_title="α (Separation Parameter)",
        yaxis_title="Execution Time (s)",
        yaxis2_title="Relative Error (Log Scale)",
        yaxis2_type="log",
        yaxis2_exponentformat="e",
        yaxis3_title="Count of BFs",
    )

    return fig
end

# ==============================================================================
# --- SETUP & MULTI-CONFIGURATION EXECUTION ---
# ==============================================================================
h = 0.05
lambda = 10 * h
k = 2 * pi / lambda
op = Maxwell3D.singlelayer(; wavenumber=k)
m = meshsphere(1.0, h)
X = raviartthomas(m)
N = length(X)

BLAS.set_num_threads(1)
tol = 1e-3

# 1. Define the configurations to iterate through
tree_types = [:KMeansTree, :BisectionTree, :TwoNTree]
estimators = [:Geometric, :Butterfly] # 🚀 NEW: Iterate over both Estimator Strategies

# Define the criteria and their appropriate α sweep ranges
criteria_configs = [
    (:isFarFunctor, 0.0:0.1:1.5), (:CenterDistanceAdmissibility, 0.8:0.1:2.0)
]

for tree_type in tree_types
    println("\n**************************************************")
    println(" 🌲 Building Tree Architecture: $tree_type")
    println("**************************************************")

    # 2. Instantiate the corresponding tree dynamically
    if tree_type == :KMeansTree
        tree = KMeansTree(X.pos, 2; minvalues=100)
    elseif tree_type == :BisectionTree
        tree = ButterflyFactorizations.build_bisection_tree(X.pos; max_points=100)
    elseif tree_type == :TwoNTree
        tree = H2Trees.TwoNTree(X, h)
    end
    blktree = H2Trees.BlockTree(tree, tree)

    for (criterion, alpha_range) in criteria_configs
        for est_type in estimators
            println("\n--------------------------------------------------")
            println(" 🔬 Testing: $tree_type | $criterion | $est_type Estimator")
            println("--------------------------------------------------")

            # 3. Run the benchmark
            results = benchmark_and_validate_alpha(
                op,
                X,
                X,
                blktree,
                k;
                compressor=ButterflyFactorizations.PartialQR(),
                tol=tol,
                alpha_range=alpha_range,
                criterion=criterion,
                rankestimator_type=est_type, # 🚀 Pass estimator type
                scheduler=OhMyThreads.DynamicScheduler(),
                farfield_only=false,
            )

            # 4. Generate the plot
            fig = plot_alpha_performance_and_accuracy(results, tol)

            # Update the plot title to reflect the full configuration
            relayout!(
                fig;
                title_text="Runtime & Accuracy Profiling ($tree_type | $criterion | $est_type)",
            )

            # 5. Save the plot with a descriptive, dynamic filename
            filename = "alpha_benchmark_$(tree_type)_$(criterion)_$(est_type).html"
            savefig(fig, filename)
            println("\n✅ Saved plot to: $filename")

            # 6. Display in the active REPL / IDE
            display(fig)
        end
    end
end
