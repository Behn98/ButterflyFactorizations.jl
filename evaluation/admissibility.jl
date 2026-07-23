using BenchmarkTools
using Printf
using PlotlyJS

function benchmark_alpha_schedule(
    operator,
    testspace,
    trialspace,
    tree::BlockTree,
    k::Float64;
    compressor=ButterflyFactorizations.PartialQR(),
    tol=1e-3,
    alpha_range=1.0:0.5:3.0,
    scheduler=OhMyThreads.DynamicScheduler(),
)
    # Store benchmark results for thesis analysis
    benchmark_results = NamedTuple{
        (:alpha, :time_near, :time_far, :time_total, :num_near, :num_far),
        Tuple{Float64,Float64,Float64,Float64,Int,Int},
    }[]

    println("==================================================")
    println(" Starting Alpha Runtime & Memory Benchmark Sweep")
    println("==================================================")

    for α in alpha_range
        @printf("Testing α = %.2f ... \n", α)

        # 1. Benchmark Tree Traversal & Near-Field Assembly
        # We wrap this to isolate the near-field phase time
        time_near = @elapsed begin
            nearmatrix_near = AbstractKernelMatrix(
                operator, testspace, trialspace; type=:near
            )
            farints, nearints = nearandfar(tree, α; unbalancedints=true, leafcom=true)

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

            nears = if !isempty(nearints)
                BlockSparseMatrix(
                    blocks,
                    test_indices,
                    trial_indices,
                    size(nearmatrix_near);
                    scheduler=scheduler,
                )
            else
                sparse(zeros(ComplexF64, size(nearmatrix_near)...))
            end
        end

        # 2. Benchmark Far-Field Butterfly Assembly Phase
        time_far = @elapsed begin
            nearmatrix_far = AbstractKernelMatrix(
                operator, testspace, trialspace; type=:far
            )
            fly = Vector{ButterflyFactorization{ComplexF64,typeof(tree)}}(
                undef, length(farints)
            )

            let nearmatrix_far = nearmatrix_far
                @tasks for i in eachindex(farints)
                    @set scheduler = scheduler
                    (NO, NS) = farints[i]
                    fly[i] = assemble_BF(
                        nearmatrix_far,
                        tree,
                        NO,
                        NS,
                        k,
                        tol;
                        compressor=compressor,
                        scheduler=OhMyThreads.SerialScheduler(), # Inner serial loop for thread safety per block
                    )
                end
            end
        end

        time_total = time_near + time_far

        @printf(
            " -> Near: %.3fs | Far: %.3fs | Total: %.3fs (Near blocks: %d, Far blocks: %d)\n",
            time_near,
            time_far,
            time_total,
            length(nearints),
            length(farints)
        )

        push!(
            benchmark_results,
            (
                alpha=α,
                time_near=time_near,
                time_far=time_far,
                time_total=time_total,
                num_near=length(nearints),
                num_far=length(farints),
            ),
        )
    end

    return benchmark_results
end

function plot_alpha_runtime_benchmark(results)
    alphas = [r.alpha for r in results]
    t_near = [r.time_near for r in results]
    t_far = [r.time_far for r in results]
    t_total = [r.time_total for r in results]

    # Find optimal execution speed point
    min_idx = argmin(t_total)
    opt_alpha = alphas[min_idx]
    opt_time = t_total[min_idx]

    fig = make_subplots(;
        rows=1,
        cols=1,
        subplot_titles=["Assembly Wall-Clock Time vs. Admissibility Parameter (α)"],
    )

    trace_near = scatter(;
        x=alphas,
        y=t_near,
        mode="lines+markers",
        name="Near-Field Phase (Dense)",
        line=attr(; color="firebrick", width=2),
    )
    trace_far = scatter(;
        x=alphas,
        y=t_far,
        mode="lines+markers",
        name="Far-Field Phase (Butterfly)",
        line=attr(; color="royalblue", width=2),
    )
    trace_total = scatter(;
        x=alphas,
        y=t_total,
        mode="lines+markers",
        name="Total Assembly Time",
        line=attr(; color="black", width=4, dash="solid"),
    )
    trace_opt = scatter(;
        x=[opt_alpha],
        y=[opt_time],
        mode="markers",
        marker=attr(;
            symbol="star", size=16, color="gold", line=attr(; color="black", width=1)
        ),
        name="Optimal α = $opt_alpha ($(round(opt_time, digits=2))s)",
    )

    add_trace!(fig, trace_near)
    add_trace!(fig, trace_far)
    add_trace!(fig, trace_total)
    add_trace!(fig, trace_opt)

    relayout!(
        fig;
        title_text="Butterfly Factorization: Runtime Performance Profiling",
        height=550,
        width=800,
        template="plotly_white",
        xaxis_title="α (Separation Parameter)",
        yaxis_title="Execution Time (Seconds)",
        hovermode="x unified",
    )

    return fig
end
