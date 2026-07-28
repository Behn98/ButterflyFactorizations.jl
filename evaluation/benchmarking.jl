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

# --- Fitting Helper Functions ---
f_nlogn(N) = N * log2(N)
f_nlog2n(N) = N * (log2(N))^2

function isnearwrap(treea, treeb, nodea, nodeb; α=1.5)
    return (!(ButterflyFactorizations.isFarFunctor(α)(treea, treeb, nodea, nodeb)))
end

"""
    fit_scaling_factor(N_vec, Y_vec, scaling_func)

Computes the least-squares optimal constant `c` for Y ≈ c * scaling_func(N).
"""
function fit_scaling_factor(
    N_vec::Vector{Int}, Y_vec::Vector{Float64}, scaling_func::Function
)
    f_vals = scaling_func.(N_vec)
    return sum(Y_vec .* f_vals) / sum(f_vals .^ 2)
end

# Helper function to compute memory footprint of Butterfly matrix entries
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

function validate_farfield_accuracy(
    operator, testspace, trialspace, BF::PetrovGalerkinBF; n_samples=20
)
    # We only need the far-field kernel evaluator
    kernelmatrix = ButterflyFactorizations.AbstractKernelMatrix(
        operator, testspace, trialspace; type=:far
    )

    # Pick a random subset of blocks to test so we don't wait forever
    n_total = length(BF.BFs)
    sample_indices = shuffle(1:n_total)[1:min(n_samples, n_total)]

    max_err = 0.0
    avg_err = 0.0

    println("\n--- Far-Field Accuracy Validation ($n_samples blocks) ---")

    for i in sample_indices
        (node_s, node_o) = ButterflyFactorizations.getNSNO(BF.BFs[i])

        # 1. Get the exact global indices for this block
        test_idx = H2Trees.values(BF.tree.testcluster, node_o)
        trial_idx = H2Trees.values(BF.tree.trialcluster, node_s)

        # 2. Assemble the EXACT dense block using BEAST
        Z_exact = zeros(ComplexF64, length(test_idx), length(trial_idx))
        kernelmatrix(Z_exact, test_idx, trial_idx)

        # 3. Generate a random input vector
        x = randn(ComplexF64, length(trial_idx))

        # 4. Compute exact matvec
        y_exact = Z_exact * x

        # 5. Compute Butterfly matvec
        # Assuming `fly[i]` supports multiplication: `mul!(y_bf, fly[i], x)`
        # If your Butterfly uses an allocated `*`:
        x_tmp = zeros(ComplexF64, length(trialspace))
        x_tmp[trial_idx] .= x
        y_bf = zeros(ComplexF64, length(testspace))
        mul!(y_bf, BF.BFs[i], x_tmp)

        # 6. Calculate relative error
        rel_err = norm(y_exact - y_bf[test_idx]) / norm(y_exact)

        max_err = max(max_err, rel_err)
        avg_err += rel_err
    end

    avg_err /= length(sample_indices)

    println("Maximum Relative Error: $(round(max_err, sigdigits=4))")
    println("Average Relative Error: $(round(avg_err, sigdigits=4))")
    println("------------------------------------------------\n")

    return avg_err, max_err
end

# Helper to build mesh based on shape symbol or custom function
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

# Helper to instantiate BEAST operator based on IE type
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
    ie_type::Symbol=:EFIE,
    checkfarfieldaccuracy::Bool=false,
    treekind::Symbol=:KMeansTree,
    unbalancedints::Bool=false,
    csv_file::String="benchmark_results.csv",
    log_file_path::String="benchmark_log.txt",
    scheduler=OhMyThreads.DynamicScheduler(),
)
    BLAS.set_num_threads(1)

    # Storage arrays for PlotlyJS rendering and fitting
    N_vals = Int[]
    t_aca_vals = Float64[]
    t_bf_vals = Float64[]
    mem_aca_vals = Float64[]
    mem_bf_total_vals = Float64[]
    mem_ButterflyFactorization_Mat_vals = Float64[]
    err_bf_vals = Float64[]
    t_mv_aca_vals = Float64[]
    t_mv_bf_vals = Float64[]

    println("==========================================================")
    println(" Starting Benchmarks | Shape: $shape | IE Type: $ie_type")
    println(" Structured Data File : $csv_file")
    println(" Console Log File     : $log_file_path")
    println("==========================================================\n")

    # Initialize CSV file with updated fitted reference headers
    csv_stream = open(csv_file, "w")
    write(
        csv_stream,
        "h,N,shape,ie_type,time_aca_s,time_bf_s,ref_bf_time_nlogn,ref_bf_time_nlog2n,mem_aca_mb,mem_bf_total_mb,mem_bf_entries_mb,ref_bf_mem_nlogn,ref_bf_mem_nlog2n,rel_err_bf,mv_aca_s,mv_bf_s,ref_bf_mv_nlogn\n",
    )
    flush(csv_stream)

    # Initialize Log file
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

        lambda = 10 * h
        k = 2 * pi / lambda

        # 1. Build Geometry and Operator dynamically
        op = build_beast_operator(ie_type, k)
        m = build_benchmark_mesh(shape, h)
        X = raviartthomas(m)
        N = length(X)

        # Build Trees
        if treekind == :KMeansTree
            tree = H2Trees.KMeansTree(X.pos, 2; minvalues=100)
        elseif treekind == :BisectionTree
            tree = H2Trees.BisectionTree(X.pos; max_depth=10)
        else
            tree = H2Trees.TwoNTree(X, h)
        end
        blktree = H2Trees.BlockTree(tree, tree)
        ACAtree = tree = H2Trees.KMeansTree(X.pos, 2; minvalues=100)
        ACAblktree = H2Trees.BlockTree(ACAtree, ACAtree)
        # 2. HMatrix (ACA)
        println("\nStarting ACA ($ie_type)...")
        #if !checkfarfieldaccuracy
        t_aca = @elapsed begin
            hmat = HMatrix(
                op,
                X,
                X,
                ACAblktree;
                tol=1e-3,
                spaceordering=AdaptiveCrossApproximation.PreserveSpaceOrder(),
                scheduler=scheduler,
                maxrank=60,
            )
        end
        #=else
            t_aca = @elapsed begin
                hmat = AdaptiveCrossApproximation.farmatrix(HMatrix(op, X, X, ACAblktree; tol=1e-3,
                    spaceordering=AdaptiveCrossApproximation.PreserveSpaceOrder(),
                    scheduler=scheduler, maxrank=60, isnear=isnearwrap))
            end
        end=#
        time_aca_str = @sprintf("Time for HMatrix (ACA): %.3f seconds", t_aca)
        println(time_aca_str)
        write(log_stream, time_aca_str * "\n")

        mem_aca = Base.summarysize(hmat) / 1024^2
        mem_aca_str = @sprintf("Total memory for HMatrix (ACA): %.2f MB", mem_aca)
        println(mem_aca_str)
        write(log_stream, mem_aca_str * "\n")

        # 3. Butterfly Factorization (BF)
        println("\nStarting ButterflyFactorization ($ie_type)...")
        #if !checkfarfieldaccuracy
        t_bf = @elapsed begin
            Bfmat = ButterflyFactorizations.PetrovGalerkinBF(
                op,
                X,
                X,
                blktree,
                k;
                compressor=ButterflyFactorizations.PartialQR(),
                scheduler=scheduler,
                tol=1e-3,
                unbalancedints=unbalancedints,
            )
        end
        #=else
            t_bf = @elapsed begin
                Bfmat = ButterflyFactorizations.farmatrix(ButterflyFactorizations.PetrovGalerkinBF(op, X, X, blktree, k;
                    compressor=ButterflyFactorizations.PartialQR(), scheduler=scheduler, tol=1e-3, unbalancedints=unbalancedints, isnear=isnearwrap))
            end
        end=#
        time_bf_str = @sprintf("Time for ButterflyFactorization: %.3f seconds", t_bf)
        println(time_bf_str)
        write(log_stream, time_bf_str * "\n")
        if checkfarfieldaccuracy
            validate_farfield_accuracy(op, X, X, Bfmat; n_samples=100)
        end
        mem_bf_total = Base.summarysize(Bfmat) / 1024^2
        mem_ButterflyFactorization_Mat = bf_matrix_memory(Bfmat)
        mem_bf_str = @sprintf(
            "Total memory for ButterflyFactorization: %.2f MB | Entries only: %.2f MB",
            mem_bf_total,
            mem_ButterflyFactorization_Mat
        )
        println(mem_bf_str)
        write(log_stream, mem_bf_str * "\n")

        # 4. Reference Matrix & Accuracy Check
        println("\nComputing reference Matrix with ACA for a lower tolerance...")
        #if !checkfarfieldaccuracy
        refmat = HMatrix(
            op,
            X,
            X,
            ACAblktree;
            tol=1e-5,
            spaceordering=AdaptiveCrossApproximation.PreserveSpaceOrder(),
            scheduler=OhMyThreads.DynamicScheduler(),
            maxrank=100,
        )
        #=else
            refmat = AdaptiveCrossApproximation.farmatrix(HMatrix(op, X, X, ACAblktree; tol=1e-5,
                spaceordering=AdaptiveCrossApproximation.PreserveSpaceOrder(),
                scheduler=OhMyThreads.DynamicScheduler(), maxrank=100, isnear=isnearwrap))
        end=#

        xtest = randn(ComplexF64, size(Bfmat, 2))
        y_exact = refmat * xtest
        y_bf = Bfmat * xtest
        err_bf = norm(y_exact - y_bf) / norm(y_exact)
        err_str = @sprintf("Relative error of ButterflyFactorization mat-vec: %.2e", err_bf)
        println(err_str)
        write(log_stream, err_str * "\n")

        # 5. Benchmark Mat-Vec Products
        println("\nBenchmarking MV product for ACA...")
        t_mv_aca = @belapsed $hmat * $xtest
        mv_aca_str = @sprintf("Time for ACA mat-vec: %.5f seconds", t_mv_aca)
        println(mv_aca_str)
        write(log_stream, mv_aca_str * "\n")

        println("Benchmarking MV product for ButterflyFactorization...")
        t_mv_bf = @belapsed $Bfmat * $xtest
        mv_bf_str = @sprintf("Time for Butterfly mat-vec: %.5f seconds", t_mv_bf)
        println(mv_bf_str)
        write(log_stream, mv_bf_str * "\n")

        # Push into in-memory arrays
        push!(N_vals, N)
        push!(t_aca_vals, t_aca)
        push!(t_bf_vals, t_bf)
        push!(mem_aca_vals, mem_aca)
        push!(mem_bf_total_vals, mem_bf_total)
        push!(mem_ButterflyFactorization_Mat_vals, mem_ButterflyFactorization_Mat)
        push!(err_bf_vals, err_bf)
        push!(t_mv_aca_vals, t_mv_aca)
        push!(t_mv_bf_vals, t_mv_bf)

        # Fit Scaling Constants for the current iteration
        c_time_nlogn = fit_scaling_factor(N_vals, t_bf_vals, f_nlogn)
        c_time_nlog2n = fit_scaling_factor(N_vals, t_bf_vals, f_nlog2n)
        c_mem_nlogn = fit_scaling_factor(N_vals, mem_bf_total_vals, f_nlogn)
        c_mem_nlog2n = fit_scaling_factor(N_vals, mem_bf_total_vals, f_nlog2n)
        c_mv_nlogn = fit_scaling_factor(N_vals, t_mv_bf_vals, f_nlogn)

        # 🚀 IMMEDIATE INCREMENTAL CSV WRITE
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

        # 🚀 INCREMENTAL PLOTS SAVING
        loop_ref_time_nlogn = c_time_nlogn .* f_nlogn.(N_vals)
        loop_ref_time_nlog2n = c_time_nlog2n .* f_nlog2n.(N_vals)
        loop_ref_mem_nlogn = c_mem_nlogn .* f_nlogn.(N_vals)
        loop_ref_mem_nlog2n = c_mem_nlog2n .* f_nlog2n.(N_vals)
        loop_ref_mv_nlogn = c_mv_nlogn .* f_nlogn.(N_vals)

        p_time = plot(
            [
                scatter(;
                    x=N_vals,
                    y=t_aca_vals,
                    name="ACA",
                    mode="lines+markers",
                    line=attr(; color="firebrick"),
                ),
                scatter(;
                    x=N_vals,
                    y=t_bf_vals,
                    name="Butterfly",
                    mode="lines+markers",
                    line=attr(; color="royalblue"),
                ),
                scatter(;
                    x=N_vals,
                    y=loop_ref_time_nlogn,
                    name="O(N log N) Fit",
                    mode="lines",
                    line=attr(; color="gray", dash="dash"),
                ),
                scatter(;
                    x=N_vals,
                    y=loop_ref_time_nlog2n,
                    name="O(N log² N) Fit",
                    mode="lines",
                    line=attr(; color="black", dash="dot"),
                ),
            ],
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
            [
                scatter(;
                    x=N_vals,
                    y=mem_aca_vals,
                    name="ACA Total",
                    mode="lines+markers",
                    line=attr(; color="firebrick"),
                ),
                scatter(;
                    x=N_vals,
                    y=mem_bf_total_vals,
                    name="Butterfly Total",
                    mode="lines+markers",
                    line=attr(; color="royalblue"),
                ),
                scatter(;
                    x=N_vals,
                    y=mem_ButterflyFactorization_Mat_vals,
                    name="Butterfly Matrix Entries",
                    mode="lines+markers",
                    line=attr(; color="skyblue", dash="dash"),
                ),
                scatter(;
                    x=N_vals,
                    y=loop_ref_mem_nlogn,
                    name="O(N log N) Fit",
                    mode="lines",
                    line=attr(; color="gray", dash="dash"),
                ),
                scatter(;
                    x=N_vals,
                    y=loop_ref_mem_nlog2n,
                    name="O(N log² N) Fit",
                    mode="lines",
                    line=attr(; color="black", dash="dot"),
                ),
            ],
            Layout(;
                title="Memory Usage vs N ($shape_str, $ie_type)",
                xaxis_title="N (DOFs)",
                yaxis_title="Memory (MB)",
                yaxis_type="log",
                xaxis_type="log",
                template="plotly_white",
            ),
        )
        savefig(p_mem, "plot_memory_usage.html")

        p_mv = plot(
            [
                scatter(;
                    x=N_vals,
                    y=t_mv_aca_vals,
                    name="ACA",
                    mode="lines+markers",
                    line=attr(; color="firebrick"),
                ),
                scatter(;
                    x=N_vals,
                    y=t_mv_bf_vals,
                    name="Butterfly",
                    mode="lines+markers",
                    line=attr(; color="royalblue"),
                ),
                scatter(;
                    x=N_vals,
                    y=loop_ref_mv_nlogn,
                    name="O(N log N) Fit",
                    mode="lines",
                    line=attr(; color="gray", dash="dash"),
                ),
            ],
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
                    name="Butterfly Rel Error",
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

    # Print Final Summary Table
    header1 = "=========================================================================================================================="
    header2 = @sprintf(
        "%-6s | %-8s | %-12s | %-12s | %-12s | %-12s | %-12s | %-10s | %-10s",
        "h",
        "N",
        "Time ACA (s)",
        "Time BF (s)",
        "Mem ACA (MB)",
        "Mem BF (MB)",
        "Err",
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

    # --- FINAL RECOMPUTATION OF FITTED REFERENCE CURVES (Global Scope outside loop) ---
    final_c_time_nlogn = fit_scaling_factor(N_vals, t_bf_vals, f_nlogn)
    final_c_time_nlog2n = fit_scaling_factor(N_vals, t_bf_vals, f_nlog2n)
    final_c_mem_nlogn = fit_scaling_factor(N_vals, mem_bf_total_vals, f_nlogn)
    final_c_mem_nlog2n = fit_scaling_factor(N_vals, mem_bf_total_vals, f_nlog2n)
    final_c_mv_nlogn = fit_scaling_factor(N_vals, t_mv_bf_vals, f_nlogn)

    ref_time_nlogn = final_c_time_nlogn .* f_nlogn.(N_vals)
    ref_time_nlog2n = final_c_time_nlog2n .* f_nlog2n.(N_vals)
    ref_mem_nlogn = final_c_mem_nlogn .* f_nlogn.(N_vals)
    ref_mem_nlog2n = final_c_mem_nlog2n .* f_nlog2n.(N_vals)
    ref_mv_nlogn = final_c_mv_nlogn .* f_nlogn.(N_vals)

    shape_str = shape isa Symbol ? string(shape) : "custom"

    # Construct Final Returnable Plot Objects
    p_time = plot(
        [
            scatter(;
                x=N_vals,
                y=t_aca_vals,
                name="ACA",
                mode="lines+markers",
                line=attr(; color="firebrick"),
            ),
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
        ],
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
        [
            scatter(;
                x=N_vals,
                y=mem_aca_vals,
                name="ACA Total",
                mode="lines+markers",
                line=attr(; color="firebrick"),
            ),
            scatter(;
                x=N_vals,
                y=mem_bf_total_vals,
                name="Butterfly Total",
                mode="lines+markers",
                line=attr(; color="royalblue"),
            ),
            scatter(;
                x=N_vals,
                y=mem_ButterflyFactorization_Mat_vals,
                name="Butterfly Matrix Entries",
                mode="lines+markers",
                line=attr(; color="skyblue", dash="dash"),
            ),
            scatter(;
                x=N_vals,
                y=ref_mem_nlogn,
                name="O(N log N) Fit",
                mode="lines",
                line=attr(; color="gray", dash="dash"),
            ),
            scatter(;
                x=N_vals,
                y=ref_mem_nlog2n,
                name="O(N log² N) Fit",
                mode="lines",
                line=attr(; color="black", dash="dot"),
            ),
        ],
        Layout(;
            title="Memory Usage vs N ($shape_str, $ie_type)",
            xaxis_title="N (DOFs)",
            yaxis_title="Memory (MB)",
            yaxis_type="log",
            xaxis_type="log",
            template="plotly_white",
        ),
    )
    savefig(p_mem, "plot_memory_usage.html")

    p_mv = plot(
        [
            scatter(;
                x=N_vals,
                y=t_mv_aca_vals,
                name="ACA",
                mode="lines+markers",
                line=attr(; color="firebrick"),
            ),
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
        ],
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
                name="Butterfly Rel Error",
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

    println("\nAll benchmarks complete! Data saved to '$csv_file'.")

    return p_time, p_mem, p_mv, p_err
end

h_values = [0.10, 0.08, 0.06] #, 0.06, 0.03, 0.02, 0.01, 0.005
p_time, p_mem, p_mv, p_err = run_benchmarks(
    h_values;
    shape=:sphere,
    ie_type=:EFIE,
    checkfarfieldaccuracy=true,
    unbalancedints=true,
    treekind=:KMeansTree,
    scheduler=OhMyThreads.DynamicScheduler(),
    csv_file="benchmark_results.csv",
    log_file_path="benchmark_log.txt",
);
display(p_time)
display(p_mem)
display(p_mv)
display(p_err)
