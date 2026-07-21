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

# Hjälpfunktion för att räkna ut exakt hur mycket minne själva matriserna
# inuti BF-dictionarierna tar upp (exkluderar träd och metadata)
function bf_matrix_memory(Bfmat)
    num_elements = 0

    # Räkna element i Far-field (Butterfly blocks)
    for bf in Bfmat.BFs
        for q in values(bf.Q)
            num_elements += length(q)
        end
        for p in values(bf.P)
            num_elements += length(p)
        end
        for r_level in bf.R
            for row in values(r_level)
                for mat in values(row)
                    num_elements += length(mat)
                end
            end
        end
    end

    # Räkna element i Near-field (BlockSparseMatrix)
    #=for blk in Bfmat.nearinteractions.blocks
        num_elements += length(blk)
    end=#

    # Räkna element i Near-field (SparseMatrixCSC)
    num_elements += SparseArrays.nnz(Bfmat.nearinteractions)

    # 16 bytes per ComplexF64 (2 * 8 bytes)
    mem_bytes = num_elements * 16
    return mem_bytes / 1024^2 # Returnera i MB
end

function run_benchmarks(h_values)
    BLAS.set_num_threads(1)

    # Arrayer för att spara data till plottarna
    N_vals = Int[]
    t_aca_vals = Float64[]
    t_bf_vals = Float64[]
    mem_aca_vals = Float64[]
    mem_bf_total_vals = Float64[]
    mem_bf_mats_vals = Float64[]
    err_bf_vals = Float64[]
    t_mv_aca_vals = Float64[]
    t_mv_bf_vals = Float64[]

    println("Starting benchmarks...")
    println(
        "Note: The @time outputs below show the detailed RAM allocations and execution times.",
    )
    println(
        "Results will be saved incrementally to 'benchmark_log.txt' and plots to HTML files.\n",
    )

    # Öppna en loggfil för att spara data löpande (inkrementellt ifall det kraschar mitt i)
    log_file = open("benchmark_log.txt", "w")
    write(log_file, "Starting benchmarks...\n")
    flush(log_file)

    i = 1
    for h in h_values
        round_str = @sprintf(
            "==================================================== Round %-1d ==========================================================",
            i
        )
        println(round_str)
        write(log_file, round_str * "\n")
        flush(log_file)

        lambda = 10 * h
        k = 2 * pi / lambda
        op = Maxwell3D.singlelayer(; wavenumber=k)
        m = meshsphere(1.0, h)
        X = raviartthomas(m)
        N = length(X)

        # Bygg träd
        tree = H2Trees.KMeansTree(X.pos, 2; minvalues=100)
        blktree = H2Trees.BlockTree(tree, tree)

        # 1. HMatrix (ACA)
        println("\nStarting ACA...\n")
        t_aca = @elapsed begin
            hmat = HMatrix(
                op,
                X,
                X,
                blktree;
                tol=1e-3,
                spaceordering=AdaptiveCrossApproximation.PreserveSpaceOrder(),
                scheduler=DynamicScheduler(),
                maxrank=60,
            )
        end
        time_aca_str = @sprintf("\nTime for HMatrix (ACA): %.3f seconds \n", t_aca)
        println(time_aca_str)
        write(log_file, time_aca_str * "\n")
        mem_aca = Base.summarysize(hmat) / 1024^2
        mem_aca_str = @sprintf("\nTotal memory for HMatrix (ACA): %.2f MB \n", mem_aca)
        println(mem_aca_str)
        write(log_file, mem_aca_str * "\n")

        # 2. Butterfly (BF)
        println("\nStarting ButterflyFactorization...\n")
        t_bf = @elapsed begin
            Bfmat = ButterflyFactorizations.PetrovGalerkinBF(
                op,
                X,
                X,
                blktree,
                k;
                compressor=ButterflyFactorizations.PartialQR(),
                tol=1e-3,
                α=1.5,
            )
        end
        println("\nFlattening Butterfly Structures...\n")
        Bfmat2 = ButterflyFactorizations.flattenmatrix(Bfmat)
        time_bf_str = @sprintf("\nTime for ButterflyFactorization: %.3f seconds \n", t_bf)
        println(time_bf_str)
        write(log_file, time_bf_str * "\n")
        mem_bf_total = Base.summarysize(Bfmat) / 1024^2
        mem_bf_mats = bf_matrix_memory(Bfmat)
        mem_bf_str = @sprintf(
            "\nTotal memory for ButterflyFactorization: %.2f MB\n\nMemory for Butterfly matrix entries only: %.2f MB \n",
            mem_bf_total,
            mem_bf_mats
        )
        println(mem_bf_str)
        write(log_file, mem_bf_str * "\n")

        # 3. Dense matrix & Accuracy
        println("\nComputing reference Matrix with ACA for a lower tolerance...\n")
        refmat = HMatrix(
            op,
            X,
            X,
            blktree;
            tol=1e-5,
            spaceordering=AdaptiveCrossApproximation.PreserveSpaceOrder(),
            scheduler=DynamicScheduler(),
            maxrank=100,
        )
        xtest = randn(ComplexF64, size(Bfmat, 2))

        y_exact = refmat * xtest
        y_bf = Bfmat2 * xtest
        err_bf = norm(y_exact - y_bf) / norm(y_exact)
        err_str = @sprintf(
            "Relative error of ButterflyFactorization mat-vec: %.2e\n", err_bf
        )
        println(err_str)
        write(log_file, err_str * "\n")

        # 4. Benchmarka mat-vec-produkterna
        println("Benchmarking MV product for ACA...\n")
        t_mv_aca = @belapsed $hmat * $xtest
        mv_aca_str = @sprintf("Time for ACA mat-vec: %.5f seconds\n", t_mv_aca)
        println(mv_aca_str)
        write(log_file, mv_aca_str * "\n")

        println("Benchmarking MV product for ButterflyFactorization...\n")
        t_mv_bf = @belapsed $Bfmat2 * $xtest
        mv_bf_str = @sprintf("Time for Butterfly mat-vec: %.5f seconds\n", t_mv_bf)
        println(mv_bf_str)
        write(log_file, mv_bf_str * "\n")

        # Spara data till arrayer
        push!(N_vals, N)
        push!(t_aca_vals, t_aca)
        push!(t_bf_vals, t_bf)
        push!(mem_aca_vals, mem_aca)
        push!(mem_bf_total_vals, mem_bf_total)
        push!(mem_bf_mats_vals, mem_bf_mats)
        push!(err_bf_vals, err_bf)
        push!(t_mv_aca_vals, t_mv_aca)
        push!(t_mv_bf_vals, t_mv_bf)

        end_str = @sprintf(
            "================================================ End of Round %-1d =======================================================\n",
            i
        )
        println(end_str)
        write(log_file, end_str * "\n")
        flush(log_file)
        i += 1
    end

    # Skriv ut sluttabeln till både terminalen och loggfilen
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
    println(header1)
    println(header2)
    println(header1)
    write(
        log_file,
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
        write(log_file, row_str * "\n")
    end
    println(header1)
    write(log_file, header1 * "\n")
    close(log_file)

    # Skapa och SPARA plottar med PlotlyJS
    println("\nSaving plots to disk...")

    p_time = plot(
        [
            scatter(; x=N_vals, y=t_aca_vals, name="ACA", mode="lines+markers"),
            scatter(; x=N_vals, y=t_bf_vals, name="Butterfly", mode="lines+markers"),
        ],
        Layout(;
            title="Build Time vs N",
            xaxis_title="N (DOFs)",
            yaxis_title="Time (s)",
            yaxis_type="log",
            xaxis_type="log",
        ),
    )
    savefig(p_time, "plot_build_time.html")

    p_mem = plot(
        [
            scatter(; x=N_vals, y=mem_aca_vals, name="ACA Total", mode="lines+markers"),
            scatter(;
                x=N_vals, y=mem_bf_total_vals, name="Butterfly Total", mode="lines+markers"
            ),
            scatter(;
                x=N_vals,
                y=mem_bf_mats_vals,
                name="Butterfly Dictionaries Only",
                mode="lines+markers",
                line_dash="dash",
            ),
        ],
        Layout(;
            title="Memory Usage vs N",
            xaxis_title="N (DOFs)",
            yaxis_title="Memory (MB)",
            yaxis_type="log",
            xaxis_type="log",
        ),
    )
    savefig(p_mem, "plot_memory_usage.html")

    p_mv = plot(
        [
            scatter(; x=N_vals, y=t_mv_aca_vals, name="ACA", mode="lines+markers"),
            scatter(; x=N_vals, y=t_mv_bf_vals, name="Butterfly", mode="lines+markers"),
        ],
        Layout(;
            title="Mat-Vec Product Time vs N",
            xaxis_title="N (DOFs)",
            yaxis_title="Time (s)",
            yaxis_type="log",
            xaxis_type="log",
        ),
    )
    savefig(p_mv, "plot_mat_vec_time.html")

    p_err = plot(
        [
            scatter(;
                x=N_vals,
                y=err_bf_vals,
                name="Butterfly Relative Error",
                mode="lines+markers",
            ),
        ],
        Layout(;
            title="Accuracy vs N",
            xaxis_title="N (DOFs)",
            yaxis_title="Relative Error",
            yaxis_type="log",
            xaxis_type="log",
        ),
    )
    savefig(p_err, "plot_accuracy.html")

    println("All plots successfully saved as interactive HTML files!")

    return p_time, p_mem, p_mv, p_err
end

h_values = [0.10, 0.08, 0.06, 0.03, 0.02, 0.01, 0.005]
p_time, p_mem, p_mv, p_err = run_benchmarks(h_values);
