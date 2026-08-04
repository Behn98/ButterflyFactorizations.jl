@testitem "Butterfly splitmultiplication tests" begin
    using Test
    using H2Trees
    using CompScienceMeshes
    using BEAST
    using ButterflyFactorizations
    using StaticArrays
    using LinearAlgebra
    using LowRankApprox
    using OhMyThreads

    # =========================================================================
    # 1. Geometry & Problem Setup
    # =========================================================================
    @info "Initializing geometry, functional spaces, and trees..."
    BLAS.set_num_threads(1)
    lambda = 1.0
    k = 2 * pi / lambda
    tol = 1e-4

    # Geometry: 3 separated spheres (Observer T, Intermediate U, Source V)
    x = meshsphere(0.25, lambda / 10)
    y = translate(x, SVector(3.0, 0.0, 0.0))
    z = translate(x, SVector(-3.0, 0.0, 0.0))

    # Spaces & Operator
    T = raviartthomas(x)
    U = raviartthomas(y)
    V = raviartthomas(z)
    op = Maxwell3D.singlelayer(; wavenumber=k)

    # Trees
    Ttree = ButterflyFactorizations.build_bisection_tree(T.pos; max_points=50)
    Utree = ButterflyFactorizations.build_bisection_tree(U.pos; max_points=50)
    Vtree = ButterflyFactorizations.build_bisection_tree(V.pos; max_points=50)

    blktree_left  = H2Trees.BlockTree(Ttree, Utree)
    blktree_right = H2Trees.BlockTree(Utree, Vtree)

    farassembler_left  = ButterflyFactorizations.AbstractKernelMatrix(op, T, U)
    farassembler_right = ButterflyFactorizations.AbstractKernelMatrix(op, U, V)

    @info "Assembling Dense Ground Truth Matrices (this may take a moment)..."
    A_left  = assemble(op, T, U)
    A_right = assemble(op, U, V)
    A_prod  = A_left * A_right

    # Random Test Vectors
    x_rand        = randn(ComplexF64, length(V))
    y_exact_right = A_right * x_rand
    y_exact_prod  = A_prod * x_rand

    @info "Setup complete. Degrees of Freedom: T=$(length(T)), U=$(length(U)), V=$(length(V))"

    # =========================================================================
    # 2. Test Set: Factorization & Assembly Accuracy
    # =========================================================================
    @testset "Single & Cluster Assembly" begin
        @info "Starting Test 1: Single & Cluster Assembly..."

        # 1. Single Higher-Rank BF (Right)
        bf_right = ButterflyFactorizations.assemble_BF(
            farassembler_right,
            blktree_right,
            1,
            1,
            k,
            tol;
            compressor=ButterflyFactorizations.PartialQR(),
            scheduler=OhMyThreads.DynamicScheduler(),
        )

        y_bf_right = bf_right * x_rand
        rel_err_right = norm(y_exact_right - y_bf_right) / norm(y_exact_right)
        @test rel_err_right < 10 * tol

        # 2. Cluster Assembly (Left)
        schildren = collect(H2Trees.children(blktree_left.trialcluster, 1))
        ochildren = collect(H2Trees.children(blktree_left.testcluster, 1))
        bf_cluster = Matrix{ButterflyFactorizations.ButterflyFactorization}(
            undef, length(ochildren), length(schildren)
        )

        for (i, oc) in enumerate(ochildren)
            for (j, sc) in enumerate(schildren)
                bf_cluster[i, j] = ButterflyFactorizations.assemble_BF(
                    farassembler_left,
                    blktree_left,
                    oc,
                    sc,
                    k,
                    tol;
                    compressor=ButterflyFactorizations.PartialQR(),
                    scheduler=OhMyThreads.DynamicScheduler(),
                )
            end
        end

        # Test cluster-vector product
        x_inter = randn(ComplexF64, length(U))
        y_exact_left = A_left * x_inter
        y_cluster = zeros(ComplexF64, length(T))

        for i in eachindex(ochildren), j in eachindex(schildren)
            mul!(y_cluster, bf_cluster[i, j], x_inter, 1, 1)
        end
        rel_err_left = norm(y_exact_left - y_cluster) / norm(y_exact_left)

        @test rel_err_left < 10 * tol
        @info "Test 1 Passed."
    end

    # =========================================================================
    # 3. Test Set: Operator Addition (`add_eqbfs`)
    # =========================================================================
    @testset "Operator Addition (add_eqbfs)" begin
        @info "Starting Test 2: Operator Addition (`add_eqbfs`)..."

        bf_right_1 = ButterflyFactorizations.assemble_BF(
            farassembler_right,
            blktree_right,
            1,
            1,
            k,
            tol;
            compressor=ButterflyFactorizations.PartialQR(),
        )
        bf_right_2 = ButterflyFactorizations.assemble_BF(
            farassembler_right,
            blktree_right,
            1,
            1,
            k,
            tol;
            compressor=ButterflyFactorizations.PartialQR(),
        )

        # Compute (A_right + A_right) via BF addition
        bf_sum = ButterflyFactorizations.add_eqbfs(bf_right_1, bf_right_2, tol)

        y_sum_exact = 2.0 .* (A_right * x_rand)
        y_sum_bf    = bf_sum * x_rand

        rel_err_sum = norm(y_sum_exact - y_sum_bf) / norm(y_sum_exact)
        @test rel_err_sum < 10 * tol
        @info "Test 2 Passed."
    end

    # =========================================================================
    # 4. Test Set: Direct Operator Multiplication (`mulBFs`)
    # =========================================================================
    @testset "Direct Operator Multiplication (mulBFs)" begin
        @info "Starting Test 3: Direct Operator Multiplication (`mulBFs`)..."

        bf_left_single = ButterflyFactorizations.assemble_BF(
            farassembler_left,
            blktree_left,
            1,
            1,
            k,
            tol;
            compressor=ButterflyFactorizations.PartialQR(),
        )
        bf_right_single = ButterflyFactorizations.assemble_BF(
            farassembler_right,
            blktree_right,
            1,
            1,
            k,
            tol;
            compressor=ButterflyFactorizations.PartialQR(),
        )

        # Direct operator product
        @info "  -> Multiplying matrices natively..."
        bf_prod_direct = ButterflyFactorizations.mulBFs(
            bf_left_single, bf_right_single, tol
        )

        y_direct_bf = bf_prod_direct * x_rand
        rel_err_direct = norm(y_exact_prod - y_direct_bf) / norm(y_exact_prod)

        @test rel_err_direct < 50 * tol
        @info "Test 3 Passed."
    end

    # =========================================================================
    # 5. Test Set: Hierarchical Split-Multiplication (`splitmulbf`)
    # =========================================================================
    @testset "Hierarchical Split-Multiplication (splitmulbf)" begin
        @info "Starting Test 4: Hierarchical Split-Multiplication (`splitmulbf`)..."

        schildren = collect(H2Trees.children(blktree_left.trialcluster, 1))
        ochildren = collect(H2Trees.children(blktree_left.testcluster, 1))

        bf_cluster = Matrix{ButterflyFactorizations.ButterflyFactorization}(
            undef, length(ochildren), length(schildren)
        )
        for (i, oc) in enumerate(ochildren), (j, sc) in enumerate(schildren)
            bf_cluster[i, j] = ButterflyFactorizations.assemble_BF(
                farassembler_left,
                blktree_left,
                oc,
                sc,
                k,
                tol;
                compressor=ButterflyFactorizations.PartialQR(),
            )
        end

        higherkBF = ButterflyFactorizations.assemble_BF(
            farassembler_right,
            blktree_right,
            1,
            1,
            k,
            tol;
            compressor=ButterflyFactorizations.PartialQR(),
        )

        # Execute Split-Multiplication
        @info "  -> Executing splitmulbf algorithm..."
        splitbfprod = ButterflyFactorizations.splitmulbf(bf_cluster, higherkBF, 1e-5)

        y_split_bf = splitbfprod * x_rand
        rel_err_split = norm(y_exact_prod - y_split_bf) / norm(y_exact_prod)

        # Verify Accuracy
        @test rel_err_split < 50 * tol

        # Memory assertion: Ensure compression actually happened
        mem_dense = Base.summarysize(A_prod)
        mem_bf    = Base.summarysize(splitbfprod)
        @test mem_bf < mem_dense

        @info "Test 4 Passed."
        @info "[Test Metrics] Split-Mul Relative Error: $rel_err_split"
        @info "[Test Metrics] Compression Ratio: $(mem_dense / mem_bf)x"
    end

    @info "All Butterfly Factorization tests completed successfully!"
end
