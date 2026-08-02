@testitem "Testing Blockassembly subroutines for pruned trees" begin
    using Test
    using BEAST
    using OhMyThreads
    using CompScienceMeshes
    using ParallelKMeans
    using H2Trees
    using ButterflyFactorizations
    using StaticArrays
    using LinearAlgebra

    # =========================================================================
    # 1. Geometry and Spaces (Scaled for CI Performance)
    # =========================================================================
    # Force single-threaded BLAS to prevent oversubscription on CI runners
    BLAS.set_num_threads(1)

    h = 0.1
    lambda = 1.0
    k = 2 * pi / lambda

    op = Maxwell3D.singlelayer(; wavenumber=k)

    @info "Generating meshes and Raviart-Thomas spaces..."
    # Create two well-separated spheres
    m1 = meshsphere(0.25, h)
    X1 = raviartthomas(m1)

    m2 = translate(m1, SVector(5.0, 0.0, 0.0))
    X2 = raviartthomas(m2)

    @info "Degrees of Freedom per space: $(length(X1))"

    # =========================================================================
    # 2. Tree Construction (Testing Different Partitioning Strategies)
    # =========================================================================
    @info "Constructing KMeans and TwoN pruned trees..."
    # Strategy 1: Data-driven K-Means clustering
    # minvalues=100 ensures the trees are aggressively pruned (shallow)
    Ttree_kmeans = H2Trees.KMeansTree(X1.pos, 2; minvalues=50)
    Stree_kmeans = H2Trees.KMeansTree(X2.pos, 2; minvalues=50)
    blktree_kmeans = H2Trees.BlockTree(Ttree_kmeans, Stree_kmeans)

    # Strategy 2: Octree-style spatial bisection
    Ttree_twon = H2Trees.TwoNTree(X1, h; minvalues=100)
    Stree_twon = H2Trees.TwoNTree(X2, h; minvalues=100)
    blktree_twon = H2Trees.BlockTree(Ttree_twon, Stree_twon)

    # =========================================================================
    # 3. Dense Ground Truth Assembly
    # =========================================================================
    @info "Assembling Dense Ground Truth Matrix..."
    @time A_exact = assemble(op, X1, X2)

    # =========================================================================
    # 4. Global Butterfly Operator Assembly
    # =========================================================================
    target_tol = 1e-4

    @info "Assembling PetrovGalerkinBF via KMeansTree..."
    Bfmat_kmeans = ButterflyFactorizations.PetrovGalerkinBF(
        op,
        X1,
        X2,
        blktree_kmeans,
        k;
        compressor=ButterflyFactorizations.PartialQR(),
        tol=target_tol,
        scheduler=OhMyThreads.StaticScheduler(),
    )

    @info "Assembling PetrovGalerkinBF via TwoNTree..."
    Bfmat_twon = ButterflyFactorizations.PetrovGalerkinBF(
        op,
        X1,
        X2,
        blktree_twon,
        k;
        compressor=ButterflyFactorizations.PartialQR(),
        tol=target_tol,
        scheduler=OhMyThreads.StaticScheduler(),
    )

    # =========================================================================
    # 5. Verification & Memory Profiling
    # =========================================================================
    @info "Evaluating Matrix-Vector Products..."

    # Generate a random excitation vector
    x_in = randn(ComplexF64, size(A_exact, 2))

    y_exact = A_exact * x_in
    y_approx_kmeans = Bfmat_kmeans * x_in
    y_approx_twon = Bfmat_twon * x_in

    err_kmeans = norm(y_exact - y_approx_kmeans) / norm(y_exact)
    err_twon = norm(y_exact - y_approx_twon) / norm(y_exact)

    @info """
    --- Memory Footprint Analysis ---
    Dense Matrix       : $(round(Base.summarysize(A_exact) / 1024^2, digits=3)) MB
    K-Means BF         : $(round(Base.summarysize(Bfmat_kmeans) / 1024^2, digits=3)) MB
    TwoN BF            : $(round(Base.summarysize(Bfmat_twon) / 1024^2, digits=3)) MB
    """

    @info """
    --- Accuracy Results ---
    Target Tolerance   : $target_tol
    K-Means Error      : $(round(err_kmeans, sigdigits=4))
    TwoN Error         : $(round(err_twon, sigdigits=4))
    """

    # Ensure the algebraic assembly maintains the target precision
    # (Allowing a factor of 10 for compression error accumulation across the hierarchy)
    @test err_kmeans < target_tol * 10
    @test err_twon < target_tol * 10
end
