@testitem "Algebraic Multiplication of Butterfly Factorizations" begin
    using Test
    using H2Trees
    using CompScienceMeshes
    using BEAST
    using ButterflyFactorizations
    using StaticArrays
    using LinearAlgebra
    using OhMyThreads
    # =========================================================================
    # 1. Geometry and Spaces (Scaled for CI Performance)
    # =========================================================================
    # Force single-threaded BLAS to prevent oversubscription on CI runners
    BLAS.set_num_threads(1)

    lambda = 1.0
    k = 2 * pi / lambda

    # Create three distinct, well-separated spheres.
    # Mesh size (0.15) is kept deliberately coarse to ensure rapid CI execution
    # while still generating enough elements to build a multi-level tree.
    @info "Generating meshes and Raviart-Thomas spaces..."
    x = meshsphere(0.25, 0.15)                   # Centered at 0
    y = translate(x, SVector(3.0, 0.0, 0.0))     # Centered at 3
    z = translate(x, SVector(-3.0, 0.0, 0.0))    # Centered at -3

    # Define the function spaces.
    # Note the strict mapping convention: Operators map from Column Space (Source)
    # to Row Space (Observer).
    T = raviartthomas(x) # Row space (Observer) for Left Operator
    U = raviartthomas(y) # Column space for Left, Row space for Right
    V = raviartthomas(z) # Column space (Source) for Right Operator

    @info "Degrees of Freedom per space: $(length(T))"

    # Standard Single Layer Maxwell operator
    op = Maxwell3D.singlelayer(; wavenumber=k)

    # =========================================================================
    # 2. Tree Construction and BlockTrees
    # =========================================================================
    # Create BisectionTrees for all three spaces.
    # max_points=5 ensures the tree gets deep enough to test the R-factor hierarchy
    Ttree = ButterflyFactorizations.build_bisection_tree(T.pos; max_points=5)
    Utree = ButterflyFactorizations.build_bisection_tree(U.pos; max_points=5) # <-- The critical shared intermediate tree!
    Vtree = ButterflyFactorizations.build_bisection_tree(V.pos; max_points=5)

    # Left Operator: Maps from U (Column Space) -> T (Row Space)
    blktree_left = H2Trees.BlockTree(Ttree, Utree)
    farassembler_left = ButterflyFactorizations.AbstractKernelMatrix(op, T, U)

    # Right Operator: Maps from V (Column Space) -> U (Row Space)
    blktree_right = H2Trees.BlockTree(Utree, Vtree)
    farassembler_right = ButterflyFactorizations.AbstractKernelMatrix(op, U, V)

    # =========================================================================
    # 3. Assembly of Dense Ground Truth Matrices
    # =========================================================================
    @info "Assembling Dense Ground Truth Matrices..."
    A_left = assemble(op, T, U)
    A_right = assemble(op, U, V)

    # The ground truth matrix product
    A_prod = A_left * A_right

    # =========================================================================
    # 4. Assembly of Butterfly Factorizations
    # =========================================================================
    tol = 1e-4

    @info "Assembling Left Butterfly Factorization..."
    Bfly_left = ButterflyFactorizations.assemble_BF(
        farassembler_left,
        blktree_left,
        1,
        1,
        k,
        tol;
        scheduler=OhMyThreads.StaticScheduler(),
    )

    @info "Assembling Right Butterfly Factorization..."
    Bfly_right = ButterflyFactorizations.assemble_BF(
        farassembler_right,
        blktree_right,
        1,
        1,
        k,
        tol;
        scheduler=OhMyThreads.StaticScheduler(),
    )

    # =========================================================================
    # 5. Testing the Multiplication Pipeline
    # =========================================================================
    @info "Multiplying Butterflies (Structural Algebraic Product)..."

    # 1. Compute the recompressed algebraic product (O(N log N) complexity)
    Bfly_prod = Bfly_left * Bfly_right

    # 2. Compute the trivial (uncompressed) cascade for baseline comparison
    trivprod = ButterflyFactorizations.trivialmul(Bfly_left, Bfly_right)

    # Generate a random excitation vector on the far-right column space (V)
    v_in = randn(ComplexF64, length(V))

    # Evaluate the cascading operations
    y_dense = A_prod * v_in                     # 1. True dense matrix multiplication
    y_bf_chain = Bfly_left * (Bfly_right * v_in) # 2. Sequential matrix-vector products
    y_bf_product = Bfly_prod * v_in             # 3. Our compressed algebraic operator product
    y_bf_trivprod = trivprod * v_in             # 4. Uncompressed algebraic cascade

    # =========================================================================
    # 6. Verification & Memory Profiling
    # =========================================================================
    err_chain = norm(y_bf_chain - y_dense) / norm(y_dense)
    err_product = norm(y_bf_product - y_dense) / norm(y_dense)
    err_trivprod = norm(y_bf_trivprod - y_dense) / norm(y_dense)

    @info """
    --- Matrix-Vector Evaluation Accuracy ---
    Tolerance target: $tol
    Error (Chained mat-vec)    : $(round(err_chain, sigdigits=4))
    Error (Trivial product)    : $(round(err_trivprod, sigdigits=4))
    Error (Recompressed prod)  : $(round(err_product, sigdigits=4))
    """

    @info """
    --- Memory Footprint Analysis ---
    Dense product memory : $(round(Base.summarysize(A_prod) / 1024^2, digits=3)) MB
    Left BF memory       : $(round(Base.summarysize(Bfly_left) / 1024^2, digits=3)) MB
    Right BF memory      : $(round(Base.summarysize(Bfly_right) / 1024^2, digits=3)) MB
    Trivial prod memory  : $(round(Base.summarysize(trivprod) / 1024^2, digits=3)) MB
    Recompressed memory  : $(round(Base.summarysize(Bfly_prod) / 1024^2, digits=3)) MB
    """

    # Ensure the algebraic multiplication maintains the target precision
    # (Allowing a factor of 10 for error accumulation across deep trees)
    @test err_chain < tol * 10
    @test err_trivprod < tol * 10
    @test err_product < tol * 10

    # Ensure the recompression actually saved memory compared to the trivial cascade
    @test Base.summarysize(Bfly_prod) < Base.summarysize(trivprod)
end
