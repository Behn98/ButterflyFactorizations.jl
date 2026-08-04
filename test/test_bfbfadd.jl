@testitem "Algebraic Addition of Butterfly Factorizations" begin
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

    # Create two distinct, well-separated spheres.
    # Mesh size (0.15) is kept coarse for fast CI execution while
    # generating enough DOFs to build a multi-level tree.
    @info "Generating meshes and Raviart-Thomas spaces..."
    x = meshsphere(0.25, 0.15)                   # Centered at 0
    y = translate(x, SVector(5.0, 0.0, 0.0))     # Centered at 5

    # Define the Raviart-Thomas spaces
    T = raviartthomas(x) # Row space (Observer)
    U = raviartthomas(y) # Column space (Source)

    @info "Degrees of Freedom per space: $(length(T))"

    # Standard Single Layer Maxwell operator
    op = Maxwell3D.singlelayer(; wavenumber=k)

    # =========================================================================
    # 2. Tree Construction and BlockTrees
    # =========================================================================
    Ttree = ButterflyFactorizations.build_bisection_tree(T.pos; max_points=20)
    Utree = ButterflyFactorizations.build_bisection_tree(U.pos; max_points=20)

    # Operator: Maps from U (Column Space) -> T (Row Space)
    blktree = H2Trees.BlockTree(Ttree, Utree)
    farassembler = ButterflyFactorizations.AbstractKernelMatrix(op, T, U)

    # =========================================================================
    # 3. Assembly of Dense Ground Truth Matrices
    # =========================================================================
    @info "Assembling Dense Ground Truth Matrices..."
    A = assemble(op, T, U)

    # The ground truth matrix addition
    A_sum = A + A

    # =========================================================================
    # 4. Assembly of Butterfly Factorizations
    # =========================================================================
    tol = 1e-4

    @info "Assembling Initial Butterfly Factorization..."
    Bfly = ButterflyFactorizations.assemble_BF(
        farassembler, blktree, 1, 1, k, tol; scheduler=OhMyThreads.StaticScheduler()
    )

    # =========================================================================
    # 5. Testing the Addition Pipeline
    # =========================================================================
    @info "Adding Butterflies (Structural Algebraic Sum)..."

    # This routes to your overloaded `+` operator (add_eqbfs)
    Bfly_sum = Bfly + Bfly

    # Generate a random excitation vector on the column space (U)
    v_in = randn(ComplexF64, length(U))

    # Evaluate the cascading operations
    y_dense = A_sum * v_in                     # 1. True dense matrix multiplication
    y_bf_chain = (Bfly * v_in) + (Bfly * v_in) # 2. Sequential matrix-vector products
    y_bf_sum = Bfly_sum * v_in                 # 3. Our new compressed algebraic operator sum

    # =========================================================================
    # 6. Verification & Memory Profiling
    # =========================================================================
    err_chain = norm(y_bf_chain - y_dense) / norm(y_dense)
    err_sum = norm(y_bf_sum - y_dense) / norm(y_dense)

    @info """
    --- Matrix-Vector Evaluation Accuracy ---
    Tolerance target: $tol
    Error (Chained mat-vec)    : $(round(err_chain, sigdigits=4))
    Error (Recompressed sum)   : $(round(err_sum, sigdigits=4))
    """

    @info """
    --- Memory Footprint Analysis ---
    Dense sum memory     : $(round(Base.summarysize(A_sum) / 1024^2, digits=3)) MB
    Single BF memory     : $(round(Base.summarysize(Bfly) / 1024^2, digits=3)) MB
    Recompressed memory  : $(round(Base.summarysize(Bfly_sum) / 1024^2, digits=3)) MB
    """

    # Ensure the algebraic addition maintains the target precision
    @test err_chain < tol * 10
    @test err_sum < tol * 10
end
