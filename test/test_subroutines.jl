@testitem "Testing Blockassembly Subroutines (Balanced & Unbalanced Trees)" begin
    using Test
    using H2Trees
    using CompScienceMeshes
    using BEAST
    using OhMyThreads
    using ButterflyFactorizations
    using StaticArrays
    using LinearAlgebra

    # =========================================================================
    # 1. Geometry and Spaces (Scaled for CI Performance)
    # =========================================================================
    # Force single-threaded BLAS to prevent oversubscription on CI runners
    BLAS.set_num_threads(1)

    lambda = 1.0
    k = 2 * pi / lambda

    @info "Generating meshes and Raviart-Thomas spaces..."
    x = meshsphere(0.25, lambda / 10)
    y = translate(x, SVector(5.0, 0.0, 0.0))

    # Duplicate geometries to test isolated reference spaces
    x2 = meshsphere(0.25, lambda / 10)
    y2 = translate(x2, SVector(5.0, 0.0, 0.0))

    op = Maxwell3D.singlelayer(; wavenumber=k)

    T = raviartthomas(x)
    U = raviartthomas(y)
    T2 = raviartthomas(x2)
    U2 = raviartthomas(y2)

    @info "Degrees of Freedom per space: $(length(T))"

    # =========================================================================
    # 2. Tree Construction (Balanced vs. Unbalanced)
    # =========================================================================
    @info "Constructing hierarchical block trees..."

    # Configuration 1 & 2: Symmetrical, Balanced Trees
    tree1 = TwoNTree(T, U, lambda / 10) # Maps U -> T
    tree2 = TwoNTree(U, T, lambda / 10) # Maps T -> U

    # Configuration 3: Unbalanced Tree (Deep Observer, Shallow Source)
    tree31 = TwoNTree(U2.pos, lambda / 1.25) # Shallow
    tree32 = TwoNTree(T.pos, lambda / 10)    # Deep
    tree3  = BlockTree(tree31, tree32)       # Maps T -> U2

    # Configuration 4: Unbalanced Tree (Shallow Observer, Deep Source)
    tree41 = TwoNTree(U.pos, lambda / 10)    # Deep
    tree42 = TwoNTree(T2.pos, lambda / 1.25) # Shallow
    tree4  = BlockTree(tree41, tree42)       # Maps T2 -> U

    # Define the isolated far-field kernel matrix evaluators
    farassembler1 = ButterflyFactorizations.AbstractKernelMatrix(op, T, U)
    farassembler2 = ButterflyFactorizations.AbstractKernelMatrix(op, U, T)
    farassembler3 = ButterflyFactorizations.AbstractKernelMatrix(op, U2, T)
    farassembler4 = ButterflyFactorizations.AbstractKernelMatrix(op, U, T2)

    # =========================================================================
    # 3. Dense Ground Truth Assembly
    # =========================================================================
    @info "Assembling Dense Ground Truth Matrices..."
    A1 = assemble(op, T, U)
    A2 = assemble(op, U, T)
    A3 = assemble(op, U2, T)
    A4 = assemble(op, U, T2)

    # Generate semantically correct input vectors matching the Source (Trial) spaces
    x_in_U  = randn(ComplexF64, length(U))
    x_in_T  = randn(ComplexF64, length(T))
    x_in_T2 = randn(ComplexF64, length(T2))

    y_exact1 = A1 * x_in_U
    y_exact2 = A2 * x_in_T
    y_exact3 = A3 * x_in_T
    y_exact4 = A4 * x_in_T2

    # =========================================================================
    # 4. Butterfly Factorization Assembly (Flat Array & Sparse Matrix)
    # =========================================================================
    target_tol = 1e-3

    @info "Assembling Flat-Array Butterfly Blocks..."
    Bfly1 = ButterflyFactorizations.assemble_BF(
        farassembler1, tree1, 1, 1, k, target_tol; scheduler=OhMyThreads.StaticScheduler()
    )
    Bfly2 = ButterflyFactorizations.assemble_BF(
        farassembler2, tree2, 1, 1, k, target_tol; scheduler=OhMyThreads.StaticScheduler()
    )
    Bfly3 = ButterflyFactorizations.assemble_BF(
        farassembler3, tree3, 1, 1, k, target_tol; scheduler=OhMyThreads.StaticScheduler()
    )
    Bfly4 = ButterflyFactorizations.assemble_BF(
        farassembler4, tree4, 1, 1, k, target_tol; scheduler=OhMyThreads.StaticScheduler()
    )

    @info "Assembling Sparse-Matrix Butterfly Blocks..."
    Bfly1m = ButterflyFactorizations.assemble_BF_Mat(
        farassembler1, tree1, 1, 1, k, target_tol
    )
    Bfly2m = ButterflyFactorizations.assemble_BF_Mat(
        farassembler2, tree2, 1, 1, k, target_tol
    )
    Bfly3m = ButterflyFactorizations.assemble_BF_Mat(
        farassembler3, tree3, 1, 1, k, target_tol
    )
    Bfly4m = ButterflyFactorizations.assemble_BF_Mat(
        farassembler4, tree4, 1, 1, k, target_tol
    )

    # =========================================================================
    # 5. Verification & Accuracy Profiling
    # =========================================================================
    @info "Evaluating Flat-Array Matrix-Vector Products..."

    y_test1 = zeros(ComplexF64, size(A1, 1))
    y_test2 = zeros(ComplexF64, size(A2, 1))
    y_test3 = zeros(ComplexF64, size(A3, 1))
    y_test4 = zeros(ComplexF64, size(A4, 1))

    # Evaluate Flat-Array versions directly
    mul!(y_test1, Bfly1, x_in_U)
    mul!(y_test2, Bfly2, x_in_T)
    mul!(y_test3, Bfly3, x_in_T)
    mul!(y_test4, Bfly4, x_in_T2)

    err1_flat = norm(y_test1 - y_exact1) / norm(y_exact1)
    err2_flat = norm(y_test2 - y_exact2) / norm(y_exact2)
    err3_flat = norm(y_test3 - y_exact3) / norm(y_exact3)
    err4_flat = norm(y_test4 - y_exact4) / norm(y_exact4)

    @info "Evaluating Sparse-Matrix Products (Applying Permutations)..."

    # Zero out buffers before re-use
    fill!.((y_test1, y_test2, y_test3, y_test4), 0.0)

    # Evaluate Sparse-Matrix versions using explicitly sliced index permutations
    # (Since these are isolated local blocks, we must route the row/col mappings manually)
    @views mul!(y_test1[Bfly1m.permP], Bfly1m, x_in_U[Bfly1m.permQ])
    @views mul!(y_test2[Bfly2m.permP], Bfly2m, x_in_T[Bfly2m.permQ])
    @views mul!(y_test3[Bfly3m.permP], Bfly3m, x_in_T[Bfly3m.permQ])
    @views mul!(y_test4[Bfly4m.permP], Bfly4m, x_in_T2[Bfly4m.permQ])

    err1_mat = norm(y_test1 - y_exact1) / norm(y_exact1)
    err2_mat = norm(y_test2 - y_exact2) / norm(y_exact2)
    err3_mat = norm(y_test3 - y_exact3) / norm(y_exact3)
    err4_mat = norm(y_test4 - y_exact4) / norm(y_exact4)

    @info """
    --- Matrix-Vector Evaluation Accuracy ---
    Target Tolerance: $target_tol

    Configuration 1 (Balanced U -> T):
      Flat Array Error : $(round(err1_flat, sigdigits=4))
      Sparse Mat Error : $(round(err1_mat, sigdigits=4))

    Configuration 2 (Balanced T -> U):
      Flat Array Error : $(round(err2_flat, sigdigits=4))
      Sparse Mat Error : $(round(err2_mat, sigdigits=4))

    Configuration 3 (Unbalanced Deep->Shallow):
      Flat Array Error : $(round(err3_flat, sigdigits=4))
      Sparse Mat Error : $(round(err3_mat, sigdigits=4))

    Configuration 4 (Unbalanced Shallow->Deep):
      Flat Array Error : $(round(err4_flat, sigdigits=4))
      Sparse Mat Error : $(round(err4_mat, sigdigits=4))
    """

    # Assertions (Testing both Flat and Sparse versions against target error margin)
    @test err1_flat < target_tol * 10
    @test err2_flat < target_tol * 10
    @test err3_flat < target_tol * 10
    @test err4_flat < target_tol * 10

    @test err1_mat < target_tol * 10
    @test err2_mat < target_tol * 10
    @test err3_mat < target_tol * 10
    @test err4_mat < target_tol * 10
end
