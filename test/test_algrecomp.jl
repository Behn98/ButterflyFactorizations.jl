@testitem "Testing algebraic recompression of single butterfly compressed blocks" begin
    using Test
    using H2Trees
    using CompScienceMeshes
    using OhMyThreads
    using BEAST
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

    x2 = meshsphere(0.25, lambda / 10)
    y2 = translate(x2, SVector(5.0, 0.0, 0.0))

    op = Maxwell3D.singlelayer(; wavenumber=k)

    T = raviartthomas(x)
    U = raviartthomas(y)
    T2 = raviartthomas(x2)
    U2 = raviartthomas(y2)

    @info "Degrees of Freedom: T=$(length(T)), U=$(length(U)), T2=$(length(T2)), U2=$(length(U2))"

    # =========================================================================
    # 2. Tree Construction and AbstractKernelMatrix Assembly
    # =========================================================================
    @info "Constructing hierarchical block trees and AbstractKernelMatrices..."
    tree1 = TwoNTree(T, U, lambda / 10)
    tree2 = TwoNTree(U, T, lambda / 10)
    tree3 = TwoNTree(U2, T, lambda / 10)
    tree4 = TwoNTree(U, T2, lambda / 10)

    # 🚀 REPLACED: Manual BEAST.blockassembler closures replaced with AbstractKernelMatrix
    farassembler1 = ButterflyFactorizations.AbstractKernelMatrix(op, T, U)
    farassembler2 = ButterflyFactorizations.AbstractKernelMatrix(op, U, T)
    farassembler3 = ButterflyFactorizations.AbstractKernelMatrix(op, U2, T)
    farassembler4 = ButterflyFactorizations.AbstractKernelMatrix(op, U, T2)

    # =========================================================================
    # 3. Dense Ground Truth Assembly
    # =========================================================================
    @info "Assembling Dense Ground Truth Matrices..."
    A1 = assemble(op, T, U)  # Maps U  -> T
    A2 = assemble(op, U, T)  # Maps T  -> U
    A3 = assemble(op, U2, T) # Maps T  -> U2
    A4 = assemble(op, U, T2) # Maps T2 -> U

    # Generate semantically correct input vectors matching the Source (Trial) spaces
    v_in_U = randn(ComplexF64, length(U))
    v_in_T = randn(ComplexF64, length(T))
    v_in_T2 = randn(ComplexF64, length(T2))

    y_exact1 = A1 * v_in_U
    y_exact2 = A2 * v_in_T
    y_exact3 = A3 * v_in_T
    y_exact4 = A4 * v_in_T2

    # =========================================================================
    # 4. Assembly of High-Fidelity Butterfly Factorizations
    # =========================================================================
    tol_init = 1e-4

    @info "Assembling Initial High-Fidelity Butterfly Blocks (tol = $tol_init)..."
    Bfly1 = ButterflyFactorizations.assemble_BF(
        farassembler1, tree1, 1, 1, k, tol_init; scheduler=OhMyThreads.StaticScheduler()
    )
    size1 = Base.summarysize(Bfly1)

    Bfly2 = ButterflyFactorizations.assemble_BF(
        farassembler2, tree2, 1, 1, k, tol_init; scheduler=OhMyThreads.StaticScheduler()
    )
    size2 = Base.summarysize(Bfly2)

    Bfly3 = ButterflyFactorizations.assemble_BF(
        farassembler3, tree3, 1, 1, k, tol_init; scheduler=OhMyThreads.StaticScheduler()
    )
    size3 = Base.summarysize(Bfly3)

    Bfly4 = ButterflyFactorizations.assemble_BF(
        farassembler4, tree4, 1, 1, k, tol_init; scheduler=OhMyThreads.StaticScheduler()
    )
    size4 = Base.summarysize(Bfly4)

    # =========================================================================
    # 5. Algebraic Recompression
    # =========================================================================
    tol_recomp = 1e-2

    @info "Applying Algebraic Recompression (tol = $tol_recomp)..."
    RBfly1 = ButterflyFactorizations.recompress_BF(Bfly1, tol_recomp)
    size1r = Base.summarysize(RBfly1)

    RBfly2 = ButterflyFactorizations.recompress_BF(Bfly2, tol_recomp)
    size2r = Base.summarysize(RBfly2)

    RBfly3 = ButterflyFactorizations.recompress_BF(Bfly3, tol_recomp)
    size3r = Base.summarysize(RBfly3)

    RBfly4 = ButterflyFactorizations.recompress_BF(Bfly4, tol_recomp)
    size4r = Base.summarysize(RBfly4)

    @info """
    --- Recompression Memory Savings ---
    BF1 (U  -> T) : $(round(size1 / 1024^2, digits=3)) MB  ->  $(round(size1r / 1024^2, digits=3)) MB
    BF2 (T  -> U) : $(round(size2 / 1024^2, digits=3)) MB  ->  $(round(size2r / 1024^2, digits=3)) MB
    BF3 (T  -> U2): $(round(size3 / 1024^2, digits=3)) MB  ->  $(round(size3r / 1024^2, digits=3)) MB
    BF4 (T2 -> U) : $(round(size4 / 1024^2, digits=3)) MB  ->  $(round(size4r / 1024^2, digits=3)) MB
    """

    # Recompression should strictly reduce or maintain memory size
    @test size1r <= size1
    @test size2r <= size2
    @test size3r <= size3
    @test size4r <= size4

    # =========================================================================
    # 6. Verification & Accuracy Profiling
    # =========================================================================
    @info "Evaluating Recompressed Matrix-Vector Products..."

    y_test1 = zeros(ComplexF64, size(A1, 1))
    y_test2 = zeros(ComplexF64, size(A2, 1))
    y_test3 = zeros(ComplexF64, size(A3, 1))
    y_test4 = zeros(ComplexF64, size(A4, 1))

    # Evaluate the algebraically truncated operators
    mul!(y_test1, RBfly1, v_in_U)
    mul!(y_test2, RBfly2, v_in_T)
    mul!(y_test3, RBfly3, v_in_T)
    mul!(y_test4, RBfly4, v_in_T2)

    err1 = norm(y_test1 - y_exact1) / norm(y_exact1)
    err2 = norm(y_test2 - y_exact2) / norm(y_exact2)
    err3 = norm(y_test3 - y_exact3) / norm(y_exact3)
    err4 = norm(y_test4 - y_exact4) / norm(y_exact4)

    @info """
    --- Matrix-Vector Evaluation Accuracy ---
    Recompression Target: $tol_recomp
    Error BF1           : $(round(err1, sigdigits=4))
    Error BF2           : $(round(err2, sigdigits=4))
    Error BF3           : $(round(err3, sigdigits=4))
    Error BF4           : $(round(err4, sigdigits=4))
    """

    # The error of the recompressed operator is bounded by the recompression
    # tolerance `tol_recomp`, not the initial high-fidelity assembly tolerance.
    @test err1 < tol_recomp * 10
    @test err2 < tol_recomp * 10
    @test err3 < tol_recomp * 10
    @test err4 < tol_recomp * 10
end
