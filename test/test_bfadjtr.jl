@testitem "Testing Adjoint and Transposed Butterflies" begin
    using Test
    using H2Trees
    using CompScienceMeshes
    using OhMyThreads
    using BEAST
    using ButterflyFactorizations
    using StaticArrays
    using LinearAlgebra
    using ParallelKMeans

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

    # Balanced
    tree1 = TwoNTree(T, U, lambda / 10)
    tree2 = TwoNTree(U, T, lambda / 10)
    # Unbalanced
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

    # Generate semantically correct input vectors matching the specific Codomain (Row) spaces
    # Because A' and transpose(A) map backwards: Row Space -> Column Space
    v_in_T = randn(ComplexF64, length(T))
    v_in_U = randn(ComplexF64, length(U))
    v_in_T2 = randn(ComplexF64, length(T2))
    v_in_U2 = randn(ComplexF64, length(U2))

    @info "Evaluating exact dense Adjoint and Transpose reference vectors..."
    # Adjoints (Hermitian Transpose)
    y_exact_adj1 = A1' * v_in_T  # Maps T  -> U
    y_exact_adj2 = A2' * v_in_U  # Maps U  -> T
    y_exact_adj3 = A3' * v_in_U2 # Maps U2 -> T
    y_exact_adj4 = A4' * v_in_U  # Maps U  -> T2

    # Transposes
    y_exact_trn1 = transpose(A1) * v_in_T
    y_exact_trn2 = transpose(A2) * v_in_U
    y_exact_trn3 = transpose(A3) * v_in_U2
    y_exact_trn4 = transpose(A4) * v_in_U

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
    # 5. Testing Adjoints (Flat & Sparse)
    # =========================================================================
    @info "Testing Adjoints (A')..."

    y_test1 = zeros(ComplexF64, size(A1, 2))
    y_test2 = zeros(ComplexF64, size(A2, 2))
    y_test3 = zeros(ComplexF64, size(A3, 2))
    y_test4 = zeros(ComplexF64, size(A4, 2))

    # Flat Array Adjoints
    mul!(y_test1, Bfly1', v_in_T)
    mul!(y_test2, Bfly2', v_in_U)
    mul!(y_test3, Bfly3', v_in_U2)
    mul!(y_test4, Bfly4', v_in_U)

    @test norm(y_test1 - y_exact_adj1) / norm(y_exact_adj1) < 1e-2
    @test norm(y_test2 - y_exact_adj2) / norm(y_exact_adj2) < 1e-2
    @test norm(y_test3 - y_exact_adj3) / norm(y_exact_adj3) < 1e-2
    @test norm(y_test4 - y_exact_adj4) / norm(y_exact_adj4) < 1e-2

    # Sparse Matrix Adjoints
    fill!.((y_test1, y_test2, y_test3, y_test4), 0.0)
    B1m_adj, B2m_adj, B3m_adj, B4m_adj = Bfly1m', Bfly2m', Bfly3m', Bfly4m'

    # Note: We query permP and permQ on the newly returned adjoint objects
    @views mul!(y_test1[B1m_adj.permP], B1m_adj, v_in_T[B1m_adj.permQ])
    @views mul!(y_test2[B2m_adj.permP], B2m_adj, v_in_U[B2m_adj.permQ])
    @views mul!(y_test3[B3m_adj.permP], B3m_adj, v_in_U2[B3m_adj.permQ])
    @views mul!(y_test4[B4m_adj.permP], B4m_adj, v_in_U[B4m_adj.permQ])

    @test norm(y_test1 - y_exact_adj1) / norm(y_exact_adj1) < 1e-2
    @test norm(y_test2 - y_exact_adj2) / norm(y_exact_adj2) < 1e-2
    @test norm(y_test3 - y_exact_adj3) / norm(y_exact_adj3) < 1e-2
    @test norm(y_test4 - y_exact_adj4) / norm(y_exact_adj4) < 1e-2

    # =========================================================================
    # 6. Testing Transposes (Flat & Sparse)
    # =========================================================================
    @info "Testing Transposes (transpose(A))..."

    fill!.((y_test1, y_test2, y_test3, y_test4), 0.0)

    # Flat Array Transposes
    mul!(y_test1, transpose(Bfly1), v_in_T)
    mul!(y_test2, transpose(Bfly2), v_in_U)
    mul!(y_test3, transpose(Bfly3), v_in_U2)
    mul!(y_test4, transpose(Bfly4), v_in_U)

    @test norm(y_test1 - y_exact_trn1) / norm(y_exact_trn1) < 1e-2
    @test norm(y_test2 - y_exact_trn2) / norm(y_exact_trn2) < 1e-2
    @test norm(y_test3 - y_exact_trn3) / norm(y_exact_trn3) < 1e-2
    @test norm(y_test4 - y_exact_trn4) / norm(y_exact_trn4) < 1e-2

    # Sparse Matrix Transposes
    fill!.((y_test1, y_test2, y_test3, y_test4), 0.0)
    B1m_trn, B2m_trn, B3m_trn, B4m_trn = transpose(Bfly1m),
    transpose(Bfly2m), transpose(Bfly3m),
    transpose(Bfly4m)

    @views mul!(y_test1[B1m_trn.permP], B1m_trn, v_in_T[B1m_trn.permQ])
    @views mul!(y_test2[B2m_trn.permP], B2m_trn, v_in_U[B2m_trn.permQ])
    @views mul!(y_test3[B3m_trn.permP], B3m_trn, v_in_U2[B3m_trn.permQ])
    @views mul!(y_test4[B4m_trn.permP], B4m_trn, v_in_U[B4m_trn.permQ])

    @test norm(y_test1 - y_exact_trn1) / norm(y_exact_trn1) < 1e-2
    @test norm(y_test2 - y_exact_trn2) / norm(y_exact_trn2) < 1e-2
    @test norm(y_test3 - y_exact_trn3) / norm(y_exact_trn3) < 1e-2
    @test norm(y_test4 - y_exact_trn4) / norm(y_exact_trn4) < 1e-2
end
