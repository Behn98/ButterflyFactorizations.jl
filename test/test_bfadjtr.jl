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

    #========================================================================
    =========================================================================
                            Geometry and Operators
    =========================================================================
    =========================================================================#
    lambda = 1.0
    k = 2 * pi / lambda
    x = meshsphere(0.25, lambda / 10)
    y = translate(x, SVector(5.0, 0.0, 0.0))
    x2 = meshsphere(0.25, lambda / 10)
    y2 = translate(x2, SVector(5.0, 0.0, 0.0))
    op = Maxwell3D.singlelayer(; wavenumber=k)
    T = raviartthomas(x)
    U = raviartthomas(y)
    T2 = raviartthomas(x2)
    U2 = raviartthomas(y2)
    length(T)
    length(T2)

    ##
    #========================================================================
    =========================================================================
                    Tree construction  and Kernelmatrix assembly
    =========================================================================
    =========================================================================#

    tree1 = TwoNTree(T, U, lambda / 10)     #testspace, trialspace
    tree2 = TwoNTree(U, T, lambda / 10)
    tree3 = TwoNTree(U2, T, lambda / 10)
    tree4 = TwoNTree(U, T2, lambda / 10)

    @views farasm = BEAST.blockassembler(op, T, U)
    @views function farassembler1(Z, tdata, sdata)
        @views store(v, m, n) = (Z[m, n] += v)
        return farasm(tdata, sdata, store)
    end

    @views farasm2 = BEAST.blockassembler(op, U, T)
    @views function farassembler2(Z, tdata, sdata)
        @views store(v, m, n) = (Z[m, n] += v)
        return farasm2(tdata, sdata, store)
    end

    @views farasm3 = BEAST.blockassembler(op, U2, T)
    @views function farassembler3(Z, tdata, sdata)
        @views store(v, m, n) = (Z[m, n] += v)
        return farasm3(tdata, sdata, store)
    end

    @views farasm4 = BEAST.blockassembler(op, U, T2)
    @views function farassembler4(Z, tdata, sdata)
        @views store(v, m, n) = (Z[m, n] += v)
        return farasm4(tdata, sdata, store)
    end

    #========================================================================
    =========================================================================
                        Assembly of Matrices and Vectors
    =========================================================================
    =========================================================================#
    A1 = assemble(op, T, U)
    A2 = assemble(op, U, T)
    A3 = assemble(op, U2, T)
    A4 = assemble(op, U, T2)

    x_t = randn(ComplexF64, length(T))
    x_t2 = randn(ComplexF64, length(T2))

    x_s1a = A1' * x_t
    x_s2a = A2' * x_t
    x_s3a = A3' * x_t2
    x_s4a = A4' * x_t

    x_s1t = transpose(A1) * x_t
    x_s2t = transpose(A2) * x_t
    x_s3t = transpose(A3) * x_t2
    x_s4t = transpose(A4) * x_t

    #========================================================================
    =========================================================================
                            Buttefly routines calling
    =========================================================================
    =========================================================================#

    Bfly1 = ButterflyFactorizations.subroutine_BF(
        farassembler1, tree1, 1, 1, k, 10^(-3); scheduler=OhMyThreads.DynamicScheduler()
    )
    Bfly2 = ButterflyFactorizations.subroutine_BF(
        farassembler2, tree2, 1, 1, k, 10^(-3); scheduler=OhMyThreads.DynamicScheduler()
    )
    Bfly3 = ButterflyFactorizations.subroutine_BF(
        farassembler3, tree3, 1, 1, k, 10^(-3); scheduler=OhMyThreads.DynamicScheduler()
    )
    Bfly4 = ButterflyFactorizations.subroutine_BF(
        farassembler4, tree4, 1, 1, k, 10^(-3); scheduler=OhMyThreads.DynamicScheduler()
    )

    Bfly1m = ButterflyFactorizations.subroutine_BF_mats(
        farassembler1, tree1, 1, 1, k, 10^(-3)
    )
    Bfly2m = ButterflyFactorizations.subroutine_BF_mats(
        farassembler2, tree2, 1, 1, k, 10^(-3)
    )
    Bfly3m = ButterflyFactorizations.subroutine_BF_mats(
        farassembler3, tree3, 1, 1, k, 10^(-3)
    )
    Bfly4m = ButterflyFactorizations.subroutine_BF_mats(
        farassembler4, tree4, 1, 1, k, 10^(-3)
    )

    x_test1 = zeros(ComplexF64, size(A1, 2))
    x_test2 = zeros(ComplexF64, size(A2, 2))
    x_test3 = zeros(ComplexF64, size(A3, 2))
    x_test4 = zeros(ComplexF64, size(A4, 2))

    @views mul!(x_test1, Bfly1', x_t)
    @views mul!(x_test2, Bfly2', x_t)
    @views mul!(x_test3, Bfly3', x_t2)
    @views mul!(x_test4, Bfly4', x_t)

    @test norm(x_test1 - x_s1a) / norm(x_s1a) < 10^(-2)
    @test norm(x_test2 - x_s2a) / norm(x_s2a) < 10^(-2)
    @test norm(x_test3 - x_s3a) / norm(x_s3a) < 10^(-2)
    @test norm(x_test4 - x_s4a) / norm(x_s4a) < 10^(-2)

    @views mul!(x_test1[Bfly1m'.PermP], Bfly1m', x_t[Bfly1m'.PermQ])
    @views mul!(x_test2[Bfly2m'.PermP], Bfly2m', x_t[Bfly2m'.PermQ])
    @views mul!(x_test3[Bfly3m'.PermP], Bfly3m', x_t2[Bfly3m'.PermQ])
    @views mul!(x_test4[Bfly4m'.PermP], Bfly4m', x_t[Bfly4m'.PermQ])

    @test norm(x_test1 - x_s1a) / norm(x_s1a) < 10^(-2)
    @test norm(x_test2 - x_s2a) / norm(x_s2a) < 10^(-2)
    @test norm(x_test3 - x_s3a) / norm(x_s3a) < 10^(-2)
    @test norm(x_test4 - x_s4a) / norm(x_s4a) < 10^(-2)

    @views mul!(x_test1, transpose(Bfly1), x_t)
    @views mul!(x_test2, transpose(Bfly2), x_t)
    @views mul!(x_test3, transpose(Bfly3), x_t2)
    @views mul!(x_test4, transpose(Bfly4), x_t)

    @test norm(x_test1 - x_s1t) / norm(x_s1t) < 10^(-2)
    @test norm(x_test2 - x_s2t) / norm(x_s2t) < 10^(-2)
    @test norm(x_test3 - x_s3t) / norm(x_s3t) < 10^(-2)
    @test norm(x_test4 - x_s4t) / norm(x_s4t) < 10^(-2)

    @views mul!(x_test1[Bfly1m.PermQ], transpose(Bfly1m), x_t[Bfly1m.PermP])
    @views mul!(x_test2[Bfly2m.PermQ], transpose(Bfly2m), x_t[Bfly2m.PermP])
    @views mul!(x_test3[Bfly3m.PermQ], transpose(Bfly3m), x_t2[Bfly3m.PermP])
    @views mul!(x_test4[Bfly4m.PermQ], transpose(Bfly4m), x_t[Bfly4m.PermP])

    @test norm(x_test1 - x_s1t) / norm(x_s1t) < 10^(-2)
    @test norm(x_test2 - x_s2t) / norm(x_s2t) < 10^(-2)
    @test norm(x_test3 - x_s3t) / norm(x_s3t) < 10^(-2)
    @test norm(x_test4 - x_s4t) / norm(x_s4t) < 10^(-2)
end
