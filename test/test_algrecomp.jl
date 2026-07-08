@testitem "Testing algebraic recompression of single butterfly compressed blocks" begin
    using Test
    using H2Trees
    using CompScienceMeshes
    using OhMyThreads
    using BEAST
    using ButterflyFactorizations
    using StaticArrays
    using LinearAlgebra

    #========================================================================
    =========================================================================
                            Geometry and Operators
    =========================================================================
    =========================================================================#

    lambda = 1.0
    k = 2 * pi / lambda
    x = meshsphere(0.5, lambda / 10)
    y = translate(x, SVector(5.0, 0.0, 0.0))
    x2 = meshsphere(0.5, lambda / 10)
    y2 = translate(x2, SVector(5.0, 0.0, 0.0))
    op = Maxwell3D.singlelayer(; wavenumber=k)
    T = raviartthomas(x)
    U = raviartthomas(y)
    T2 = raviartthomas(x2)
    U2 = raviartthomas(y2)

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

    x_s1 = A1 * x_t
    x_s2 = A2 * x_t
    x_s3 = A3 * x_t
    x_s4 = A4 * x_t2

    #========================================================================
    =========================================================================
                            Buttefly routines calling
    =========================================================================
    =========================================================================#

    Bfly1 = ButterflyFactorizations.subroutine_BF(
        farassembler1, tree1, 1, 1, k, 10^(-4); scheduler=OhMyThreads.DynamicScheduler()
    )
    size1 = Base.summarysize(Bfly1)
    Bfly2 = ButterflyFactorizations.subroutine_BF(
        farassembler2, tree2, 1, 1, k, 10^(-4); scheduler=OhMyThreads.DynamicScheduler()
    )
    size2 = Base.summarysize(Bfly2)
    Bfly3 = ButterflyFactorizations.subroutine_BF(
        farassembler3, tree3, 1, 1, k, 10^(-4); scheduler=OhMyThreads.DynamicScheduler()
    )
    size3 = Base.summarysize(Bfly3)
    Bfly4 = ButterflyFactorizations.subroutine_BF(
        farassembler4, tree4, 1, 1, k, 10^(-4); scheduler=OhMyThreads.DynamicScheduler()
    )
    size4 = Base.summarysize(Bfly4)

    RBfly1 = ButterflyFactorizations.recompress_BF(Bfly1, 10^(-2))
    size1r = Base.summarysize(RBfly1)
    RBfly2 = ButterflyFactorizations.recompress_BF(Bfly2, 10^(-2))
    size2r = Base.summarysize(RBfly2)
    RBfly3 = ButterflyFactorizations.recompress_BF(Bfly3, 10^(-2))
    size3r = Base.summarysize(RBfly3)
    RBfly4 = ButterflyFactorizations.recompress_BF(Bfly4, 10^(-2))
    size4r = Base.summarysize(RBfly4)

    @test size1r < size1
    @test size2r < size2
    @test size3r < size3
    @test size4r < size4

    x_test1 = zeros(ComplexF64, size(A1, 1))
    x_test2 = zeros(ComplexF64, size(A2, 1))
    x_test3 = zeros(ComplexF64, size(A3, 1))
    x_test4 = zeros(ComplexF64, size(A4, 1))

    @views mul!(x_test1, RBfly1, x_t)
    @views mul!(x_test2, RBfly2, x_t)
    @views mul!(x_test3, RBfly3, x_t)
    @views mul!(x_test4, RBfly4, x_t2)

    @test norm(x_test1 - x_s1) / norm(x_s1) < 10^(-1)
    @test norm(x_test2 - x_s2) / norm(x_s2) < 10^(-1)
    @test norm(x_test3 - x_s3) / norm(x_s3) < 10^(-1)
    @test norm(x_test4 - x_s4) / norm(x_s4) < 10^(-1)
end
