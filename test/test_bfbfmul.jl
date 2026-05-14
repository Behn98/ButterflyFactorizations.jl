@testitem "Testing Addition of Butterflies" begin
    using Test
    using H2Trees
    using CompScienceMeshes
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
    x = meshsphere(1.0, lambda / 10)
    y = translate(x, SVector(5.0, 0.0, 0.0))
    op = Maxwell3D.singlelayer(; wavenumber=k)
    T = raviartthomas(x)
    U = raviartthomas(y)

    #========================================================================
    =========================================================================
                    Tree construction  and Kernelmatrix assembly
    =========================================================================
    =========================================================================#

    tree1 = TwoNTree(T, U, lambda / 10)     #testspace, trialspace
    tree2 = TwoNTree(U, T, lambda / 10)

    go1 = H2Trees.values(tree1.testcluster, H2Trees.root(tree1.testcluster))
    go2 = H2Trees.values(tree2.testcluster, H2Trees.root(tree2.testcluster))

    gs1 = H2Trees.values(tree1.trialcluster, H2Trees.root(tree1.trialcluster))
    gs2 = H2Trees.values(tree2.trialcluster, H2Trees.root(tree2.trialcluster))

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

    #========================================================================
    =========================================================================
                        Assembly of Matrices and Vectors
    =========================================================================
    =========================================================================#

    A1 = assemble(op, T, U)
    A2 = assemble(op, U, T)

    x_t = randn(ComplexF64, length(T))

    x_s1 = (A1 * A2) * x_t

    #========================================================================
    =========================================================================
                            Buttefly routines calling
    =========================================================================
    =========================================================================#

    Bfly1 = ButterflyFactorizations.subroutine_BF(farassembler1, tree1, 1, 1, k, 10^(-4))
    Bfly2 = ButterflyFactorizations.subroutine_BF(farassembler2, tree2, 1, 1, k, 10^(-4))

    Bfly1A = ButterflyFactorizations.mulBFs(Bfly1, Bfly2, 10^-3)

    x_bfly1 = zeros(ComplexF64, size(Bfly1, 1))

    @views mul!(x_bfly1[go2], Bfly1A, x_t[gs1])

    @test norm(x_bfly1 - x_s1) / norm(x_s1) < 10^(-2)
end
