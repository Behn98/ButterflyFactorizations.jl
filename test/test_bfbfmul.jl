@testitem "Testing Addition of Butterflies" begin
    using Test
    using H2Trees
    using OhMyThreads
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

    go1 = H2Trees.values(tree1.testcluster, H2Trees.root(tree1.testcluster))

    gs1 = H2Trees.values(tree1.trialcluster, H2Trees.root(tree1.trialcluster))

    @views farasm = BEAST.blockassembler(op, T, U)
    @views function farassembler1(Z, tdata, sdata)
        @views store(v, m, n) = (Z[m, n] += v)
        return farasm(tdata, sdata, store)
    end

    #========================================================================
    =========================================================================
                        Assembly of Matrices and Vectors
    =========================================================================
    =========================================================================#

    A1 = assemble(op, T, U)

    x_t = randn(ComplexF64, length(T))

    x_s1 = (A1 * A1) * x_t

    #========================================================================
    =========================================================================
                            Buttefly routines calling
    =========================================================================
    =========================================================================#

    Bfly1 = ButterflyFactorizations.subroutine_BF(farassembler1, tree1, 1, 1, k, 10^(-4))

    Bfly1A = ButterflyFactorizations.mulBFs(Bfly1, Bfly1, 10^-3)

    x_bfly1 = zeros(ComplexF64, size(Bfly1, 1))

    @views mul!(x_bfly1[go1], Bfly1A, x_t[gs1])

    @test norm(x_bfly1 - x_s1) / norm(x_s1) < 10^(-2)
end
