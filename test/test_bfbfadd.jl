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

    go1 = H2Trees.values(tree1.testcluster, H2Trees.root(tree1.testcluster))
    go2 = H2Trees.values(tree2.testcluster, H2Trees.root(tree2.testcluster))
    go3 = H2Trees.values(tree3.testcluster, H2Trees.root(tree3.testcluster))
    go4 = H2Trees.values(tree4.testcluster, H2Trees.root(tree4.testcluster))

    gs1 = H2Trees.values(tree1.trialcluster, H2Trees.root(tree1.trialcluster))
    gs2 = H2Trees.values(tree2.trialcluster, H2Trees.root(tree2.trialcluster))
    gs3 = H2Trees.values(tree3.trialcluster, H2Trees.root(tree3.trialcluster))
    gs4 = H2Trees.values(tree4.trialcluster, H2Trees.root(tree4.trialcluster))

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

    x_s1 = (A1 + A1) * x_t
    x_s2 = (A2 + A2) * x_t
    x_s3 = (A3 + A3) * x_t
    x_s4 = (A4 + A4) * x_t2

    #========================================================================
    =========================================================================
                            Buttefly routines calling
    =========================================================================
    =========================================================================#

    Bfly1 = ButterflyFactorizations.subroutine_BF(farassembler1, tree1, 1, 1, k, 10^(-4))
    Bfly2 = ButterflyFactorizations.subroutine_BF(farassembler2, tree2, 1, 1, k, 10^(-4))
    Bfly3 = ButterflyFactorizations.subroutine_BF(farassembler3, tree3, 1, 1, k, 10^(-4))
    Bfly4 = ButterflyFactorizations.subroutine_BF(farassembler4, tree4, 1, 1, k, 10^(-4))

    Bfly1A = ButterflyFactorizations.add_eqbfs(Bfly1, Bfly1, 10^-3)
    Bfly2A = ButterflyFactorizations.add_eqbfs(Bfly2, Bfly2, 10^-3)
    Bfly3A = ButterflyFactorizations.add_eqbfs(Bfly3, Bfly3, 10^-3)
    Bfly4A = ButterflyFactorizations.add_eqbfs(Bfly4, Bfly4, 10^-3)

    x_bfly1 = zeros(ComplexF64, size(Bfly1, 1))
    x_bfly2 = zeros(ComplexF64, size(Bfly2, 1))
    x_bfly3 = zeros(ComplexF64, size(Bfly3, 1))
    x_bfly4 = zeros(ComplexF64, size(Bfly4, 1))

    @views mul!(x_bfly1[go1], Bfly1A, x_t[gs1])
    @views mul!(x_bfly2[go2], Bfly2A, x_t[gs2])
    @views mul!(x_bfly3[go3], Bfly3A, x_t[gs3])
    @views mul!(x_bfly4[go4], Bfly4A, x_t2[gs4])

    @test norm(x_bfly1 - x_s1) / norm(x_s1) < 10^(-3)
    @test norm(x_bfly2 - x_s2) / norm(x_s2) < 10^(-3)
    @test norm(x_bfly3 - x_s3) / norm(x_s3) < 10^(-3)
    @test norm(x_bfly4 - x_s4) / norm(x_s4) < 10^(-3)
end
