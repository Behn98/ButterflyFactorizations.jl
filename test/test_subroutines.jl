@testitem "Testing Blockassembly subroutines" begin
    using Test
    using H2Trees
    using CompScienceMeshes
    using BEAST
    using OhMyThreads
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
    length(T2)
    ##
    #========================================================================
    =========================================================================
                    Tree construction  and Kernelmatrix assembly
    =========================================================================
    =========================================================================#

    tree1 = TwoNTree(T, U, lambda / 10)     #testspace, trialspace
    tree2 = TwoNTree(U, T, lambda / 10)
    tree31 = TwoNTree(U2.pos, lambda / 2.5)
    tree32 = TwoNTree(T.pos, lambda / 10)
    tree3 = BlockTree(tree31, tree32)
    tree41 = TwoNTree(U.pos, lambda / 10)
    tree42 = TwoNTree(T2.pos, lambda / 2.5)
    tree4 = BlockTree(tree41, tree42)

    go1 = H2Trees.values(tree1.testcluster, H2Trees.root(tree1.testcluster))
    go2 = H2Trees.values(tree2.testcluster, H2Trees.root(tree2.testcluster))
    go3 = H2Trees.values(tree3.testcluster, H2Trees.root(tree3.testcluster))
    go4 = H2Trees.values(tree4.testcluster, H2Trees.root(tree4.testcluster))

    gs1 = H2Trees.values(tree1.trialcluster, H2Trees.root(tree1.trialcluster))
    gs2 = H2Trees.values(tree2.trialcluster, H2Trees.root(tree2.trialcluster))
    gs3 = H2Trees.values(tree3.trialcluster, H2Trees.root(tree3.trialcluster))
    gs4 = H2Trees.values(tree4.trialcluster, H2Trees.root(tree4.trialcluster))

    farassembler1 = ButterflyFactorizations.AbstractKernelMatrix(op, T, U)
    #=
    @views function farassembler1(Z, tdata, sdata)
        @views store(v, m, n) = (Z[m, n] += v)

        # Skapa blockassemblern LOKALT inuti anropet så att den är thread-safe!
        farasm_local = BEAST.blockassembler(op, T, U)
        return farasm_local(tdata, sdata, store)
    end
    =#
    farassembler2 = ButterflyFactorizations.AbstractKernelMatrix(op, U, T)
    #=
    @views function farassembler2(Z, tdata, sdata)
        @views store(v, m, n) = (Z[m, n] += v)

        # Skapa blockassemblern LOKALT inuti anropet så att den är thread-safe!
        farasm_local = BEAST.blockassembler(op, U, T)
        return farasm_local(tdata, sdata, store)
    end
    =#
    farassembler3 = ButterflyFactorizations.AbstractKernelMatrix(op, U2, T)
    #=
    @views function farassembler3(Z, tdata, sdata)
        @views store(v, m, n) = (Z[m, n] += v)

        # Skapa blockassemblern LOKALT inuti anropet så att den är thread-safe!
        farasm_local = BEAST.blockassembler(op, U2, T)
        return farasm_local(tdata, sdata, store)
    end
    =#
    farassembler4 = ButterflyFactorizations.AbstractKernelMatrix(op, U, T2)
    #=
    @views function farassembler4(Z, tdata, sdata)
        @views store(v, m, n) = (Z[m, n] += v)

        # Skapa blockassemblern LOKALT inuti anropet så att den är thread-safe!
        farasm_local = BEAST.blockassembler(op, U, T2)
        return farasm_local(tdata, sdata, store)
    end
    =#
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

    Bfly1 = ButterflyFactorizations.subroutine_BF(farassembler1, tree1, 1, 1, k, 10^(-3))
    Bfly2 = ButterflyFactorizations.subroutine_BF(farassembler2, tree2, 1, 1, k, 10^(-3))
    Bfly3 = ButterflyFactorizations.subroutine_BF(farassembler3, tree3, 1, 1, k, 10^(-3))
    Bfly4 = ButterflyFactorizations.subroutine_BF(farassembler4, tree4, 1, 1, k, 10^(-3))

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

    x_test1 = zeros(ComplexF64, size(A1, 1))
    x_test2 = zeros(ComplexF64, size(A2, 1))
    x_test3 = zeros(ComplexF64, size(A3, 1))
    x_test4 = zeros(ComplexF64, size(A4, 1))

    @views mul!(x_test1[go1], Bfly1, x_t[gs1])
    @views mul!(x_test2[go2], Bfly2, x_t[gs2])
    @views mul!(x_test3[go3], Bfly3, x_t[gs3])
    @views mul!(x_test4[go4], Bfly4, x_t2[gs4])

    @test norm(x_test1 - x_s1) / norm(x_s1) < 10^(-2)
    @test norm(x_test2 - x_s2) / norm(x_s2) < 10^(-2)
    @test norm(x_test3 - x_s3) / norm(x_s3) < 10^(-2)
    @test norm(x_test4 - x_s4) / norm(x_s4) < 10^(-2)

    @views mul!(x_test1[Bfly1m.PermP], Bfly1m, x_t[Bfly1m.PermQ])
    @views mul!(x_test2[Bfly2m.PermP], Bfly2m, x_t[Bfly2m.PermQ])
    @views mul!(x_test3[Bfly3m.PermP], Bfly3m, x_t[Bfly3m.PermQ])
    @views mul!(x_test4[Bfly4m.PermP], Bfly4m, x_t2[Bfly4m.PermQ])

    @test norm(x_test1 - x_s1) / norm(x_s1) < 10^(-2)
    @test norm(x_test2 - x_s2) / norm(x_s2) < 10^(-2)
    @test norm(x_test3 - x_s3) / norm(x_s3) < 10^(-2)
    @test norm(x_test4 - x_s4) / norm(x_s4) < 10^(-2)
end
