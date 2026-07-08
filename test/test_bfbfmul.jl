@testitem "Testing Multiplication of Butterflies" begin
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
    x = meshsphere(0.25, lambda / 10)
    y = translate(x, SVector(5.0, 0.0, 0.0))
    z = translate(x, SVector(8.0, 0.0, 0.0))
    op = Maxwell3D.singlelayer(; wavenumber=k)
    T = raviartthomas(x)
    U = raviartthomas(y)
    V = raviartthomas(z)
    #========================================================================
    =========================================================================
                    Tree construction  and Kernelmatrix assembly
    =========================================================================
    =========================================================================#

    tree1 = TwoNTree(T, U, lambda / 10)     #testspace, trialspace
    tree2 = TwoNTree(U, V, lambda / 10)

    @views farasm1 = BEAST.blockassembler(op, T, U)
    @views function farassembler1(Z, tdata, sdata)
        @views store(v, m, n) = (Z[m, n] += v)
        return farasm1(tdata, sdata, store)
    end

    @views farasm2 = BEAST.blockassembler(op, U, V)
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
    A2 = assemble(op, U, V)
    x_t = randn(ComplexF64, length(T))

    x_s1 = A1 * (A2 * x_t)

    #========================================================================
    =========================================================================
                            Buttefly routines calling
    =========================================================================
    =========================================================================#

    Bfly1 = ButterflyFactorizations.subroutine_BF(farassembler1, tree1, 1, 1, k, 10^(-4))
    Bfly2 = ButterflyFactorizations.subroutine_BF(farassembler2, tree2, 1, 1, k, 10^(-4))

    Bfly1A = ButterflyFactorizations.mulBFs(Bfly1, Bfly2, 10^-4)

    x_bfly1 = zeros(ComplexF64, size(Bfly1, 1))

    @views mul!(x_bfly1, Bfly1A, x_t)

    @test norm(x_bfly1 - x_s1) / norm(x_s1) < 10^(-3)
end
#=
@testitem "Testing Multiplication of a low rank Butterfly cluster with a higher rank Butterfly" begin
    using H2Trees
    using Test
    using CompScienceMeshes
    using BEAST
    using ButterflyFactorizations
    using StaticArrays
    using LinearAlgebra
    using ParallelKMeans
    using BlockSparseMatrices
    using LinearMaps
    using Random
    using OhMyThreads

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
    kernelmatrix1 = ButterflyFactorizations.AbstractKernelMatrix(op, T, U)
    firstlvl = collect(H2Trees.children(tree1.trialcluster, 1))
    parent1 = firstlvl[1]
    parent2 = firstlvl[2]
    schildren = collect(H2Trees.children(tree1.trialcluster, parent2))
    gsc = sort!(H2Trees.values(tree1.trialcluster, parent2))
    ochildren = collect(H2Trees.children(tree1.testcluster, parent2))
    goc = sort!(H2Trees.values(tree1.testcluster, parent2))
    Bfcluster = Matrix{ButterflyFactorizations.BF}(
        undef, length(ochildren), length(schildren)
    )
    higherkBF = ButterflyFactorizations.subroutine_BF(
        kernelmatrix1,
        tree1,
        parent2,
        parent1,
        k,
        1e-4;
        compressor=ButterflyFactorizations.PartialQR(),
        scheduler=OhMyThreads.DynamicScheduler(),
    )
    gsk = sort!(H2Trees.values(tree1.trialcluster, 2))
    gok = sort!(H2Trees.values(tree1.testcluster, 73))
    for (i, oc) in enumerate(ochildren)
        for (j, sc) in enumerate(schildren)
            Bfcluster[i, j] = ButterflyFactorizations.subroutine_BF(
                kernelmatrix1,
                tree1,
                oc,
                sc,
                k,
                1e-4;
                compressor=ButterflyFactorizations.PartialQR(),
                scheduler=OhMyThreads.DynamicScheduler(),
            )
        end
    end
    A1 = zeros(ComplexF64, length(goc), length(gsc))
    kernelmatrix1(A1, goc, gsc)
    A2 = zeros(ComplexF64, length(gok), length(gsk))
    kernelmatrix1(A2, gok, gsk)
    x_t = randn(ComplexF64, length(T))
    y_exact1 = zeros(ComplexF64, length(T))
    y_exact2 = zeros(ComplexF64, length(T))
    y_exact1[goc] = A1 * x_t[gsc]
    y_exact2[gok] = A2 * x_t[gsk]

    ycluster = zeros(ComplexF64, length(T))
    for i in eachindex(ochildren)
        for j in eachindex(schildren)
            ycluster .+= Bfcluster[i, j] * x_t
        end
    end

    @test reldif1 = norm(y_exact1 - ycluster) / norm(y_exact1) < 1e-3

    ycluster2 = higherkBF * x_t
    @test reldif2 = norm(y_exact2 - ycluster2) / norm(y_exact2) < 1e-3

    y_exact3 = zeros(ComplexF64, length(T))
    y_exact3[goc] = A1 * y_exact2[gsc]
    ycluster3 = zeros(ComplexF64, length(T))
    for i in eachindex(ochildren)
        for j in eachindex(schildren)
            ycluster3 .+= Bfcluster[i, j] * ycluster2
        end
    end
    @test reldif3 = norm(y_exact3 - ycluster3) / norm(y_exact3) < 1e-3

    splitbfprod = ButterflyFactorizations.splitmulbf(Bfcluster, higherkBF, 1e-2)
    y_cluster4 = splitbfprod * x_t

    @test reldif4 = norm(y_exact3 - y_cluster4) / norm(y_exact3) < 1e-1
end
=#
