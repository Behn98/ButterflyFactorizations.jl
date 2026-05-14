@testitem "Testing custom Blockconstructions" begin
    using ButterflyFactorizations
    using LinearAlgebra
    using SparseArrays
    using BlockSparseMatrices
    using Test

    M1 = rand(ComplexF64, 14, 12)
    M2 = rand(ComplexF64, 12, 10)
    M = ButterflyFactorizations.blockdiag(M1, M2)
    @test size(M) == (26, 22)

    M = SparseArrays.sparse(M)
    sparseM = ButterflyFactorizations.sparse_blockdiag(M, M2)
    @test size(sparseM) == (38, 32)
    @test issparse(sparseM)
    storedentries = size(M1)[1] * size(M1)[2] + (size(M2)[1] * size(M2)[2]) * 2
    @test nnz(sparseM) == storedentries

    M3 = hcat(M[1:size(M2)[1], :], M2)
    sparseM3 = ButterflyFactorizations.sparse_vcat(M3, sparseM)
    @test size(sparseM3) == (50, 32)
    @test issparse(sparseM3)
    @test nnz(sparseM3) == nnz(M3) + nnz(sparseM)

    M1 = BlockSparseMatrix([M1], [1:14], [1:12], (14, 12))
    M2 = BlockSparseMatrix([M2], [1:12], [1:10], (12, 10))
    M = ButterflyFactorizations.blocksparse_blockdiag(M1, M2)
    @test size(M) == (26, 22)
    @test size(M.blocks)[1] == 2
    @test size(M.blocks[1]) == (14, 12)
    @test size(M.blocks[2]) == (12, 10)
    @test M.colindices[1] == 1:12
    @test M.colindices[2] == 13:22
    @test M.rowindices[1] == 1:14
    @test M.rowindices[2] == 15:26

    M = ButterflyFactorizations.blocksparse_vcat(M1[:, 1:10], M2)
    @test size(M) == (26, 10)
    @test size(M.blocks)[1] == 2
    @test size(M.blocks[1]) == (14, 10)
    @test size(M.blocks[2]) == (12, 10)
    @test M.rowindices[1] == 1:14
    @test M.rowindices[2] == 15:26
end

@testitem "Testing Subdictionary functionality" begin
    using Test
    using Random
    using ButterflyFactorizations
    R1 = Dict{Int,Dict{Int,Matrix{ComplexF64}}}()
    R2 = Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},Matrix{ComplexF64}}}()
    idx1 = rand(Int, 10)
    entry = [rand(ComplexF64, 2, 2) for _ in 1:10]
    idx2 = [(rand(Int), rand(Int)) for _ in 1:10]
    for i in randperm(10)
        R1[idx1[i]] = Dict{Int,Matrix{ComplexF64}}()
        R1[idx1[i]][idx1[11 - i]] = entry[i]
        R2[idx2[i]] = Dict{Tuple{Int,Int},Matrix{ComplexF64}}()
        R2[idx2[i]][idx2[11 - i]] = entry[i]
    end
    for i in 1:10
        @test ButterflyFactorizations.getsubdict!(R1, idx1[i]) == R1[idx1[i]]
        @test ButterflyFactorizations.getsubdict!(R2, idx2[i]) == R2[idx2[i]]
    end

    rows = []
    col_idx = idx2[2]
    for row in keys(R2)
        if haskey(R2[row], col_idx)
            push!(rows, row)
        end
    end
    @test rows == ButterflyFactorizations.find_rows_for_column(R2, col_idx)
    rows = []
    col_idx = idx1[2]
    for row in keys(R1)
        if haskey(R1[row], col_idx)
            push!(rows, row)
        end
    end
    @test rows == ButterflyFactorizations.find_rows_for_column(R1, col_idx)
end

@testitem "Testing Subdictionary functionality" begin
    using Test
    using H2Trees
    using CompScienceMeshes
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
    x = meshsphere(1.0, lambda / 10)
    y = translate(x, SVector(5.0, 0.0, 0.0))
    op = Maxwell3D.singlelayer(; wavenumber=k)
    T = raviartthomas(x)
    U = raviartthomas(y)

    ##
    #========================================================================
    =========================================================================
                                Tree construction
    =========================================================================
    =========================================================================#

    tree1 = H2Trees.KMeansTree(T.pos, 2; minvalues=100)
    tree1 = H2Trees.BlockTree(tree1, tree1)
    tree2 = TwoNTree(T, U, lambda / 10)
    tree3 = TwoNTree(T, T, lambda / 10)
    for i in eachindex(tree1.testcluster.nodesatlevel)
        @test issetequal(
            ButterflyFactorizations.h2treelevels(
                tree1.testcluster, H2Trees.root(tree1.testcluster)
            )[i],
            tree1.testcluster.nodesatlevel[i],
        )
    end
    for i in eachindex(tree1.trialcluster.nodesatlevel)
        @test issetequal(
            ButterflyFactorizations.h2treelevels(
                tree1.trialcluster, H2Trees.root(tree1.trialcluster)
            )[i],
            tree1.trialcluster.nodesatlevel[i],
        )
    end
    for i in eachindex(tree2.testcluster.nodesatlevel)
        @test issetequal(
            ButterflyFactorizations.h2treelevels(
                tree2.testcluster, H2Trees.root(tree2.testcluster)
            )[i],
            tree2.testcluster.nodesatlevel[i],
        )
    end
    for i in eachindex(tree2.trialcluster.nodesatlevel)
        @test issetequal(
            ButterflyFactorizations.h2treelevels(
                tree2.trialcluster, H2Trees.root(tree2.trialcluster)
            )[i],
            tree2.trialcluster.nodesatlevel[i],
        )
    end
    for i in eachindex(tree3.testcluster.nodesatlevel)
        @test issetequal(
            ButterflyFactorizations.h2treelevels(
                tree3.testcluster, H2Trees.root(tree3.testcluster)
            )[i],
            tree3.testcluster.nodesatlevel[i],
        )
    end
    for i in eachindex(tree3.trialcluster.nodesatlevel)
        @test issetequal(
            ButterflyFactorizations.h2treelevels(
                tree3.trialcluster, H2Trees.root(tree3.trialcluster)
            )[i],
            tree3.trialcluster.nodesatlevel[i],
        )
    end

    leaves = H2Trees.leaves
    paddedtlvls = ButterflyFactorizations.traverseandpad(
        tree1.testcluster, H2Trees.root(tree1.testcluster)
    )
    ml = length(paddedtlvls)
    @test issetequal(paddedtlvls[ml], leaves(tree1.testcluster))
end
