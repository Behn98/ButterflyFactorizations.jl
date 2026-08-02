@testitem "Testing Auxiliary Matrix & Block Constructions" begin
    using Test
    using ButterflyFactorizations
    using LinearAlgebra
    using SparseArrays
    using BlockSparseMatrices

    @info "Testing Dense Block Diagonalization..."
    M1 = rand(ComplexF64, 14, 12)
    M2 = rand(ComplexF64, 12, 10)

    M_diag = ButterflyFactorizations.blockdiag(M1, M2)
    @test size(M_diag) == (26, 22)

    @info "Testing Sparse Block Manipulations..."
    M_sparse1 = SparseArrays.sparse(M_diag)

    # Sparse Block Diagonal
    sparse_M = ButterflyFactorizations.sparse_blockdiag(M_sparse1, M2)
    @test size(sparse_M) == (38, 32)
    @test issparse(sparse_M)

    # Check non-zero entries (nnz) are perfectly preserved
    stored_entries = size(M1, 1) * size(M1, 2) + (size(M2, 1) * size(M2, 2)) * 2
    @test nnz(sparse_M) == stored_entries

    # Sparse Vertical Concatenation
    M3 = hcat(M_diag[1:size(M2, 1), :], M2)
    sparse_M3 = ButterflyFactorizations.sparse_vcat(M3, sparse_M)
    @test size(sparse_M3) == (50, 32)
    @test issparse(sparse_M3)

    # 🚀 FIX: Convert dense M3 to sparse just to safely query `nnz`
    @test nnz(sparse_M3) == nnz(sparse(M3)) + nnz(sparse_M)

    @info "Testing BlockSparseMatrix Concatenations..."
    BSM1 = BlockSparseMatrix([M1], [1:14], [1:12], (14, 12))
    BSM2 = BlockSparseMatrix([M2], [1:12], [1:10], (12, 10))

    # BlockSparse Block Diagonal
    BSM_diag = ButterflyFactorizations.blocksparse_blockdiag(BSM1, BSM2)
    @test size(BSM_diag) == (26, 22)
    @test length(BSM_diag.blocks) == 2
    @test size(BSM_diag.blocks[1]) == (14, 12)
    @test size(BSM_diag.blocks[2]) == (12, 10)
    @test BSM_diag.colindices[1] == 1:12
    @test BSM_diag.colindices[2] == 13:22
    @test BSM_diag.rowindices[1] == 1:14
    @test BSM_diag.rowindices[2] == 15:26

    # BlockSparse Vertical Concatenation
    BSM_vcat = ButterflyFactorizations.blocksparse_vcat(BSM1[:, 1:10], BSM2)
    @test size(BSM_vcat) == (26, 10)
    @test length(BSM_vcat.blocks) == 2
    @test size(BSM_vcat.blocks[1]) == (14, 10)
    @test size(BSM_vcat.blocks[2]) == (12, 10)
    @test BSM_vcat.rowindices[1] == 1:14
    @test BSM_vcat.rowindices[2] == 15:26
end

@testitem "Testing Tree Interface and Traversal Utilities" begin
    using Test
    using H2Trees
    using CompScienceMeshes
    using BEAST
    using ButterflyFactorizations
    using StaticArrays
    using LinearAlgebra
    using ParallelKMeans
    # =========================================================================
    # Geometry and Operators
    # =========================================================================
    lambda = 1.0
    k = 2 * pi / lambda

    # Scaled down mesh sizes for rapid CI testing
    x = meshsphere(0.25, lambda / 10)
    y = translate(x, SVector(5.0, 0.0, 0.0))

    T = raviartthomas(x)
    U = raviartthomas(y)

    # =========================================================================
    # Tree construction
    # =========================================================================
    @info "Constructing hierarchical trees for level verification..."

    tree1_base = H2Trees.KMeansTree(T.pos, 2; minvalues=100)
    tree1 = H2Trees.BlockTree(tree1_base, tree1_base)

    tree2 = TwoNTree(T, U, lambda / 10)
    tree3 = TwoNTree(T, T, lambda / 10)

    # =========================================================================
    # Traversal Verification
    # =========================================================================
    @info "Verifying tree level traversals match underlying H2Trees structures..."

    function verify_tree_levels(tree_cluster)
        root = ButterflyFactorizations.cluster_root(tree_cluster)
        extracted_levels = ButterflyFactorizations.treelevels(tree_cluster, root)

        for i in eachindex(tree_cluster.nodesatlevel)
            @test issetequal(extracted_levels[i], tree_cluster.nodesatlevel[i])
        end
    end

    # Test all Test/Trial combinations across K-Means and TwoN trees
    verify_tree_levels(tree1.testcluster)
    verify_tree_levels(tree1.trialcluster)
    verify_tree_levels(tree2.testcluster)
    verify_tree_levels(tree2.trialcluster)
    verify_tree_levels(tree3.testcluster)
    verify_tree_levels(tree3.trialcluster)

    @info "Testing tree padding functionality..."
    leaves = H2Trees.leaves
    padded_tlvls = ButterflyFactorizations.traverseandpad(
        tree1.testcluster, ButterflyFactorizations.cluster_root(tree1.testcluster)
    )
    max_level = length(padded_tlvls)

    # Ensure that padding correctly drops all terminal leaves to the maximum depth
    @test issetequal(padded_tlvls[max_level], leaves(tree1.testcluster))
end
