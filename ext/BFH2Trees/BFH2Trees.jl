module BFH2Trees

using ButterflyFactorizations
using H2Trees

ButterflyFactorizations.cluster_center(tree::H2Trees.H2ClusterTree, node::Int) =
    H2Trees.center(tree, node)

ButterflyFactorizations.cluster_radius(tree::H2Trees.TwoNTree, node::Int) =
    H2Trees.halfsize(tree, node)

ButterflyFactorizations.cluster_radius(tree::H2Trees.BisectionTree, node::Int) =
    H2Trees.halfsize(tree, node)

ButterflyFactorizations.cluster_radius(tree::H2Trees.BoundingBallTree, node::Int) =
    H2Trees.radius(tree, node)

ButterflyFactorizations.cluster_children(tree::H2Trees.H2ClusterTree, node::Int) =
    H2Trees.children(tree, node)

ButterflyFactorizations.cluster_isleaf(tree::H2Trees.H2ClusterTree, node::Int) =
    H2Trees.isleaf(tree, node)

ButterflyFactorizations.cluster_root(tree::H2Trees.H2ClusterTree) = H2Trees.root(tree)

ButterflyFactorizations.cluster_testtree(tree::H2Trees.BlockTree) = H2Trees.testtree(tree)

ButterflyFactorizations.cluster_trialtree(tree::H2Trees.BlockTree) = H2Trees.trialtree(tree)

ButterflyFactorizations.cluster_parent(tree::H2Trees.H2ClusterTree, node::Int) =
    H2Trees.parent(tree, node)

ButterflyFactorizations.cluster_values(tree::H2Trees.H2ClusterTree, node::Int) =
    H2Trees.values(tree, node)

include("isfarfunctor.jl")

end
