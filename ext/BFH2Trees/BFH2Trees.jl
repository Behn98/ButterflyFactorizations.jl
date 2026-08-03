module BFH2Trees

using ButterflyFactorizations
using LinearAlgebra
using H2Trees
using StaticArrays

include("bisection_tree.jl")

# --- Center & Radius Mappings ---
ButterflyFactorizations.cluster_center(tree::H2Trees.H2ClusterTree, node::Int) =
    H2Trees.center(tree, node)

# TwoNTree returns AABB half-width, so we multiply by sqrt(3) to get the bounding sphere
ButterflyFactorizations.cluster_radius(tree::H2Trees.TwoNTree, node::Int) =
    H2Trees.halfsize(tree, node) * sqrt(3)

# BisectionTree and BoundingBallTree natively return the bounding sphere radius
ButterflyFactorizations.cluster_radius(tree::BisectionTree, node::Int) =
    H2Trees.halfsize(tree, node) * sqrt(3)

ButterflyFactorizations.cluster_radius(tree::H2Trees.BoundingBallTree, node::Int) =
    H2Trees.radius(tree, node)

# --- Hierarchy & Value Mappings ---
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

ButterflyFactorizations.cluster_levels(tree::H2Trees.H2ClusterTree) = H2Trees.levels(tree)

ButterflyFactorizations.cluster_blktree(
    stree::H2Trees.H2ClusterTree, ttree::H2Trees.H2ClusterTree
) = H2Trees.BlockTree(stree, ttree)

include("h2admissible.jl")
include("h2parameters.jl")
include("h2rankest.jl")

end
