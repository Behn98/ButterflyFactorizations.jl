#------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------#
#---------MUST BE IMPLEMENTED FOR ANY TREE TYPE TO BE USED WITH THIS PACKAGE!!!!!----------#
#------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------#

abstract type AbstractTreeParameters end

"""
    TreeParameters{T} <: AbstractTreeParameters

A container for the mathematical tuning parameters required by the Butterfly Factorization for a specific tree geometry.

# Fields

  - `α::T`: The separation parameter for standard far-field admissibility.
  - `C::T`: The geometric scaling constant for the directional rank estimator.
  - `Cε::T`: The algebraic padding constant for the rank estimator based on tolerance.
  - `β::T`: The center-to-center distance multiplier for alternative admissibility criteria.
"""
struct TreeParameters{T} <: AbstractTreeParameters
    α::T
    C::T
    Cε::T
    β::T
end

"""
    tree_parameters(tree) -> TreeParameters

Retrieve the default mathematical and geometric parameters (`α`, `C`, `Cε`, `β`) associated with a given tree type.
"""
function tree_parameters(tree)
    return error("`tree_parameters` not implemented for $(typeof(tree))")
end

"""
    cluster_center(tree, node) -> SVector

Return the geometric center coordinate of the specified `node` within the `tree`.
"""
function cluster_center(tree, node)
    return error("`cluster_center` not implemented for $(typeof(tree))")
end

"""
    cluster_radius(tree, node) -> Float64

Return the bounding radius (or maximum half-width) of the specified `node` within the `tree`.
"""
function cluster_radius(tree, node)
    return error("`cluster_radius` not implemented for $(typeof(tree))")
end

"""
    cluster_children(tree, node) -> Iterable

Return an iterable collection of the child node indices for the specified `node`.
"""
function cluster_children(tree, node)
    return error("`cluster_children` not implemented for $(typeof(tree))")
end

"""
    cluster_isleaf(tree, node) -> Bool

Evaluate whether the specified `node` is a leaf (i.e., has no children) in the `tree`.
"""
function cluster_isleaf(tree, node)
    return error("`cluster_isleaf` not implemented for $(typeof(tree))")
end

"""
    cluster_root(tree) -> Int

Return the index or identifier of the root node for the given `tree`.
"""
function cluster_root(tree)
    return error("`cluster_root` not implemented for $(typeof(tree))")
end

"""
    cluster_testtree(blktree)

Extract the observer (test) tree from a coupled block-tree structure.
"""
function cluster_testtree(tree)
    return error("`cluster_testtree` not implemented for $(typeof(tree))")
end

"""
    cluster_trialtree(blktree)

Extract the source (trial) tree from a coupled block-tree structure.
"""
function cluster_trialtree(tree)
    return error("`cluster_trialtree` not implemented for $(typeof(tree))")
end

"""
    cluster_parent(tree, node) -> Int

Return the index or identifier of the parent node for the specified `node`.
"""
function cluster_parent(tree, node)
    return error("`cluster_parent` not implemented for $(typeof(tree))")
end

"""
    cluster_values(tree, node) -> Vector{Int}

Return the global indices of the basis functions (or points) contained within the specified `node`.
"""
function cluster_values(tree, node)
    return error("`cluster_values` not implemented for $(typeof(tree))")
end

"""
    cluster_blktree(stree, ttree)

Construct a coupled block-tree hierarchy from a given source (trial) tree and observer (test) tree.
"""
function cluster_blktree(stree, ttree)
    return error(
        "`cluster_blktree` not implemented for $(typeof(stree)) and $(typeof(ttree))"
    )
end

"""
    cluster_levels(tree)

Return the level structure of the tree, typically as a collection of nodes grouped by their depth.
"""
function cluster_levels(tree)
    return error("`cluster_levels` not implemented for $(typeof(tree))")
end

"""
    largernode(tree1, tree2, node1, node2) -> Bool

Determine if `node1` in `tree1` is spatially larger than `node2` in `tree2`.
By default, this evaluates `cluster_radius(tree1, node1) >= cluster_radius(tree2, node2)`.
"""
largernode(tree1, tree2, node1, node2) =
    cluster_radius(tree1, node1) >= cluster_radius(tree2, node2)

"""
    build_bisection_tree(positions; kwargs...)

Builds a bisection tree for Butterfly algebra to work. The Implementation of this function
is provided in the BFH2Trees.jl file in the ext directory.
"""
function build_bisection_tree(positions; kwargs...)
    return error("`build_bisection_tree` not implemented for $(typeof(positions))")
end
