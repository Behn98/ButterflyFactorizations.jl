"""
    nearandfar(tree, admissible; kwargs...)
    nearandfar(tree, α::Float64; kwargs...)

Traverses a block-tree and categorizes interactions between source and observer
clusters into "near-field" and "far-field" lists.

This function acts as the entry point for the dual-tree traversal algorithm.
It uses the admissibility condition (e.g., `isFarFunctor`) to separate interactions.
If a raw `Float64` is provided instead of a functor, it automatically wraps it
in an `isFarFunctor`.

**Arguments:**

  - `tree`: A strictly coupled `BlockTree` containing both the test and trial trees.
  - `admissible`: The functor (or float `α`) used to evaluate geometric separation.

**Keyword Arguments:**

  - `minbflvl`: The minimum allowable depth for far-field compression (default: 3).
  - `unbalancedints`: If `true`, allows splitting only the functionally larger node.
  - `leafcomp`: If `false`, leaf nodes will be forced to near-field even if admissible.
  - `leafimbalance`: If `true`, allows interaction between a leaf and a non-leaf node.

**Returns:**

  - `farinteractions`: A vector of tuples `(observer_node_id, source_node_id)` for admissible interactions.
  - `nearinteractions`: A vector of tuples `(observer_node_id, source_node_id)` for non-admissible interactions.
"""
function nearandfar(
    tree, admissible; minbflvl=3, unbalancedints=false, leafcomp=true, leafimbalance=true
)
    srctree = cluster_trialtree(tree)
    tsttree = cluster_testtree(tree)
    node_o = cluster_root(tsttree)
    node_s = cluster_root(srctree)

    nearinteractions = Vector{Tuple{Int64,Int64}}()
    farinteractions = Vector{Tuple{Int64,Int64}}()

    totaltreeheight = min(length(cluster_levels(tsttree)), length(cluster_levels(srctree)))

    process_nodes!(
        srctree,
        tsttree,
        node_o,
        node_s,
        admissible,
        farinteractions,
        nearinteractions,
        totaltreeheight;
        minbflvl=minbflvl,
        unbalancedints=unbalancedints,
        leafcomp=leafcomp,
        leafimbalance=leafimbalance,
    )

    return farinteractions, nearinteractions
end

# Backwards-compatible wrapper that automatically creates `isFarFunctor`
function nearandfar(tree, α::Float64; kwargs...)
    return nearandfar(tree, isFarFunctor(α); kwargs...)
end

"""
    process_nodes!(srctree, tsttree, node_o, node_s, admissible, farinteractions, nearinteractions, currentheight; kwargs...)

Recursively analyzes the interaction between a source node and an observer node.

  - If the nodes are `admissible` (well-separated), their IDs are added to `farinteractions`.
  - If they are not admissible but hit recursion limits (e.g., both are leaves, or `minbflvl` is reached), they are added to `nearinteractions`.
  - If they are not admissible and can be split, they are subdivided into their children and the process repeats.
"""
function process_nodes!(
    srctree::T2,
    tsttree::T1,
    node_o,
    node_s,
    admissible,
    farinteractions,
    nearinteractions,
    currentheight;
    leafcomp=true,
    unbalancedints=false,
    leafimbalance=true,
    minbflvl=1,
) where {T1,T2}

    # 1. Calculate clearly defined state variables
    is_leaf_o = cluster_isleaf(tsttree, node_o)
    is_leaf_s = cluster_isleaf(srctree, node_s)
    any_leaf = is_leaf_o || is_leaf_s
    both_leaves = is_leaf_o && is_leaf_s
    is_admissible = admissible(srctree, tsttree, node_s, node_o)

    # 2. Global Minimum Depth Override
    # If we hit the minimum allowed depth and leaf compression is off, force into near-field
    if (currentheight <= minbflvl && !leafcomp)
        push!(nearinteractions, (node_o, node_s))
        return nothing
    end

    # 3. Admissible Handling
    if is_admissible
        if (any_leaf && !leafcomp)
            push!(nearinteractions, (node_o, node_s))
        else
            push!(farinteractions, (node_o, node_s))
        end
        return nothing
    end

    # 4. Non-Admissible Fallbacks
    if (both_leaves || (any_leaf && !leafimbalance))
        push!(nearinteractions, (node_o, node_s))
        return nothing
    end

    # 5. Split and Recurse
    if unbalancedints
        # Split the functionally larger node
        if (
            largernode(tsttree, srctree, node_o, node_s) && !cluster_isleaf(tsttree, node_o)
        ) || cluster_isleaf(srctree, node_s)
            for child_o in cluster_children(tsttree, node_o)
                process_nodes!(
                    srctree,
                    tsttree,
                    child_o,
                    node_s,
                    admissible,
                    farinteractions,
                    nearinteractions,
                    currentheight - 1;
                    minbflvl=minbflvl,
                    leafcomp=leafcomp,
                    unbalancedints=unbalancedints,
                    leafimbalance=leafimbalance,
                )
            end
        else
            for child_s in cluster_children(srctree, node_s)
                process_nodes!(
                    srctree,
                    tsttree,
                    node_o,
                    child_s,
                    admissible,
                    farinteractions,
                    nearinteractions,
                    currentheight - 1;
                    minbflvl=minbflvl,
                    leafcomp=leafcomp,
                    unbalancedints=unbalancedints,
                    leafimbalance=leafimbalance,
                )
            end
        end
    else
        # Balanced splitting
        if cluster_isleaf(tsttree, node_o)
            for child_s in cluster_children(srctree, node_s)
                process_nodes!(
                    srctree,
                    tsttree,
                    node_o,
                    child_s,
                    admissible,
                    farinteractions,
                    nearinteractions,
                    currentheight - 1;
                    minbflvl=minbflvl,
                    leafcomp=leafcomp,
                    unbalancedints=unbalancedints,
                    leafimbalance=leafimbalance,
                )
            end
        elseif cluster_isleaf(srctree, node_s)
            for child_o in cluster_children(tsttree, node_o)
                process_nodes!(
                    srctree,
                    tsttree,
                    child_o,
                    node_s,
                    admissible,
                    farinteractions,
                    nearinteractions,
                    currentheight - 1;
                    minbflvl=minbflvl,
                    leafcomp=leafcomp,
                    unbalancedints=unbalancedints,
                    leafimbalance=leafimbalance,
                )
            end
        else
            for child_o in cluster_children(tsttree, node_o)
                for child_s in cluster_children(srctree, node_s)
                    process_nodes!(
                        srctree,
                        tsttree,
                        child_o,
                        child_s,
                        admissible,
                        farinteractions,
                        nearinteractions,
                        currentheight - 1;
                        minbflvl=minbflvl,
                        leafcomp=leafcomp,
                        unbalancedints=unbalancedints,
                        leafimbalance=leafimbalance,
                    )
                end
            end
        end
    end
end
