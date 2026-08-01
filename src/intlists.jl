"""
    nearandfar(tree::H2Trees.BlockTree, α)

Traverses a block-tree and categorizes interactions between source and observer
clusters into "near-field" and "far-field" lists.

This function acts as the entry point for the dual-tree traversal algorithm.
It uses the admissibility condition (`isFarFunctor`) to separate interactions.

**Arguments:**

  - `tree`: A strictly coupled `BlockTree` containing both the test and trial trees.
  - `α`: The geometric separation parameter.

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

#Backwards-compatible wrapper that automatically creates `isFarFunctor`
# if a Float64 (α) is passed instead.
function nearandfar(tree, α::Float64; kwargs...)
    return nearandfar(tree, isFarFunctor(α); kwargs...)
end

"""
    process_nodes!(srctree, tsttree, node_o, node_s, admissible, farinteractions, nearinteractions; leafcomp=true, unbalancedints=true)

Recursively analyzes the interaction between a source node and an observer node.

  - If the nodes are `admissible` (well-separated), they are added to the `farinteractions` list.
  - If they are not admissible but both are leaf nodes, their direct mesh indices are
    extracted and appended to the near-field lists (`nearsv` and `nearov`).
  - If they are not admissible and can be split, the functionally larger node is subdivided
    into its children, and the process repeats.

**Arguments:**

  - `srctree`, `tsttree`: The tree hierarchies.

  - `node_o`, `node_s`: Current observer and source node IDs.

  - `admissible`: The `isFarFunctor` used to check geometric separation.

  - `farinteractions`: Accumulator dictionary for far-field node pairs.

  - `nearinteractions`: Accumulator vector for near-field node pairs.

      + `leafcomp`: If `false`, leaf nodes will not be compressed even if admissible.
      + `unbalancedints`: If `false`, both nodes must be split simultaneously, even if one is larger.
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
    if admissible(srctree, tsttree, node_s, node_o) && (
        !(
            (cluster_isleaf(tsttree, node_o) || cluster_isleaf(srctree, node_s))&&currentheight>minbflvl &&
            !leafcomp
        )
    )
        push!(farinteractions, (node_o, node_s))
        return nothing
    elseif (cluster_isleaf(tsttree, node_o) && cluster_isleaf(srctree, node_s)) ||
        (
            !leafimbalance &&
            (cluster_isleaf(tsttree, node_o) || cluster_isleaf(srctree, node_s))
        ) ||
        (currentheight <= minbflvl && !leafcomp)
        push!(nearinteractions, (node_o, node_s))
        return nothing
    end
    # split the larger node
    if unbalancedints
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
                    currentheight-1;
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
                    currentheight-1;
                    minbflvl=minbflvl,
                    leafcomp=leafcomp,
                    unbalancedints=unbalancedints,
                    leafimbalance=leafimbalance,
                )
            end
        end
    else
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
                    currentheight-1;
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
                    currentheight-1;
                    minbflvl=minbflvl,
                    leafcomp=leafcomp,
                    unbalancedints=unbalancedints,
                    leafimbalance=leafimbalance,
                )
            end
        else
            for child_o in collect(cluster_children(tsttree, node_o))
                for child_s in cluster_children(srctree, node_s)
                    process_nodes!(
                        srctree,
                        tsttree,
                        child_o,
                        child_s,
                        admissible,
                        farinteractions,
                        nearinteractions,
                        currentheight-1;
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
