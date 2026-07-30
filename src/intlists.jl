"""
    isFarFunctor

A functor (callable struct) used to determine if two bounding boxes (clusters)
in a hierarchical tree are well-separated ("far") enough to be compressed.

**Fields:**

  - `α::Float64`: The separation parameter. A larger `α` forces clusters to be
    further apart before they are considered admissible for low-rank approximation.
"""
struct isFarFunctor
    α::Float64
    isFarFunctor(α) = new(α)
end

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
function nearandfar(tree, α; unbalancedints=false, leafcomp=true, leafimbalance=true)
    admissible = isFarFunctor(α)
    srctree = cluster_trialtree(tree)
    tsttree = cluster_testtree(tree)
    node_o = cluster_root(tsttree)
    node_s = cluster_root(srctree)
    nearinteractions = Vector{Tuple{Int64,Int64}}()         #observernodeid --> sourcenodeid
    farinteractions = Vector{Tuple{Int64,Int64}}()          #observernodeid --> sourcenodeid
    process_nodes!(
        srctree,
        tsttree,
        node_o,
        node_s,
        admissible,
        farinteractions,
        nearinteractions;
        unbalancedints=unbalancedints,
        leafcomp=leafcomp,
        leafimbalance=leafimbalance,
    )
    return farinteractions, nearinteractions
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
    admissible::isFarFunctor,
    farinteractions,
    nearinteractions;
    leafcomp=true,
    unbalancedints=false,
    leafimbalance=true,
) where {T1,T2}
    if admissible(srctree, tsttree, node_s, node_o) &&
        (!(cluster_isleaf(tsttree, node_o) && cluster_isleaf(srctree, node_s) && !leafcomp))
        push!(farinteractions, (node_o, node_s))
        return nothing
    elseif (cluster_isleaf(tsttree, node_o) && cluster_isleaf(srctree, node_s)) || (
        !leafimbalance &&
        (cluster_isleaf(tsttree, node_o) || cluster_isleaf(srctree, node_s))
    )
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
                    nearinteractions;
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
                    nearinteractions;
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
                    nearinteractions;
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
                    nearinteractions;
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
                        nearinteractions;
                        leafcomp=leafcomp,
                        unbalancedints=unbalancedints,
                        leafimbalance=leafimbalance,
                    )
                end
            end
        end
    end
end
