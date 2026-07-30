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

  - `nearov`: A `Vector` containing the global observer indices for near-field blocks.
  - `nearsv`: A `Vector` containing the global source indices for near-field blocks.
  - `farinteractions`: A `Dict` mapping an observer node ID to a list of source node
    IDs that are well-separated from it.
"""
function nearandfar(tree, α; unbalancedints=false, leafcom=true)
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
        allowunbalancedfints=unbalancedints,
        allowleafcompression=leafcom,
    )
    return farinteractions, nearinteractions
end

"""
    process_nodes!(srctree, tsttree, node_o, node_s, admissible, far, nearsv, nearov; allowleafcompression=true, allowunbalancedfints=true)

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

  - `nearsv`, `nearov`: Accumulator vectors for near-field global indices.

      + `allowleafcompression`: If `false`, leaf nodes will not be compressed even if admissible.
      + `allowunbalancedfints`: If `false`, both nodes must be split simultaneously, even if one is larger.
"""
function process_nodes!(
    srctree::T,
    tsttree::T,
    node_o,
    node_s,
    admissible::isFarFunctor,
    farinteractions,
    nearinteractions;
    allowleafcompression=true,
    allowunbalancedfints=false,
) where {T}
    if admissible(srctree, tsttree, node_s, node_o) && (
        !(
            cluster_isleaf(tsttree, node_o) &&
            cluster_isleaf(srctree, node_s) &&
            !allowleafcompression
        )
    )
        push!(farinteractions, (node_o, node_s))
        return nothing
    elseif cluster_isleaf(tsttree, node_o) && cluster_isleaf(srctree, node_s)
        #push!(nearsv, cluster_values(srctree, node_s))
        #push!(nearov, cluster_values(tsttree, node_o))
        push!(nearinteractions, (node_o, node_s))
        return nothing
    end
    # split the larger node
    if allowunbalancedfints
        if (
            largernode(tsttree, srctree, node_o, node_s) && !cluster_isleaf(tsttree, node_o)
        ) || cluster_isleaf(srctree, node_s)
            for child_o in collect(cluster_children(tsttree, node_o))
                process_nodes!(
                    srctree,
                    tsttree,
                    child_o,
                    node_s,
                    admissible,
                    farinteractions,
                    nearinteractions;
                    allowleafcompression=allowleafcompression,
                    allowunbalancedfints=allowunbalancedfints,
                )
            end
        else
            for child_s in collect(cluster_children(srctree, node_s))
                process_nodes!(
                    srctree,
                    tsttree,
                    node_o,
                    child_s,
                    admissible,
                    farinteractions,
                    nearinteractions;
                    allowleafcompression=allowleafcompression,
                    allowunbalancedfints=allowunbalancedfints,
                )
            end
        end
    else
        if !cluster_isleaf(tsttree, node_o) && !cluster_isleaf(srctree, node_s)
            for child_o in collect(cluster_children(tsttree, node_o))
                for child_s in collect(cluster_children(srctree, node_s))
                    process_nodes!(
                        srctree,
                        tsttree,
                        child_o,
                        child_s,
                        admissible,
                        farinteractions,
                        nearinteractions;
                        allowleafcompression=allowleafcompression,
                        allowunbalancedfints=allowunbalancedfints,
                    )
                end
            end
        elseif !cluster_isleaf(tsttree, node_o)
            for child_o in collect(cluster_children(tsttree, node_o))
                process_nodes!(
                    srctree,
                    tsttree,
                    child_o,
                    node_s,
                    admissible,
                    farinteractions,
                    nearinteractions;
                    allowleafcompression=allowleafcompression,
                    allowunbalancedfints=allowunbalancedfints,
                )
            end
        else
            for child_s in collect(cluster_children(srctree, node_s))
                process_nodes!(
                    srctree,
                    tsttree,
                    node_o,
                    child_s,
                    admissible,
                    farinteractions,
                    nearinteractions;
                    allowleafcompression=allowleafcompression,
                    allowunbalancedfints=allowunbalancedfints,
                )
            end
        end
    end
end
