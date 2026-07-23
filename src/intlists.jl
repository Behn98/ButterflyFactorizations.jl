import H2Trees: isleaf, testtree, trialtree, root, children

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
    (t::isFarFunctor)(srctree, tsttree, snode, onode)

Evaluates the admissibility condition between a source node and an observer node.

The rule checks if the distance between the bounding box centers is sufficiently
larger than the sum of their physical sizes, scaled by the separation parameter `α`
and the maximum half-size `W` of the two clusters.

**Arguments:**

  - `srctree`: The tree structure containing source (trial) clusters.
  - `tsttree`: The tree structure containing observer (test) clusters.
  - `snode`: The ID of the source node to evaluate.
  - `onode`: The ID of the observer node to evaluate.

**Returns:**

  - `true` if the nodes are well-separated (admissible for far-field compression).
  - `false` if they are too close (must be treated as near-field or split further).
"""
function (t::isFarFunctor)(
    srctree::H2Trees.TwoNTree, tsttree::H2Trees.TwoNTree, snode, onode
)
    ocenter = H2Trees.center(tsttree, onode)
    olength = H2Trees.halfsize(tsttree, onode) # Assuming this is half-width
    scenter = H2Trees.center(srctree, snode)
    slength = H2Trees.halfsize(srctree, snode)

    W = max(slength, olength)
    target_dist = t.α * W

    # 1. Fast Axis-Aligned Bounding Box (AABB) Distance
    # Calculate exact distance between the two boxes
    mind_sq = 0.0
    for i in 1:3
        # Distance between intervals along axis i
        dist_axis = abs(ocenter[i] - scenter[i]) - (slength + olength)
        if dist_axis > 0.0
            mind_sq += dist_axis^2
        end
    end

    # If the closest points of the cubes are further than α * W, they are far-field
    return sqrt(mind_sq) > target_dist
end

function (t::isFarFunctor)(
    srctree::H2Trees.BoundingBallTree, tsttree::H2Trees.BoundingBallTree, snode, onode
)
    ocenter = H2Trees.center(tsttree, onode)
    scenter = H2Trees.center(srctree, snode)
    olength = H2Trees.radius(tsttree, onode)
    slength = H2Trees.radius(srctree, snode)

    dist = norm(scenter - ocenter)

    # If you want standard H-matrix condition:
    W = max(slength, olength)
    return dist - (olength + slength) > t.α * W

    # OR: If doing pure Butterfly and needing a relative gap condition, use:
    # return dist > (1 + t.α) * (olength + slength)
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
function nearandfar(tree::H2Trees.BlockTree, α; unbalancedints=true, leafcom=true)
    admissible = isFarFunctor(α)
    srctree = trialtree(tree)
    tsttree = testtree(tree)
    node_o = root(tsttree)
    node_s = root(srctree)
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

function largernode(tsttree::H2Trees.TwoNTree, srctree::H2Trees.TwoNTree, node_o, node_s)
    return H2Trees.halfsize(tsttree, node_o) >= H2Trees.halfsize(srctree, node_s)
end

function largernode(
    tsttree::H2Trees.BoundingBallTree, srctree::H2Trees.BoundingBallTree, node_o, node_s
)
    return H2Trees.radius(tsttree, node_o) >= H2Trees.radius(srctree, node_s)
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
    allowunbalancedfints=true,
) where {T}
    if admissible(srctree, tsttree, node_s, node_o) &&
        (!(isleaf(tsttree, node_o) && isleaf(srctree, node_s) && !allowleafcompression))
        push!(farinteractions, (node_o, node_s))
        return nothing
    elseif isleaf(tsttree, node_o) && isleaf(srctree, node_s)
        #push!(nearsv, H2Trees.values(srctree, node_s))
        #push!(nearov, H2Trees.values(tsttree, node_o))
        push!(nearinteractions, (node_o, node_s))
        return nothing
    end
    # split the larger node
    if allowunbalancedfints
        if (largernode(tsttree, srctree, node_o, node_s) && !isleaf(tsttree, node_o)) ||
            isleaf(srctree, node_s)
            for child_o in collect(children(tsttree, node_o))
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
            for child_s in collect(children(srctree, node_s))
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
        if !isleaf(tsttree, node_o) && !isleaf(srctree, node_s)
            for child_o in collect(children(tsttree, node_o))
                for child_s in collect(children(srctree, node_s))
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
        elseif !isleaf(tsttree, node_o)
            for child_o in collect(children(tsttree, node_o))
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
            for child_s in collect(children(srctree, node_s))
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
