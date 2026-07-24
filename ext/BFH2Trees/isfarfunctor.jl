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
function (t::ButterflyFactorizations.isFarFunctor)(
    srctree::H2Trees.TwoNTree, tsttree::H2Trees.TwoNTree, snode::Int, onode::Int
)
    ocenter = ButterflyFactorizations.cluster_center(tsttree, onode)
    olength = ButterflyFactorizations.cluster_radius(tsttree, onode)
    scenter = ButterflyFactorizations.cluster_center(srctree, snode)
    slength = ButterflyFactorizations.cluster_radius(srctree, snode)

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

function (t::ButterflyFactorizations.isFarFunctor)(
    srctree::H2Trees.BoundingBallTree,
    tsttree::H2Trees.BoundingBallTree,
    snode::Int,
    onode::Int,
)
    ocenter = ButterflyFactorizations.cluster_center(tsttree, onode)
    scenter = ButterflyFactorizations.cluster_center(srctree, snode)
    olength = ButterflyFactorizations.cluster_radius(tsttree, onode)
    slength = ButterflyFactorizations.cluster_radius(srctree, snode)

    dist = norm(scenter - ocenter)

    # If you want standard H-matrix condition:
    W = max(slength, olength)
    return dist - (olength + slength) > t.α * W

    # OR: If doing pure Butterfly and needing a relative gap condition, use:
    # return dist > (1 + t.α) * (olength + slength)
end

function (t::ButterflyFactorizations.isFarFunctor)(
    srctree::H2Trees.BisectionTree, tsttree::H2Trees.BisectionTree, snode::Int, onode::Int
)
    ocenter = ButterflyFactorizations.cluster_center(tsttree, onode)
    olength = ButterflyFactorizations.cluster_radius(tsttree, onode)
    scenter = ButterflyFactorizations.cluster_center(srctree, snode)
    slength = ButterflyFactorizations.cluster_radius(srctree, snode)

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
