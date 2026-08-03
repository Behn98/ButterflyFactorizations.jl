"""
    (t::isFarFunctor)(srctree, tsttree, snode, onode)

Evaluates the admissibility condition between a source node and an observer node for `TwoNTree` and `BisectionTree`.

The rule checks if the distance between the bounding boxes is sufficiently larger
than the sum of their physical sizes, scaled by the separation parameter `α`
and the maximum half-size `W` of the two clusters. Contains a singularity interceptor
to prevent zero-volume node overlaps from blowing up integral evaluations.
"""
function (t::ButterflyFactorizations.isFarFunctor)(
    srctree::Union{H2Trees.TwoNTree,BisectionTree},
    tsttree::Union{H2Trees.TwoNTree,BisectionTree},
    snode::Int,
    onode::Int,
)
    ocenter = ButterflyFactorizations.cluster_center(tsttree, onode)
    olength = ButterflyFactorizations.cluster_radius(tsttree, onode)
    scenter = ButterflyFactorizations.cluster_center(srctree, snode)
    slength = ButterflyFactorizations.cluster_radius(srctree, snode)

    W = max(slength, olength)
    target_dist = t.α * W

    # Fast Axis-Aligned Bounding Box (AABB) Distance
    mind_sq = 0.0
    for i in 1:3
        dist_axis = abs(ocenter[i] - scenter[i]) - (slength + olength)
        if dist_axis > 0.0
            mind_sq += dist_axis^2
        end
    end

    # 🚀 SINGULARITY INTERCEPTOR: Force physically touching nodes to near-field
    if mind_sq < 1e-10
        return false
    end

    return sqrt(mind_sq) > target_dist
end

"""
    (t::isFarFunctor)(srctree::BoundingBallTree, tsttree::BoundingBallTree, snode, onode)

Evaluates standard H-matrix admissibility for bounding ball clusters based strictly on center-to-center distance.
"""
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
    W = max(slength, olength)

    if dist < 1e-5
        return false
    end

    return dist - (olength + slength) > t.α * W
end

"""
    (t::CenterDistanceAdmissibility)(srctree, tsttree, snode, onode)

An alternative admissibility condition relying entirely on the macroscopic center-to-center
distance, rendering it immune to overlapping origin traps.
"""
function (t::ButterflyFactorizations.CenterDistanceAdmissibility)(
    srctree::BisectionTree, tsttree::BisectionTree, snode::Int, onode::Int
)
    ocenter = ButterflyFactorizations.cluster_center(tsttree, onode)
    scenter = ButterflyFactorizations.cluster_center(srctree, snode)

    olength = ButterflyFactorizations.cluster_radius(tsttree, onode)
    slength = ButterflyFactorizations.cluster_radius(srctree, snode)

    dist_centers = norm(scenter - ocenter)

    # 🚀 SINGULARITY INTERCEPTOR
    if dist_centers < 1e-5
        return false
    end

    return dist_centers > t.β * (slength + olength)
end
