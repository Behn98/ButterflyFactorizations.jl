"""
    estimate_rank_3d(k, trialT::H2ClusterTree, testT::H2ClusterTree, Snode, Onode, ε; kwargs...)

Extends the core rank estimation logic to natively unpack `H2Trees.H2ClusterTree` structures.
Extracts the macroscopic centers and calibrated radii to predict the mathematical rank
required to hit the tolerance `ε`.
"""
function ButterflyFactorizations.estimate_rank_3d(
    k,
    trialT::H2Trees.H2ClusterTree,
    testT::H2Trees.H2ClusterTree,
    Snode::Int,
    Onode::Int,
    ε::Float64;
    C=1.5,
    Cε=1.5,
    Rmin=3,
)
    c_s = ButterflyFactorizations.cluster_center(trialT, Snode)
    c_o = ButterflyFactorizations.cluster_center(testT, Onode)
    a_s = ButterflyFactorizations.cluster_radius(trialT, Snode)
    a_o = ButterflyFactorizations.cluster_radius(testT, Onode)

    # 1. Use the actual center-to-center distance!
    d = norm(c_s .- c_o)

    # Safe-guard against origin collisions, NO sphere-gap subtraction!
    d_safe = max(d, 1e-4)

    # 2. Isolated predictor variables using pure macroscopic distance
    x1 = (k * (a_s * a_o) / d_safe)^2
    x2 = log(1 / ε)

    R = ceil(Int, C * x1 + Cε * x2)
    n_otilde = max(R, Rmin)

    return ButterflyFactorizations.RankEstimate(n_otilde, x1, x2)
end
