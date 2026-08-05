# ==============================================================================
# Rank Estimator Abstractions
# ==============================================================================

abstract type AbstractRankEstimator end

"""
    GeometricRankEstimator(C, Cε; Rmin=3)

Standard rank estimator using macroscopic cluster distances and wavenumbers.
Matches the original FMM / H-matrix geometric scaling behavior.
"""
struct GeometricRankEstimator <: AbstractRankEstimator
    C::Float64
    Cε::Float64
    Rmin::Int
end
GeometricRankEstimator(C, Cε; Rmin=3) = GeometricRankEstimator(C, Cε, Rmin)

function (est::GeometricRankEstimator)(k, trialT, testT, Snode::Int, Onode::Int, ε::Float64)
    return estimate_rank_3d(
        k, trialT, testT, Snode, Onode, ε; C=est.C, Cε=est.Cε, Rmin=est.Rmin
    )
end

"""
    ButterflyRankEstimator(Cε; Rmin=10)

Optimized rank estimator for Butterfly Factorization blocks. Assumes rank is bounded
independently of wavenumber k due to complementary cluster size admissibility.
"""
struct ButterflyRankEstimator <: AbstractRankEstimator
    Cε::Float64
    Rmin::Int
end
ButterflyRankEstimator(Cε; Rmin=10) = ButterflyRankEstimator(Cε, Rmin)

function (est::ButterflyRankEstimator)(k, trialT, testT, Snode::Int, Onode::Int, ε::Float64)
    return estimate_rank_butterfly(ε; Cε=est.Cε, Rmin=est.Rmin)
end

"""
    estimate_rank_3d(k, c_s, c_o, a_s, a_o, ε; kwargs...)

Estimates the necessary rank of interaction between a source and an observer bounding box
in 3D space to maintain a given tolerance `ε`.
"""
function estimate_rank_3d(
    k,
    c_s::SVector,
    c_o::SVector,
    a_s::Float64,
    a_o::Float64,
    ε::Float64;
    C=1.0,
    Cε=3.0,
    Rmin=3,
)
    d = norm(c_s .- c_o)
    d_safe = max(d, 1e-4)
    R_geom = C * (k * (a_s * a_o) / d_safe)^2
    R_tol = Cε * log(1 / ε)
    R = ceil(Int, R_geom + R_tol)
    return max(R, Rmin)
end

"""
    estimate_rank_3d(k, trialT, testT, Snode, Onode, ε; kwargs...)
"""
function estimate_rank_3d(
    k, trialT, testT, Snode::Int, Onode::Int, ε::Float64; C=1.5, Cε=1.5, Rmin=3
)
    c_s = cluster_center(trialT, Snode)
    c_o = cluster_center(testT, Onode)
    a_s = cluster_radius(trialT, Snode)
    a_o = cluster_radius(testT, Onode)

    d = norm(c_s .- c_o)
    d_safe = max(d, 1e-4)

    x1 = (k * (a_s * a_o) / d_safe)^2
    x2 = log(1 / ε)

    R = ceil(Int, C * x1 + Cε * x2)
    n_otilde = max(R, Rmin)

    return RankEstimate(n_otilde, x1, x2)
end

"""
    estimate_rank_butterfly(ε; Cε=4.0, Rmin=10)
"""
function estimate_rank_butterfly(ε::Float64; Cε=4.0, Rmin=10)
    x2 = log(1 / ε)
    R = ceil(Int, Cε * x2)
    n_otilde = max(R, Rmin)
    return RankEstimate(n_otilde, 1.0, x2)
end
