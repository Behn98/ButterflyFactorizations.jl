"""
    isFarFunctor

A functor (callable struct) used to determine if two bounding boxes (clusters)
in a hierarchical tree are well-separated ("far") enough to be compressed.

# Fields:

  - `α::Float64`: The separation parameter. A larger `α` forces clusters to be
    further apart before they are considered admissible for low-rank approximation.
"""
struct isFarFunctor <: AbstractAdmissibility
    α::Float64
end

"""
    CenterDistanceAdmissibility

An alternative admissibility functor that evaluates cluster separation based
strictly on the macroscopic center-to-center distance, completely avoiding
the physical sizes in the gap calculation to prevent overlapping origin traps.

# Fields:

  - `β::Float64`: The separation multiplier.
"""
struct CenterDistanceAdmissibility <: AbstractAdmissibility
    β::Float64
end
