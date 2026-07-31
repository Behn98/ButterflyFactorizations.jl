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
end

struct CenterDistanceAdmissibility
    β::Float64
end
