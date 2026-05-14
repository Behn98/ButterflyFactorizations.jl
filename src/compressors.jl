abstract type Abstractcompressor end

"""
    PartialQR <: Abstractcompressor

A type representing the Partial QR compression strategy for low-rank approximations.
"""

struct PartialQR <: Abstractcompressor
    PartialQR() = new()
end

"""
    (t::PartialQR)(farassembler, src_index, obs_index, n_otilde, ε)

Executes a low-rank approximation of a matrix block using a Partial Pivoted QR
decomposition.

To avoid assembling the full dense matrix, this functor randomly samples `n_otilde` rows
from the observer space and evaluates only those interactions against the full source space.
A pivoted QR decomposition is then applied to find a basis and an active set of column
indices (the "skeleton").

**Arguments:**
- `farassembler`: A function that assembles entries of the interaction matrix.
- `src_index`: Vector of global indices for the source (trial) cluster.
- `obs_index`: Vector of global indices for the observer (test) cluster.
- `n_otilde`: The number of rows to sample randomly.
- `ε`: The relative tolerance used to determine the rank truncation.

**Returns:**
- `tmp`: The compressed coefficient matrix (size `r × n_src`).
- `k`: The optimal skeleton of source indices selected by the pivot strategy.
- `r`: The estimated mathematical rank of the block.
"""

function (t::PartialQR)(
    farassembler, src_index::Vector{Int}, obs_index::Vector{Int}, n_otilde::Int, ε::Float64
)
    n_obs = length(obs_index)
    n_src = length(src_index)
    n_otilde = min(n_otilde, n_obs)

    # --- random row sampling (type stable) ---
    idx = randperm(n_obs)
    row = @view obs_index[idx[1:n_otilde]]
    col = src_index  # full view, no copy

    # --- assemble Z ---
    Z = zeros(ComplexF64, n_otilde, n_src)
    farassembler(Z, row, col)

    # --- pivoted QR (LAPACK-backed) ---
    Fqr = pqr(Z; rtol=ε)

    Q = Fqr[1]
    R = Fqr[2]
    P = Fqr[3]

    r = size(Q, 2)

    # --- views to avoid allocations ---
    Q1 = @view Q[:, 1:r]
    R11 = UpperTriangular(@view R[1:r, 1:r])

    # --- compute q_ks without inv ---
    # tmp = Q1' * Z
    tmp = Matrix{ComplexF64}(undef, r, n_src)
    mul!(tmp, adjoint(Q1), Z)

    # q_ks = R11 \ tmp
    ldiv!(R11, tmp)

    k = src_index[P[1:r]]

    return tmp, k, r
end

"""
    estimate_rank_3d(k, c_s, c_o, a_s, a_o, ε; kwargs...)

Estimates the necessary rank of interaction between a source and an observer bounding box
in 3D space to maintain a given tolerance `ε`.

The formula combines a geometric separation estimate based on the physical sizes and
distances of the bounding boxes, augmented by an algebraic padding term derived from the
desired precision.

**Arguments:**

  - `k`: Wavenumber of the physical problem.
  - `c_s`, `c_o`: Centers of the source and observer clusters (SVector).
  - `a_s`, `a_o`: Half-sizes (radii) of the source and observer clusters.
  - `ε`: Desired precision tolerance.

**Keyword Arguments:**

  - `C`: Scaling factor for the geometric rank term (default: 1.0).
  - `Cε`: Scaling factor for the tolerance padding term (default: 3.0).
  - `Rmin`: Minimum allowable rank (default: 5).

**Returns:**
An `Int` representing the conservatively estimated rank `r` for the block.
"""
function estimate_rank_3d(
    k,
    c_s::SVector,
    c_o::SVector,
    a_s::Float64,
    a_o::Float64,
    ε::Float64,
    ;
    C=1.0,
    Cε=3.0,
    Rmin=5,
)

    # Center separation
    d = norm(c_s .- c_o)

    # Minimum separation (avoid singular or near-field cases)
    dmin = max(d - 0.5 * (a_s + a_o), 1e-12)

    # Geometric directional rank estimate
    R_geom = C * k * (a_s * a_o) / dmin

    # Tolerance-dependent padding
    R_tol = Cε * log(1 / ε)

    # Final rank
    R = ceil(Int, R_geom + R_tol)

    return max(R, Rmin)
end

function estimate_rank_3d(
    k,
    trialT::H2Trees.TwoNTree,
    testT::H2Trees.TwoNTree,
    Snode::Int,
    Onode::Int,
    ε::Float64;
    C=1.0,
    Cε=3.0,
    Rmin=3,
)
    center = H2Trees.center
    halfsize = H2Trees.halfsize

    # Extract geometric information from the tree nodes
    c_s = center(trialT, Snode)
    c_o = center(testT, Onode)
    a_s = halfsize(trialT, Snode)
    a_o = halfsize(testT, Onode)

    # Center separation
    d = norm(c_s .- c_o)

    # Minimum separation (avoid singular or near-field cases)
    dmin = max(d - 0.5 * (a_s + a_o), 1e-12)

    # Geometric directional rank estimate
    R_geom = C * k * (a_s * a_o) / dmin

    # Tolerance-dependent padding
    R_tol = Cε * log(1 / ε)

    # Final rank
    R = ceil(Int, R_geom + R_tol)

    return max(R, Rmin)
end

function estimate_rank_3d(
    k,
    trialT::H2Trees.BoundingBallTree,
    testT::H2Trees.BoundingBallTree,
    Snode::Int,
    Onode::Int,
    ε::Float64;
    C=1.0,
    Cε=3.0,
    Rmin=3,
)
    center = H2Trees.center
    radius = H2Trees.radius

    # Extract geometric information from the tree nodes
    c_s = center(trialT, Snode)
    c_o = center(testT, Onode)
    a_s = radius(trialT, Snode)
    a_o = radius(testT, Onode)

    # Center separation
    d = norm(c_s .- c_o)

    # Minimum separation (avoid singular or near-field cases)
    dmin = max(d - 0.5 * (a_s + a_o), 1e-12)

    # Geometric directional rank estimate
    R_geom = C * k * (a_s * a_o) / dmin

    # Tolerance-dependent padding
    R_tol = Cε * log(1 / ε)

    # Final rank
    R = ceil(Int, R_geom + R_tol)

    return max(R, Rmin)
end
