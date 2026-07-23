abstract type Abstractcompressor end

"""

PartialQR <: Abstractcompressor


A type representing the Partial QR compression strategy for low-rank approximations.

"""

struct PartialQR{L} <: Abstractcompressor
    logger::L
end

# Default constructor: No logger = zero overhead!
PartialQR() = PartialQR(nothing)

# Compiler eliminates this branch when logger === nothing
@inline log_rank!(::Nothing, rank_est, r) = nothing

# Executed ONLY when logger is active
@inline function log_rank!(logger::RankLogger, est::RankEstimate, r::Int)
    tid = Threads.threadid()
    return push!(logger.buffers[tid], (est.x1, est.x2, r))
end

@inline function log_rank!(logger::RankLogger, est::Int, r::Int)
    # Fallback if plain Int was passed without geometric terms
    return nothing
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
    farassembler,
    src_index::Vector{Int},
    obs_index::Vector{Int},
    rank_est, # Can be Int or RankEstimate
    ε::Float64,
)
    n_obs = length(obs_index)
    n_src = length(src_index)

    # 1. Early exit if index sets are empty
    if n_obs == 0 || n_src == 0
        log_rank!(t.logger, rank_est, 0)
        return Matrix{ComplexF64}(undef, 0, n_src), Int[], 0
    end

    n_otilde_guess = get_n_otilde(rank_est)
    n_otilde = min(max(n_otilde_guess, 10), n_obs)

    shuffled_obs = obs_index[randperm(n_obs)]

    # 🚀 FIX 1: Must use `zeros`! BEAST accumulates (+=) into this matrix.
    # `undef` leaves memory garbage (including NaNs) which ruins the assembly.
    Z = zeros(ComplexF64, n_otilde, n_src)

    current_rows = @view shuffled_obs[1:n_otilde]
    farassembler(Z, current_rows, src_index)

    rows_evaluated = n_otilde

    while true
        # --- Pivoted QR ---
        Zpqr = deepcopy(Z)
        Fqr = pqr(Zpqr; rtol=ε)
        Q, R, P = Fqr[1], Fqr[2], Fqr[3]

        # 🚀 TRUNCATE RANK BASED ON DIAGONAL OF R
        r = 0
        if size(R, 1) > 0 && size(R, 2) > 0
            R11_abs = abs(R[1, 1])

            # 🚀 FIX 2: Scale-invariant tolerance!
            # If the block is naturally tiny (e.g. 1e-16), this scales the cutoff proportionally,
            # preventing SingularException without accidentally wiping out valid small blocks.
            tol = R11_abs * max(ε, eps(Float64))

            max_r = min(size(Q, 2), size(R, 1), size(R, 2))

            for i in 1:max_r
                if abs(R[i, i]) > tol
                    r = i
                else
                    break
                end
            end
        end

        # Adaptive check: expand sample if rank maxed out sampled rows
        if r > floor(Int, 0.8 * rows_evaluated) && rows_evaluated < n_obs
            new_target = min(rows_evaluated * 2, n_obs)
            new_rows_count = new_target - rows_evaluated

            # 🚀 FIX 1 (cont.): Must also be `zeros` here.
            Z_new = zeros(ComplexF64, new_rows_count, n_src)
            new_rows_idx = @view shuffled_obs[(rows_evaluated + 1):new_target]

            farassembler(Z_new, new_rows_idx, src_index)

            Z = vcat(Z, Z_new)
            rows_evaluated = new_target
            continue
        end

        log_rank!(t.logger, rank_est, r)

        if r == 0
            @show "WARNING: Rank-0 block detected. This may indicate a numerical issue or a degenerate block."
            return Matrix{ComplexF64}(undef, 0, n_src), Int[], 0
        end

        Q1 = @view Q[:, 1:r]
        R11 = UpperTriangular(@view R[1:r, 1:r])

        # `tmp` is completely overwritten by `mul!`, so `undef` is perfectly safe and fast here.
        tmp = Matrix{ComplexF64}(undef, r, n_src)
        mul!(tmp, Q1', Z)
        ldiv!(R11, tmp)

        k = src_index[P[1:r]]
        return tmp, k, r
    end
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
    Rmin=3,
)

    # Center separation

    d = norm(c_s .- c_o)

    # Minimum separation (avoid singular or near-field cases)

    dmin = max(d - 0.5 * (a_s + a_o), 1e-4)

    # Geometric directional rank estimate

    R_geom = C * (k * (a_s * a_o) / dmin)^2

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
    C=73.4706,
    Cε=0.6361,
    Rmin=3,
)
    center = H2Trees.center
    halfsize = H2Trees.halfsize

    c_s = center(trialT, Snode)
    c_o = center(testT, Onode)
    a_s = halfsize(trialT, Snode)
    a_o = halfsize(testT, Onode)

    d = norm(c_s .- c_o)
    dmin = max(d - 0.5 * (a_s + a_o), 1e-4)

    # Isolated predictor variables
    x1 = (k * (a_s * a_o) / dmin)^2
    x2 = log(1 / ε)

    R = ceil(Int, C * x1 + Cε * x2)
    n_otilde = max(R, Rmin)

    return RankEstimate(n_otilde, x1, x2)
end

function estimate_rank_3d(
    k,
    trialT::H2Trees.BoundingBallTree,
    testT::H2Trees.BoundingBallTree,
    Snode::Int,
    Onode::Int,
    ε::Float64;

    C=0.8237,

    Cε=1.7178,

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

    # Minimum separation: Radierna MÅSTE subtraheras i sin helhet!

    # Bytt ut 1e-12 mot 1e-4 för att förhindra IntegerOverflow vid överlappande löv

    dmin = max(d - (a_s + a_o), 1e-4)

    # Isolated predictor variables
    x1 = (k * (a_s * a_o) / dmin)^2
    x2 = log(1 / ε)

    R = ceil(Int, C * x1 + Cε * x2)
    n_otilde = max(R, Rmin)

    return RankEstimate(n_otilde, x1, x2)
end
