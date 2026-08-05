"""
    Abstractcompressor

Abstract base type for all low-rank approximation strategies used in the Butterfly Factorization.
"""
abstract type Abstractcompressor end

"""
    PartialQRWorkspace{T}

A thread-safe, pre-allocated workspace containing dense matrix buffers.
This prevents the `PartialQR` compressor from reallocating large memory blocks
during the randomized sampling phase of the factorization.
"""
struct PartialQRWorkspace{T}
    buffers::Vector{Matrix{T}}

    function PartialQRWorkspace{T}() where {T}
        # Use maxthreadid() to safely account for interactive/main threads in modern Julia
        n_buffers = if isdefined(Threads, :maxthreadid)
            Threads.maxthreadid()
        else
            Threads.nthreads() + 1
        end

        return new([Matrix{T}(undef, 256, 256) for _ in 1:n_buffers])
    end
end

"""
    PartialQR{L, T} <: Abstractcompressor

A type representing the Partial Pivoted QR compression strategy for low-rank approximations.

# Fields

  - `logger::L`: An optional thread-safe logger to record estimated vs. actual ranks.
  - `workspace::PartialQRWorkspace{T}`: Pre-allocated buffers to prevent memory allocation during sampling.
"""
struct PartialQR{L,T} <: Abstractcompressor
    logger::L
    workspace::PartialQRWorkspace{T}
end

# Default constructor: initializes the workspace automatically with ComplexF64
PartialQR() = PartialQR(nothing, PartialQRWorkspace{ComplexF64}())
PartialQR(logger::L) where {L} = PartialQR(logger, PartialQRWorkspace{ComplexF64}())
# Type-flexible constructor for real or custom precision arithmetic
PartialQR{T}(logger::L=nothing) where {T,L} = PartialQR(logger, PartialQRWorkspace{T}())

# Compiler eliminates this branch when logger === nothing
@inline log_rank!(::Nothing, rank_est, r) = nothing

"""
    log_rank!(logger, est, r)

Records the rank predictor variables and the final computed rank `r` into the thread-local
buffer of the `RankLogger`. If no logger is provided (or if a raw `Int` is passed instead
of a `RankEstimate`), this function compiles away or safely falls back to a no-op.
"""
@inline function log_rank!(logger::RankLogger, est::RankEstimate, r::Int)
    tid = Threads.threadid()
    return push!(logger.buffers[tid], (est.x1, est.x2, r))
end

@inline function log_rank!(logger::RankLogger, est::Int, r::Int)
    # Fallback if a plain Int was passed without geometric terms
    return nothing
end

"""
    (t::PartialQR)(farassembler, src_index, obs_index, rank_est, ε; adaptive=false)

Executes a low-rank approximation of a matrix block using a Partial Pivoted QR decomposition.

To avoid assembling the full dense matrix, this functor randomly samples rows from the
observer space based on `rank_est`. A pivoted QR decomposition is applied to this sample
to find a basis and an active set of column indices (the "skeleton"). If `adaptive=true`,
the algorithm will dynamically sample more rows if the required rank hits the sample limit.
"""
function (t::PartialQR{L,T})(
    farassembler,
    src_index::Vector{Int},
    obs_index::Vector{Int},
    rank_est,
    ε::Float64;
    adaptive=false,
) where {L,T}
    n_obs = length(obs_index)
    n_src = length(src_index)

    # Early exit
    if n_obs == 0 || n_src == 0
        log_rank!(t.logger, rank_est, 0)
        return Matrix{T}(undef, 0, n_src), Int[], 0
    end

    n_otilde_guess = get_n_otilde(rank_est)
    n_otilde = min(max(n_otilde_guess, 10), n_obs)

    shuffled_obs = obs_index[randperm(n_obs)]

    # --- NO-ALLOCATION BUFFER MANAGEMENT ---
    tid = Threads.threadid()
    buffer = t.workspace.buffers[tid]

    # Dynamically grow the thread's buffer if the current block is larger than expected
    if size(buffer, 1) < n_obs || size(buffer, 2) < n_src
        new_rows = max(size(buffer, 1), n_obs)
        new_cols = max(size(buffer, 2), n_src)
        t.workspace.buffers[tid] = Matrix{T}(undef, new_rows, new_cols)
        buffer = t.workspace.buffers[tid]
    end

    # Create a view for the initial sample size and explicitly zero it out
    Z = view(buffer, 1:n_otilde, 1:n_src)
    fill!(Z, zero(T))

    current_rows = @view shuffled_obs[1:n_otilde]
    farassembler(Z, current_rows, src_index)

    rows_evaluated = n_otilde

    while true
        # --- Pivoted QR ---
        # Z is a view. calling copy(Z) creates an exact-sized dense Matrix
        # for pqr to safely mutate, without copying the massive background buffer.
        Zpqr = copy(Z)
        Fqr = pqr(Zpqr; rtol=ε)
        Q, R, P = Fqr[1], Fqr[2], Fqr[3]
        r = size(Q, 2)

        # Adaptive check: expand sample if rank maxed out sampled rows
        if ((r > floor(Int, 0.8 * rows_evaluated) && rows_evaluated < n_obs) && adaptive)
            new_target = min(rows_evaluated * 2, n_obs)

            # Instead of vcat, we just expand our view window into the buffer!
            Z_expanded = view(buffer, 1:new_target, 1:n_src)

            # Isolate the newly added rows and zero them out for BEAST
            Z_new_section = view(buffer, (rows_evaluated + 1):new_target, 1:n_src)
            fill!(Z_new_section, zero(T))

            new_rows_idx = @view shuffled_obs[(rows_evaluated + 1):new_target]
            farassembler(Z_new_section, new_rows_idx, src_index)

            # Update Z to point to the expanded matrix for the next loop iteration
            Z = Z_expanded
            rows_evaluated = new_target
            continue
        end

        log_rank!(t.logger, rank_est, r)

        Q1 = @view Q[:, 1:r]
        R11 = UpperTriangular(@view R[1:r, 1:r])

        tmp = Matrix{T}(undef, r, n_src)
        mul!(tmp, Q1', Z)
        ldiv!(R11, tmp)

        k = src_index[P[1:r]]
        return tmp, k, r
    end
end
