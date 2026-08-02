"""
    ButterflyBlock{T}

Represents a single low-rank or identity block within a Butterfly Factorization.

To achieve maximum performance during the matrix-vector product, this struct stores
pre-computed execution pointers (`out_ptr` and `in_ptr`). This allows the flat-array
mat-vec algorithm to read and write directly to contiguous memory without expensive
dictionary lookups.

# Fields

  - `obs_out::Int`: The output(row) observer (test) node index.
  - `src_out::Int`: The output(row) source (trial) node index.
  - `obs_in::Int`: The input(col) observer (test) node index.
  - `src_in::Int`: The input(col) source (trial) node index.
  - `data::Union{Matrix{T},UniformScaling{Bool}}`: The matrix data (or identity scaling).
  - `out_ptr::Int`: Memory index for the output vector block in the flat workspace.
  - `in_ptr::Int`: Memory index for the input vector block in the flat workspace.
"""
struct ButterflyBlock{T}
    obs_out::Int
    src_out::Int
    obs_in::Int
    src_in::Int
    data::Union{Matrix{T},UniformScaling{Bool}}
    out_ptr::Int
    in_ptr::Int
end

# Convenient inner/outer constructor for raw blocks during assembly
function ButterflyBlock(
    obs_out::Int,
    src_out::Int,
    obs_in::Int,
    src_in::Int,
    data::Matrix{T},
    out_ptr::Int=0,
    in_ptr::Int=0,
) where {T}
    return ButterflyBlock{T}(obs_out, src_out, obs_in, src_in, data, out_ptr, in_ptr)
end

# Convenience constructor fallback for UniformScaling without explicit type parameters
function ButterflyBlock(
    obs_out::Int,
    src_out::Int,
    obs_in::Int,
    src_in::Int,
    data::UniformScaling{Bool},
    out_ptr::Int=0,
    in_ptr::Int=0,
)
    return ButterflyBlock{ComplexF64}(
        obs_out, src_out, obs_in, src_in, data, out_ptr, in_ptr
    )
end

"""
    ButterflyLevel{T}

A container for all `ButterflyBlock`s that belong to a specific intermediate depth (R-level)
in the Butterfly Factorization.
"""
struct ButterflyLevel{T}
    blocks::Vector{ButterflyBlock{T}}
end

"""
    ButterflyFactorization{T,M}

The core mathematical container for a single factorized block interaction between
a far-field source and observer cluster.

The factorization is represented as a sequence of sparse block matrices:
\$\$ M \\approx P \\cdot R_L \\cdot R_{L-1} \\cdots R_1 \\cdot Q \$\$

The inner constructor automatically wires up the `in_ptr` and `out_ptr` integers
for every block. This transforms the complex hierarchical tree traversal into a
lightning-fast linear memory cascade.

# Fields

  - `Q::Vector{ButterflyBlock{T}}`: The leaf-level observer basis factors.
  - `R::Vector{ButterflyLevel{T}}`: The intermediate hierarchical transfer factors.
  - `P::Vector{ButterflyBlock{T}}`: The leaf-level source basis factors.
  - `tree::M`: The hierarchical tree structure.
  - `k::Float64`: The physical wavenumber.
  - `τ::Float64`: The relative tolerance used during compression.
"""
struct ButterflyFactorization{T,M}
    Q::Vector{ButterflyBlock{T}}
    R::Vector{ButterflyLevel{T}}
    P::Vector{ButterflyBlock{T}}
    tree::M
    k::Float64
    τ::Float64

    function ButterflyFactorization(
        Q::Vector{ButterflyBlock{T}},
        R::Vector{ButterflyLevel{T}},
        P::Vector{ButterflyBlock{T}},
        tree::M,
        k::Float64,
        τ::Float64,
    ) where {T,M}
        num_levels = length(R) + 1

        # ONE shared map per level ensures output pointers perfectly match input pointers!
        level_maps = [Dict{Tuple{Int,Int},Int}() for _ in 1:num_levels]

        function get_ptr!(lvl_idx, key)
            m = level_maps[lvl_idx]
            return get!(m, key, length(m) + 1)
        end

        indexed_Q = map(Q) do b
            out_k = (b.obs_out, b.src_out)
            return ButterflyBlock(
                b.obs_out, b.src_out, b.obs_in, b.src_in, b.data, get_ptr!(1, out_k), 0
            )
        end

        indexed_R = map(enumerate(R)) do (l, level)
            indexed_blocks = map(level.blocks) do b
                in_k  = (b.obs_in, b.src_in)
                out_k = (b.obs_out, b.src_out)

                in_p  = get_ptr!(l, in_k)       # Read from level l
                out_p = get_ptr!(l + 1, out_k)  # Write to level l+1

                return ButterflyBlock(
                    b.obs_out, b.src_out, b.obs_in, b.src_in, b.data, out_p, in_p
                )
            end
            return ButterflyLevel(indexed_blocks)
        end

        indexed_P = map(P) do b
            in_k = (b.obs_in, b.src_in)
            in_p = get_ptr!(num_levels, in_k)   # Read from final level
            return ButterflyBlock(b.obs_out, b.src_out, b.obs_in, b.src_in, b.data, 0, in_p)
        end

        return new{T,M}(indexed_Q, indexed_R, indexed_P, tree, k, τ)
    end
end

"""
    ButterflyWorkspace{T}

A dictionary-based workspace used to store intermediate skeleton vectors during
mat-vec evaluation. Mostly used for legacy or unindexed evaluation modes.
"""
struct ButterflyWorkspace{T}
    level_buffers::Vector{Dict{Tuple{Int,Int},Vector{T}}}
end

"""
    ThreadButterflyWorkspace{T}

A highly optimized, pre-allocated flat-array workspace designed for thread-safe,
zero-allocation matrix-vector products.
"""
struct ThreadButterflyWorkspace{T}
    level_buffers::Vector{Vector{Vector{T}}}
    level_lengths::Vector{Vector{Int}}
end

"""
    ButterflyFactorization_Mat{T}

A legacy or dense wrapper container for a fully instantiated Butterfly matrix.
Maintains the block permutation mappings required to map the hierarchical data
back to the original physical degrees of freedom.
"""
struct ButterflyFactorization_Mat{T}
    Q::AbstractMatrix{T}
    R::Vector{AbstractMatrix{T}}
    P::AbstractMatrix{T}
    ns::Int64 # Source dimension
    no::Int64 # Observer dimension
    k::Float64
    τ::Float64
    permP::Vector{Int}
    permQ::Vector{Int}

    ButterflyFactorization_Mat(
        Q::AbstractMatrix{T},
        R::Vector{<:AbstractMatrix{T}},
        P::AbstractMatrix{T},
        ns::Int64,
        no::Int64,
        k::Float64,
        τ::Float64,
        permP::Vector{Int},
        permQ::Vector{Int},
    ) where {T} = new{T}(Q, R, P, ns, no, k, τ, permP, permQ)
end

"""
    RankEstimate

A lightweight container holding the estimated rank `n_otilde` along with the raw
geometric (`x1`) and algebraic (`x2`) predictor variables used to compute it.
"""
struct RankEstimate
    n_otilde::Int
    x1::Float64
    x2::Float64
end

"""
    RankLogger

A thread-safe logger used during parallel assembly to record the true ranks
computed by the compressor against the initial geometric rank estimates.
Used for dynamic auto-calibration of the rank predictor constants.
"""
struct RankLogger
    buffers::Vector{Vector{Tuple{Float64,Float64,Int}}}

    function RankLogger()
        # Uses maxthreadid() if available to prevent crashes with dynamic thread pools
        n_bufs = if isdefined(Threads, :maxthreadid)
            Threads.maxthreadid()
        else
            Threads.nthreads() + 1
        end
        return new([Vector{Tuple{Float64,Float64,Int}}() for _ in 1:n_bufs])
    end
end
