# A single block in the butterfly factorization
struct ButterflyBlock{T}
    obs_out::Int  # e.g., Ochild
    src_out::Int  # e.g., Svert
    obs_in::Int   # e.g., Overt
    src_in::Int   # e.g., Schild
    data::Union{Matrix{T},UniformScaling{Bool}}

    # Execution pointers for flat-array workspaces (0 = unindexed/algebraic mode)
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
# An entire level l of the R factors
struct ButterflyLevel{T}
    blocks::Vector{ButterflyBlock{T}}
end
# In bfstructs.jl

struct ButterflyFactorization{T,M}
    Q::Vector{ButterflyBlock{T}}
    R::Vector{ButterflyLevel{T}}
    P::Vector{ButterflyBlock{T}}
    tree::M
    k::Float64
    τ::Float64

    # Inner constructor intercepts the raw blocks
    function ButterflyFactorization(
        Q::Vector{ButterflyBlock{T}},
        R::Vector{ButterflyLevel{T}},
        P::Vector{ButterflyBlock{T}},
        tree::M,
        k::Float64,
        τ::Float64,
    ) where {T,M}
        num_levels = length(R) + 1

        # 🚀 ONE shared map per level ensures output pointers perfectly match input pointers!
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

        # 'new' is Julia's internal keyword for building the struct inside an inner constructor
        return new{T,M}(indexed_Q, indexed_R, indexed_P, tree, k, τ)
    end
end

struct ButterflyWorkspace{T}
    # Pre-allocated vector buffers for intermediate skeleton states at each level.
    # Indexed as: levels[l][(obs_node, src_node)]
    level_buffers::Vector{Dict{Tuple{Int,Int},Vector{T}}}
end

struct ThreadButterflyWorkspace{T}
    level_buffers::Vector{Vector{Vector{T}}}
    level_lengths::Vector{Vector{Int}}
end

struct ButterflyFactorization_Mat
    Q::AbstractMatrix{ComplexF64}
    R::Vector{AbstractMatrix{ComplexF64}}
    P::AbstractMatrix{ComplexF64}
    NS::Int64
    NO::Int64
    k::Float64
    τ::Float64
    PermP::Vector{Int}
    PermQ::Vector{Int}
    ButterflyFactorization_Mat(Q, R, P, NS, NO, k, τ, PermP, PermQ) =
        new(Q, R, P, NS, NO, k, τ, PermP, PermQ)
end

# Lightweight container holding the initial guess and the raw predictor variables
struct RankEstimate
    n_otilde::Int
    x1::Float64   # (k * a_s * a_o / dmin)^2
    x2::Float64   # log(1 / ε)
end

# Thread-safe logger for parallel block assembly
struct RankLogger
    buffers::Vector{Vector{Tuple{Float64,Float64,Int}}}
    RankLogger() =
        new([Vector{Tuple{Float64,Float64,Int}}() for _ in 1:(Threads.nthreads() + 1)])
end
