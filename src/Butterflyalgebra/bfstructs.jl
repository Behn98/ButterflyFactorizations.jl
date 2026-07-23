# A single block in the butterfly factorization
struct ButterflyBlock{T}
    # CODOMAIN (Output / Row-equivalent)
    # The skeleton this block maps TO
    obs_out::Int  # e.g., Ochild
    src_out::Int  # e.g., Svert

    # DOMAIN (Input / Column-equivalent)
    # The skeleton this block maps FROM
    obs_in::Int   # e.g., Overt
    src_in::Int   # e.g., Schild

    data::Union{Matrix{T},UniformScaling{Bool}}
end

ButterflyBlock(
    obs_out::Int, src_out::Int, obs_in::Int, src_in::Int, data::Matrix{T}
) where {T} = ButterflyBlock{T}(obs_out, src_out, obs_in, src_in, data)

ButterflyBlock(
    obs_out::Int, src_out::Int, obs_in::Int, src_in::Int, data::UniformScaling{Bool}
) = ButterflyBlock{ComplexF64}(obs_out, src_out, obs_in, src_in, data)

# An entire level l of the R factors
struct ButterflyLevel{T}
    blocks::Vector{ButterflyBlock{T}}
end

# The complete factorization
struct ButterflyFactorization{T,M}
    Q::Vector{ButterflyBlock{T}}       # Leaf level
    R::Vector{ButterflyLevel{T}}       # Levels 1 to L-1
    P::Vector{ButterflyBlock{T}}       # Root/Observer leaf level
    tree::M
    k::Float64
    τ::Float64
end

struct ButterflyWorkspace{T}
    # Pre-allocated vector buffers for intermediate skeleton states at each level.
    # Indexed as: levels[l][(obs_node, src_node)]
    level_buffers::Vector{Dict{Tuple{Int,Int},Vector{T}}}
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
