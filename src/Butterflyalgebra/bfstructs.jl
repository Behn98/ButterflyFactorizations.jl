# ==========================================
# 1. Parameterized Factor Struct Definitions
# ==========================================

struct R_factor{T,M<:AbstractMatrix}
    Dict::Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},M}}
    slvl::Tuple{Int,Int}
    olvl::Tuple{Int,Int}
    rowstree::T
    rowotree::T
    colstree::T
    colotree::T
end

struct Q_factor{T,M<:AbstractMatrix}
    Dict::Dict{Tuple{Int,Int},M}
    stree::T
end

struct P_factor{T,M<:AbstractMatrix}
    Dict::Dict{Tuple{Int,Int},M}
    otree::T
end

# We introduce a new parameter 'M' for the Matrix/Array type
struct BF{T,M<:AbstractMatrix}
    Q::Dict{Tuple{Int,Int},M}
    R::Vector{Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},M}}}
    P::Dict{Tuple{Int,Int},M}
    dim::Tuple{Int,Int}
    NS::Int64
    NO::Int64
    k::Float64
    τ::Float64
    stree::T
    otree::T

    # Inner constructor updated with both parameters
    BF{T,M}(Q, R, P, dim, NS, NO, k, τ, stree::T, otree::T) where {T,M} =
        new{T,M}(Q, R, P, dim, NS, NO, k, τ, stree, otree)
end

struct AlgBF{T,M<:AbstractMatrix}
    dim::Tuple{Int,Int}
    Q::Q_factor{T}   # Assumed to internalize M or be generic enough
    R::Vector{R_factor{T}}
    P::P_factor{T}

    # 1. Base inner constructor
    AlgBF{T,M}(dim, Q, R, P) where {T,M} = new{T,M}(dim, Q, R, P)

    # 2. Constructor converting BF to AlgBF (Extracts both T and M automatically!)
    function AlgBF(Butterfly::BF{T,M}) where {T,M}
        Q = Q_factor(Butterfly.Q, Butterfly.stree)
        lr = length(Butterfly.R)
        R_vec = Vector{R_factor{T}}(undef, lr)
        for l in eachindex(Butterfly.R)
            R_vec[l] = R_factor(
                Butterfly.R[l],
                (lr - (l - 2), lr - (l - 1)),
                (l, l + 1),
                Butterfly.stree,
                Butterfly.otree,
                Butterfly.stree,
                Butterfly.otree,
            )
        end
        P = P_factor(Butterfly.P, Butterfly.otree)
        return new{T,M}(Butterfly.dim, Q, R_vec, P)
    end

    # 3. Constructor copying/modifying an existing AlgBF with new dictionaries
    function AlgBF(
        BFalg::AlgBF{T,M},
        Q::Dict{Tuple{Int,Int},M},
        R::Vector{Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},M}}},
        P::Dict{Tuple{Int,Int},M},
    ) where {T,M}
        Q_f = Q_factor(Q, BFalg.Q.stree)
        R_factors = Vector{R_factor{T}}(undef, length(R))
        for l in eachindex(R)
            R_factors[l] = R_factor(
                R[l],
                BFalg.R[l].slvl,
                BFalg.R[l].olvl,
                BFalg.R[l].rowstree,
                BFalg.R[l].rowotree,
                BFalg.R[l].colstree,
                BFalg.R[l].colotree,
            )
        end
        P_f = P_factor(P, BFalg.P.otree)
        return new{T,M}(BFalg.dim, Q_f, R_factors, P_f)
    end
end

struct BF_Mats
    Q::AbstractMatrix{ComplexF64}
    R::Vector{AbstractMatrix{ComplexF64}}
    P::AbstractMatrix{ComplexF64}
    NS::Int64
    NO::Int64
    k::Float64
    τ::Float64
    PermP::Vector{Int}
    PermQ::Vector{Int}
    BF_Mats(Q, R, P, NS, NO, k, τ, PermP, PermQ) = new(Q, R, P, NS, NO, k, τ, PermP, PermQ)
end
#AbstractMatrix for SparseArrays, BlockSparseMatrix for BlockSparseMatrices

# En helt platt CSR-nivå för R-faktorerna
struct FlatLinearLayer
    row_ptr::Vector{Int32}       # Var i col_idx/blocks en viss block-rad startar
    col_idx::Vector{Int32}       # Vilket block-kolumnindex varje block har
    blocks::Vector{Matrix{ComplexF64}} # Själva matrisblocken

    row_offsets::Vector{Int}     # Skalärt startindex i ut-vektorn för varje block-rad
    col_offsets::Vector{Int}     # Skalärt startindex i in-vektorn för varje block-kolumn

    in_size::Int                 # Total skalär storlek på in-vektorn för denna nivå
    out_size::Int                # Total skalär storlek på ut-vektorn för denna nivå
end

# Platt representation av Q (läser från globala x via perm, skriver till R[1]:s indata)
struct FlatQLayer
    blocks::Vector{Matrix{ComplexF64}}
    col_offsets::Vector{Int}     # Var i R[1]:s indata-vektor detta block ska skrivas
    perm::Vector{Vector{Int}}    # Globala käll-index (PermQ) för varje block
end

# Platt representation av P (läser från R[end]:s utdata, skriver till globala y via perm)
struct FlatPLayer
    blocks::Vector{Matrix{ComplexF64}}
    row_offsets::Vector{Int}     # Var i R[end]:s utdata-vektor detta block ska läsas ifrån
    perm::Vector{Vector{Int}}    # Globala observatörs-index (PermP) för varje block
end

# Den slutgiltiga, optimerade BF-strukturen
struct FlatBF
    Q::FlatQLayer
    R::Vector{FlatLinearLayer}
    P::FlatPLayer
    dim::Tuple{Int,Int}   # (totala observatörer, totala källor)
    NS::Int64
    NO::Int64
    layer_vectors::Vector{Vector{ComplexF64}} # NYTT: Pre-allocated workspace
end

struct BFSTAT
    Q_ratios::Vector{Tuple{Int,Float64}}
    R_ratios::Vector{Vector{Tuple{Int,Float64}}}
end

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
