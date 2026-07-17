# We introduce a new parameter 'M' for the Matrix/Array type
const RKey = Tuple{Int,Int}
const CKey = Tuple{Int,Int}

struct BF{T,M<:AbstractMatrix}
    Q::Dict{RKey,M}
    R::Vector{Dict{RKey,Dict{CKey,M}}}
    P::Dict{CKey,M}
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

struct R_factor{M<:AbstractMatrix}
    row_spaces::Dict{RKey,Vector{CKey}}
    col_spaces::Dict{CKey,Vector{RKey}}

    # 2. The Flat Block Locator (Fast Access)
    block_map::Dict{Tuple{RKey,CKey},Tuple{Int,Int,Int}}

    # 3. The Data Payload (Unchanged)
    inverse_map::Vector{Matrix{Tuple{RKey,CKey}}}
    elementblocks::Vector{Matrix{M}}
end

struct Q_factor{T,M<:AbstractMatrix}
    dict::Dict{RKey,M}
    stree::T
    otree::T
end

struct P_factor{T,M<:AbstractMatrix}
    dict::Dict{CKey,M}
    stree::T
    otree::T
end

struct AlgBF{T,M<:AbstractMatrix,S<:AbstractMatrix}#algebraic butterfly factorization
    dim::Tuple{Int,Int}
    Q::Q_factor{T,M}   # Assumed to internalize M or be generic enough
    R::Vector{R_factor{S}}
    P::P_factor{T,M}
    k::Float64
    τ::Float64

    # 1. Base inner constructor
    AlgBF{T,M,S}(dim, Q, R, P, k, τ) where {T,M,S} = new{T,M,S}(dim, Q, R, P, k, τ)
end

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

# ==========================================
# 1. Structural Zero
# ==========================================
struct StructuralZero{T} <: AbstractMatrix{T}
    rows::Int
    cols::Int
end

# ==========================================
# 2. Structural Identity
# ==========================================
struct StructuralIdentity{T} <: AbstractMatrix{T}
    n::Int
end

const BlockType = Union{
    Matrix{ComplexF64},StructuralZero{ComplexF64},StructuralIdentity{ComplexF64}
}
