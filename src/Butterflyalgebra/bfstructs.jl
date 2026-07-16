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

struct R_factor{T,M<:AbstractMatrix}
    dict::Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},Matrix{M}}}#Matrix of Matrices
    slvl::Tuple{Int,Int}
    olvl::Tuple{Int,Int}
    rowstree::T
    rowotree::T
    colstree::T
    colotree::T
end

struct Q_factor{T,M<:AbstractMatrix}
    dict::Dict{Tuple{Int,Int},M}
    stree::T
    otree::T
end

struct P_factor{T,M<:AbstractMatrix}
    dict::Dict{Tuple{Int,Int},M}
    stree::T
    otree::T
end

struct AlgBF{T,M<:AbstractMatrix}#algebraic butterfly factorization
    dim::Tuple{Int,Int}
    Q::Q_factor{T,M}   # Assumed to internalize M or be generic enough
    R::Vector{R_factor{T,M}}
    P::P_factor{T,M}
    k::Float64
    τ::Float64

    # 1. Base inner constructor
    AlgBF{T,M}(dim, Q, R, P, k, τ) where {T,M} = new{T,M}(dim, Q, R, P, k, τ)
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
