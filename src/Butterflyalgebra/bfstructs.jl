struct R_factor{T}
    Dict::Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},Matrix{ComplexF64}}}
    slvl::Tuple{Int,Int}
    olvl::Tuple{Int,Int}
    rowstree::T
    rowotree::T
    colstree::T
    colotree::T
end

struct Q_factor{T}
    Dict::Dict{Int,Matrix{ComplexF64}}
    stree::T
end

struct P_factor{T}
    Dict::Dict{Int,Matrix{ComplexF64}}
    otree::T
end

struct BF{T}
    Q::Dict{Int,Matrix{ComplexF64}}
    R::Vector{Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},Matrix{ComplexF64}}}}
    P::Dict{Int,Matrix{ComplexF64}}
    PermQ::Dict{Int,Vector{Int}}
    PermP::Dict{Int,Vector{Int}}
    dim::Tuple{Int,Int}
    NS::Int64
    NO::Int64
    k::Float64
    τ::Float64
    stree::T
    otree::T
    BF{T}(Q, R, P, PermQ, PermP, dim, NS, NO, k, τ, stree::T, otree::T) where {T} =
        new{T}(Q, R, P, PermQ, PermP, dim, NS, NO, k, τ, stree, otree)
end

# Helper outer constructor to automatically infer T
function BF(Q, R, P, PermQ, PermP, dim, NS, NO, k, τ, stree::T, otree::T) where {T}
    return BF{T}(Q, R, P, PermQ, PermP, dim, NS, NO, k, τ, stree, otree)
end

struct AlgBF{T}
    dim::Tuple{Int,Int}
    Q::Q_factor{T}
    R::Vector{R_factor{T}}
    P::P_factor{T}
    AlgBF{T}(dim, Q, R, P) where {T} = new{T}(dim, Q, R, P)
    # Constructor converting BF to AlgBF
    function AlgBF(Butterfly::BF{T}) where {T}
        Q = Q_factor(Butterfly.Q, Butterfly.stree)
        lr = length(Butterfly.R)
        R = Vector{R_factor{T}}(undef, lr)
        for l in eachindex(Butterfly.R)
            R[l] = R_factor(
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
        return new{T}(Butterfly.dim, Q, R, P)
    end

    # Constructor copying/modifying an existing AlgBF
    function AlgBF(
        BFalg::AlgBF{T},
        Q::Dict{Int,Matrix{ComplexF64}},
        R::Vector{Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},Matrix{ComplexF64}}}},
        P::Dict{Int,Matrix{ComplexF64}},
    ) where {T}
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
        return new{T}(BFalg.dim, Q_f, R_factors, P_f)
    end
end

function AlgBF(dim, Q::Q_factor{T}, R::AbstractVector, P::P_factor{T}) where {T}
    return AlgBF{T}(dim, Q, R, P)
end

function BF(algBF::AlgBF{T}, PermQ, PermP, NS, NO, k, τ, stree::T, otree::T) where {T}
    return BF{T}(
        algBF.Q.Dict,
        [r.Dict for r in algBF.R],
        algBF.P.Dict,
        PermQ,
        PermP,
        algBF.dim,
        NS,
        NO,
        k,
        τ,
        stree,
        otree,
    )
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
end

struct BFSTAT
    Q_ratios::Vector{Tuple{Int,Float64}}
    R_ratios::Vector{Vector{Tuple{Int,Float64}}}
end
