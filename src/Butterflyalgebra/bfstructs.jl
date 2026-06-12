struct AlgBF
    dim::Tuple{Int,Int}
    Q::Dict{Int,Matrix{ComplexF64}}
    R::Vector{Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},AbstractMatrix{ComplexF64}}}}
    P::Dict{Int,Matrix{ComplexF64}}
end

struct BF
    Q::Dict{Int,AbstractMatrix{ComplexF64}}
    R::Vector{Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},AbstractMatrix{ComplexF64}}}}
    P::Dict{Int,AbstractMatrix{ComplexF64}}
    PermQ::Dict{Int,Vector{Int}}
    PermP::Dict{Int,Vector{Int}}
    dim::Tuple{Int,Int}
    NS::Int64
    NO::Int64
    k::Float64
    τ::Float64
    tree::H2Trees.BlockTree
    BF(Q, R, P, PermQ, PermP, dim, NS, NO, k, τ, tree) =
        new(Q, R, P, PermQ, PermP, dim, NS, NO, k, τ, tree)
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
