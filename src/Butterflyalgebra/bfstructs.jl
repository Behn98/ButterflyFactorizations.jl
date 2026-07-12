import Base: show
# ==========================================
# 1. Parameterized Factor Struct Definitions
# ==========================================

struct R_factor{T,M<:AbstractMatrix}
    dict::Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},Tuple{Int,Int,Int}}}#(E,row,col))
    elementblocks::Vector{Matrix{M}}
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

# ==========================================
# 2. Outer Constructors for Type Inference
# ==========================================

# These helpers allow you to instantiate factors without explicitly typing out {T, M}
function R_factor(
    dict::Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},M}},
    slvl,
    olvl,
    rst::T,
    rot::T,
    cst::T,
    cot::T,
) where {T,M}
    rowgrps = group_identical_colspaces(dict)
    eidx = 1
    elementblocks = Vector{Matrix{M}}()
    mapping = Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},Tuple{Int,Int,Int}}}()
    for (col_space, rows) in rowgrps
        matrix_grid, r_labels, c_labels = build_matrix_from_group(dict, rows, col_space)
        push!(elementblocks, matrix_grid)
        for (i, row) in enumerate(r_labels)
            if !haskey(mapping, row)
                mapping[row] = Dict{Tuple{Int,Int},Tuple{Int,Int,Int}}()
            end
            for (j, col) in enumerate(c_labels)
                mapping[row][col] = (eidx, i, j)
            end
        end
        eidx += 1
    end
    return R_factor{T,M}(mapping, elementblocks, slvl, olvl, rst, rot, cst, cot)
end

function R_factor(
    dict::Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},Tuple{Int,Int,Int}}},
    elementblocks::Vector{Matrix{M}},
    slvl,
    olvl,
    rst::T,
    rot::T,
    cst::T,
    cot::T,
) where {T,M}
    return R_factor{T,M}(dict, elementblocks, slvl, olvl, rst, rot, cst, cot)
end

function (t::R_factor{T,M})(row::Tuple{Int,Int}, col::Tuple{Int,Int}) where {T,M}
    if haskey(t.dict, row) && haskey(t.dict[row], col)
        (eidx, i, j) = t.dict[row][col]
        return t.elementblocks[eidx][i, j]
    else
        return zero(M)
    end
end

function Q_factor(dict::Dict{Tuple{Int,Int},M}, stree::T, otree::T) where {T,M}
    return Q_factor{T,M}(dict, stree, otree)
end

function P_factor(dict::Dict{Tuple{Int,Int},M}, otree::T, stree::T) where {T,M}
    return P_factor{T,M}(dict, stree, otree)
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

# An outer constructor helper so you don't always have to pass T and M explicitly
function BF(
    Q::Dict{Tuple{Int,Int},M}, R, P, dim, NS, NO, k, τ, stree::T, otree::T
) where {T,M}
    return BF{T,M}(Q, R, P, dim, NS, NO, k, τ, stree, otree)
end

struct AlgBF{T,M<:AbstractMatrix}#algebraic butterfly factorization
    dim::Tuple{Int,Int}
    Q::Q_factor{T,M}   # Assumed to internalize M or be generic enough
    R::Vector{R_factor{T,M}}
    P::P_factor{T,M}

    # 1. Base inner constructor
    AlgBF{T,M}(dim, Q, R, P) where {T,M} = new{T,M}(dim, Q, R, P)

    # 2. Constructor converting BF to AlgBF (Extracts both T and M automatically!)
    function AlgBF(Butterfly::BF{T,M}) where {T,M}
        Q = Q_factor(Butterfly.Q, Butterfly.stree, Butterfly.otree)
        lr = length(Butterfly.R)
        R_vec = Vector{R_factor{T,M}}(undef, lr)
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
        P = P_factor(Butterfly.P, Butterfly.otree, Butterfly.stree)
        return new{T,M}(Butterfly.dim, Q, R_vec, P)
    end

    # 3. Constructor copying/modifying an existing AlgBF with new dictionaries
    function AlgBF(
        BFalg::AlgBF{T,M},
        Q::Dict{Tuple{Int,Int},M},
        R::Vector{Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},M}}},
        P::Dict{Tuple{Int,Int},M},
    ) where {T,M}
        Q_f = Q_factor(Q, BFalg.Q.stree, BFalg.Q.otree)
        R_factors = Vector{R_factor{T,M}}(undef, length(R))
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
        P_f = P_factor(P, BFalg.P.otree, BFalg.P.stree)
        return new{T,M}(BFalg.dim, Q_f, R_factors, P_f)
    end
end

# --- Updated Companion Outer Constructors & Helpers ---

function AlgBF(
    dim, Q::Q_factor{T,M}, R::AbstractVector{R_factor{T,M}}, P::P_factor{T,M}
) where {T,M}
    return AlgBF{T,M}(dim, Q, R, P)
end

# Conversion back from AlgBF to BF
function BF(algBF::AlgBF{T,M}, k, τ) where {T,M}
    NS = first(keys(algBF.P.dict))[2]
    NO = first(keys(algBF.Q.dict))[1]
    stree = algBF.Q.stree
    otree = algBF.P.otree
    newR = Vector{Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},M}}}(undef, length(algBF.R))
    for l in eachindex(algBF.R)
        newR[l] = Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},M}}()
        for (row, col_dict) in algBF.R[l].dict
            newR[l][row] = Dict{Tuple{Int,Int},M}()
            for (col, (eidx, i, j)) in col_dict
                newR[l][row][col] = algBF.R[l].elementblocks[eidx][i, j]
            end
        end
    end
    return BF{T,M}(algBF.Q.dict, newR, algBF.P.dict, algBF.dim, NS, NO, k, τ, stree, otree)
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

# 1-line summary
function show(io::IO, alg::AlgBF{T,M}) where {T,M}
    return print(io, "AlgBF{$T, $M}(dim=$(alg.dim))")
end

# Multi-line REPL presentation
function show(io::IO, mime::MIME"text/plain", alg::AlgBF{T,M}) where {T,M}
    println(io, "Algebraic Butterfly Factorization (AlgBF)")
    println(io, "  Tree Type:   ", T)
    println(io, "  Matrix Type: ", M)
    println(io, "  Dimension:   ", alg.dim)
    return print(
        io,
        "  Structure:   Q (",
        typeof(alg.Q),
        "), R (",
        length(alg.R),
        " levels), P (",
        typeof(alg.P),
        ")",
    )
end

# --- Q_factor Display ---
function show(io::IO, q::Q_factor{T,M}) where {T,M}
    return print(io, "Q_factor{$T, $M}(keys=$(length(q.dict)))")
end

function show(io::IO, ::MIME"text/plain", q::Q_factor{T,M}) where {T,M}
    println(io, "Butterfly Q_factor")
    println(io, "  Matrix Type: ", M)
    println(io, "  Tree Type:   ", T)
    return print(io, "  Storage:     ", length(q.dict), " blocks mapped in Dict")
end

# --- P_factor Display ---
function show(io::IO, p::P_factor{T,M}) where {T,M}
    return print(io, "P_factor{$T, $M}(keys=$(length(p.dict)))")
end

function show(io::IO, ::MIME"text/plain", p::P_factor{T,M}) where {T,M}
    println(io, "Butterfly P_factor")
    println(io, "  Matrix Type: ", M)
    println(io, "  Tree Type:   ", T)
    return print(io, "  Storage:     ", length(p.dict), " blocks mapped in Dict")
end

# --- R_factor Display ---
function show(io::IO, r::R_factor{T,M}) where {T,M}
    return print(io, "R_factor{$T, $M}(slvl=$(r.slvl), olvl=$(r.olvl))")
end

function show(io::IO, ::MIME"text/plain", r::R_factor{T,M}) where {T,M}
    println(io, "Butterfly R_factor (Middle Level)")
    println(io, "  Matrix Type: ", M)
    println(io, "  Levels (s/o):", r.slvl, " / ", r.olvl)
    return print(io, "  Storage:     ", length(r.dict), " top-level block keys")
end

# 1. This controls the 1-line display (e.g., when inside arrays)
function show(io::IO, bf::BF{T,M}) where {T,M}
    return print(io, "BF{$T, $M}(dim=$(bf.dim), k=$(bf.k))")
end

# 2. This controls the multi-line display when a BF object is returned in the REPL
function show(io::IO, mime::MIME"text/plain", bf::BF{T,M}) where {T,M}
    println(io, "Butterfly Factorization (BF)")
    println(io, "  Tree Type:   ", T)
    println(io, "  Matrix Type: ", M)
    println(io, "  Dimension:   ", bf.dim)
    println(io, "  NS / NO:     ", bf.NS, " / ", bf.NO)
    println(io, "  k / τ:       ", bf.k, " / ", bf.τ)
    return print(
        io,
        "  Storage:     Q ($(length(bf.Q)) keys), R ($(length(bf.R)) levels), P ($(length(bf.P)) keys)",
    )
end

# 1. This controls the 1-line display (e.g., when inside arrays)
function show(io::IO, bf::BF_Mats)
    return print(io, "BF_Mats(dim=$(bf.dim), k=$(bf.k))")
end

# 2. This controls the multi-line display when a BF object is returned in the REPL
function show(io::IO, mime::MIME"text/plain", bf::BF_Mats)
    println(io, "Butterfly Factorization (BF_Mats)")
    println(io, "  Dimension:   ", bf.dim)
    println(io, "  NS / NO:     ", bf.NS, " / ", bf.NO)
    println(io, "  k / τ:       ", bf.k, " / ", bf.τ)
    return print(
        io,
        "  Storage:     Q ($(Base.summarysize(bf.Q)/1024) KB), R ($(length(bf.R)) levels), P ($(Base.summarysize(bf.P)/1024) KB)",
    )
end
