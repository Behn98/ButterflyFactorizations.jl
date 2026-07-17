import Base: size, getindex, *, +, adjoint

function (t::R_factor{M})(row::Tuple{Int,Int}, col::Tuple{Int,Int}) where {M}
    if haskey(t.dict, row) && haskey(t.dict[row], col)
        (eidx, i, j) = t.dict[row][col]
        return t.elementblocks[eidx][i, j]
    else
        return zero(M)
    end
end

# ==========================================
# 2. Outer Constructors for Type Inference
# ==========================================

# These helpers allow you to instantiate factors without explicitly typing out {T, M}
function R_factor(dict::Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},M}}) where {M}
    rowgrps = group_identical_colspaces(dict)
    eidx = 1
    elementblocks = Vector{Matrix{M}}()
    rowmap = Dict{RKey,Vector{CKey}}()
    colmap = Dict{CKey,Vector{RKey}}()
    mapping = Dict{Tuple{RKey,CKey},Tuple{Int,Int,Int}}()
    invmapping = Vector{Matrix{Tuple{RKey,CKey}}}()
    for (col_space, rows) in rowgrps
        matrix_grid, r_labels, c_labels = build_matrix_from_group(dict, rows, col_space)
        push!(elementblocks, matrix_grid)
        invmappinge = Matrix{Tuple{RKey,CKey}}(undef, length(r_labels), length(c_labels))
        for row in r_labels
            rowmap[row] = c_labels
        end
        for col in c_labels
            colmap[col] = r_labels
        end
        for (i, row) in enumerate(r_labels)
            for (j, col) in enumerate(c_labels)
                mapping[(row, col)] = (eidx, i, j)
                invmappinge[i, j] = (row, col)
            end
        end
        push!(invmapping, invmappinge)
        eidx += 1
    end
    return R_factor{M}(rowmap, colmap, mapping, invmapping, elementblocks)
end

function R_factor(
    rowmap::Dict{RKey,Vector{CKey}},
    colmap::Dict{CKey,Vector{RKey}},
    mapping::Dict{Tuple{RKey,CKey},Tuple{Int,Int,Int}},
    invmapping::Vector{Matrix{Tuple{RKey,CKey}}},
    elementblocks::Vector{Matrix{M}},
) where {M}
    return R_factor{M}(rowmap, colmap, mapping, invmapping, elementblocks)
end

function Q_factor(dict::Dict{RKey,M}, stree::T, otree::T) where {T,M}
    return Q_factor{T,M}(dict, stree, otree)
end

function P_factor(dict::Dict{CKey,M}, otree::T, stree::T) where {T,M}
    return P_factor{T,M}(dict, stree, otree)
end

# An outer constructor helper so you don't always have to pass T and M explicitly
function BF(
    Q::Dict{Tuple{Int,Int},M}, R, P, dim, NS, NO, k, τ, stree::T, otree::T
) where {T,M}
    return BF{T,M}(Q, R, P, dim, NS, NO, k, τ, stree, otree)
end

# --- Updated Companion Outer Constructors & Helpers ---

function AlgBF(
    dim, Q::Q_factor{T,M}, R::AbstractVector{R_factor{S}}, P::P_factor{T,M}, k, τ
) where {T,M,S}
    return AlgBF{T,M,S}(dim, Q, R, P, k, τ)
end
#=
function AlgBF(
    dim, Q::Q_factor{T,M}, R::AbstractVector{R_factor{M}}, P::P_factor{T,M}, k, τ
) where {T,M}
    return AlgBF{T,M,M}(dim, Q, R, P, k, τ)
end
=#
function AlgBF(Butterfly::BF{T,M}) where {T,M}
    Q = Q_factor(Butterfly.Q, Butterfly.stree, Butterfly.otree)
    lr = length(Butterfly.R)
    R_vec = Vector{R_factor{M}}(undef, lr)
    for l in eachindex(Butterfly.R)
        R_vec[l] = R_factor(Butterfly.R[l])
    end
    P = P_factor(Butterfly.P, Butterfly.otree, Butterfly.stree)
    return AlgBF{T,M,M}(Butterfly.dim, Q, R_vec, P, Butterfly.k, Butterfly.τ)
end

function AlgBF(
    BFalg::AlgBF{T,M,S},
    Q::Dict{RKey,M},
    R::Vector{Dict{RKey,Dict{CKey,S}}},
    P::Dict{CKey,M},
) where {T,M,S}
    Q_f = Q_factor(Q, BFalg.Q.stree, BFalg.Q.otree)
    R_factors = Vector{R_factor{S}}(undef, length(R))
    for l in eachindex(R)
        R_factors[l] = R_factor(R[l])
    end
    P_f = P_factor(P, BFalg.P.otree, BFalg.P.stree)
    return AlgBF{T,M,S}(BFalg.dim, Q_f, R_factors, P_f, BFalg.k, BFalg.τ)
end

function AlgBF(
    BFalg::AlgBF{T,M,S}, Q::Dict{RKey,M}, R::Vector{R_factor{S}}, P::Dict{CKey,M}
) where {T,M,S}
    Q_f = Q_factor(Q, BFalg.Q.stree, BFalg.Q.otree)
    P_f = P_factor(P, BFalg.P.otree, BFalg.P.stree)
    return AlgBF{T,M,S}(BFalg.dim, Q_f, R, P_f, BFalg.k, BFalg.τ)
end

# Conversion back from AlgBF to BF
function BF(algBF::AlgBF{T,M,S}, k, τ) where {T,M,S}
    NS = first(keys(algBF.P.dict))[2]
    NO = first(keys(algBF.Q.dict))[1]
    stree = algBF.Q.stree
    otree = algBF.P.otree
    newR = Vector{Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},M}}}(undef, length(algBF.R))
    for l in eachindex(algBF.R)
        newR[l] = Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},M}}()
        for ((row, col), (eidx, i, j)) in algBF.R[l].block_map
            if !haskey(newR[l], row)
                newR[l][row] = Dict{Tuple{Int,Int},M}()
            end
            if typeof(algBF.R[l].elementblocks[eidx][i, j]) == StructuralIdentity
                block = algBF.R[l].elementblocks[eidx][i, j]
                newR[l][row][col] = Matrix{eltype(block)}(I, size(block, 1), size(block, 2))
            elseif typeof(algBF.R[l].elementblocks[eidx][i, j]) == StructuralZero
                newR[l][row][col] = Matrix{eltype(block)}(eltype(block), 0, 0)
            else
                newR[l][row][col] = algBF.R[l].elementblocks[eidx][i, j]
            end
        end
    end
    return BF{T,M}(algBF.Q.dict, newR, algBF.P.dict, algBF.dim, NS, NO, k, τ, stree, otree)
end

size(Z::StructuralZero) = (Z.rows, Z.cols)
# If a solver desperately tries to read a scalar, give it a zero safely
getindex(Z::StructuralZero{T}, i::Int, j::Int) where {T} = zero(T)

# Fast Transpose / Adjoint
adjoint(Z::StructuralZero{T}) where {T} = StructuralZero{T}(Z.cols, Z.rows)

size(I::StructuralIdentity) = (I.n, I.n)
getindex(I::StructuralIdentity{T}, i::Int, j::Int) where {T} = i == j ? one(T) : zero(T)

adjoint(I::StructuralIdentity{T}) where {T} = I

# ------------------------------------------
# Multiplication Overloads
# ------------------------------------------

# Any Matrix * Zero = Zero (Propagate the new correct dimensions!)
*(Z::StructuralZero{T}, A::AbstractMatrix) where {T} =
    StructuralZero{T}(size(Z, 1), size(A, 2))
*(A::AbstractMatrix, Z::StructuralZero{T}) where {T} =
    StructuralZero{T}(size(A, 1), size(Z, 2))

# Zero * Zero = Zero
*(Z1::StructuralZero{T}, Z2::StructuralZero{T}) where {T} =
    StructuralZero{T}(size(Z1, 1), size(Z2, 2))

# Any Matrix * Identity = Matrix
*(I::StructuralIdentity, A::AbstractMatrix) = A
*(A::AbstractMatrix, I::StructuralIdentity) = A

# Identity * Identity = Identity
*(I1::StructuralIdentity{T}, I2::StructuralIdentity{T}) where {T} = I1

# ------------------------------------------
# Addition Overloads
# ------------------------------------------

# Any Matrix + Zero = Matrix
+(Z::StructuralZero, A::AbstractMatrix) = A
+(A::AbstractMatrix, Z::StructuralZero) = A

# Zero + Zero = Zero
+(Z1::StructuralZero{T}, Z2::StructuralZero{T}) where {T} = Z1
