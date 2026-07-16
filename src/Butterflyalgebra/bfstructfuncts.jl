import Base: size, getindex, *, +, adjoint

function (t::R_factor{T,M})(row::Tuple{Int,Int}, col::Tuple{Int,Int}) where {T,M}
    if haskey(t.dict, row) && haskey(t.dict[row], col)
        (eidx, j) = t.dict[row][col]
        return t.elementblocks[eidx][:, j]
    else
        return zero(M)
    end
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
    return R_factor{T,M}(mapping, invmapping, elementblocks, slvl, olvl, rst, rot, cst, cot)
end

function Q_factor(dict::Dict{Tuple{Int,Int},M}, stree::T, otree::T) where {T,M}
    return Q_factor{T,M}(dict, stree, otree)
end

function P_factor(dict::Dict{Tuple{Int,Int},M}, otree::T, stree::T) where {T,M}
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
    dim, Q::Q_factor{T,M}, R::AbstractVector{R_factor{T,M}}, P::P_factor{T,M}
) where {T,M}
    return AlgBF{T,M}(dim, Q, R, P)
end

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
    return AlgBF{T,M}(Butterfly.dim, Q, R_vec, P)
end

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
    return AlgBF{T,M}(BFalg.dim, Q_f, R_factors, P_f)
end

function AlgBF(
    BFalg::AlgBF{T,M},
    Q::Dict{Tuple{Int,Int},M},
    R::Vector{R_factor{T,M}},
    P::Dict{Tuple{Int,Int},M},
) where {T,M}
    Q_f = Q_factor(Q, BFalg.Q.stree, BFalg.Q.otree)
    P_f = P_factor(P, BFalg.P.otree, BFalg.P.stree)
    return AlgBF{T,M}(BFalg.dim, Q_f, R, P_f, BFalg.k, BFalg.τ)
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

const BlockType = Union{
    Matrix{ComplexF64},StructuralZero{ComplexF64},StructuralIdentity{ComplexF64}
}
