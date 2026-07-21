# These helpers allow you to instantiate factors without explicitly typing out {T, M}
function R_factor(
    Dict::Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},M}},
    slvl,
    olvl,
    rst::T,
    rot::T,
    cst::T,
    cot::T,
) where {T,M}
    return R_factor{T,M}(Dict, slvl, olvl, rst, rot, cst, cot)
end

function Q_factor(Dict::Dict{Tuple{Int,Int},M}, stree::T) where {T,M}
    return Q_factor{T,M}(Dict, stree)
end

function P_factor(Dict::Dict{Tuple{Int,Int},M}, otree::T) where {T,M}
    return P_factor{T,M}(Dict, otree)
end

# An outer constructor helper so you don't always have to pass T and M explicitly
function BF(
    Q::Dict{Tuple{Int,Int},M}, R, P, dim, NS, NO, k, τ, stree::T, otree::T
) where {T,M}
    return BF{T,M}(Q, R, P, dim, NS, NO, k, τ, stree, otree)
end

# --- Updated Companion Outer Constructors & Helpers ---

function AlgBF(dim, Q::Q_factor{T}, R::AbstractVector, P::P_factor{T}) where {T}
    # Fallback to standard dense Matrix if type M can't be inferred directly here
    # (Though usually you'll want to pass the type explicitly or rely on the BF converter)
    return AlgBF{T,Matrix{ComplexF64}}(dim, Q, R, P)
end

# Conversion back from AlgBF to BF
function BF(algBF::AlgBF{T,M}, NS, NO, k, τ, stree::T, otree::T) where {T,M}
    return BF{T,M}(
        algBF.Q.Dict,
        [r.Dict for r in algBF.R],
        algBF.P.Dict,
        algBF.dim,
        NS,
        NO,
        k,
        τ,
        stree,
        otree,
    )
end

# The sorting key now perfectly identifies the exact interaction pair
function block_key(b::ButterflyBlock)
    return (b.obs_out, b.src_out, b.obs_in, b.src_in)
end

function getNSNO(BFactorization::ButterflyFactorization)
    return block_key(BFactorization.P[1])[4], block_key(BFactorization.Q[1])[1]
end
