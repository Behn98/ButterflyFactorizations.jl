import LinearAlgebra: mul!, adjoint, transpose

function Base.adjoint(t::ButterflyFactorizations.ButterflyFactorization_Mat)
    return ButterflyFactorization_Mat(
        t.P',                                                      # Q becomes P'
        AbstractMatrix{ComplexF64}[r' for r in Iterators.reverse(t.R)], # Reverse and map R
        t.Q',                                                      # P becomes Q'
        t.NO,                                                      # NS and NO swap roles
        t.NS,
        t.k,
        t.τ,
        t.PermQ,                                                   # Permutations swap roles
        t.PermP,
    )
end

function Base.transpose(t::ButterflyFactorizations.ButterflyFactorization_Mat)
    return ButterflyFactorization_Mat(
        transpose(t.P),
        AbstractMatrix{ComplexF64}[transpose(r) for r in Iterators.reverse(t.R)],
        transpose(t.Q),
        t.NO,
        t.NS,
        t.k,
        t.τ,
        t.PermQ,
        t.PermP,
    )
end

# Helper function to flip domain/codomain and transform matrix data
function transform_block(b::ButterflyBlock{T}, op) where {T}
    # Handle UniformScaling (Identity): I' = I, transpose(I) = I
    new_data = b.data isa UniformScaling ? b.data : Matrix(op(b.data))

    return ButterflyBlock(
        b.obs_in,   # new obs_out (was in)
        b.src_in,   # new src_out (was in)
        b.obs_out,  # new obs_in  (was out)
        b.src_out,  # new src_in  (was out)
        new_data,
    )
end

"""
    Base.adjoint(BF::ButterflyFactorization)

Explicitly constructs the Adjoint (Hermitian transpose) ButterflyFactorization.
"""
function Base.adjoint(BF::ButterflyFactorization{T,M}) where {T,M}
    # 1. P_adj becomes Q'
    Q_adj = [transform_block(b, adjoint) for b in BF.P]

    # 2. Reversed R levels
    L_minus_1 = length(BF.R)
    R_adj = Vector{ButterflyLevel{T}}(undef, L_minus_1)

    for l in 1:L_minus_1
        # Map level l to level (L - l) of the original factorization
        orig_level = BF.R[L_minus_1 - l + 1]

        new_blocks = [transform_block(b, adjoint) for b in orig_level.blocks]
        sort!(new_blocks; by=block_key)

        R_adj[l] = ButterflyLevel(new_blocks)
    end

    # 3. Q_adj becomes P'
    P_adj = [transform_block(b, adjoint) for b in BF.Q]

    return ButterflyFactorization(Q_adj, R_adj, P_adj, BF.tree, BF.k, BF.τ)
end

"""
    Base.transpose(BF::ButterflyFactorization)

Explicitly constructs the Transpose ButterflyFactorization.
"""
function Base.transpose(BF::ButterflyFactorization{T,M}) where {T,M}
    Q_trans = [transform_block(b, transpose) for b in BF.P]

    L_minus_1 = length(BF.R)
    R_trans = Vector{ButterflyLevel{T}}(undef, L_minus_1)

    for l in 1:L_minus_1
        orig_level = BF.R[L_minus_1 - l + 1]

        new_blocks = [transform_block(b, transpose) for b in orig_level.blocks]
        sort!(new_blocks; by=block_key)

        R_trans[l] = ButterflyLevel(new_blocks)
    end

    P_trans = [transform_block(b, transpose) for b in BF.Q]

    return ButterflyFactorization(Q_trans, R_trans, P_trans, BF.tree, BF.k, BF.τ)
end
