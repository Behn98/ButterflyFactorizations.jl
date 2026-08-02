import LinearAlgebra: mul!, adjoint, transpose

"""
    transform_block(b::ButterflyBlock{T}, op) where {T}

Applies a linear algebra operation (like `adjoint` or `transpose`) to a single
`ButterflyBlock`.

It structurally swaps the input and output routing keys to reflect the inverted
domain and codomain, and mathematically applies the operation to the underlying
matrix data. Identity blocks (`UniformScaling`) are passed through safely.
"""
function transform_block(b::ButterflyBlock{T}, op) where {T}
    # Handle UniformScaling (Identity): I' = I, transpose(I) = I
    new_data = b.data isa UniformScaling ? b.data : Matrix(op(b.data))

    return ButterflyBlock(
        b.src_in,   # new obs_out (was in)
        b.obs_in,   # new src_out (was in)
        b.src_out,  # new obs_in  (was out)
        b.obs_out,  # new src_in  (was out)
        new_data,
    )
end

"""
    reverse_tree(blktree)

Reverses a coupled block-tree by swapping the source (trial) and observer (test) trees.
Required for mathematically transposing or taking the adjoint of a Butterfly Factorization.
"""
function reverse_tree(blktree)
    # Reverse the block tree by swapping source and target trees
    return cluster_blktree(cluster_trialtree(blktree), cluster_testtree(blktree))
end

"""
    Base.adjoint(BF::ButterflyFactorization)

Explicitly constructs the Adjoint (Hermitian transpose) of a flat-array ButterflyFactorization.
Reverses the cascaded order of the factors and computes the adjoint of every block.
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
        R_adj[l] = ButterflyLevel(new_blocks)
    end

    # 3. Q_adj becomes P'
    P_adj = [transform_block(b, adjoint) for b in BF.Q]

    return ButterflyFactorization(Q_adj, R_adj, P_adj, reverse_tree(BF.tree), BF.k, BF.τ)
end

"""
    Base.transpose(BF::ButterflyFactorization)

Explicitly constructs the Transpose of a flat-array ButterflyFactorization.
Reverses the cascaded order of the factors and computes the transpose of every block.
"""
function Base.transpose(BF::ButterflyFactorization{T,M}) where {T,M}
    Q_trans = [transform_block(b, transpose) for b in BF.P]

    L_minus_1 = length(BF.R)
    R_trans = Vector{ButterflyLevel{T}}(undef, L_minus_1)

    for l in 1:L_minus_1
        orig_level = BF.R[L_minus_1 - l + 1]

        new_blocks = [transform_block(b, transpose) for b in orig_level.blocks]
        R_trans[l] = ButterflyLevel(new_blocks)
    end

    P_trans = [transform_block(b, transpose) for b in BF.Q]

    return ButterflyFactorization(
        Q_trans, R_trans, P_trans, reverse_tree(BF.tree), BF.k, BF.τ
    )
end

"""
    Base.adjoint(t::ButterflyFactorization_Mat)

Explicitly constructs the Adjoint (Hermitian transpose) of a sparse-matrix ButterflyFactorization.
"""
function Base.adjoint(t::ButterflyFactorizations.ButterflyFactorization_Mat{T}) where {T}
    return ButterflyFactorization_Mat(
        t.P',                                           # Q becomes P'
        AbstractMatrix{T}[r' for r in Iterators.reverse(t.R)], # Reverse and map R
        t.Q',                                           # P becomes Q'
        t.no,                                           # ns and no swap roles
        t.ns,
        t.k,
        t.τ,
        t.permQ,                                        # Permutations swap roles
        t.permP,
    )
end

"""
    Base.transpose(t::ButterflyFactorization_Mat)

Explicitly constructs the Transpose of a sparse-matrix ButterflyFactorization.
"""
function Base.transpose(t::ButterflyFactorizations.ButterflyFactorization_Mat{T}) where {T}
    return ButterflyFactorization_Mat(
        transpose(t.P),
        AbstractMatrix{T}[transpose(r) for r in Iterators.reverse(t.R)],
        transpose(t.Q),
        t.no,
        t.ns,
        t.k,
        t.τ,
        t.permQ,
        t.permP,
    )
end
