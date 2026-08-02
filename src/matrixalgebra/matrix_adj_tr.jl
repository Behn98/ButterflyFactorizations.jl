import Base: adjoint, transpose
"""
    Base.adjoint(A::PetrovGalerkinBF)

Constructs the Hermitian transpose (Adjoint) of the global Butterfly operator.
Safely reverses the underlying tree structures, block interactions, and near-field matrices.
"""
function Base.adjoint(A::ButterflyFactorizations.PetrovGalerkinBF{T}) where {T}
    # Adjoint all individual far-field blocks
    adj_BFs = map(adjoint, A.BFs)

    # Reconstruct the global operator (inner constructor handles ThreadWorkspaces automatically)
    return PetrovGalerkinBF{T}(
        adjoint(A.nearinteractions),
        reverse_tree(A.tree),             # Swap global test/trial trees
        adj_BFs,
        reverse(A.dim),
        sparse(transpose(A.near_lookup)), # Transpose is fine for mapping integer indices
        sparse(transpose(A.far_lookup)),
    )
end

"""
    Base.transpose(A::PetrovGalerkinBF)

Constructs the Transpose of the global Butterfly operator.
"""
function Base.transpose(A::ButterflyFactorizations.PetrovGalerkinBF{T}) where {T}
    # Transpose all individual far-field blocks
    trans_BFs = map(transpose, A.BFs)

    return PetrovGalerkinBF{T}(
        transpose(A.nearinteractions),
        reverse_tree(A.tree),             # Swap global test/trial trees
        trans_BFs,
        reverse(A.dim),
        sparse(transpose(A.near_lookup)),
        sparse(transpose(A.far_lookup)),
    )
end
