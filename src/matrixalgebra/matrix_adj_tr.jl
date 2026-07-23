function Base.adjoint(A::PetrovGalerkinBF{T}) where {T}
    adj_BFs = map(adjoint, A.BFs)
    adj_workspaces = map(ButterflyWorkspace, adj_BFs) # Allocate once!

    return PetrovGalerkinBF{T}(
        adjoint(A.nearinteractions),
        A.tree,
        adj_BFs,
        adj_workspaces,
        reverse(A.dim),
        sparse(transpose(A.near_lookup)), # transpose is fine for real/integer indices
        sparse(transpose(A.far_lookup)),
    )
end

function Base.transpose(A::PetrovGalerkinBF{T}) where {T}
    trans_BFs = map(transpose, A.BFs)
    trans_workspaces = map(ButterflyWorkspace, trans_BFs) # Allocate once!

    return PetrovGalerkinBF{T}(
        transpose(A.nearinteractions),
        A.tree,
        trans_BFs,
        trans_workspaces,
        reverse(A.dim),
        sparse(transpose(A.near_lookup)),
        sparse(transpose(A.far_lookup)),
    )
end
