function flattenmatrix(mat::PetrovGalerkinBF)
    return FlatPGBF{eltype(mat)}(
        mat.nearinteractions,
        mat.dim,
        mat.tree,
        mat.farinteractions,
        [flatten_bf(bf) for bf in mat.BFs],
    )
end
