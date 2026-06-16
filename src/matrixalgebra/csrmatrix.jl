function flattenmatrix(mat::PetrovGalerkinBF)
    return FlatPGBF{eltype(mat)}(
        mat.nearinteractions, mat.dim, mat.tree, [flatten_bf(bf) for bf in mat.BFs]
    )
end
