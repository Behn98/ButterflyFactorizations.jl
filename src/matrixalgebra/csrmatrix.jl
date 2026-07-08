function flattenmatrix(mat::PetrovGalerkinBF)
    return FlatPGBF{eltype(mat)}(
        mat.nearinteractions, mat.dim, mat.tree, [FlatBF(bf) for bf in mat.BFs]
    )
end
