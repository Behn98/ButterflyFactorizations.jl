"""
    flattenmatrix(mat::PetrovGalerkinBF)

    Converts a Petrov-Galerkin Butterfly Factorization (PetrovGalerkinBF) into a Flat
    Petrov-Galerkin Butterfly Factorization (FlatPGBF). The resulting FlatPGBF has the same
    near interactions, dimensions, and tree structure as the original PetrovGalerkinBF, but
    its butterfly factors are flattened into a vector of FlatBF objects.
"""
function flattenmatrix(mat::PetrovGalerkinBF)
    return FlatPGBF{eltype(mat)}(
        mat.nearinteractions, mat.dim, mat.tree, [FlatBF(bf) for bf in mat.BFs]
    )
end
