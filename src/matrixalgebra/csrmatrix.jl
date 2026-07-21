"""
    flattenmatrix(mat::PetrovGalerkinBF) -> FlatPGBF

Convert a dictionary-based [`PetrovGalerkinBF`](@ref) into its flattened
[`FlatPGBF`](@ref) representation.

This keeps the same near-field interactions, geometry, and tree structure, while
replacing each hierarchical butterfly block in `mat.BFs` with its flattened
`FlatBF` counterpart for more efficient downstream evaluation.

# Arguments

  - `mat::PetrovGalerkinBF`: The Petrov-Galerkin butterfly factorization to flatten.

# Returns

  - `FlatPGBF`: A flattened Petrov-Galerkin butterfly factorization with the same
    near-field matrix, dimensions, and tree, but with `FlatBF` blocks.
"""
function flattenmatrix(mat::PetrovGalerkinBF)
    return FlatPGBF{eltype(mat)}(
        mat.nearinteractions, mat.dim, mat.tree, [FlatBF(bf) for bf in mat.BFs]
    )
end
