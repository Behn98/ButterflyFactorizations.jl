# ------------------------------------------------------------------
# Matrix-Matrix Extensions for ButterflyFactorization
# ------------------------------------------------------------------

"""
    mul!(Y::AbstractMatrix, BF::ButterflyFactorization, X::AbstractMatrix, α=1, β=0)

Computes `Y = α * BF * X + β * Y` column-by-column using raw dictionary evaluation.
"""
function LinearAlgebra.mul!(
    Y::AbstractMatrix,
    BF::ButterflyFactorization{T,M},
    X::AbstractMatrix,
    α::Number=1,
    β::Number=0;
) where {T,M}
    @assert size(X, 2) == size(Y, 2) "Number of columns in X and Y must match."

    for j in 1:size(X, 2)
        x_col = view(X, :, j)
        y_col = view(Y, :, j)
        mul!(y_col, BF, x_col, α, β)
    end

    return Y
end

"""
    mul!(Y, BF, X, ws::Union{ButterflyWorkspace, ThreadButterflyWorkspace}, α=1, β=0)

Computes the matrix-matrix product column-by-column while reusing the provided workspace.
"""
function LinearAlgebra.mul!(
    Y::AbstractMatrix{T},
    BF::ButterflyFactorization{T,M},
    X::AbstractMatrix{T},
    ws::Union{ButterflyWorkspace{T},ThreadButterflyWorkspace{T}},
    α::Number=1,
    β::Number=0;
) where {T,M}
    @assert size(X, 2) == size(Y, 2) "Number of columns in X and Y must match."

    for j in 1:size(X, 2)
        x_col = view(X, :, j)
        y_col = view(Y, :, j)
        mul!(y_col, BF, x_col, ws, α, β)
    end

    return Y
end

"""
    *(BF::ButterflyFactorization, X::AbstractMatrix)

Allocates the output matrix and computes `Y = BF * X`.
Automatically initializes a `ThreadButterflyWorkspace` and reuses it across all columns
for zero-allocation inner loops.
"""
function Base.:*(BF::ButterflyFactorization{T,M}, X::AbstractMatrix) where {T,M}
    rows, _ = size(BF)
    cols = size(X, 2)
    T_val = promote_type(T, eltype(X))
    Y = zeros(T_val, rows, cols)

    # Initialize workspace exactly once
    ws = ThreadButterflyWorkspace{T_val}()

    for j in 1:cols
        x_col = view(X, :, j)
        y_col = view(Y, :, j)
        mul!(y_col, BF, x_col, ws)
    end

    return Y
end
