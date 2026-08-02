import Base: *, size, eltype, length

# ------------------------------------------------------------------
# ButterflyFactorization_Mat Overloads
# ------------------------------------------------------------------

"""
    Base.size(A::ButterflyFactorization_Mat) -> Tuple{Int, Int}
    Base.size(A::ButterflyFactorization_Mat, dim::Integer) -> Int

Returns the spatial dimensions (rows, columns) of the fully assembled sparse matrix.
"""
Base.size(A::ButterflyFactorizations.ButterflyFactorization_Mat) =
    (size(A.P, 1), size(A.Q, 2))

function Base.size(A::ButterflyFactorizations.ButterflyFactorization_Mat, dim::Integer)
    if dim == 1
        return size(A.P, 1)
    elseif dim == 2
        return size(A.Q, 2)
    else
        throw(ArgumentError("dim must be either 1 or 2"))
    end
end

"""
    Base.length(A::ButterflyFactorization_Mat) -> Int

Returns the total number of sequential cascaded matrix factors (Q, R levels, and P)
rather than the number of matrix elements.
"""
function Base.length(A::ButterflyFactorizations.ButterflyFactorization_Mat)
    return length(A.R) + 2
end

"""
    Base.eltype(A::ButterflyFactorization_Mat)

Extracts the numeric type of the underlying matrix data.
"""
Base.eltype(::Type{<:ButterflyFactorizations.ButterflyFactorization_Mat{T}}) where {T} = T
Base.eltype(A::ButterflyFactorizations.ButterflyFactorization_Mat) = eltype(typeof(A))

# ------------------------------------------------------------------
# ButterflyFactorization (Flat Array) Overloads
# ------------------------------------------------------------------

"""
    get_bf_size(BF::ButterflyFactorization) -> Tuple{Int, Int}

Infers the global dense spatial size of the factorized matrix by querying the
maximum index stored in the root nodes of the test (row) and trial (column) trees.
"""
function get_bf_size(BF::ButterflyFactorization)
    trialT = cluster_trialtree(BF.tree)
    testT  = cluster_testtree(BF.tree)

    # 🚀 OPTIMIZATION: Query the root node directly instead of looping over all blocks!
    # The root node contains the global indices for the entire space.
    rows = maximum(cluster_values(testT, cluster_root(testT)))
    cols = maximum(cluster_values(trialT, cluster_root(trialT)))

    return (rows, cols)
end

"""
    Base.size(BF::ButterflyFactorization) -> Tuple{Int, Int}
    Base.size(BF::ButterflyFactorization, dim::Integer) -> Int

Returns the spatial dimensions (rows, columns) mapped by the factorization.
"""
Base.size(BF::ButterflyFactorization) = get_bf_size(BF)
Base.size(BF::ButterflyFactorization, dim::Integer) = get_bf_size(BF)[dim]

"""
    Base.length(BF::ButterflyFactorization) -> Int

Returns the total number of sequential cascaded factor levels (Q, R levels, and P).
"""
function Base.length(BF::ButterflyFactorization)
    return length(BF.R) + 2
end

"""
    Base.eltype(BF::ButterflyFactorization)

Extracts the numeric type of the underlying block data (e.g., `ComplexF64`, `Float64`).
"""
Base.eltype(::Type{<:ButterflyFactorization{T}}) where {T} = T
Base.eltype(BF::ButterflyFactorization) = eltype(typeof(BF))
