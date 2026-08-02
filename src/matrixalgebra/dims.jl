import Base: size

# ------------------------------------------------------------------
# Size Overloads
# ------------------------------------------------------------------

"""
    Base.size(A::PetrovGalerkinBF) -> Tuple{Int, Int}
    Base.size(A::PetrovGalerkinBF, dim::Integer) -> Int

Returns the spatial dimensions (rows, columns) of the global Butterfly operator.
"""
Base.size(A::ButterflyFactorizations.PetrovGalerkinBF) = A.dim

function Base.size(A::ButterflyFactorizations.PetrovGalerkinBF, dim::Integer)
    if dim == 1
        return A.dim[1]
    elseif dim == 2
        return A.dim[2]
    else
        throw(ArgumentError("dim must be either 1 or 2"))
    end
end

"""
    Base.size(A::PetrovGalerkinBF_Mat) -> Tuple{Int, Int}
    Base.size(A::PetrovGalerkinBF_Mat, dim::Integer) -> Int

Returns the spatial dimensions (rows, columns) of the sparse-matrix global operator.
"""
Base.size(A::ButterflyFactorizations.PetrovGalerkinBF_Mat) = A.dim

function Base.size(A::ButterflyFactorizations.PetrovGalerkinBF_Mat, dim::Integer)
    if dim == 1
        return A.dim[1]
    elseif dim == 2
        return A.dim[2]
    else
        throw(ArgumentError("dim must be either 1 or 2"))
    end
end
