function Base.size(A::ButterflyFactorizations.PetrovGalerkinBF, dim=nothing)
    if dim === nothing
        return (A.dim[1], A.dim[2])
    elseif dim == 1
        return A.dim[1]
    elseif dim == 2
        return A.dim[2]
    else
        error("dim must be either 1 or 2")
    end
end

function Base.size(
    A::Adjoint{<:Any,<:ButterflyFactorizations.PetrovGalerkinBF}, dim=nothing
)
    P = parent(A)
    if dim === nothing
        return (P.dim[2], P.dim[1])
    elseif dim == 1
        return P.dim[2]
    elseif dim == 2
        return P.dim[1]
    else
        error("dim must be either 1 or 2")
    end
end

function Base.size(A::ButterflyFactorizations.PetrovGalerkinBF_Mat, dim=nothing)
    if dim === nothing
        return (A.dim[1], A.dim[2])
    elseif dim == 1
        return A.dim[1]
    elseif dim == 2
        return A.dim[2]
    else
        error("dim must be either 1 or 2")
    end
end

function Base.size(
    A::Adjoint{<:Any,<:ButterflyFactorizations.PetrovGalerkinBF_Mat}, dim=nothing
)
    P = parent(A)
    if dim === nothing
        return (P.dim[2], P.dim[1])
    elseif dim == 1
        return P.dim[2]
    elseif dim == 2
        return P.dim[1]
    else
        error("dim must be either 1 or 2")
    end
end

function Base.size(A::AbstractBlockView, dim=nothing)
    if dim === nothing
        return A.dim
    elseif dim == 1
        return A.dim[1]
    elseif dim == 2
        return A.dim[2]
    else
        error("dim must be either 1 or 2")
    end
end
