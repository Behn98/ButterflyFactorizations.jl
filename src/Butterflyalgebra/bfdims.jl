function Base.size(A::ButterflyFactorizations.AlgBF, dim=nothing)
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

function Base.size(A::ButterflyFactorizations.BF, dim=nothing)
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

function Base.length(A::ButterflyFactorizations.BF)
    return length(A.R) + 2
end

function Base.size(A::ButterflyFactorizations.FlatBF, dim=nothing)
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

function Base.size(A::ButterflyFactorizations.BF_Mats, dim=nothing)
    if dim === nothing
        return (size(A.P)[1], size(A.Q)[2])
    elseif dim == 1
        return size(A.P)[1]
    elseif dim == 2
        return size(A.Q)[2]
    else
        error("dim must be either 1 or 2")
    end
end

function Base.length(A::ButterflyFactorizations.BF_Mats)
    return length(A.R) + 2
end
