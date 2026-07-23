import Base: *, size, eltype

function Base.size(A::ButterflyFactorizations.ButterflyFactorization_Mat, dim=nothing)
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

function Base.length(A::ButterflyFactorizations.ButterflyFactorization_Mat)
    return length(A.R) + 2
end

# Helper to infer global size by scanning the max indices in the leaf trees
function get_bf_size(BF::ButterflyFactorization)
    trialT = trialtree(BF.tree)
    testT  = testtree(BF.tree)

    # Infer number of rows (Observer dimension)
    rows = 0
    for block in BF.P
        rows = max(rows, maximum(values(testT, block.obs_out)))
    end

    # Infer number of columns (Source dimension)
    cols = 0
    for block in BF.Q
        cols = max(cols, maximum(values(trialT, block.src_in)))
    end

    return (rows, cols)
end

Base.size(BF::ButterflyFactorization) = get_bf_size(BF)
Base.size(BF::ButterflyFactorization, dim::Int) = get_bf_size(BF)[dim]
