import LinearAlgebra: mul!, adjoint, transpose
using LinearMaps: LinearMaps

# ------------------------------------------------------------------
# Forward Matrix-Vector Product (Used for A, A', and transpose(A))
# ------------------------------------------------------------------
@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat, A::PetrovGalerkinBF{T}, x::AbstractVector{T}
) where {T}
    LinearMaps.check_dim_mul(y, A, x)
    fill!(y, zero(T))

    # 1. Near Interactions
    y .+= A.nearinteractions * x

    # 2. Far Interactions
    n_chunks = min(length(A.BFs), Threads.nthreads() * 4)
    chunk_size = cld(length(A.BFs), n_chunks)
    y_locals = Vector{Vector{T}}(undef, n_chunks)

    @tasks for c in 1:n_chunks
        @set scheduler = DynamicScheduler()
        y_local = zeros(T, length(y))
        start_idx = (c - 1) * chunk_size + 1
        end_idx = min(c * chunk_size, length(A.BFs))

        for i in start_idx:end_idx
            bf = A.BFs[i]
            bfw = A.workspaces[i]

            mul!(y_local, bf, x, bfw, 1, 1)
        end
        y_locals[c] = y_local
    end

    # 3. Reduction
    for c in 1:n_chunks
        y .+= y_locals[c]
    end

    return y
end

# ... (Här börjar dina orörda funktioner för PetrovGalerkinBF_Mat) ...
@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat,
    A::ButterflyFactorizations.PetrovGalerkinBF_Mat,
    x::AbstractVector{T},
) where {T}
    LinearMaps.check_dim_mul(y, A, x)
    fill!(y, zero(T))
    y .+= A.nearinteractions * x

    for i in eachindex(A.BFs)
        y[A.BFs[i].PermP] .+= applyButterflyFactorization_Mat(A.BFs[i], x[A.BFs[i].PermQ])
    end
    return y
end

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat,
    At::LinearMaps.TransposeMap{<:Any,<:ButterflyFactorizations.PetrovGalerkinBF_Mat},
    x::AbstractVector{T},
) where {T}
    LinearMaps.check_dim_mul(y, At.lmap, x)
    fill!(y, zero(T))
    y .+= transpose(At.lmap.nearinteractions) * x
    for i in eachindex(At.lmap.BFs)
        y[At.lmap.BFs[i].PermQ] .+= applyButterflyFactorization_Mat(
            transpose(At.lmap.BFs[i]), x[At.lmap.BFs[i].PermP]
        )
    end
    return y
end

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat,
    At::LinearMaps.AdjointMap{<:Any,<:ButterflyFactorizations.PetrovGalerkinBF_Mat},
    x::AbstractVector{T},
) where {T}
    LinearMaps.check_dim_mul(y, At.lmap, x)
    fill!(y, zero(T))
    y .+= adjoint(At.lmap.nearinteractions) * x
    for i in eachindex(At.lmap.BFs)
        y[At.lmap.BFs[i].PermQ] .+= applyButterflyFactorization_Mat(
            At.lmap.BFs[i]', x[At.lmap.BFs[i].PermP]
        )
    end
    return y
end
