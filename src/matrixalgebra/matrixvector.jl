import LinearAlgebra: mul!, adjoint, transpose
using LinearMaps: LinearMaps

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat, A::ButterflyFactorizations.PetrovGalerkinBF, x::AbstractVector{T}
) where {T}
    LinearMaps.check_dim_mul(y, A, x)
    fill!(y, zero(T))
    y_near = A.nearinteractions * x
    y .+= y_near

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
            res = apply_BF(bf, x; scheduler=OhMyThreads.SerialScheduler())

            y_local .+= res
        end
        y_locals[c] = y_local
    end

    for c in 1:n_chunks
        y .+= y_locals[c]
    end
    return y
end

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat,
    At::LinearMaps.AdjointMap{<:Any,<:ButterflyFactorizations.PetrovGalerkinBF},
    x::AbstractVector{T},
) where {T}
    LinearMaps.check_dim_mul(y, At.lmap, x)
    fill!(y, zero(T))
    y .+= adjoint(At.lmap.nearinteractions) * x

    n_chunks = min(length(At.lmap.BFs), Threads.nthreads() * 4)
    chunk_size = cld(length(At.lmap.BFs), n_chunks)
    y_locals = Vector{Vector{T}}(undef, n_chunks)

    @tasks for c in 1:n_chunks
        @set scheduler = DynamicScheduler()
        y_local = zeros(T, length(y))
        start_idx = (c - 1) * chunk_size + 1
        end_idx = min(c * chunk_size, length(At.lmap.BFs))

        for i in start_idx:end_idx
            bf = At.lmap.BFs[i]

            res = apply_BF(bf', x; scheduler=OhMyThreads.SerialScheduler())

            y_local .+= res
        end
        y_locals[c] = y_local
    end

    for c in 1:n_chunks
        y .+= y_locals[c]
    end
    return y
end

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat,
    At::LinearMaps.TransposeMap{<:Any,<:ButterflyFactorizations.PetrovGalerkinBF},
    x::AbstractVector{T},
) where {T}
    LinearMaps.check_dim_mul(y, At.lmap, x)
    fill!(y, zero(T))
    y .+= transpose(At.lmap.nearinteractions) * x

    n_chunks = min(length(At.lmap.BFs), Threads.nthreads() * 4)
    chunk_size = cld(length(At.lmap.BFs), n_chunks)
    y_locals = Vector{Vector{T}}(undef, n_chunks)

    @tasks for c in 1:n_chunks
        @set scheduler = DynamicScheduler()
        y_local = zeros(T, length(y))
        start_idx = (c - 1) * chunk_size + 1
        end_idx = min(c * chunk_size, length(At.lmap.BFs))

        for i in start_idx:end_idx
            bf = At.lmap.BFs[i]

            res = apply_BF(transpose(bf), x; scheduler=OhMyThreads.SerialScheduler())

            y_local .+= res
        end
        y_locals[c] = y_local
    end

    for c in 1:n_chunks
        y .+= y_locals[c]
    end
    return y
end

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat, A::ButterflyFactorizations.FlatPGBF, x::AbstractVector{T}
) where {T}
    LinearMaps.check_dim_mul(y, A, x)
    fill!(y, zero(T))
    y_near = A.nearinteractions * x
    y .+= y_near

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

            mul_flat_bf!(y_local, bf, x)
        end
        y_locals[c] = y_local
    end

    for c in 1:n_chunks
        y .+= y_locals[c]
    end
    return y
end

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat,
    At::LinearMaps.TransposeMap{<:Any,<:ButterflyFactorizations.FlatPGBF},
    x::AbstractVector{T},
) where {T}
    LinearMaps.check_dim_mul(y, At.lmap, x)
    fill!(y, zero(T))
    y .+= transpose(At.lmap.nearinteractions) * x

    n_chunks = min(length(At.lmap.BFs), Threads.nthreads() * 4)
    chunk_size = cld(length(At.lmap.BFs), n_chunks)
    y_locals = Vector{Vector{T}}(undef, n_chunks)

    @tasks for c in 1:n_chunks
        @set scheduler = DynamicScheduler()
        y_local = zeros(T, length(y))
        start_idx = (c - 1) * chunk_size + 1
        end_idx = min(c * chunk_size, length(At.lmap.BFs))

        for i in start_idx:end_idx
            bf = At.lmap.BFs[i]

            res = mul_flat_bf(transpose(bf), x; scheduler=OhMyThreads.SerialScheduler())

            y_local .+= res
        end
        y_locals[c] = y_local
    end

    for c in 1:n_chunks
        y .+= y_locals[c]
    end
    return y
end

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat,
    At::LinearMaps.AdjointMap{<:Any,<:ButterflyFactorizations.FlatPGBF},
    x::AbstractVector{T},
) where {T}
    LinearMaps.check_dim_mul(y, At.lmap, x)
    fill!(y, zero(T))
    y .+= adjoint(At.lmap.nearinteractions) * x

    n_chunks = min(length(At.lmap.BFs), Threads.nthreads() * 4)
    chunk_size = cld(length(At.lmap.BFs), n_chunks)
    y_locals = Vector{Vector{T}}(undef, n_chunks)

    @tasks for c in 1:n_chunks
        @set scheduler = DynamicScheduler()
        y_local = zeros(T, length(y))
        start_idx = (c - 1) * chunk_size + 1
        end_idx = min(c * chunk_size, length(At.lmap.BFs))

        for i in start_idx:end_idx
            bf = At.lmap.BFs[i]

            res = mul_flat_bf(bf', x; scheduler=OhMyThreads.SerialScheduler())

            y_local .+= res
        end
        y_locals[c] = y_local
    end

    for c in 1:n_chunks
        y .+= y_locals[c]
    end
    return y
end

# ... (Här börjar dina orörda funktioner för PetrovGalerkinBF_mats) ...
@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat,
    A::ButterflyFactorizations.PetrovGalerkinBF_mats,
    x::AbstractVector{T},
) where {T}
    LinearMaps.check_dim_mul(y, A, x)
    fill!(y, zero(T))
    y .+= A.nearinteractions * x

    for i in eachindex(A.BFs)
        y[A.BFs[i].PermP] .+= applyBF_Mats(A.BFs[i], x[A.BFs[i].PermQ])
    end
    return y
end

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat,
    At::LinearMaps.TransposeMap{<:Any,<:ButterflyFactorizations.PetrovGalerkinBF_mats},
    x::AbstractVector{T},
) where {T}
    LinearMaps.check_dim_mul(y, At.lmap, x)
    fill!(y, zero(T))
    y .+= transpose(At.lmap.nearinteractions) * x
    for i in eachindex(At.lmap.BFs)
        y[At.lmap.BFs[i].PermQ] .+= applyBF_Mats(
            transpose(At.lmap.BFs[i]), x[At.lmap.BFs[i].PermP]
        )
    end
    return y
end

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat,
    At::LinearMaps.AdjointMap{<:Any,<:ButterflyFactorizations.PetrovGalerkinBF_mats},
    x::AbstractVector{T},
) where {T}
    LinearMaps.check_dim_mul(y, At.lmap, x)
    fill!(y, zero(T))
    y .+= adjoint(At.lmap.nearinteractions) * x
    for i in eachindex(At.lmap.BFs)
        y[At.lmap.BFs[i].PermQ] .+= applyBF_Mats(At.lmap.BFs[i]', x[At.lmap.BFs[i].PermP])
    end
    return y
end

# ------------------------------------------------------------------
# Forward Matrix-Vector Product
# ------------------------------------------------------------------
@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat, A::FlatPGBF2{T}, x::AbstractVector{T}
) where {T}
    LinearMaps.check_dim_mul(y, A, x)
    fill!(y, zero(T))
    y .+= A.nearinteractions * x

    n_chunks = min(length(A.BFs), Threads.nthreads() * 4)
    chunk_size = cld(length(A.BFs), n_chunks)
    y_locals = Vector{Vector{T}}(undef, n_chunks)

    @tasks for c in 1:n_chunks
        @set scheduler = DynamicScheduler()
        y_local = zeros(T, length(y))
        start_idx = (c - 1) * chunk_size + 1
        end_idx = min(c * chunk_size, length(A.BFs))

        # Create a local workspace per chunk for zero-allocation performance
        # (Assuming you initialized workspaces or are falling back to the allocation engine)
        for i in start_idx:end_idx
            bf = A.BFs[i]
            # If using workspaces: mul!(y_local, bf, x, ws, 1, 1)
            # Falling back to standard operator:
            mul!(y_local, bf, x, 1, 1)
        end
        y_locals[c] = y_local
    end

    for c in 1:n_chunks
        y .+= y_locals[c]
    end
    return y
end

# ------------------------------------------------------------------
# Adjoint Matrix-Vector Product (BF')
# ------------------------------------------------------------------
@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat,
    At::LinearMaps.AdjointMap{<:Any,<:FlatPGBF2{T}},
    x::AbstractVector{T},
) where {T}
    A = At.lmap
    LinearMaps.check_dim_mul(y, At, x)
    fill!(y, zero(T))
    y .+= adjoint(A.nearinteractions) * x

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
            bf_adj = adjoint(bf) # Leverages our explicit ButterflyFactorization adjoint constructor!
            mul!(y_local, bf_adj, x, 1, 1)
        end
        y_locals[c] = y_local
    end

    for c in 1:n_chunks
        y .+= y_locals[c]
    end
    return y
end

# ------------------------------------------------------------------
# Transpose Matrix-Vector Product (transpose(BF))
# ------------------------------------------------------------------
@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat,
    At::LinearMaps.TransposeMap{<:Any,<:FlatPGBF2{T}},
    x::AbstractVector{T},
) where {T}
    A = At.lmap
    LinearMaps.check_dim_mul(y, At, x)
    fill!(y, zero(T))
    y .+= transpose(A.nearinteractions) * x

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
            bf_trans = transpose(bf)
            mul!(y_local, bf_trans, x, 1, 1)
        end
        y_locals[c] = y_local
    end

    for c in 1:n_chunks
        y .+= y_locals[c]
    end
    return y
end
