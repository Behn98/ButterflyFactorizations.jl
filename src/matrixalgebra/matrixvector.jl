import LinearAlgebra: mul!, adjoint, transpose
using LinearMaps: LinearMaps

# ------------------------------------------------------------------
# Forward Matrix-Vector Product (Flat Array BF)
# ------------------------------------------------------------------

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat,
    A::ButterflyFactorizations.PetrovGalerkinBF{T},
    x::AbstractVector{T};
    scheduler=OhMyThreads.StaticScheduler(), # 🚀 StaticScheduler is required for threadid() safety
) where {T}
    LinearMaps.check_dim_mul(y, A, x)

    # 1. Near Interactions (Zero-allocation evaluation)
    # mul! with β=0 completely overwrites `y` without allocating a temporary vector
    mul!(y, A.nearinteractions, x, 1, 0)

    # 2. Far Interactions (Looping over all the individual BFs)
    if !isempty(A.BFs)
        # Clear out thread buffers
        for buf in A.y_thread_buffers
            fill!(buf, zero(T))
        end

        @tasks for i in 1:length(A.BFs)
            @set scheduler = scheduler

            # What OS thread is executing this specific iteration?
            tid = Threads.threadid()

            # Grab the workspace and y-buffer assigned to this thread
            y_local = A.y_thread_buffers[tid]
            ws_local = A.thread_workspaces[tid]
            bf = A.BFs[i]

            # The thread re-uses its personal workspace to evaluate this BF block.
            mul!(y_local, bf, x, ws_local, 1, 1)
        end

        # 3. Reduction: Sum all the thread local vectors back into the main `y`
        for buf in A.y_thread_buffers
            y .+= buf
        end
    end

    return y
end

# ------------------------------------------------------------------
# Legacy / Sparse Matrix Overloads
# ------------------------------------------------------------------

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat,
    A::ButterflyFactorizations.PetrovGalerkinBF_Mat,
    x::AbstractVector{T},
) where {T}
    LinearMaps.check_dim_mul(y, A, x)

    # Zero-allocation near-field
    mul!(y, A.nearinteractions, x, 1, 0)

    for i in eachindex(A.BFs)
        y[A.BFs[i].permP] .+= applyButterflyFactorization_Mat(A.BFs[i], x[A.BFs[i].permQ])
    end
    return y
end

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat,
    At::LinearMaps.TransposeMap{<:Any,<:ButterflyFactorizations.PetrovGalerkinBF_Mat},
    x::AbstractVector{T},
) where {T}
    LinearMaps.check_dim_mul(y, At.lmap, x)

    mul!(y, transpose(At.lmap.nearinteractions), x, 1, 0)

    for i in eachindex(At.lmap.BFs)
        y[At.lmap.BFs[i].permQ] .+= applyButterflyFactorization_Mat(
            transpose(At.lmap.BFs[i]), x[At.lmap.BFs[i].permP]
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

    mul!(y, adjoint(At.lmap.nearinteractions), x, 1, 0)

    for i in eachindex(At.lmap.BFs)
        y[At.lmap.BFs[i].permQ] .+= applyButterflyFactorization_Mat(
            At.lmap.BFs[i]', x[At.lmap.BFs[i].permP]
        )
    end
    return y
end
