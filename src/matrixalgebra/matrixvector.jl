import LinearAlgebra: mul!, adjoint, transpose
using LinearMaps: LinearMaps

# ------------------------------------------------------------------
# Forward Matrix-Vector Product (Used for A, A', and transpose(A))
# ------------------------------------------------------------------
@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat,
    A::PetrovGalerkinBF{T},
    x::AbstractVector{T};
    scheduler=OhMyThreads.DynamicScheduler(),
) where {T}
    LinearMaps.check_dim_mul(y, A, x)
    fill!(y, zero(T))

    # 1. Near Interactions
    y .+= A.nearinteractions * x

    # 2. Far Interactions
    # 2. Far Interactions
    if !isempty(A.BFs)
        # Zero out all persistent thread buffers before we begin
        for buf in A.y_thread_buffers
            fill!(buf, zero(T))
        end

        @tasks for i in 1:length(A.BFs)
            @set scheduler = scheduler

            # Grab the buffer belonging to the current thread running this task
            tid = Threads.threadid()
            y_local = A.y_thread_buffers[tid]

            bf = A.BFs[i]
            bfw = A.workspaces[i]

            # Accumulate directly into the thread's persistent buffer!
            mul!(y_local, bf, x, bfw, 1, 1)
        end

        # 3. Reduction
        # Sum all thread buffers back into the main output vector `y`
        for buf in A.y_thread_buffers
            y .+= buf
        end
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
