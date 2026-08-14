import LinearAlgebra: mul!, adjoint, transpose
using LinearMaps: LinearMaps

# ------------------------------------------------------------------
# Forward Matrix-Vector Product (Flat Array BF)
# ------------------------------------------------------------------

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat,
    A::ButterflyFactorizations.PetrovGalerkinBF{T},
    x::AbstractVector{T},
    α::Number=1,
    β::Number=0,
) where {T}
    LinearMaps.check_dim_mul(y, A, x)

    # 1. Handle scaling or zeroing of the output vector y based on β
    if β == 0
        fill!(y, zero(T))
    elseif β != 1
        rmul!(y, β)
    end

    # 2. Evaluate Near-Field (accumulate with α scaling if needed, or standard 1, 1 if β was handled)
    # To be fully general with α and β:
    # We can compute the application into a temporary or handle near/far carefully.
    # A cleaner approach for LinearMaps compatibility:

    # Let's compute near-field contribution
    # (If α != 1, we scale accordingly)
    if α == 1
        mul!(y, A.nearinteractions, x, 1, 1) # adds to y
    else
        # Temporary buffer or direct scaling
        y_near = A.nearinteractions * x
        y .+= α .* y_near
    end

    # 3. Far-Field Butterfly Evaluation
    # 3. Far-Field Butterfly Evaluation
    if !isempty(A.BFs)
        # Zero out all buffers before starting
        for buf in A.y_thread_buffers
            fill!(buf, zero(T))
        end

        @tasks for i in 1:length(A.BFs)
            @set scheduler = OhMyThreads.DynamicScheduler()

            # Safely "check out" a workspace ID (blocks if none available)
            buf_id = take!(A.buffer_pool)

            try
                y_local = A.y_thread_buffers[buf_id]
                ws_local = A.thread_workspaces[buf_id]

                # Evaluate this specific butterfly
                mul!(y_local, A.BFs[i], x, ws_local, 1, 1)
            finally
                # ALWAYS return the ID to the pool, even if an error occurs
                put!(A.buffer_pool, buf_id)
            end
        end

        # Reduction: Sum the thread-local vectors back into the global y vector
        for buf in A.y_thread_buffers
            if α == 1
                y .+= buf
            else
                y .+= α .* buf
            end
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
    α::Number=1,
    β::Number=0,
) where {T}
    if β == 0
        fill!(y, zero(T))
    elseif β != 1
        rmul!(y, β)
    end
    LinearMaps.check_dim_mul(y, A, x)

    # Zero-allocation near-field
    mul!(y, A.nearinteractions, x, α, β)

    for i in eachindex(A.BFs)
        y[A.BFs[i].permP] .+= applyButterflyFactorization_Mat(A.BFs[i], x[A.BFs[i].permQ])
    end
    return y
end

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat,
    At::LinearMaps.TransposeMap{<:Any,<:ButterflyFactorizations.PetrovGalerkinBF_Mat},
    x::AbstractVector{T},
    α::Number=1,
    β::Number=0,
) where {T}
    LinearMaps.check_dim_mul(y, At.lmap, x)

    if β == 0
        fill!(y, zero(T))
    elseif β != 1
        rmul!(y, β)
    end

    mul!(y, transpose(At.lmap.nearinteractions), x, α, β)

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
    α::Number=1,
    β::Number=0,
) where {T}
    LinearMaps.check_dim_mul(y, At.lmap, x)

    if β == 0
        fill!(y, zero(T))
    elseif β != 1
        rmul!(y, β)
    end

    mul!(y, adjoint(At.lmap.nearinteractions), x, α, β)

    for i in eachindex(At.lmap.BFs)
        y[At.lmap.BFs[i].permQ] .+= applyButterflyFactorization_Mat(
            At.lmap.BFs[i]', x[At.lmap.BFs[i].permP]
        )
    end
    return y
end
