import Base: *, size, eltype

"""
    mul!(y, BF::ButterflyFactorization, x, α=1, β=0)

Computes y = α * BF * x + β * y.
Traverses the butterfly factorization: Q -> R^1 -> ... -> R^{L-1} -> P.
"""
function LinearAlgebra.mul!(
    y::AbstractVector,
    BF::ButterflyFactorization,
    x::AbstractVector,
    α::Number=1,
    β::Number=0;
    scheduler=OhMyThreads.DynamicScheduler(),
)
    trialT = cluster_trialtree(BF.tree)
    testT  = cluster_testtree(BF.tree)

    # Scale the initial y vector
    if β != 1
        β == 0 ? fill!(y, 0) : rmul!(y, β)
    end

    # Intermediate skeleton states
    T_val = promote_type(ComplexF64, eltype(x))
    x_curr = Dict{Tuple{Int,Int},Vector{T_val}}()
    x_next = Dict{Tuple{Int,Int},Vector{T_val}}()

    # -----------------------------------------------------------
    # 1. Forward Q: Leaf Sources -> Source Skeletons
    # -----------------------------------------------------------
    for block in BF.Q
        src_idx = cluster_values(trialT, block.src_in)
        v_in = x[src_idx]

        # block.data is either a dense Matrix or UniformScaling (I)
        x_next[(block.obs_out, block.src_out)] = block.data * v_in
    end

    # -----------------------------------------------------------
    # 2. Traverse R levels (Basis changing)
    # -----------------------------------------------------------
    for level in BF.R
        x_curr = x_next
        x_next = Dict{Tuple{Int,Int},Vector{T_val}}()

        for block in level.blocks
            in_key  = (block.obs_in, block.src_in)
            out_key = (block.obs_out, block.src_out)

            v_in = x_curr[in_key]
            v_out_part = block.data * v_in

            if haskey(x_next, out_key)
                x_next[out_key] .+= v_out_part
            else
                x_next[out_key] = v_out_part
            end
        end
    end

    # -----------------------------------------------------------
    # 3. Forward P: Observer Skeletons -> Leaf Observers
    # -----------------------------------------------------------
    x_curr = x_next
    for block in BF.P
        in_key  = (block.obs_in, block.src_in)
        obs_idx = cluster_values(testT, block.obs_out)

        v_in = x_curr[in_key]

        if α == 1
            y[obs_idx] .+= block.data * v_in
        else
            y[obs_idx] .+= α .* (block.data * v_in)
        end
    end

    return y
end

function ButterflyWorkspace(BF::ButterflyFactorization{T}) where {T}
    num_levels = length(BF.R) + 1
    buffers = [Dict{Tuple{Int,Int},Vector{T}}() for _ in 1:num_levels]

    # Pre-allocate exact buffer sizes for Level 1 (after Q)
    for b in BF.Q
        key = (b.obs_out, b.src_out)
        size_out = b.data isa UniformScaling ? 0 : size(b.data, 1) # row dim
        buffers[1][key] = Vector{T}(undef, size_out)
    end

    # Pre-allocate for intermediate R levels
    for (l, level) in enumerate(BF.R)
        for b in level.blocks
            key = (b.obs_out, b.src_out)
            if !haskey(buffers[l + 1], key)
                size_out = if b.data isa UniformScaling
                    size(buffers[l][(b.obs_in, b.src_in)], 1)
                else
                    size(b.data, 1)
                end
                buffers[l + 1][key] = Vector{T}(undef, size_out)
            end
        end
    end

    return ButterflyWorkspace(buffers)
end

function LinearAlgebra.mul!(
    y::AbstractVector{T},
    BF::ButterflyFactorization{T,M},
    x::AbstractVector{T},
    ws::ButterflyWorkspace{T},
    α::Number=1,
    β::Number=0;
    scheduler=OhMyThreads.DynamicScheduler(),
) where {T,M}
    trialT = cluster_trialtree(BF.tree)
    testT  = cluster_testtree(BF.tree)

    # Initialize y
    if β != 1
        β == 0 ? fill!(y, 0) : rmul!(y, β)
    end

    # -----------------------------------------------------------
    # 1. Leaf Level Q
    # -----------------------------------------------------------
    buf_Q = ws.level_buffers[1]
    for block in BF.Q
        src_idx = cluster_values(trialT, block.src_in)
        v_out   = buf_Q[(block.obs_out, block.src_out)]

        v_in = view(x, src_idx)

        if block.data isa UniformScaling
            copyto!(v_out, v_in)
        else
            mul!(v_out, block.data, v_in) # Zero-allocation BLAS GEMV
        end
    end

    # -----------------------------------------------------------
    # 2. Intermediate R Levels
    # -----------------------------------------------------------
    for (l, level) in enumerate(BF.R)
        buf_in  = ws.level_buffers[l]
        buf_out = ws.level_buffers[l + 1]

        # Reset output buffers to zero before accumulating
        for v in Base.values(buf_out)
            fill!(v, 0)
        end

        for block in level.blocks
            v_in  = buf_in[(block.obs_in, block.src_in)]
            v_out = buf_out[(block.obs_out, block.src_out)]

            if block.data isa UniformScaling
                v_out .+= v_in
            else
                # Accumulate block.data * v_in into v_out
                mul!(v_out, block.data, v_in, 1, 1)
            end
        end
    end

    # -----------------------------------------------------------
    # 3. Leaf Level P
    # -----------------------------------------------------------
    last_buf = ws.level_buffers[end]
    for block in BF.P
        v_in    = last_buf[(block.obs_in, block.src_in)]
        obs_idx = cluster_values(testT, block.obs_out)

        y_view = view(y, obs_idx)

        if block.data isa UniformScaling
            y_view .+= α .* v_in
        else
            mul!(y_view, block.data, v_in, α, 1) # Accumulate directly into y!
        end
    end

    return y
end

function LinearAlgebra.mul!(
    y::AbstractVector{T},
    BF::ButterflyFactorization{T,M},
    x::AbstractVector{T},
    ws::ThreadButterflyWorkspace{T},
    α::Number=1,
    β::Number=0;
) where {T,M}
    trialT = cluster_trialtree(BF.tree)
    testT  = cluster_testtree(BF.tree)

    if β != 1
        β == 0 ? fill!(y, 0) : rmul!(y, β)
    end

    num_levels = length(BF.R) + 1

    # Safely expand workspace depth if this BF > 20 levels
    if length(ws.level_buffers) < num_levels
        old_len = length(ws.level_buffers)
        resize!(ws.level_buffers, num_levels)
        resize!(ws.level_lengths, num_levels)
        for i in (old_len + 1):num_levels
            ws.level_buffers[i] = Vector{Vector{T}}()
            ws.level_lengths[i] = Vector{Int}()
        end
    end

    # ---------------------------------------------
    # 1. Leaf Level Q
    # ---------------------------------------------
    buf_Q = ws.level_buffers[1]
    len_Q = ws.level_lengths[1]

    for block in BF.Q
        src_idx = cluster_values(trialT, block.src_in)
        v_in = view(x, src_idx)

        req_size = block.data isa UniformScaling ? length(src_idx) : size(block.data, 1)
        v_out = get_buffer_and_set_len!(buf_Q, len_Q, block.out_ptr, req_size)
        v_out_view = view(v_out, 1:req_size)

        if block.data isa UniformScaling
            copyto!(v_out_view, v_in)
        else
            mul!(v_out_view, block.data, v_in)
        end
    end

    # ---------------------------------------------
    # 2. Intermediate R Levels
    # ---------------------------------------------
    for (l, level) in enumerate(BF.R)
        buf_in = ws.level_buffers[l]
        len_in = ws.level_lengths[l]

        buf_out = ws.level_buffers[l + 1]
        len_out = ws.level_lengths[l + 1]

        # Reset outputs first
        for block in level.blocks
            in_size = len_in[block.in_ptr]
            req_size = block.data isa UniformScaling ? in_size : size(block.data, 1)

            v_out = get_buffer_and_set_len!(buf_out, len_out, block.out_ptr, req_size)
            fill!(view(v_out, 1:req_size), 0)
        end

        # Multiply and accumulate
        for block in level.blocks
            in_size = len_in[block.in_ptr]
            v_in = view(buf_in[block.in_ptr], 1:in_size) # 🚀 Exact valid size

            req_size = block.data isa UniformScaling ? in_size : size(block.data, 1)
            v_out = view(buf_out[block.out_ptr], 1:req_size)

            if block.data isa UniformScaling
                v_out .+= v_in
            else
                mul!(v_out, block.data, v_in, 1, 1)
            end
        end
    end

    # ---------------------------------------------
    # 3. Leaf Level P
    # ---------------------------------------------
    last_buf = ws.level_buffers[num_levels]
    last_len = ws.level_lengths[num_levels]

    for block in BF.P
        in_size = last_len[block.in_ptr]
        v_in = view(last_buf[block.in_ptr], 1:in_size) # 🚀 Exact valid size

        obs_idx = cluster_values(testT, block.obs_out)
        y_view = view(y, obs_idx)

        if block.data isa UniformScaling
            y_view .+= α .* v_in
        else
            mul!(y_view, block.data, v_in, α, 1)
        end
    end

    return y
end

# Enable `y = BF * x`
function Base.:*(BF::ButterflyFactorization, x::AbstractVector)
    rows, _ = size(BF)
    T_val = promote_type(ComplexF64, eltype(x))
    y = zeros(T_val, rows)
    mul!(y, BF, x)
    return y
end

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat, Butterfly::ButterflyFactorization_Mat, x::AbstractVector{T}
) where {T}
    LinearMaps.check_dim_mul(y, Butterfly, x)
    fill!(y, zero(T))
    y .= applyButterflyFactorization_Mat(Butterfly, x)
    return y
end

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat,
    At::LinearMaps.TransposeMap{<:Any,<:ButterflyFactorizations.ButterflyFactorization_Mat},
    x::AbstractVector{T},
) where {T}
    LinearMaps.check_dim_mul(y, transposed(At.lmap), x)
    fill!(y, zero(T))
    y .= applyButterflyFactorization_Mat(transpose(At.lmap), x)
    return y
end

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat,
    At::LinearMaps.AdjointMap{<:Any,<:ButterflyFactorizations.ButterflyFactorization_Mat},
    x::AbstractVector{T},
) where {T}
    LinearMaps.check_dim_mul(y, At.lmap, x)
    fill!(y, zero(T))
    y .= applyButterflyFactorization_Mat(At.lmap', x)
    return y
end

function Base.:*(Butterfly::ButterflyFactorization_Mat, x::AbstractVector)
    return applyButterflyFactorization_Mat(Butterfly, x)
end

"""
    applyButterflyFactorization_Mat(t:: ButterflyFactorization_Mat, v::AbstractVector)

Applies the sequential matrix operations (`Q`, `R` layers, and `P`) of a
` ButterflyFactorization_Mat` factorization to a vector `v` and returns the output vector.
"""
function applyButterflyFactorization_Mat(t::ButterflyFactorization_Mat, v::AbstractVector)
    y = v
    y = t.Q * y
    for R_block in t.R
        y = R_block * y
    end
    result = zeros(ComplexF64, size(t, 1))
    result = t.P * y
    return result
end
