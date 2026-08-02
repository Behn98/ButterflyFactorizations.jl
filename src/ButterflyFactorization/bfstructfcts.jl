"""
    block_key(b::ButterflyBlock) -> NTuple{4, Int}

Extracts the unique 4-tuple routing key `(obs_out, src_out, obs_in, src_in)` from a `ButterflyBlock`.
This key perfectly identifies the exact interaction pair this block maps to within the tree structure.
"""
function block_key(b::ButterflyBlock)
    return (b.obs_out, b.src_out, b.obs_in, b.src_in)
end

"""
    getnsno(BFactorization::ButterflyFactorization) -> Tuple{Int, Int}

Retrieves the global root node IDs for the source (trial) and observer (test) clusters
from a fully assembled Butterfly Factorization.
"""
function getnsno(BFactorization::ButterflyFactorization)
    # Extracts the source root from the first P block, and observer root from the first Q block
    return block_key(BFactorization.P[1])[4], block_key(BFactorization.Q[1])[1]
end

"""
    getrowidx(block::ButterflyBlock) -> Tuple{Int, Int}

Extracts the `(obs_out, src_out)` output routing key, representing the row-space mapping.
"""
function getrowidx(block::ButterflyBlock)
    return (block.obs_out, block.src_out)
end

"""
    getcolidx(block::ButterflyBlock) -> Tuple{Int, Int}

Extracts the `(obs_in, src_in)` input routing key, representing the column-space mapping.
"""
function getcolidx(block::ButterflyBlock)
    return (block.obs_in, block.src_in)
end

"""
    blockdiag(blk_A::ButterflyBlock{T}, blk_B::ButterflyBlock{T}) where {T}

Merges two `ButterflyBlock`s that share the exact same routing keys into a single
block-diagonal `ButterflyBlock`. This is highly useful for compressing and grouping
independent interactions.
"""
function blockdiag(blk_A::ButterflyBlock{T}, blk_B::ButterflyBlock{T}) where {T}
    @assert blk_A.obs_out == blk_B.obs_out && blk_A.src_out == blk_B.src_out
    @assert blk_A.obs_in == blk_B.obs_in && blk_A.src_in == blk_B.src_in

    new_data = blockdiag(blk_A.data, blk_B.data)

    return ButterflyBlock(
        blk_A.obs_out, blk_A.src_out, blk_A.obs_in, blk_A.src_in, new_data
    )
end

"""
    clear!(logger::RankLogger)

Empties all thread-local buffers within the `RankLogger`, preparing it for a new
assembly pass without reallocating the underlying vectors.
"""
function clear!(logger::RankLogger)
    for buf in logger.buffers
        empty!(buf)
    end
end

"""
    get_n_otilde(e::Union{RankEstimate, Int}) -> Int

Safely extracts the scalar rank guess from either a `RankEstimate` struct or a raw integer.
"""
get_n_otilde(e::RankEstimate) = e.n_otilde
get_n_otilde(n::Int) = n

"""
    ThreadButterflyWorkspace{T}(max_levels::Int=20)

Initializes a pre-allocated workspace designed to hold intermediate state vectors
during a matrix-free mat-vec operation. Defaults to supporting up to a depth of 20 levels.
"""
function ThreadButterflyWorkspace{T}(max_levels::Int=20) where {T}
    level_buffers = [Vector{Vector{T}}() for _ in 1:max_levels]
    level_lengths = [Vector{Int}() for _ in 1:max_levels]
    return ThreadButterflyWorkspace{T}(level_buffers, level_lengths)
end

"""
    get_buffer_and_set_len!(pool, lens, ptr, required_size)

Dynamically manages memory within a thread's flat-array memory pool.

If the requested pointer `ptr` exceeds current tracking limits, it expands the pool.
If the physical buffer at `ptr` is too small, it calls `resize!` to grow it.
Crucially, it records the exact mathematical length required for this block into `lens`,
allowing subsequent mat-vec operations to only read valid data.

**Returns:**

  - The appropriately sized memory buffer `Vector{T}`.
"""
@inline function get_buffer_and_set_len!(
    pool::Vector{Vector{T}}, lens::Vector{Int}, ptr::Int, required_size::Int
) where {T}
    # 1. Expand pointer slots if needed
    if ptr > length(pool)
        old_len = length(pool)
        resize!(pool, ptr)
        resize!(lens, ptr)
        for i in (old_len + 1):ptr
            pool[i] = Vector{T}()
            lens[i] = 0
        end
    end

    # 2. Grab buffer and ensure physical capacity
    buf = pool[ptr]
    if length(buf) < required_size
        resize!(buf, required_size)
    end

    # 3. Record the EXACT mathematical length written to this pointer
    lens[ptr] = required_size

    return buf
end

# ------------------------------------------------------------------
# Helpers for memory & rank analysis
# ------------------------------------------------------------------

"""
    format_bytes(bytes::Real) -> String

Converts a raw byte count into a human-readable string (e.g., "KiB", "MiB", "GiB").
"""
function format_bytes(bytes::Real)
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    i = 1
    b = Float64(bytes)
    while b >= 1024 && i < length(units)
        b /= 1024
        i += 1
    end
    return string(round(b; digits=2), " ", units[i])
end

"""
    block_stats(blocks::Vector{ButterflyBlock{T}})

Aggregates compression statistics for a specific vector of `ButterflyBlock`s.

**Returns:**

  - A `NamedTuple` containing the count of blocks, number of identity matrices,
    minimum/maximum/average ranks, and total memory footprint in bytes.
"""
function block_stats(blocks::Vector{ButterflyBlock{T}}) where {T}
    num_blocks = length(blocks)
    if num_blocks == 0
        return (count=0, ident=0, min_r=0, max_r=0, avg_r=0.0, bytes=0)
    end

    identities = 0
    ranks = Int[]
    sizehint!(ranks, num_blocks)
    total_bytes = 0

    for b in blocks
        if b.data isa UniformScaling
            identities += 1
        else
            r, c = size(b.data)
            push!(ranks, r)
            total_bytes += sizeof(b.data)
        end
    end

    min_r = isempty(ranks) ? 0 : minimum(ranks)
    max_r = isempty(ranks) ? 0 : maximum(ranks)
    avg_r = isempty(ranks) ? 0.0 : sum(ranks) / length(ranks)

    return (
        count=num_blocks,
        ident=identities,
        min_r=min_r,
        max_r=max_r,
        avg_r=avg_r,
        bytes=total_bytes,
    )
end

"""
    print_table_row(io, lvl_name, count, ident_pct, avg_r, max_r)

Helper to print cleanly aligned table rows for diagnostic outputs without relying on
external formatting packages like `Printf`.
"""
function print_table_row(
    io::IO, lvl_name::String, count::Int, ident_pct::Float64, avg_r::Float64, max_r::Int
)
    col1 = rpad(lvl_name, 7)
    col2 = lpad(string(count), 7)
    col3 = lpad(string(round(ident_pct; digits=1), "%"), 11)
    col4 = lpad(string(round(avg_r; digits=1)), 9)
    col5 = lpad(string(max_r), 9)

    return println(
        io, "     │ ", col1, " │ ", col2, " │ ", col3, " │ ", col4, " │ ", col5, " │"
    )
end
