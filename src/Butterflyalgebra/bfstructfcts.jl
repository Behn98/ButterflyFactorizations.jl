# The sorting key now perfectly identifies the exact interaction pair
function block_key(b::ButterflyBlock)
    return (b.obs_out, b.src_out, b.obs_in, b.src_in)
end

function getNSNO(BFactorization::ButterflyFactorization)
    return block_key(BFactorization.P[1])[4], block_key(BFactorization.Q[1])[1]
end

# ------------------------------------------------------------------
# Helpers for memory & rank analysis
# ------------------------------------------------------------------

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

# Helper to print aligned table rows without Printf
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
