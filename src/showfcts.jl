import Base: show

function Base.show(io::IO, b::ButterflyBlock{T}) where {T}
    data_str =
        b.data isa UniformScaling ? "I" : string(size(b.data, 1), "×", size(b.data, 2))
    return print(
        io,
        "ButterflyBlock((",
        b.obs_out,
        ", ",
        b.src_out,
        ") ← (",
        b.obs_in,
        ", ",
        b.src_in,
        ") | ",
        data_str,
        ")",
    )
end

function Base.show(io::IO, level::ButterflyLevel{T}) where {T}
    stats = block_stats(level.blocks)
    print(io, "ButterflyLevel(", stats.count, " blocks")
    if stats.ident > 0
        print(io, ", ", stats.ident, " frozen")
    end
    if stats.count > stats.ident
        print(io, ", r_avg=", round(stats.avg_r; digits=1), ", r_max=", stats.max_r)
    end
    return print(io, ")")
end

function Base.show(io::IO, ::MIME"text/plain", level::ButterflyLevel{T}) where {T}
    stats = block_stats(level.blocks)
    println(io, "ButterflyLevel{", T, "} with ", stats.count, " blocks:")
    println(
        io,
        "  • Dense blocks : ",
        stats.count - stats.ident,
        " (avg rank: ",
        round(stats.avg_r; digits=1),
        ", max: ",
        stats.max_r,
        ")",
    )
    println(io, "  • Frozen (I)   : ", stats.ident)
    return print(io, "  • Memory       : ", format_bytes(stats.bytes))
end

function Base.show(io::IO, BF::ButterflyFactorization{T}) where {T}
    m, n = size(BF)
    return print(
        io,
        m,
        "×",
        n,
        " ButterflyFactorization{",
        T,
        "} (L=",
        length(BF.R)+1,
        ", k=",
        BF.k,
        ", τ=",
        BF.τ,
        ")",
    )
end

function Base.show(io::IO, ::MIME"text/plain", BF::ButterflyFactorization{T}) where {T}
    m, n = size(BF)
    num_R_levels = length(BF.R)

    q_stats = block_stats(BF.Q)
    p_stats = block_stats(BF.P)

    total_bytes = q_stats.bytes + p_stats.bytes
    for lvl in BF.R
        total_bytes += block_stats(lvl.blocks).bytes
    end

    println(io, "ButterflyFactorization{", T, "}")
    println(io, "  ├─ Dimensions : ", m, " × ", n)
    println(io, "  ├─ Parameters : k = ", BF.k, ", τ = ", BF.τ)
    println(
        io,
        "  ├─ Total Depth: ",
        num_R_levels + 1,
        " levels (Q, R¹..R",
        num_R_levels,
        ", P)",
    )
    println(io, "  ├─ Matrix Storage: ~", format_bytes(total_bytes))
    println(io, "  └─ Level Breakdown:")

    # Table Header
    println(io, "     ┌─────────┬─────────┬─────────────┬───────────┬───────────┐")
    println(io, "     │ Level   │ Blocks  │ Identity %  │ Avg Rank  │ Max Rank  │")
    println(io, "     ├─────────┼─────────┼─────────────┼───────────┼───────────┤")

    # Q Level
    q_pct = q_stats.count == 0 ? 0.0 : (q_stats.ident / q_stats.count) * 100.0
    print_table_row(io, "Q", q_stats.count, q_pct, q_stats.avg_r, q_stats.max_r)

    # Intermediate R Levels
    for (i, lvl) in enumerate(BF.R)
        st = block_stats(lvl.blocks)
        ident_pct = st.count == 0 ? 0.0 : (st.ident / st.count) * 100.0
        print_table_row(io, string("R", i), st.count, ident_pct, st.avg_r, st.max_r)
    end

    # P Level
    p_pct = p_stats.count == 0 ? 0.0 : (p_stats.ident / p_stats.count) * 100.0
    print_table_row(io, "P", p_stats.count, p_pct, p_stats.avg_r, p_stats.max_r)

    return print(io, "     └─────────┴─────────┴─────────────┴───────────┴───────────┘")
end

function Base.show(io::IO, A::ButterflyFactorizations.PetrovGalerkinBF{T}) where {T}
    m, n = size(A)
    return print(io, m, "×", n, " PetrovGalerkinBF{", T, "}(far=", length(A.BFs), ")")
end

function Base.show(
    io::IO, ::MIME"text/plain", A::ButterflyFactorizations.PetrovGalerkinBF{T}
) where {T}
    m, n = size(A)
    println(io, "PetrovGalerkinBF{", T, "}")
    println(io, "  Dimensions : ", m, " × ", n)
    println(io, "  Near field : ", typeof(A.nearinteractions))
    println(io, "  Flat BFs    : ", length(A.BFs))
    println(io, "  Tree type   : ", typeof(A.tree))
    println(
        io, "  Near lookup : nnz=", nnz(A.near_lookup), ", type=", typeof(A.near_lookup)
    )
    return print(
        io, "  Far lookup  : nnz=", nnz(A.far_lookup), ", type=", typeof(A.far_lookup)
    )
end
