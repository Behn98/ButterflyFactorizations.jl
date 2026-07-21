import Base: show
# 1-line summary
function show(io::IO, alg::AlgBF{T,M}) where {T,M}
    return print(io, "AlgBF{$T, $M}(dim=$(alg.dim))")
end

# Multi-line REPL presentation
function show(io::IO, mime::MIME"text/plain", alg::AlgBF{T,M}) where {T,M}
    println(io, "Algorithmic Butterfly Factorization (AlgBF)")
    println(io, "  Tree Type:   ", T)
    println(io, "  Matrix Type: ", M)
    println(io, "  Dimension:   ", alg.dim)
    return print(
        io,
        "  Structure:   Q (",
        typeof(alg.Q),
        "), R (",
        length(alg.R),
        " levels), P (",
        typeof(alg.P),
        ")",
    )
end

# --- Q_factor Display ---
function show(io::IO, q::Q_factor{T,M}) where {T,M}
    return print(io, "Q_factor{$T, $M}(keys=$(length(q.Dict)))")
end

function show(io::IO, ::MIME"text/plain", q::Q_factor{T,M}) where {T,M}
    println(io, "Butterfly Q_factor")
    println(io, "  Matrix Type: ", M)
    println(io, "  Tree Type:   ", T)
    return print(io, "  Storage:     ", length(q.Dict), " blocks mapped in Dict")
end

# --- P_factor Display ---
function show(io::IO, p::P_factor{T,M}) where {T,M}
    return print(io, "P_factor{$T, $M}(keys=$(length(p.Dict)))")
end

function show(io::IO, ::MIME"text/plain", p::P_factor{T,M}) where {T,M}
    println(io, "Butterfly P_factor")
    println(io, "  Matrix Type: ", M)
    println(io, "  Tree Type:   ", T)
    return print(io, "  Storage:     ", length(p.Dict), " blocks mapped in Dict")
end

# --- R_factor Display ---
function show(io::IO, r::R_factor{T,M}) where {T,M}
    return print(io, "R_factor{$T, $M}(slvl=$(r.slvl), olvl=$(r.olvl))")
end

function show(io::IO, ::MIME"text/plain", r::R_factor{T,M}) where {T,M}
    println(io, "Butterfly R_factor (Middle Level)")
    println(io, "  Matrix Type: ", M)
    println(io, "  Levels (s/o):", r.slvl, " / ", r.olvl)
    return print(io, "  Storage:     ", length(r.Dict), " top-level block keys")
end

# 1. This controls the 1-line display (e.g., when inside arrays)
function show(io::IO, bf::BF{T,M}) where {T,M}
    return print(io, "BF{$T, $M}(dim=$(bf.dim), k=$(bf.k))")
end

# 2. This controls the multi-line display when a BF object is returned in the REPL
function show(io::IO, mime::MIME"text/plain", bf::BF{T,M}) where {T,M}
    println(io, "Butterfly Factorization (BF)")
    println(io, "  Tree Type:   ", T)
    println(io, "  Matrix Type: ", M)
    println(io, "  Dimension:   ", bf.dim)
    println(io, "  NS / NO:     ", bf.NS, " / ", bf.NO)
    println(io, "  k / τ:       ", bf.k, " / ", bf.τ)
    return print(
        io,
        "  Storage:     Q ($(length(bf.Q)) keys), R ($(length(bf.R)) levels), P ($(length(bf.P)) keys)",
    )
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

# ------------------------------------------------------------------
# 1. ButterflyBlock
# ------------------------------------------------------------------

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

# ------------------------------------------------------------------
# 2. ButterflyLevel
# ------------------------------------------------------------------

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

# ------------------------------------------------------------------
# 3. ButterflyFactorization
# ------------------------------------------------------------------

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

# ------------------------------------------------------------------
# 4. FlatBF
# ------------------------------------------------------------------

function Base.show(io::IO, bf::FlatBF)
    return print(
        io,
        "FlatBF(dim=",
        bf.dim,
        ", levels=",
        length(bf.R),
        ", NS=",
        bf.NS,
        ", NO=",
        bf.NO,
        ")",
    )
end

function Base.show(io::IO, ::MIME"text/plain", bf::FlatBF)
    println(io, "FlatBF")
    println(io, "  Dimensions : ", bf.dim)
    println(io, "  NS / NO    : ", bf.NS, " / ", bf.NO)
    println(io, "  Levels     : ", length(bf.R))
    println(io, "  Workspaces : ", length(bf.layer_vectors))
    println(io, "  Q blocks   : ", length(bf.Q.blocks))
    println(io, "  R levels   : ", length(bf.R))
    return print(io, "  P blocks   : ", length(bf.P.blocks))
end

# ------------------------------------------------------------------
# 5. Petrov-Galerkin wrappers
# ------------------------------------------------------------------

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
    println(io, "  Far blocks  : ", length(A.BFs))
    println(io, "  Tree type   : ", typeof(A.tree))
    println(
        io, "  Near lookup : nnz=", nnz(A.near_lookup), ", type=", typeof(A.near_lookup)
    )
    return print(
        io, "  Far lookup  : nnz=", nnz(A.far_lookup), ", type=", typeof(A.far_lookup)
    )
end

function Base.show(io::IO, A::ButterflyFactorizations.FlatPGBF{T}) where {T}
    m, n = size(A)
    return print(io, m, "×", n, " FlatPGBF{", T, "}(far=", length(A.BFs), ")")
end

function Base.show(
    io::IO, ::MIME"text/plain", A::ButterflyFactorizations.FlatPGBF{T}
) where {T}
    m, n = size(A)
    println(io, "FlatPGBF{", T, "}")
    println(io, "  Dimensions : ", m, " × ", n)
    println(io, "  Near field : ", typeof(A.nearinteractions))
    println(io, "  Flat BFs    : ", length(A.BFs))
    println(io, "  Tree type   : ", typeof(A.tree))
    return print(io, "  Note        : uses flattened ButterflyFactorization blocks")
end

function Base.show(io::IO, A::ButterflyFactorizations.FlatPGBF2{T}) where {T}
    m, n = size(A)
    return print(io, m, "×", n, " FlatPGBF2{", T, "}(far=", length(A.BFs), ")")
end

function Base.show(
    io::IO, ::MIME"text/plain", A::ButterflyFactorizations.FlatPGBF2{T}
) where {T}
    m, n = size(A)
    println(io, "FlatPGBF2{", T, "}")
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
