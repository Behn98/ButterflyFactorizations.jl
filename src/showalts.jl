import Base: show

# 1-line summary
function show(io::IO, alg::AlgBF{T,M}) where {T,M}
    return print(io, "AlgBF{$T, $M}(dim=$(alg.dim))")
end

# Multi-line REPL presentation
function show(io::IO, mime::MIME"text/plain", alg::AlgBF{T,M}) where {T,M}
    println(io, "Algebraic Butterfly Factorization (AlgBF)")
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
    return print(io, "Q_factor{$T, $M}(keys=$(length(q.dict)))")
end

function show(io::IO, ::MIME"text/plain", q::Q_factor{T,M}) where {T,M}
    println(io, "Butterfly Q_factor")
    println(io, "  Matrix Type: ", M)
    println(io, "  Tree Type:   ", T)
    return print(io, "  Storage:     ", length(q.dict), " blocks mapped in Dict")
end

# --- P_factor Display ---
function show(io::IO, p::P_factor{T,M}) where {T,M}
    return print(io, "P_factor{$T, $M}(keys=$(length(p.dict)))")
end

function show(io::IO, ::MIME"text/plain", p::P_factor{T,M}) where {T,M}
    println(io, "Butterfly P_factor")
    println(io, "  Matrix Type: ", M)
    println(io, "  Tree Type:   ", T)
    return print(io, "  Storage:     ", length(p.dict), " blocks mapped in Dict")
end

# --- R_factor Display ---
function show(io::IO, r::R_factor{T,M}) where {T,M}
    return print(io, "R_factor{$T, $M}(slvl=$(r.slvl), olvl=$(r.olvl))")
end

function show(io::IO, ::MIME"text/plain", r::R_factor{T,M}) where {T,M}
    println(io, "Butterfly R_factor (Middle Level)")
    println(io, "  Matrix Type: ", M)
    println(io, "  Levels (s/o):", r.slvl, " / ", r.olvl)
    return print(io, "  Storage:     ", length(r.dict), " top-level block keys")
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

# 1. This controls the 1-line display (e.g., when inside arrays)
function show(io::IO, bf::BF_Mats)
    return print(io, "BF_Mats(dim=$(bf.dim), k=$(bf.k))")
end

# 2. This controls the multi-line display when a BF object is returned in the REPL
function show(io::IO, mime::MIME"text/plain", bf::BF_Mats)
    println(io, "Butterfly Factorization (BF_Mats)")
    println(io, "  Dimension:   ", bf.dim)
    println(io, "  NS / NO:     ", bf.NS, " / ", bf.NO)
    println(io, "  k / τ:       ", bf.k, " / ", bf.τ)
    return print(
        io,
        "  Storage:     Q ($(Base.summarysize(bf.Q)/1024) KB), R ($(length(bf.R)) levels), P ($(Base.summarysize(bf.P)/1024) KB)",
    )
end
