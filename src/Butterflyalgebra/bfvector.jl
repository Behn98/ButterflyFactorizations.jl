@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat, Butterfly::BF, x::AbstractVector{T}
) where {T}
    LinearMaps.check_dim_mul(y, Butterfly, x)
    fill!(y, zero(T))
    y .= apply_BF(Butterfly, x)
    return y
end

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat,
    At::LinearMaps.TransposeMap{<:Any,<:ButterflyFactorizations.BF},
    x::AbstractVector{T},
) where {T}
    LinearMaps.check_dim_mul(y, transposed(At.lmap), x)
    fill!(y, zero(T))
    y .= applyBF(transpose(At.lmap), x)
    return y
end

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat,
    At::LinearMaps.AdjointMap{<:Any,<:ButterflyFactorizations.BF},
    x::AbstractVector{T},
) where {T}
    LinearMaps.check_dim_mul(y, At.lmap, x)
    fill!(y, zero(T))
    y .= applyBF(At.lmap', x)
    return y
end

function Base.:*(Butterfly::BF, x::AbstractVector)
    return apply_BF(Butterfly, x)
end

"""
    apply_BF(Butterfly::BF, v::AbstractVector{ComplexF64})

Applies the sequence of Butterfly Factorization factors (`Q`, `R`, and `P`) to a vector `v`
and returns the resulting vector. Do note that this function is optimized for the structure
of `BF` and uses a "ping-pong" strategy to minimize memory allocations during the sequential
application of the `R` factors. The indices of the dictionaries are matching the row and
column indices of the skeleton associated with the underlying Matrix structure thus we can
perform a block by block matrix vector product using the well known algebra for matrices,
while retaining the semantical structure of the corresponding clusters.
"""
function apply_BF(Butterfly::BF, v::AbstractVector{ComplexF64})
    Q = Butterfly.Q
    R = Butterfly.R
    P = Butterfly.P
    NO = Butterfly.NO
    NS = Butterfly.NS
    PermQ = Butterfly.PermQ
    PermP = Butterfly.PermP

    # ------------------------------------------------------------
    # Leaf initialization (Q)
    # ------------------------------------------------------------
    q_keys = collect(keys(Q))
    q_results = tmap(q_keys) do Sleaf
        srcvals = PermQ[Sleaf]
        # Allokera thread-lokal vektor och beräkna direkt
        out = Vector{ComplexF64}(undef, size(Q[Sleaf], 1))
        @views mul!(out, Q[Sleaf], v[srcvals])
        ((NO, Sleaf), out)
    end

    # Skapa current-ordboken blixtsnabbt från listan av Pairs
    coeffs_current = Dict{Tuple{Int,Int},Vector{ComplexF64}}(q_results)

    # Step 2: Sequentially apply R factors (Ping-Pong strategi)
    # Själva nivåerna KAN INTE parallelliseras, men blocken PÅ en nivå kan!
    for l in eachindex(R)
        r_keys = collect(keys(R[l]))
        r_results = let coeffs_current = coeffs_current
            tmap(r_keys) do row
                cols = collect(keys(R[l][row]))

                # Skapa ny vektor för första kolumn-bidraget
                out = Vector{ComplexF64}(undef, size(R[l][row][cols[1]], 1))
                @views mul!(out, R[l][row][cols[1]], coeffs_current[cols[1]])

                # Om fler kolumner ska slås ihop för denna "row"
                if length(cols) > 1
                    coeff_temp = Vector{ComplexF64}(undef, length(out))
                    for i in 2:length(cols)
                        @views mul!(coeff_temp, R[l][row][cols[i]], coeffs_current[cols[i]])
                        out .+= coeff_temp
                    end
                end

                (row, out)
            end
        end # SLUT PÅ let-BLOCKET

        # Uppdatera next till current inför nästa nivå
        coeffs_current = Dict{Tuple{Int,Int},Vector{ComplexF64}}(r_results)
    end

    # Step 3: Apply P to the result from the last R factor
    # ------------------------------------------------------------
    # Final assembly
    # ------------------------------------------------------------
    p_keys = collect(keys(P))
    p_results = let coeffs_current = coeffs_current
        tmap(p_keys) do Oleaf
            inds = PermP[Oleaf]
            out = Vector{ComplexF64}(undef, size(P[Oleaf], 1))
            # Kör Mat-vekt mult på den lokala out
            mul!(out, P[Oleaf], coeffs_current[(Oleaf, NS)])
            (inds, out)
        end
    end

    # Säker ihopslagnig på slutet i Main tråden!
    result = zeros(ComplexF64, Butterfly.dim[1])
    for (inds, out) in p_results
        result[inds] .= out
    end

    return result
end

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat, Butterfly::BF_Mats, x::AbstractVector{T}
) where {T}
    LinearMaps.check_dim_mul(y, Butterfly, x)
    fill!(y, zero(T))
    y .= applyBF_Mats(Butterfly, x)
    return y
end

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat,
    At::LinearMaps.TransposeMap{<:Any,<:ButterflyFactorizations.BF_Mats},
    x::AbstractVector{T},
) where {T}
    LinearMaps.check_dim_mul(y, transposed(At.lmap), x)
    fill!(y, zero(T))
    y .= applyBF_Mats(transpose(At.lmap), x)
    return y
end

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat,
    At::LinearMaps.AdjointMap{<:Any,<:ButterflyFactorizations.BF_Mats},
    x::AbstractVector{T},
) where {T}
    LinearMaps.check_dim_mul(y, At.lmap, x)
    fill!(y, zero(T))
    y .= applyBF_Mats(At.lmap', x)
    return y
end

function Base.:*(Butterfly::BF_Mats, x::AbstractVector)
    return applyBF_Mats(Butterfly, x)
end

"""
    applyBF_Mats(t::BF_Mats, v::AbstractVector{ComplexF64})

Applies the sequential matrix operations (`Q`, `R` layers, and `P`) of a
`BF_Mats` factorization to a vector `v` and returns the output vector.
"""
function applyBF_Mats(t::BF_Mats, v::AbstractVector{ComplexF64})
    y = v
    y = t.Q * y
    for R_block in t.R
        y = R_block * y
    end
    result = zeros(ComplexF64, size(t, 1))
    result = t.P * y
    return result
end
