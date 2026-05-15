@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat, Butterfly::BF, x::AbstractVector{T}
) where {T}
    LinearMaps.check_dim_mul(y, Butterfly, x)
    result = apply_BF(Butterfly, x)
    copyto!(y, result)
    return nothing
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

    # Endast en dict för nuvarande nivå behövs initialt
    coeffs_current = Dict{Tuple{Int,Int},Vector{ComplexF64}}()

    # ------------------------------------------------------------
    # Leaf initialization (Q)
    # ------------------------------------------------------------
    for Sleaf in keys(Q)
        srcvals = PermQ[Sleaf]

        # Allokera och beräkna direkt
        coeffs_current[(NO, Sleaf)] = Vector{ComplexF64}(undef, size(Q[Sleaf], 1))
        @views mul!(coeffs_current[(NO, Sleaf)], Q[Sleaf], v[srcvals])
    end

    # Step 2: Sequentially apply R factors (Ping-Pong strategi)
    for l in eachindex(R)
        coeffs_next = Dict{Tuple{Int,Int},Vector{ComplexF64}}()

        for row in keys(R[l])
            for col in keys(R[l][row])
                if !haskey(coeffs_next, row)
                    # Skapa ny vektor för första kolumn-bidraget
                    coeffs_next[row] = Vector{ComplexF64}(undef, size(R[l][row][col], 1))
                    @views mul!(coeffs_next[row], R[l][row][col], coeffs_current[col])
                else
                    # Addera till existerande vektor vektor med temporär minneshantering
                    coeff_temp = Vector{ComplexF64}(undef, size(R[l][row][col], 1))
                    @views mul!(coeff_temp, R[l][row][col], coeffs_current[col])
                    coeffs_next[row] .+= coeff_temp
                end
            end
        end

        # Flytta next till current inför nästa nivå
        coeffs_current = coeffs_next
    end

    # Step 3: Apply P to the result from the last R factor
    # ------------------------------------------------------------
    # Final assembly
    # ------------------------------------------------------------
    result = zeros(ComplexF64, Butterfly.dim[1])
    for Oleaf in keys(P)
        inds = PermP[Oleaf]
        dest = @view result[inds]
        mul!(dest, P[Oleaf], coeffs_current[(Oleaf, NS)])
    end
    return result
end

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat, Butterfly::BF_Mats, x::AbstractVector{T}
) where {T}
    LinearMaps.check_dim_mul(y, Butterfly, x)
    result = applyBF_Mats(Butterfly, x)
    copyto!(y, result)
    return nothing
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
