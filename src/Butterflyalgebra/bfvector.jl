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
    apply_BF(Butterfly::BF, v::AbstractVector)

Applies the sequence of Butterfly Factorization factors (`Q`, `R`, and `P`) to a vector `v`
and returns the resulting vector. Do note that this function is optimized for the structure
of `BF` and uses a "ping-pong" strategy to minimize memory allocations during the sequential
application of the `R` factors. The indices of the dictionaries are matching the row and
column indices of the skeleton associated with the underlying Matrix structure thus we can
perform a block by block matrix vector product using the well known algebra for matrices,
while retaining the semantical structure of the corresponding clusters.
"""
function apply_BF(
    Butterfly::BF, v::AbstractVector; scheduler=OhMyThreads.DynamicScheduler()
)
    Q = Butterfly.Q
    R = Butterfly.R
    P = Butterfly.P
    NO = Butterfly.NO
    NS = Butterfly.NS
    PermQ = Butterfly.PermQ
    PermP = Butterfly.PermP

    old_blas = BLAS.get_num_threads()
    BLAS.set_num_threads(1)

    # ------------------------------------------------------------
    # Leaf initialization (Q)
    # ------------------------------------------------------------
    q_keys = collect(keys(Q))
    q_results = tmap(q_keys; scheduler=scheduler) do Sleaf
        srcvals = PermQ[Sleaf]
        out = Vector{ComplexF64}(undef, size(Q[Sleaf], 1))
        @views mul!(out, Q[Sleaf], v[srcvals])
        ((NO, Sleaf), out)
    end

    coeffs_current = Dict{Tuple{Int,Int},Vector{ComplexF64}}(q_results)

    # Step 2: Sequentially apply R factors (Ping-Pong strategi)
    for l in eachindex(R)
        r_keys = collect(keys(R[l]))
        r_results = let coeffs_current = coeffs_current
            tmap(r_keys; scheduler=scheduler) do row
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
        end

        # Uppdatera next till current inför nästa nivå
        coeffs_current = Dict{Tuple{Int,Int},Vector{ComplexF64}}(r_results)
    end

    # Step 3: Apply P to the result from the last R factor
    # ------------------------------------------------------------
    # Final assembly
    # ------------------------------------------------------------
    p_keys = collect(keys(P))
    p_results = let coeffs_current = coeffs_current
        tmap(p_keys; scheduler=scheduler) do Oleaf
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
        result[inds] .+= out
    end
    BLAS.set_num_threads(old_blas)
    return result
end

function mul_flat_bf(bf::FlatBF, x::AbstractVector; scheduler=OhMyThreads.SerialScheduler())
    # 1. Allokera temporära arbetsvektorer baserat på lagrens storlek
    layer_vectors = Vector{Vector{ComplexF64}}(undef, length(bf.R) + 1)
    layer_vectors[1] = zeros(ComplexF64, bf.R[1].in_size)
    for l in 1:length(bf.R)
        layer_vectors[l + 1] = zeros(ComplexF64, bf.R[l].out_size)
    end

    y = zeros(ComplexF64, bf.dim[1])

    # --- STEG 1: Applicera Q ---
    r1_in = layer_vectors[1]
    tforeach(1:length(bf.Q.blocks); scheduler=scheduler) do i
        B = bf.Q.blocks[i]
        c_start = bf.Q.col_offsets[i]
        p = bf.Q.perm[i]

        @views mul!(r1_in[c_start:(c_start + size(B, 1) - 1)], B, x[p], 1.0, 1.0)
    end

    # --- STEG 2: Loopa igenom alla R-nivåer ---
    for l in 1:length(bf.R)
        layer = bf.R[l]
        v_in = layer_vectors[l]
        v_out = layer_vectors[l + 1]

        tforeach(1:(length(layer.row_ptr) - 1); scheduler=scheduler) do i
            r_start = layer.row_offsets[i]

            for b in layer.row_ptr[i]:(layer.row_ptr[i + 1] - 1)
                j = layer.col_idx[b]
                c_start = layer.col_offsets[j]
                B = layer.blocks[b]
                nr, nc = size(B)

                @views mul!(
                    v_out[r_start:(r_start + nr - 1)],
                    B,
                    v_in[c_start:(c_start + nc - 1)],
                    1.0,
                    1.0,
                )
            end
        end
    end

    # --- STEG 3: Applicera P ---
    rend_out = layer_vectors[end]
    tforeach(1:length(bf.P.blocks); scheduler=scheduler) do i
        B = bf.P.blocks[i]
        r_start = bf.P.row_offsets[i]
        p = bf.P.perm[i]

        # Eftersom PermP representerar disjunkta löv krockar inte trådarna i globala y
        @views mul!(y[p], B, rend_out[r_start:(r_start + size(B, 2) - 1)], 1.0, 1.0)
    end

    return y
end

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat, Butterfly::ButterflyFactorizations.FlatBF, x::AbstractVector{T}
) where {T}
    LinearMaps.check_dim_mul(y, Butterfly, x)
    fill!(y, zero(T))
    y .= mul_flat_bf(Butterfly, x; scheduler=OhMyThreads.DynamicScheduler())
    return y
end

function Base.:*(Butterfly::ButterflyFactorizations.FlatBF, x::AbstractVector)
    return mul_flat_bf(Butterfly, x)
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
    applyBF_Mats(t::BF_Mats, v::AbstractVector)

Applies the sequential matrix operations (`Q`, `R` layers, and `P`) of a
`BF_Mats` factorization to a vector `v` and returns the output vector.
"""
function applyBF_Mats(t::BF_Mats, v::AbstractVector)
    y = v
    y = t.Q * y
    for R_block in t.R
        y = R_block * y
    end
    result = zeros(ComplexF64, size(t, 1))
    result = t.P * y
    return result
end
