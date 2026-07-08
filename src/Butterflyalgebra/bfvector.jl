@views function LinearAlgebra.mul!(
    y::AbstractVector, Butterfly::BF, x::AbstractVector{T}
) where {T}
    LinearMaps.check_dim_mul(y, Butterfly, x)
    fill!(y, zero(T))
    y .= apply_BF(Butterfly, x)
    return y
end

@views function LinearAlgebra.mul!(
    y::AbstractVector,
    At::LinearMaps.TransposeMap{<:Any,<:ButterflyFactorizations.BF},
    x::AbstractVector{T},
) where {T}
    LinearMaps.check_dim_mul(y, transposed(At.lmap), x)
    fill!(y, zero(T))
    y .= applyBF(transpose(At.lmap), x)
    return y
end

@views function LinearAlgebra.mul!(
    y::AbstractVector,
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
    otree = Butterfly.otree
    stree = Butterfly.stree

    #old_blas = BLAS.get_num_threads()
    #BLAS.set_num_threads(1)

    # ------------------------------------------------------------
    # Leaf initialization (Q)
    # ------------------------------------------------------------
    q_keys = collect(keys(Q))
    q_results = tmap(q_keys; scheduler=scheduler) do qkey
        srcvals = H2Trees.values(stree, qkey[2])
        out = Vector{ComplexF64}(undef, size(Q[qkey], 1))
        @views mul!(out, Q[qkey], v[srcvals])
        return (qkey, out)
    end

    coeffs_current = Dict{Tuple{Int,Int},Vector{ComplexF64}}(q_results)

    # Step 2: Sequentially apply R factors (Ping-Pong strategi)
    for l in eachindex(R)
        r_keys = collect(keys(R[l]))
        r_results = let coeffs_current = coeffs_current
            tmap(r_keys; scheduler=scheduler) do row
                cols = collect(keys(R[l][row]))

                # Skapa ny vektor för första kolumn-bidraget
                if !isempty(R[l][row][cols[1]])
                    out = Vector{ComplexF64}(undef, size(R[l][row][cols[1]], 1))
                    @views mul!(out, R[l][row][cols[1]], coeffs_current[cols[1]])
                else
                    out = coeffs_current[cols[1]]
                end
                # Om fler kolumner ska slås ihop för denna "row"
                if length(cols) > 1
                    coeff_temp = Vector{ComplexF64}(undef, length(out))
                    for i in 2:length(cols)
                        in_vec_next = coeffs_current[cols[i]]

                        #if isempty(R[l][row][cols[i]])
                        #    out .+= in_vec_next
                        #else
                        @views mul!(coeff_temp, R[l][row][cols[i]], in_vec_next)
                        out .+= coeff_temp
                        #end
                    end
                end

                return (row, out)
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
        tmap(p_keys; scheduler=scheduler) do pkey
            if !haskey(coeffs_current, pkey)
                println(
                    "Warning: No coefficients found for P key ",
                    pkey,
                    ". This may indicate a mismatch in the factorization structure.",
                )
            end
            inds = H2Trees.values(otree, pkey[1])
            out = Vector{ComplexF64}(undef, size(P[pkey], 1))
            # Kör Mat-vekt mult på den lokala out
            mul!(out, P[pkey], coeffs_current[pkey])
            return (inds, out)
        end
    end

    # Säker ihopslagnig på slutet i Main tråden!
    result = zeros(ComplexF64, length(H2Trees.values(Butterfly.otree, 1)))
    for (inds, out) in p_results
        result[inds] .+= out
    end
    return result
end

function mul_flat_bf!(
    y::AbstractVector{ComplexF64}, bf::FlatBF, x::AbstractVector{ComplexF64}
)
    # 1. Zero out the pre-allocated workspace
    for v in bf.layer_vectors
        fill!(v, zero(ComplexF64))
    end

    # Aliases for clarity
    layer_vectors = bf.layer_vectors

    # --- STEG 1: Applicera Q ---
    r1_in = layer_vectors[1]
    for i in 1:length(bf.Q.blocks)
        B = bf.Q.blocks[i]
        c_start = bf.Q.col_offsets[i]
        p = bf.Q.perm[i]

        # Fast memory mapping (consider writing a manual loop if blocks are tiny!)
        @views mul!(r1_in[c_start:(c_start + size(B, 1) - 1)], B, x[p], 1.0, 1.0)
    end

    # --- STEG 2: Loopa igenom alla R-nivåer ---
    for l in 1:length(bf.R)
        layer = bf.R[l]
        v_in = layer_vectors[l]
        v_out = layer_vectors[l + 1]

        # Native for-loops (no tforeach overhead)
        @inbounds for i in 1:(length(layer.row_ptr) - 1)
            r_start = layer.row_offsets[i]

            for b in layer.row_ptr[i]:(layer.row_ptr[i + 1] - 1)
                j = layer.col_idx[b]
                c_start = layer.col_offsets[j]
                B = layer.blocks[b]

                if isempty(B)
                    nr = if (i < length(layer.row_offsets))
                        (layer.row_offsets[i + 1] - r_start)
                    else
                        (length(v_out) - r_start + 1)
                    end
                    @views v_out[r_start:(r_start + nr - 1)] .+= v_in[c_start:(c_start + nr - 1)]
                    continue
                end

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

    # --- STEG 3: Applicera P (writes directly to global y) ---
    rend_out = layer_vectors[end]
    for i in 1:length(bf.P.blocks)
        B = bf.P.blocks[i]
        r_start = bf.P.row_offsets[i]
        p = bf.P.perm[i]

        # Safe because PermP represents disjoint leaves
        @views mul!(y[p], B, rend_out[r_start:(r_start + size(B, 2) - 1)], 1.0, 1.0)
    end

    return nothing # Writes directly to y
end

function mul_flat_bf_p(
    bf::FlatBF, x::AbstractVector{ComplexF64}; scheduler=OhMyThreads.SerialScheduler()
)
    # 1. Allokera temporära arbetsvektorer baserat på lagrens storlek
    # (Detta kan optimeras ytterligare genom att återanvända minne mellan MV-anrop)
    layer_vectors = Vector{Vector{ComplexF64}}(undef, length(bf.R) + 1)
    if length(bf.R) == 0
        # Räkna ut exakt hur stor den mellanliggande vektorn måste vara
        int_size = if isempty(bf.Q.blocks)
            0
        else
            (bf.Q.col_offsets[end] + size(bf.Q.blocks[end], 1) - 1)
        end
        layer_vectors[1] = zeros(ComplexF64, int_size)
    else
        layer_vectors[1] = zeros(ComplexF64, bf.R[1].in_size)
        for l in 1:length(bf.R)
            layer_vectors[l + 1] = zeros(ComplexF64, bf.R[l].out_size)
        end
    end
    y = zeros(ComplexF64, bf.dim[1])
    # --- STEG 1: Applicera Q ---
    r1_in = layer_vectors[1]
    tforeach(1:length(bf.Q.blocks); scheduler=scheduler) do i
        B = bf.Q.blocks[i]
        c_start = bf.Q.col_offsets[i]
        p = bf.Q.perm[i]

        # Byt ut mot BLAS gemv! för maximal prestanda:
        @views mul!(r1_in[c_start:(c_start + size(B, 1) - 1)], B, x[p], 1.0, 1.0)
    end

    # --- STEG 2: Loopa igenom alla R-nivåer ---
    for l in 1:length(bf.R)
        layer = bf.R[l]
        v_in = layer_vectors[l]
        v_out = layer_vectors[l + 1]

        # Denna loop är helt trådsäker eftersom trådarna skriver till helt olika segment i v_out!
        # CPU-prefetchern kommer att älska denna sekventiella minnesläsning.
        tforeach(1:(length(layer.row_ptr) - 1); scheduler=scheduler) do i
            r_start = layer.row_offsets[i]

            for b in layer.row_ptr[i]:(layer.row_ptr[i + 1] - 1)
                j = layer.col_idx[b]
                c_start = layer.col_offsets[j]
                B = layer.blocks[b]
                if isempty(B)
                    # FIX: Eftersom storleken av B är (0,0), räkna fram storleken
                    # genom att titta på var NÄSTA rad offset börjar.
                    # Om det är den sista raden lånar vi storleken via v_out:s längd.
                    nr = if (i < length(layer.row_offsets))
                        (layer.row_offsets[i + 1] - r_start)
                    else
                        (length(v_out) - r_start + 1)
                    end

                    # Identitetsmatris. Kopiera direkt från in till ut!
                    @views v_out[r_start:(r_start + nr - 1)] .+= v_in[c_start:(c_start + nr - 1)]
                    continue
                end
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
