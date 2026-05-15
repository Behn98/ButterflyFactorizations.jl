function recompress_BF_left(Butterfly::AlgBF, τ)
    return recompress_BF_right(Butterfly', τ)'
end

function recompress_BF(Butterfly::AlgBF, τ)
    return recompress_BF_left(recompress_BF_right(Butterfly, τ), τ)
end

"""
    recompress_BF(Butterfly::BF, τ)

Recompresses a structural Butterfly Factorization (`BF`) by extracting its algebraic
factors, recompressing them with tolerance `τ`, and restructuring the output back into a
`BF`. This process involves two main steps: first, the right factors are recompressed using
a QR-based approach, and then the left factors are recompressed by transposing the
structure, applying the same right recompression, and transposing back. The resulting `BF`
maintains the same hierarchical structure but with potentially reduced ranks in the `R`
factors, leading to improved efficiency in storage and matrix-vector products while
preserving the overall accuracy within the specified tolerance. Any of the algebraic
operations is only supported for the Dictionary versions of the Butterflies, as the
matrix-based format is not designed for algebraic manipulations and would require a complete
restructuring of the underlying data representation to support such operations effectively.
"""
function recompress_BF(Butterfly::BF, τ)
    Q = Butterfly.Q
    R = Butterfly.R
    P = Butterfly.P
    BFalg = AlgBF(Butterfly.dim, Q, R, P)
    BFalg = recompress_BF(BFalg, τ)
    return BF(
        BFalg.Q,
        BFalg.R,
        BFalg.P,
        Butterfly.PermQ,
        Butterfly.PermP,
        Butterfly.dim,
        Butterfly.NS,
        Butterfly.NO,
        Butterfly.k,
        Butterfly.τ,
    )
end

function recompress_BF_right(Butterfly::AlgBF, τ)
    Q = Butterfly.Q
    R = Butterfly.R
    P = Butterfly.P
    lr = length(R)
    for l in eachindex(R)
        lold = lr - l + 1
        R_u = Dict{Int,Dict{Int,Matrix{ComplexF64}}}()
        col = Vector{Tuple{Int,Int}}(undef, 0)
        for row in keys(R[lold])
            col = unique(append!(col, keys(R[lold][row])))
        end
        for col_idx in col
            rows_with_col = [
                row for (row, inner_dict) in R[lold] if haskey(inner_dict, col_idx)
            ]
            R_k = Vector{Matrix{ComplexF64}}()
            row_spc = Vector{Int}()
            i = 1
            for row in rows_with_col
                push!(R_k, R[lold][row][col_idx])
                push!(row_spc, size(R_k[i], 1))

                i += 1
            end
            A_k = vcat(R_k...)
            #@show size(A_k)
            QRA = pqr(A_k; rtol=τ)
            if !haskey(R_u, col_idx[1])
                R_u[col_idx[1]] = Dict{Int,Matrix{ComplexF64}}()
            end
            if haskey(R_u[col_idx[1]], col_idx[2])
                @show "col_idx already exists in R_u, this should not happen!"
            end
            R_u[col_idx[1]][col_idx[2]] = QRA[2][:, invperm(QRA[3])]
            last = 0
            j = 1
            for row in rows_with_col
                R[lold][row][col_idx] = QRA[1][(last + 1):(last + row_spc[j]), :]
                last += row_spc[j]
                j += 1
            end
        end
        if l < lr
            R[lold - 1] = update_next_level_R_right(R_u, R[lold - 1])
        else
            Q = update_next_level_R_right(R_u, Q)
        end
    end

    return AlgBF(Butterfly.dim, Q, R, P)
end

@views function update_next_level_R_right(
    R_u::Dict{Int,Dict{Int,Matrix{ComplexF64}}},
    rightfactor::Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},AbstractMatrix{ComplexF64}}},
)
    for row in keys(rightfactor)
        for col in keys(rightfactor[row])
            rightfactor[row][col] = R_u[row[1]][row[2]] * rightfactor[row][col]
        end
    end
    return rightfactor
end

@views function update_next_level_R_right(
    R_u::Dict{Int,Dict{Int,Matrix{ComplexF64}}}, rightfactor::Dict{Int,Matrix{ComplexF64}}
)
    NO = collect(keys(R_u))[1]
    for nodeS in keys(rightfactor)
        rightfactor[nodeS] = R_u[NO][nodeS] * rightfactor[nodeS]
    end
    return rightfactor
end
