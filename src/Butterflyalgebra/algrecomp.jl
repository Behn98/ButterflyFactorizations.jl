import H2Trees: values, center, halfsize, children, isleaf, trialtree, testtree
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
    BFalg = AlgBF(Butterfly)
    BFalg = recompress_BF(BFalg, τ)
    return BF(BFalg, Butterfly.k, τ)
end

function recompress_BF_right(Butterfly_init::AlgBF, τ; include_Q=false)
    Butterfly = deepcopy(Butterfly_init)
    Q = Butterfly.Q.dict
    R = Butterfly.R
    P = Butterfly.P.dict
    lr = length(R)
    include_Q ? endidx = lr : endidx = lr - 1
    for l in eachindex(R[1:endidx])
        lold = lr - l + 1

        # Bulletproof: Flatten R_u to map the full col_idx tuple directly to its matrix
        R_u = Dict{Tuple{Int,Int},Matrix{ComplexF64}}()#Dict{Tuple{Int,Int},}
        eblocks = R[lold].elementblocks
        for (eidx, eblock) in enumerate(eblocks)
            esize = size(eblock)
            # 2. Process each unique column space
            row_spc = Vector{Int}(undef, esize[1])
            for i in eachindex(row_spc)
                row_spc[i] = size(eblock[i, 1], 1)
            end
            for j in eachindex(eblock[1, :])
                col_idx = R[lold].inverse_map[eidx][1, j][2]
                A_k = vcat(eblock[:, j]...)
                QRA = pqr(A_k; rtol=τ)

                # Extract the local transfer matrix
                T_mat = QRA[2][:, invperm(QRA[3])]
                R_u[col_idx] = T_mat
                last_idx = 0
                for i in eachindex(row_spc)
                    eblock[i, j] = Matrix(    #(parent_node, col_idx[2])
                        QRA[1][(last_idx + 1):(last_idx + row_spc[i]), :],
                    )
                    last_idx += row_spc[i]
                end
            end
        end
        # 3. Propagate the accumulated R_u transformations
        if l < lr
            update_next_level_R_right(R_u, R[lold - 1])
        else
            Q = update_next_level_R_right(R_u, Q)
        end
    end

    return AlgBF(Butterfly, Q, R, P)
end

function update_next_level_R_right(
    R_u::Dict{Tuple{Int,Int},Matrix{ComplexF64}}, rightfactor::R_factor{M}
) where {M}
    for row in keys(rightfactor.row_spaces)
        # Because the row key of rightfactor is exactly the col_idx of the previous level
        if haskey(R_u, row)
            T_mat = R_u[row]
            for col in rightfactor.row_spaces[row]
                eidx, i, j = rightfactor.block_map[(row, col)]
                eblock = rightfactor.elementblocks[eidx]
                eblock[i, j] = T_mat * eblock[i, j]
            end
        end
    end
    #return rightfactor
end

# Overload 1: Updating intermediate R factors (Clean 1:1 matching)
function update_next_level_R_right(
    R_u::Dict{Tuple{Int,Int},Matrix{ComplexF64}},
    rightfactor::Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},Matrix{ComplexF64}}},
)
    for row in keys(rightfactor)
        # Because the row key of rightfactor is exactly the col_idx of the previous level
        if haskey(R_u, row)
            T_mat = R_u[row]
            for col in keys(rightfactor[row])
                rightfactor[row][col] = T_mat * rightfactor[row][col]
            end
        end
    end
    return rightfactor
end

# Overload 2: Updating the terminal Q factor
function update_next_level_R_right(
    R_u::Dict{Tuple{Int,Int},Matrix{ComplexF64}},
    rightfactor::Dict{Tuple{Int,Int},Matrix{ComplexF64}},
)
    for col_idx in keys(R_u)
        nodeS = col_idx[2] # Pull out the source leaf node ID directly
        if haskey(rightfactor, nodeS)
            rightfactor[nodeS] = R_u[col_idx] * rightfactor[nodeS]
        end
    end
    return rightfactor
end
