import H2Trees: values, center, halfsize, children, isleaf, trialtree, testtree
function recompress_BF_left(Butterfly::AlgBF, τ, tree::H2Trees.BlockTree)
    return recompress_BF_right(
        Butterfly', τ, H2Trees.BlockTree(trialtree(tree), testtree(tree))
    )'
end

function recompress_BF(Butterfly::AlgBF, τ, tree::H2Trees.BlockTree)
    return recompress_BF_left(recompress_BF_right(Butterfly, τ, tree), τ, tree)
end

"""
    recompress_BF(Butterfly::BF, τ, tree::H2Trees.BlockTree)

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
function recompress_BF(Butterfly::BF, τ, tree::H2Trees.BlockTree)
    Q = Butterfly.Q
    R = Butterfly.R
    P = Butterfly.P
    BFalg = AlgBF(Butterfly.dim, Q, R, P)
    BFalg = recompress_BF(BFalg, τ, tree)
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

@views function recompress_BF_right(Butterfly::AlgBF, τ, tree::H2Trees.BlockTree)
    Q = Butterfly.Q
    R = Butterfly.R
    P = Butterfly.P
    lr = length(R)

    for l in eachindex(R)
        lold = lr - l + 1

        # Bulletproof: Flatten R_u to map the full col_idx tuple directly to its matrix
        R_u = Dict{Tuple{Int,Int},Matrix{ComplexF64}}()

        # 1. Map column skeletons to all associated row skeletons at this level
        col_to_rows = Dict{Tuple{Int,Int},Vector{Tuple{Int,Int}}}()
        for row_skel in keys(R[lold])
            for col_idx in keys(R[lold][row_skel])
                if !haskey(col_to_rows, col_idx)
                    col_to_rows[col_idx] = Vector{Tuple{Int,Int}}()
                end
                push!(col_to_rows[col_idx], row_skel)
            end
        end

        # 2. Process each unique column space
        for (col_idx, rows_with_col) in col_to_rows
            parent_groups = group_by_parents(tree, rows_with_col)

            for (parent_node, local_rows) in parent_groups
                R_k = Vector{Matrix{ComplexF64}}()
                row_spc = Vector{Int}()

                for row_skel in local_rows
                    block = R[lold][row_skel][col_idx]
                    push!(R_k, block)
                    push!(row_spc, size(block, 1))
                end

                A_k = vcat(R_k...)
                QRA = pqr(A_k; rtol=τ)

                # Store using the unique col_idx key — no overwrites possible
                R_u[col_idx] = QRA[2][:, invperm(QRA[3])]

                last_idx = 0
                for (j, row_skel) in enumerate(local_rows)
                    R[lold][row_skel][col_idx] = Matrix(
                        QRA[1][(last_idx + 1):(last_idx + row_spc[j]), :]
                    )
                    last_idx += row_spc[j]
                end
            end
        end

        # 3. Propagate the accumulated R_u transformations
        if l < lr
            R[lold - 1] = update_next_level_R_right(R_u, R[lold - 1])
        else
            Q = update_next_level_R_right(R_u, Q)
        end
    end

    return AlgBF(Butterfly.dim, Q, R, P)
end

# Overload 1: Updating intermediate R factors (Clean 1:1 matching)
@views function update_next_level_R_right(
    R_u::Dict{Tuple{Int,Int},Matrix{ComplexF64}},
    rightfactor::Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},AbstractMatrix{ComplexF64}}},
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
@views function update_next_level_R_right(
    R_u::Dict{Tuple{Int,Int},Matrix{ComplexF64}}, rightfactor::Dict{Int,Matrix{ComplexF64}}
)
    for col_idx in keys(R_u)
        nodeS = col_idx[2] # Pull out the source leaf node ID directly
        if haskey(rightfactor, nodeS)
            rightfactor[nodeS] = R_u[col_idx] * rightfactor[nodeS]
        end
    end
    return rightfactor
end
