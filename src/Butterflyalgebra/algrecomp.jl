"""
    recompress_BF(Butterfly ::ButterflyFactorization, τ)

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
function recompress_BF(Butterfly::ButterflyFactorization, τ)
    return recompress_BF_left(recompress_BF_right(Butterfly, τ), τ)
end

function recompress_BF_left(Butterfly::ButterflyFactorization, τ)
    return recompress_BF_right(Butterfly', τ)'
end
# Helper functions to extract domains (columns) and codomains (rows)
_col_key(b::ButterflyBlock) = (b.obs_in, b.src_in)
_row_key(b::ButterflyBlock) = (b.obs_out, b.src_out)

function recompress_BF_right(
    Butterfly_init::ButterflyFactorization{T,M},
    τ::Float64;
    scheduler=OhMyThreads.SerialScheduler(),
) where {T,M}

    # 1. Deepcopy the R factors so we can safely mutate them
    R_factors = [
        ButterflyLevel([
            ButterflyBlock(b.obs_out, b.src_out, b.obs_in, b.src_in, copy(b.data)) for
            b in lvl.blocks
        ]) for lvl in Butterfly_init.R
    ]

    lr = length(R_factors)

    for l in 1:(lr - 1)
        lold = lr - l + 1
        current_blocks = R_factors[lold].blocks

        # 2. Sort to group by column space
        sort!(current_blocks; by=_col_key)

        # 3. Identify chunk boundaries sequentially (very fast O(N) pass)
        chunks = Tuple{Int,Int,Tuple{Int,Int}}[]
        n_blocks = length(current_blocks)
        i = 1
        while i <= n_blocks
            start_idx = i
            c_key = _col_key(current_blocks[i])
            while i <= n_blocks && _col_key(current_blocks[i]) == c_key
                i += 1
            end
            push!(chunks, (start_idx, i - 1, c_key))
        end

        # 4. Process chunks in parallel using OhMyThreads
        chunk_results = tmap(chunks; scheduler=scheduler) do (start_idx, end_idx, c_key)
            # Extract matrices
            blocks_to_compress = [current_blocks[k].data for k in start_idx:end_idx]
            row_sizes = [size(mat, 1) for mat in blocks_to_compress]

            # Concatenate and QR
            A_k = vcat(blocks_to_compress...)
            Q_mat, R_mat, p = pqr(A_k; rtol=τ)

            # Local transfer matrix
            T_mat = R_mat[:, invperm(p)]

            # Construct updated blocks
            new_blocks = Vector{ButterflyBlock{T}}(undef, end_idx - start_idx + 1)
            last_row = 0
            for (j, k) in enumerate(start_idx:end_idx)
                slice_rows = row_sizes[j]
                old_b = current_blocks[k]
                new_data = Matrix(Q_mat[(last_row + 1):(last_row + slice_rows), :])

                new_blocks[j] = ButterflyBlock(
                    old_b.obs_out, old_b.src_out, old_b.obs_in, old_b.src_in, new_data
                )
                last_row += slice_rows
            end

            return (c_key, T_mat, start_idx, new_blocks)
        end

        # 5. Gather results sequentially
        R_u = Dict{Tuple{Int,Int},Matrix{T}}()
        for (c_key, T_mat, start_idx, new_blocks) in chunk_results
            R_u[c_key] = T_mat
            for (j, b) in enumerate(new_blocks)
                current_blocks[start_idx + j - 1] = b
            end
        end

        # 6. Apply R_u to the next level (lold - 1) in parallel
        # We can map over the next level's blocks independently!
        next_blocks = R_factors[lold - 1].blocks

        tmap!(next_blocks, next_blocks; scheduler=scheduler) do block
            r_key = _row_key(block)
            if haskey(R_u, r_key)
                # Apply transfer matrix update
                return ButterflyBlock(
                    block.obs_out,
                    block.src_out,
                    block.obs_in,
                    block.src_in,
                    R_u[r_key] * block.data,
                )
            else
                return block
            end
        end
    end

    # Return new factorisation
    return ButterflyFactorization{T,M}(
        Butterfly_init.Q,
        R_factors,
        Butterfly_init.P,
        Butterfly_init.tree,
        Butterfly_init.k,
        Butterfly_init.τ,
    )
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
