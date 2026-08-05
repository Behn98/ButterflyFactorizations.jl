# ------------------------------------------------------------------
# ButterflyFactorization Algebraic Multiplication
# ------------------------------------------------------------------

"""
    mulBFs(BF_1::ButterflyFactorization, BF_2::ButterflyFactorization, τ::Float64) -> BF

Compute the algebraic operator product of two Butterfly Factorizations (`BF_1 * BF_2`)
and compress the resulting representation to a specified accuracy tolerance.

This algorithm pushes a "messenger" matrix through the hierarchical layers using
adjacent row-swaps and graph-based block splitting, allowing the differing tree
structures to be merged algebraically without ever inflating to a dense matrix.
"""
function mulBFs(
    BF_1_init::ButterflyFactorization{T,M},
    BF_2_init::ButterflyFactorization{T,M},
    τ::Float64,
) where {T,M}
    @assert length(BF_1_init.R) == length(BF_2_init.R) "Both BFs must have the same number of levels"
    BF_1 = deepcopy(BF_1_init)
    BF_2 = deepcopy(BF_2_init)
    L = length(BF_1.R)

    # 1. Messenger Initialization
    # M_leaf = Q_1 * P_2
    M_leaf = multiply_levels(BF_1.Q, BF_2.P)

    # M_mid = R_1[1] * M_leaf * R_2[L]
    M_mid = multiply_levels(BF_1.R[1], multiply_levels(M_leaf, BF_2.R[L]))

    # Assemble the active levels flat array:
    # [BF_1.R[L:-1:2], M_mid, BF_2.R[L-1:-1:1]]
    active_levels = Vector{ButterflyLevel{T}}()
    append!(active_levels, BF_1.R[L:-1:2])
    push!(active_levels, M_mid)
    append!(active_levels, BF_2.R[(L - 1):-1:1])

    # 3. Iterative Row Swapping and Trimming
    for m in 1:(L - 1)
        # Push the messenger level through via row-swaps
        for t in 1:m
            idx = L + 1 - t
            active_levels = swap_adjacent!(active_levels, idx)
        end

        # Multiply the aligned levels
        idx_mul = L - m
        active_levels = multiply_adjacent!(active_levels, idx_mul)

        # 4. Trimming / Recompression
        temp_BF = ButterflyFactorization(
            BF_2.Q, copy(reverse(active_levels)), BF_1.P, BF_1.tree, BF_1.k, τ
        )
        temp_BF = recompress_BF(temp_BF, τ)
        active_levels = reverse(temp_BF.R)
    end

    return ButterflyFactorization(
        BF_2.Q,
        reverse(active_levels),
        BF_1.P,
        cluster_blktree(cluster_testtree(BF_1.tree), cluster_trialtree(BF_2.tree)),
        BF_1.k,
        τ,
    )
end

# --- Array Manipulators ---

"""
    multiply_adjacent!(levels, idx)

Multiplies two adjacent hierarchical levels in the flat array and splices the
merged result back into the vector.
"""
function multiply_adjacent!(levels::Vector{ButterflyLevel{T}}, idx::Int) where {T}
    merged = multiply_levels(levels[idx], levels[idx + 1])
    splice!(levels, idx:(idx + 1), [merged])
    return levels
end

"""
    swap_adjacent!(levels, idx)

Algebraically pushes a messenger matrix through a level by extracting connected
components and splitting the blocks.
"""
function swap_adjacent!(levels::Vector{ButterflyLevel{T}}, idx::Int) where {T}
    merged = multiply_levels(levels[idx], levels[idx + 1])
    L_new, R_new = browswap_split(merged)
    splice!(levels, idx:(idx + 1), [L_new, R_new])
    return levels
end

# --- Core Matrix Operations ---

"""
    multiply_levels(A, B)

Computes the block-wise algebraic product of two Butterfly layers.
Properly aligns the column-space input domain of the left factor with the
row-space output codomain of the right factor.
"""
function multiply_levels(
    Q::Vector{<:ButterflyBlock{T}}, P::Vector{<:ButterflyBlock{T}}
) where {T}
    P_grouped = Dict{Int,ButterflyBlock{T}}()
    for b in P
        r_key = b.obs_out
        P_grouped[r_key] = b
    end

    Prod_dict = Dict{Tuple{Tuple{Int,Int},Tuple{Int,Int}},Matrix{T}}()

    for a in Q
        a_col_key = (a.obs_in, a.src_in)
        if haskey(P_grouped, a_col_key[2])
            b = P_grouped[a_col_key[2]]
            product_key = ((a.obs_out, a.src_out), (b.obs_in, b.src_in))
            Prod_dict[product_key] = a.data * b.data
        end
    end

    return ButterflyLevel([
        ButterflyBlock(row[1], row[2], col[1], col[2], data) for
        ((row, col), data) in Prod_dict
    ])
end

function multiply_levels(A::ButterflyLevel, B::Vector{<:ButterflyBlock})
    return multiply_levels(A, ButterflyLevel(B))
end

function multiply_levels(A::Vector{<:ButterflyBlock}, B::ButterflyLevel)
    return multiply_levels(ButterflyLevel(A), B)
end

function multiply_levels(A::ButterflyLevel{T}, B::ButterflyLevel{T}) where {T}
    B_grouped = Dict{Tuple{Int,Int},Vector{ButterflyBlock{T}}}()
    for b in B.blocks
        r_key = (b.obs_out, b.src_out)
        if !haskey(B_grouped, r_key)
            B_grouped[r_key] = ButterflyBlock{T}[]
        end
        push!(B_grouped[r_key], b)
    end

    P_dict = Dict{Tuple{Tuple{Int,Int},Tuple{Int,Int}},Matrix{T}}()

    for a in A.blocks
        a_col_key = (a.obs_in, a.src_in)
        if haskey(B_grouped, a_col_key)
            for b in B_grouped[a_col_key]
                product_key = ((a.obs_out, a.src_out), (b.obs_in, b.src_in))
                mat_prod = a.data * b.data
                if haskey(P_dict, product_key)
                    P_dict[product_key] .+= mat_prod
                else
                    P_dict[product_key] = mat_prod
                end
            end
        end
    end

    return ButterflyLevel([
        ButterflyBlock(row[1], row[2], col[1], col[2], data) for
        ((row, col), data) in P_dict
    ])
end

"""
    browswap_split(P::ButterflyLevel)

Resolves structural bottlenecks by grouping blocks into a flood-fill adjacency graph
and slicing them to allow messenger matrices to pass through the hierarchy.
"""
function browswap_split(P::ButterflyLevel{T}) where {T}
    n = length(P.blocks)

    # 1. Build Adjacency Graph (Flood Fill)
    row_to_blocks = Dict{Tuple{Int,Int},Vector{Int}}()
    col_to_blocks = Dict{Tuple{Int,Int},Vector{Int}}()

    for (i, b) in enumerate(P.blocks)
        rk = (b.obs_out, b.src_out)
        ck = (b.obs_in, b.src_in)
        push!(get!(row_to_blocks, rk, Int[]), i)
        push!(get!(col_to_blocks, ck, Int[]), i)
    end

    B_blocks = ButterflyBlock{T}[]
    C_blocks = ButterflyBlock{T}[]
    visited = falses(n)

    # 2. Extract Connected Components
    colcount1 = 1
    colcount2 = 1
    for i in 1:n
        if !visited[i]
            cluster = ButterflyBlock{T}[]
            queue = Int[]
            rblks = row_to_blocks[(P.blocks[i].obs_out, P.blocks[i].src_out)]
            cblks = col_to_blocks[(P.blocks[i].obs_in, P.blocks[i].src_in)]
            cols = sort!([(blk.obs_in, blk.src_in) for blk in P.blocks[rblks]])
            rows = sort!([(blk.obs_out, blk.src_out) for blk in P.blocks[cblks]])

            for row in rows
                append!(queue, row_to_blocks[row])
            end

            while !isempty(queue)
                curr = popfirst!(queue)
                push!(cluster, P.blocks[curr])
                visited[curr] = true
            end

            for (idx, row) in enumerate(rows)
                currentblks = cluster[(idx * 4 - 3):(idx * 4)]
                sort!(currentblks; by=x -> (x.obs_in, x.src_in))

                offset = 1
                for k in 1:2:3
                    ckblkcolkey = (currentblks[k].obs_in, currentblks[k].src_in)
                    ck_1blkcolkey = (currentblks[k + 1].obs_in, currentblks[k + 1].src_in)

                    newbblock = hcat(currentblks[k].data, currentblks[k + 1].data)
                    ksizes = (
                        size(currentblks[k].data, 2), size(currentblks[k + 1].data, 2)
                    )

                    ckblk = vcat(
                        Matrix{T}(I, ksizes[1], ksizes[1]), zeros(T, ksizes[2], ksizes[1])
                    )
                    ck_1blk = vcat(
                        zeros(T, ksizes[1], ksizes[2]), Matrix{T}(I, ksizes[2], ksizes[2])
                    )

                    # 1. B Blocks depend on the row
                    if idx < 3
                        push!(
                            B_blocks,
                            ButterflyBlock(
                                row[1], row[2], 1, colcount1 + offset, newbblock
                            ),
                        )
                    else
                        push!(
                            B_blocks,
                            ButterflyBlock(
                                row[1], row[2], 2, colcount2 + offset, newbblock
                            ),
                        )
                    end

                    # 2. C Blocks only route columns
                    if idx == 1
                        push!(
                            C_blocks,
                            ButterflyBlock(
                                1, colcount1 + offset, ckblkcolkey[1], ckblkcolkey[2], ckblk
                            ),
                        )
                        push!(
                            C_blocks,
                            ButterflyBlock(
                                1,
                                colcount1 + offset,
                                ck_1blkcolkey[1],
                                ck_1blkcolkey[2],
                                ck_1blk,
                            ),
                        )
                    elseif idx == 3
                        push!(
                            C_blocks,
                            ButterflyBlock(
                                2, colcount2 + offset, ckblkcolkey[1], ckblkcolkey[2], ckblk
                            ),
                        )
                        push!(
                            C_blocks,
                            ButterflyBlock(
                                2,
                                colcount2 + offset,
                                ck_1blkcolkey[1],
                                ck_1blkcolkey[2],
                                ck_1blk,
                            ),
                        )
                    end
                    offset += 1
                end
            end
            colcount1 += 2
            colcount2 += 2
        end
    end

    return ButterflyLevel(B_blocks), ButterflyLevel(C_blocks)
end

# --- Overloads ---

"""
    *(BF_1::ButterflyFactorization, BF_2::ButterflyFactorization) -> BF

Overloads the `*` operator to compute the algebraic product of two Butterfly operators.
Dynamically falls back to the maximum tolerance of the two inputs for recompression.
"""
function Base.:*(BF_1::ButterflyFactorization, BF_2::ButterflyFactorization)
    return cleanupidxs(mulBFs(BF_1, BF_2, max(BF_1.τ, BF_2.τ)))
end

"""
    cleanupidxs(BF::ButterflyFactorization)

Rebuilds the physical global indices across the hierarchical `R` factors after
an algebraic manipulation (like multiplication or recompression) destroys the
strict spatial locality mappings.
"""
function cleanupidxs(BF::ButterflyFactorization{T,M}) where {T,M}
    return cleanupidxs(BF.Q, BF.R, BF.P, BF.tree, BF.k, BF.τ)
end

function cleanupidxs(
    Q_init::Vector{ButterflyBlock{T}},
    R_init::Vector{ButterflyLevel{T}},
    P_init::Vector{ButterflyBlock{T}},
    tree::M,
    k,
    τ,
) where {T,M}
    Q = deepcopy(Q_init)
    R = deepcopy(R_init)
    P = deepcopy(P_init)
    tsttree = cluster_testtree(tree)
    trialtree = cluster_trialtree(tree)
    L = length(R)

    # =========================================================================
    # PASS 1: Forward Traversal (Trace Source / Trial Tree Keys from Q)
    # =========================================================================
    src_translation = Dict{Tuple{Int,Int},Int}()

    for b in Q
        src_translation[(b.obs_in, b.src_in)] = b.src_in
    end

    src_keys_R = [Vector{Tuple{Int,Int}}(undef, length(R[l].blocks)) for l in 1:L]

    for l in 1:L
        next_src_translation = Dict{Tuple{Int,Int},Int}()
        for (i, b) in enumerate(R[l].blocks)
            new_src_in = src_translation[(b.obs_in, b.src_in)]
            new_src_out = cluster_parent(trialtree, new_src_in)

            src_keys_R[l][i] = (new_src_out, new_src_in)
            next_src_translation[(b.obs_out, b.src_out)] = new_src_out
        end
        src_translation = next_src_translation
    end

    # =========================================================================
    # PASS 2: Backward Traversal (Trace Observer Keys & Reconstruct R)
    # =========================================================================
    obs_translation = Dict{Tuple{Int,Int},Int}()
    newP = Vector{ButterflyBlock{T}}(undef, length(P))

    for (i, b) in enumerate(P)
        true_obs_out = b.obs_out
        true_obs_in = b.obs_in
        root_src = src_translation[(b.obs_in, b.src_in)]

        newP[i] = ButterflyBlock(
            true_obs_out, root_src, true_obs_in, root_src, copy(b.data)
        )
        obs_translation[(b.obs_in, b.src_in)] = true_obs_in
    end

    newR = Vector{ButterflyLevel{T}}(undef, L)
    for l in L:-1:1
        next_obs_translation = Dict{Tuple{Int,Int},Int}()
        new_blocks = Vector{ButterflyBlock{T}}(undef, length(R[l].blocks))

        for (i, b) in enumerate(R[l].blocks)
            new_obs_out = obs_translation[(b.obs_out, b.src_out)]
            new_obs_in = cluster_parent(tsttree, new_obs_out)
            new_src_out, new_src_in = src_keys_R[l][i]

            new_blocks[i] = ButterflyBlock(
                new_obs_out, new_src_out, new_obs_in, new_src_in, copy(b.data)
            )
            next_obs_translation[(b.obs_in, b.src_in)] = new_obs_in
        end
        newR[l] = ButterflyLevel(new_blocks)
        obs_translation = next_obs_translation
    end

    newQ = Vector{ButterflyBlock{T}}(undef, length(Q))
    for (i, b) in enumerate(Q)
        true_obs_out = obs_translation[(b.obs_out, b.src_out)]
        true_obs_in = true_obs_out

        newQ[i] = ButterflyBlock(
            true_obs_out, b.src_out, true_obs_in, b.src_in, copy(b.data)
        )
    end

    return ButterflyFactorization(newQ, newR, newP, tree, k, τ)
end

"""
    trivialmul(BF_1, BF_2)

Computes the uncompressed structural product of two Butterfly Factorizations.
It cascades the factors without attempting intermediate recompression passes.
"""
function trivialmul(
    BF_1::ButterflyFactorization{T,M}, BF_2::ButterflyFactorization{T,M}
) where {T,M}
    @assert length(BF_1.R) == length(BF_2.R) "Both BFs must have the same number of levels"

    L = length(BF_1.R)
    M_leaf = multiply_levels(BF_1.Q, BF_2.P)
    M_mid = multiply_levels(BF_1.R[1], multiply_levels(M_leaf, BF_2.R[L]))

    active_levels = Vector{ButterflyLevel{T}}()
    append!(active_levels, BF_1.R[L:-1:2])
    push!(active_levels, M_mid)
    append!(active_levels, BF_2.R[(L - 1):-1:1])

    for m in 1:(L - 1)
        multiply_adjacent!(active_levels, L - m)
    end

    return ButterflyFactorization(
        BF_2.Q,
        reverse(active_levels),
        BF_1.P,
        cluster_blktree(cluster_testtree(BF_1.tree), cluster_trialtree(BF_2.tree)),
        BF_1.k,
        max(BF_1.τ, BF_2.τ),
    )
end
