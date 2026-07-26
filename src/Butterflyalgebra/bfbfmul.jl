"""
    mulBFs(BF_1::ButterflyFactorization, BF_2::ButterflyFactorization, τ::Float64) -> BF

Compute the operator product of two Butterfly Factorizations (`BF`) and compress the
resulting representation to a specified accuracy tolerance.
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
            active_levels = swap_adjacent!(active_levels, idx) # Tree no longer needed!
            println("swap done")
        end

        # Multiply the aligned levels
        idx_mul = L - m
        active_levels = multiply_adjacent!(active_levels, idx_mul)

        # 4. Trimming / Recompression
        temp_BF = ButterflyFactorization{T,M}(
            BF_2.Q, copy(reverse(active_levels)), BF_1.P, BF_1.tree, BF_1.k, τ
        )
        temp_BF = recompress_BF(temp_BF, τ)
        active_levels = reverse(temp_BF.R)
    end

    return ButterflyFactorization{T,M}(
        BF_2.Q,
        reverse(active_levels),
        BF_1.P,
        cluster_blktree(cluster_testtree(BF_1.tree), cluster_trialtree(BF_2.tree)),
        BF_1.k,
        τ,
    )
end

# --- Array Manipulators ---

function multiply_adjacent!(levels::Vector{ButterflyLevel{T}}, idx::Int) where {T}
    merged = multiply_levels(levels[idx], levels[idx + 1])
    splice!(levels, idx:(idx + 1), [merged])
    return levels
end

function swap_adjacent!(levels::Vector{ButterflyLevel{T}}, idx::Int) where {T}
    merged = multiply_levels(levels[idx], levels[idx + 1])
    L_new, R_new = browswap_split(merged)
    splice!(levels, idx:(idx + 1), [L_new, R_new])
    return levels
end

# --- Core Matrix Operations ---

# Overloads to allow native multiplication of Q and P vectors
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
    for i in 1:n
        if !visited[i]
            cluster = ButterflyBlock{T}[] #cluster is 4x4 block
            queue = Int[]
            rblks = row_to_blocks[(P.blocks[i].obs_out, P.blocks[i].src_out)]
            cblks = col_to_blocks[(P.blocks[i].obs_in, P.blocks[i].src_in)]
            cols = sort!([(blk.obs_in, blk.src_in) for blk in P.blocks[rblks]])
            rows = sort!([(blk.obs_out, blk.src_out) for blk in P.blocks[cblks]])
            if length(rows) != 4 || length(cols) != 4
                @show length(rows) length(cols)
            end
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
                i = 1
                for k in 1:2:3
                    ckblkcolkey = (currentblks[k].obs_in, currentblks[k].src_in)
                    ck_1blkcolkey = (currentblks[k + 1].obs_in, currentblks[k + 1].src_in)
                    #=if ckblkcolkey != cols[k]
                        println(
                            "Warning: Column key mismatch for ckblkcolkey: ",
                            ckblkcolkey,
                            " vs cols[",
                            k,
                            "]: ",
                            cols[k],
                        )
                    end
                    if ck_1blkcolkey != cols[k + 1]
                        println(
                            "Warning: Column key mismatch for ck_1blkcolkey: ",
                            ck_1blkcolkey,
                            " vs cols[",
                            k + 1,
                            "]: ",
                            cols[k + 1],
                        )
                    end=#
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
                    # 1. B Blocks depend on the row (Push for every idx)
                    if idx < 3
                        push!(
                            B_blocks,
                            ButterflyBlock(
                                row[1], row[2], cols[i][1], cols[i][2], newbblock
                            ),
                        )
                    else
                        push!(
                            B_blocks,
                            ButterflyBlock(
                                row[1], row[2], cols[i + 2][1], cols[i + 2][2], newbblock
                            ),
                        )
                    end

                    # 2. C Blocks only route columns (Push ONLY on the first row of each half)
                    if idx == 1
                        push!(
                            C_blocks,
                            ButterflyBlock(
                                cols[i][1],
                                cols[i][2],
                                ckblkcolkey[1],
                                ckblkcolkey[2],
                                ckblk,
                            ),
                        )
                        push!(
                            C_blocks,
                            ButterflyBlock(
                                cols[i][1],
                                cols[i][2],
                                ck_1blkcolkey[1],
                                ck_1blkcolkey[2],
                                ck_1blk,
                            ),
                        )
                    elseif idx == 3
                        push!(
                            C_blocks,
                            ButterflyBlock(
                                cols[i + 2][1],
                                cols[i + 2][2],
                                ckblkcolkey[1],
                                ckblkcolkey[2],
                                ckblk,
                            ),
                        )
                        push!(
                            C_blocks,
                            ButterflyBlock(
                                cols[i + 2][1],
                                cols[i + 2][2],
                                ck_1blkcolkey[1],
                                ck_1blkcolkey[2],
                                ck_1blk,
                            ),
                        )
                    end
                    i+=1
                end
            end
        end
    end
    return ButterflyLevel(B_blocks), ButterflyLevel(C_blocks)
end

# --- Overloads ---

function Base.:*(BF_1::ButterflyFactorization, BF_2::ButterflyFactorization)
    return mulBFs(BF_1, BF_2, max(BF_1.τ, BF_2.τ))
end

function LinearAlgebra.mul!(
    C::ButterflyFactorization, A::ButterflyFactorization, B::ButterflyFactorization
)
    # Replaces the internal fields of C with the newly computed product
    res = mulBFs(A, B, max(A.τ, B.τ))
    C.Q = res.Q
    C.R = res.R
    C.P = res.P
    C.τ = res.τ
    return C
end

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

    return ButterflyFactorization{T,M}(
        BF_2.Q,
        reverse(active_levels),
        BF_1.P,
        cluster_blktree(cluster_testtree(BF_1.tree), cluster_trialtree(BF_2.tree)),
        BF_1.k,
        max(BF_1.τ, BF_2.τ),
    )
end
