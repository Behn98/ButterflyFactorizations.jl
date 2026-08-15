function splitBF(BF_1::ButterflyFactorization{T,M}, τ::Float64) where {T,M}
    L = length(BF_1.R)
    active_levels = Vector{ButterflyLevel{T}}()
    append!(active_levels, BF_1.R[L:-1:1])

    return ButterflyFactorization(
        BF_2.Q,
        reverse(active_levels),
        BF_1.P,
        cluster_blktree(cluster_testtree(BF_1.tree), cluster_trialtree(BF_2.tree)),
        BF_1.k,
        τ,
    )
end

function split_adjacent!(levels::Vector{ButterflyLevel{T}}, idx::Int) where {T}
    merged = multiply_levels(levels[idx], levels[idx + 1])
    L_new, R_new = bsymswap(merged)
    splice!(levels, idx:(idx + 1), [L_new, R_new])
    return levels
end

function bsymswap(P::ButterflyLevel{T}) where {T}
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
                    if !iseven(idx)
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
