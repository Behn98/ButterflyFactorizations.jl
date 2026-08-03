function splitmulbf(
    bfcluster::AbstractMatrix{ButterflyFactorization{T,M}},
    higherkBF::ButterflyFactorization{T,M};
    τ=higherkBF.τ,
) where {T,M}

    # Step 1: Split the higher-level BF into child-level BFs
    srcchildren = [getnsno(childbf)[1] for childbf in bfcluster[1, :]]
    bfdiagonal = splitobsside(higherkBF; obschildren=srcchildren)
    @assert size(bfcluster, 1) == size(bfcluster, 2) "bfcluster must be square --> no unbalanced interactions allowed"
    N = size(bfcluster, 1)
    # Step 2: Multiply each child BF with the corresponding cluster BF
    intermediate_bfs = Vector{ButterflyFactorization{T,M}}(undef, N)
    for i in 1:N
        intermediate_bfs[i] = ButterflyFactorization{T,M}(
            Vector{ButterflyBlock{T}}(),
            Vector{ButterflyLevel{T}}(),
            Vector{ButterflyBlock{T}}(),
            cluster_blktree(
                cluster_testtree(bfcluster[1, 1].tree), cluster_trialtree(higherkBF.tree)
            ),
            higherkBF.k,
            τ,
        )
    end
    for i in 1:N
        for (j, child_bf) in enumerate(bfdiagonal)
            intermediate_bfs[mod(j - i, N) + 1] = merge_bfs(
                intermediate_bfs[mod(j - i, N) + 1],
                mulBFs(bfcluster[i, j], child_bf; τ=higherkBF.τ),
            )
        end
    end
    product = cleanupidxs(
        higherkBF.Q,
        [higherkBF.R[1:log2(size(bfcluster, 2))], intermediate_bfs[1].R...],
        intermediate_bfs[1].P,
        higherkBF.tree,
        higherkBF.k,
        τ,
    )
    for i in 2:N
        product += cleanupidxs(
            higherkBF.Q,
            [higherkBF.R[1:log2(size(bfcluster, 2))], intermediate_bfs[i].R...],
            intermediate_bfs[i].P,
            higherkBF.tree,
            higherkBF.k,
            τ,
        )
    end

    return merged_bf
end

function splitobsside(
    BF_init::ButterflyFactorization{T,M};
    obschildren=sort!(cluster_children(cluster_testtree(BF.tree), getNSNO(BF)[2])),
) where {T,M}
    BF = deepcopy(BF_init)
    depth=log2(length(obschildren))
    L = length(BF.R)
    bfdiagonal = Vector{ButterflyFactorization{T,M}}(undef, length(obschildren))
    for (i, ochild) in enumerate(obschildren)
        newR = Vector{ButterflyLevel}(undef, L-depth)
        idx = 1
        newrlength = length(BF.R[2].blocks)/length(obschildren)
        newRlvl = Vector{ButterflyBlock{T}}(undef, newrlength)
        rowmemory = Dict{Tuple{Int,Int},Bool}()
        for b in BF.R[depth + 1].blocks
            if b.obs_in == ochild
                newRlvl[idx] = b
                idx += 1
                if !haskey(rowmemory, getrowidx(b))
                    rowmemory[getrowidx(b)] = true
                end
            end
        end
        newR[1] = ButterflyLevel(newRlvl)
        for l in (depth + 2):L
            idx = 1
            newRlvl = Vector{ButterflyBlock{T}}(undef, newrlength)
            newrowmemory = Dict{Tuple{Int,Int},Bool}()
            for b in BF.R[l].blocks
                if haskey(rowmemory, getcolidx(b))
                    newRlvl[idx] = b
                    idx += 1
                    newrowmemory[getrowidx(b)] = true
                end
            end
            rowmemory = newrowmemory
            newR[l - 1] = ButterflyLevel(newRlvl)
        end
        newP = Vector{ButterflyBlock{T}}(undef, length(BF.P)/length(obschildren))
        idx = 1
        for b in BF.P
            if haskey(rowmemory, getcolidx(b))
                newP[idx] = b
                idx += 1
            end
        end
        bfdiagonal[i] = ButterflyFactorization(
            Vector{ButterflyBlock{T}}(), newR, newP, BF.tree, BF.k, τ
        )
    end

    return bfdiagonal
end

function merge_bfs(
    BF1::ButterflyFactorization{T,M}, BF2::ButterflyFactorization{T,M}
) where {T,M}
    newQ = vcat(BF1.Q, BF2.Q)
    newR = vcat(BF1.R, BF2.R)
    newP = vcat(BF1.P, BF2.P)
    return ButterflyFactorization(newQ, newR, newP, BF1.tree, BF1.k, BF1.τ)
end
