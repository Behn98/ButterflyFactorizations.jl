function splitmulbf(
    bfcluster_init::AbstractMatrix{ButterflyFactorization},
    higherkBF_init::ButterflyFactorization{T,M},
    τ::Float64,
) where {T,M}
    bfcluster = deepcopy(bfcluster_init)
    higherkBF = deepcopy(higherkBF_init)
    # Step 1: Split the higher-level BF into child-level BFs
    srcchildren = [getnsno(childbf)[1] for childbf in bfcluster[1, :]]
    bfdiagonal = splitobsside(higherkBF; obschildren=srcchildren)
    @assert size(bfcluster, 1) == size(bfcluster, 2) "bfcluster must be square --> no unbalanced interactions allowed"
    N = size(bfcluster, 1)
    # Step 2: Multiply each child BF with the corresponding cluster BF
    intermediate_bfs = Vector{ButterflyFactorization{T,M}}(undef, N)
    for i in 1:N
        intermediate_bfs[i] = ButterflyFactorization(
            Vector{ButterflyBlock{T}}(undef, 0),
            [
                ButterflyLevel{T}(Vector{ButterflyBlock{T}}(undef, 0)) for
                _ in 1:length(bfcluster[1, 1].R)
            ],
            #Vector{ButterflyLevel{T}}(undef, length(bfcluster[1, 1].R)),
            Vector{ButterflyBlock{T}}(undef, 0),
            cluster_blktree(
                cluster_testtree(bfcluster[1, 1].tree), cluster_trialtree(higherkBF.tree)
            ),
            higherkBF.k,
            τ,
        )
    end

    for i in 1:N
        for (j, child_bf) in enumerate(bfdiagonal)
            tempbf = partialcleanupidxs(mulBFs(bfcluster[i, j], child_bf, τ))
            obsancestor = getnsno(bfcluster[i, j])[2]
            refobs = getnsno(bfcluster[j, j])[2]
            currenttree = treelevels(cluster_testtree(tempbf.tree), obsancestor)
            reftree = treelevels(cluster_testtree(bfcluster[j, j].tree), refobs)
            treemapping = build_tree_mapping(currenttree, reftree)
            tempbf = remap_BFobservers(tempbf, treemapping)
            intermediate_bfs[mod(j - i, N) + 1] = merge_bfs(
                intermediate_bfs[mod(j - i, N) + 1], tempbf
            )
        end
    end
    product = ButterflyFactorization(
        higherkBF.Q,
        vcat(higherkBF.R[1:Int(log2(size(bfcluster, 2)))], intermediate_bfs[1].R),
        intermediate_bfs[1].P,
        intermediate_bfs[1].tree,
        higherkBF.k,
        τ,
    )
    for i in 2:N
        product += ButterflyFactorization(
            higherkBF.Q,
            vcat(higherkBF.R[1:Int(log2(size(bfcluster, 2)))], intermediate_bfs[i].R),
            intermediate_bfs[i].P,
            intermediate_bfs[i].tree,
            higherkBF.k,
            τ,
        )
    end

    return product
end

function splitobsside(
    BF_init::ButterflyFactorization{T,M};
    obschildren=sort!(cluster_children(cluster_testtree(BF.tree), getnsno(BF)[2])),
) where {T,M}
    BF = deepcopy(BF_init)
    depth=Int(log2(length(obschildren)))
    L = length(BF.R)
    bfdiagonal = Vector{ButterflyFactorization{T,M}}(undef, length(obschildren))
    for (i, ochild) in enumerate(obschildren)
        newR = Vector{ButterflyLevel{T}}(undef, L-depth)
        idx = 1
        newrlength = Int(length(BF.R[1].blocks)/length(obschildren))
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
        newR[1] = ButterflyLevel{T}(newRlvl)
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
            newR[l - 1] = ButterflyLevel{T}(newRlvl)
        end
        newP = Vector{ButterflyBlock{T}}(undef, Int(length(BF.P)/length(obschildren)))
        idx = 1
        for b in BF.P
            if haskey(rowmemory, getcolidx(b))
                newP[idx] = b
                idx += 1
            end
        end
        bfdiagonal[i] = ButterflyFactorization(
            Vector{ButterflyBlock{T}}(), newR, newP, BF.tree, BF.k, BF.τ
        )
    end

    return bfdiagonal
end

function merge_bfs(
    BF1::ButterflyFactorization{T,M}, BF2::ButterflyFactorization{T,M}
) where {T,M}
    newQ = vcat(BF1.Q, BF2.Q)
    newR = [
        ButterflyLevel{T}(vcat(BF1.R[l].blocks, BF2.R[l].blocks)) for l in 1:length(BF2.R)
    ]
    newP = vcat(BF1.P, BF2.P)
    return ButterflyFactorization(newQ, newR, newP, BF1.tree, BF1.k, BF1.τ)
end

function rename_P(BF::ButterflyFactorization{T,M}, srcancestor) where {T,M}
    newP = Vector{ButterflyBlock{T}}(undef, length(BF.P))
    for (i, b) in enumerate(BF.P)
        newP[i] = ButterflyBlock(b.obs_out, srcancestor, b.obs_in, srcancestor, b.data)
    end
    newR = [butterflylvl for butterflylvl in BF.R[1:(length(BF.R) - 1)]]
    push!(
        newR,
        ButterflyLevel{T}([
            ButterflyBlock(b.obs_out, srcancestor, b.obs_in, b.src_in, b.data) for
            b in BF.R[end].blocks
        ]),
    )
    return ButterflyFactorization(BF.Q, newR, newP, BF.tree, BF.k, BF.τ)
end

function partialcleanupidxs(BF::ButterflyFactorization{T,M}) where {T,M}
    return partialcleanupidxs(BF.Q, BF.R, BF.P, BF.tree, BF.k, BF.τ)
end

function partialcleanupidxs(
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

    for b in R[1].blocks
        if !haskey(src_translation, (b.obs_in, b.src_in))
            src_translation[(b.obs_in, b.src_in)] = b.src_in
        end
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
    for l in L:-1:2
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

    new_blocks = Vector{ButterflyBlock{T}}(undef, length(R[1].blocks))

    for (i, b) in enumerate(R[1].blocks)
        new_obs_out = obs_translation[(b.obs_out, b.src_out)]
        new_src_out, new_src_in = src_keys_R[1][i]

        new_blocks[i] = ButterflyBlock(
            new_obs_out, new_src_out, b.obs_in, b.src_in, copy(b.data)
        )
    end
    newR[1] = ButterflyLevel(new_blocks)

    newQ = Vector{ButterflyBlock{T}}(undef, length(Q))

    return ButterflyFactorization(newQ, newR, newP, tree, k, τ)
end

function build_tree_mapping(currenttree, reftree)
    mapping = Dict{Int,Int}()
    for (j, treelvl) in enumerate(currenttree)
        for (i, node) in enumerate(treelvl)
            if i <= length(reftree[j])
                mapping[node] = reftree[j][i]
            else
                mapping[node] = node
            end
        end
    end
    return mapping
end

function remap_BFobservers(tempbf::ButterflyFactorization{T,M}, treemapping) where {T,M}
    newQ = Vector{ButterflyBlock{T}}(undef, 0)
    newR = Vector{ButterflyLevel{T}}(undef, length(tempbf.R))
    for (l, lvl) in enumerate(tempbf.R[2:end])
        new_blocks = Vector{ButterflyBlock{T}}(undef, length(lvl.blocks))
        for (i, b) in enumerate(lvl.blocks)
            new_obs_out = treemapping[b.obs_out]
            new_obs_in = treemapping[b.obs_in]
            new_blocks[i] = ButterflyBlock(
                new_obs_out, b.src_out, new_obs_in, b.src_in, copy(b.data)
            )
        end
        newR[l + 1] = ButterflyLevel(new_blocks)
    end
    new_blocks = Vector{ButterflyBlock{T}}(undef, length(tempbf.R[1].blocks))
    for (i, b) in enumerate(tempbf.R[1].blocks)
        new_obs_out = treemapping[b.obs_out]
        new_blocks[i] = ButterflyBlock(
            new_obs_out, b.src_out, b.obs_in, b.src_in, copy(b.data)
        )
    end
    newR[1] = ButterflyLevel(new_blocks)
    newP = Vector{ButterflyBlock{T}}(undef, length(tempbf.P))
    for (i, b) in enumerate(tempbf.P)
        new_obs_out = treemapping[b.obs_out]
        new_obs_in = treemapping[b.obs_in]
        newP[i] = ButterflyBlock(new_obs_out, b.src_out, new_obs_in, b.src_in, copy(b.data))
    end

    return ButterflyFactorization(newQ, newR, newP, tempbf.tree, tempbf.k, tempbf.τ)
end

function algebraic_indexing(BF::ButterflyFactorization{T,M}) where {T,M} #, src_ancestor::Int
    L = length(BF.R)

    # 1. Q is kept completely stable.
    newQ = [
        ButterflyBlock(b.obs_out, b.src_out, b.obs_in, b.src_in, copy(b.data)) for b in BF.Q
    ]

    # Initialize the column translation from Q's output (which feeds into R[1])
    col_translation = Dict{Tuple{Int,Int},Tuple{Int,Int}}()
    for b in BF.Q
        col_translation[(b.obs_out, b.src_out)] = (b.obs_out, b.src_out)
    end

    newR = Vector{ButterflyLevel{T}}(undef, L)

    # 2. Traverse R[1] to R[L-1] assigning pure algebraic topological indices
    for l in 1:(L - 1)
        row_translation = Dict{Tuple{Int,Int},Tuple{Int,Int}}()

        # Find all unique old row spaces to assign them algebraic IDs
        unique_old_rows = unique([(b.obs_out, b.src_out) for b in BF.R[l].blocks])
        # Sort them to guarantee deterministic numbering across different BFs!
        sort!(unique_old_rows)#;by=x->(x[2], x[1])

        # Assign new algebraic row indices: (level, running_counter)
        for (idx, old_row) in enumerate(unique_old_rows)
            row_translation[old_row] = (l, idx)
        end

        new_blocks = Vector{ButterflyBlock{T}}(undef, length(BF.R[l].blocks))
        for (i, b) in enumerate(BF.R[l].blocks)
            # Translate column space from previous level
            new_in = col_translation[(b.obs_in, b.src_in)]
            # Translate row space to new algebraic ID
            new_out = row_translation[(b.obs_out, b.src_out)]

            new_blocks[i] = ButterflyBlock(
                new_out[1], new_out[2], new_in[1], new_in[2], copy(b.data)
            )
        end
        newR[l] = ButterflyLevel(new_blocks)

        # The row space of level l is the column space of level l+1
        col_translation = row_translation
    end

    # 3. Traverse R[L] (Last R factor)
    # Apply translation to colspace. Change src_out to src_ancestor. obs_out stays stable.
    new_blocks_L = Vector{ButterflyBlock{T}}(undef, length(BF.R[L].blocks))
    for (i, b) in enumerate(BF.R[L].blocks)
        new_in = col_translation[(b.obs_in, b.src_in)]
        # Keep physical obs_out to connect to P
        new_out = (b.obs_out, b.src_out) # (b.obs_out, src_ancestor)

        new_blocks_L[i] = ButterflyBlock(
            new_out[1], new_out[2], new_in[1], new_in[2], copy(b.data)
        )
    end
    newR[L] = ButterflyLevel(new_blocks_L)

    # 4. P is kept stable
    newP = copy(BF.P)

    return ButterflyFactorization(newQ, newR, newP, BF.tree, BF.k, BF.τ)
end
