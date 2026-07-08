function (M::PetrovGalerkinBF{T})(obs_id::Int, src_id::Int) where {T}
    # =========================================================================
    # 1. DIRECT LOOKUPS
    # =========================================================================
    bf_idx = M.far_lookup[obs_id, src_id]
    if bf_idx != 0
        return FarBlockView{T}(obs_id, src_id, size(M.BFs[bf_idx]), M.BFs[bf_idx])
    end

    near_idx = M.near_lookup[obs_id, src_id]
    if near_idx != 0
        mat = M.nearinteractions.blocks[near_idx]
        return NearBlockView{T}(obs_id, src_id, size(mat), mat)
    end

    # =========================================================================
    # 2. (ASYMMETRIC) UPWARD TRAVERSAL (Find Level-Asymmetric Butterflies)
    # =========================================================================
    # Collect the entire ancestor lineage including the node itself
    obs_ancestors = Int[obs_id]
    curr_o = obs_id
    while curr_o > 1
        curr_o = H2Trees.parent(M.tree.testcluster, curr_o)
        push!(obs_ancestors, curr_o)
    end

    src_ancestors = Int[src_id]
    curr_s = src_id
    while curr_s > 1
        curr_s = H2Trees.parent(M.tree.trialcluster, curr_s)
        push!(src_ancestors, curr_s)
    end

    lvldif = length(obs_ancestors) - length(src_ancestors)
    #in case butterflys are only built for lvl-symmetric interactions we can reduce the look
    #up for Butterflyies overshadowing the queury pair in the asymmetric part of the tree.
    #This is a common scenario when the tree is very deep and we want to avoid building many
    #butterflys for the upper levels where the ranks are typically higher.
    if lvldif > 0
        # Observer is deeper, so we need to move up the observer tree
        obs_ancestors = obs_ancestors[(lvldif + 1):end]
    elseif lvldif < 0
        # Source is deeper, so we need to move up the source tree
        src_ancestors = src_ancestors[(-lvldif + 1):end]
    end
    for i in eachindex(obs_ancestors)
        o_anc = obs_ancestors[i]
        s_anc = src_ancestors[i]

        bf_idx = M.far_lookup[o_anc, s_anc]
        if bf_idx != 0
            return FarBlockView{T}(obs_id, src_id, size(M.BFs[bf_idx]), M.BFs[bf_idx])
        end
    end
    #=
    #Asymetric butterflies --> don't adjust the ancestor lists when activating this option
    # Check the grid of all combinations of ancestors. this should be fast unless the tree
    # is very deep, and in practice we expect to find a match within the first few levels
    # up. Skip (1,1) which corresponds to checking the exact query pair again.
    for p_obs in obs_ancestors
        for p_src in src_ancestors
            (p_obs == obs_id && p_src == src_id) && continue

            ancestor_bf_idx = M.far_lookup[p_obs, p_src]
            if ancestor_bf_idx != 0
                return FarBlockView{T}(p_obs, p_src, size(M.BFs[ancestor_bf_idx]), M.BFs[ancestor_bf_idx])
            end
        end
    end
    =#

    # =========================================================================
    # 3. DOWNWARD TRAVERSAL / COMPOSITE
    # =========================================================================
    # Given that butterflys are only built with respect to source and observer clusters at
    # the same lvl in the tree hirachy we now bring the node thats higher up in the tree
    # down to the same lvl as the other node and check for butterflys at each step down. if
    # we find a butterfly we can stop dividing that node further with respect to the other
    # node, since the butterfly already captures the interaction between those subtrees. in
    # any other case we need to go all the way down to the leaves to find the near
    # interactions.
    cur_src = 1
    cur_obs = 1
    subotree = h2treelevels(M.tree.testcluster, obs_id)
    substree = h2treelevels(M.tree.trialcluster, src_id)
    if lvldif > 0
        cur_src = 1 + lvldif
    elseif lvldif < 0
        cur_obs = 1 - lvldif
    end
    Bfs = Vector{BF}()
    far_rows = Vector{Int}()
    far_cols = Vector{Int}()
    far_vals = Vector{Int}()
    nearblocks = Vector{Matrix{T}}()
    near_rows = Vector{Int}()
    near_cols = Vector{Int}()
    near_vals = Vector{Int}()
    for _ in 1:min(length(subotree), length(substree))
        for o_node in subotree[cur_obs]
            for s_node in substree[cur_src]
                bf_idx = M.far_lookup[o_node, s_node]
                if bf_idx != 0
                    push!(Bfs, M.BFs[bf_idx])
                    push!(far_rows, o_node)
                    push!(far_cols, s_node)
                    push!(far_vals, length(Bfs))
                    continue
                end

                near_idx = M.near_lookup[o_node, s_node]
                if near_idx != 0
                    mat = M.nearinteractions.blocks[near_idx]
                    push!(nearblocks, mat)
                    push!(near_rows, o_node)
                    push!(near_cols, s_node)
                    push!(near_vals, length(nearblocks))
                end
            end
        end
        cur_obs += 1
        cur_src += 1
    end
    return CompositeBlockView{T}(
        nearblocks,
        (
            length(H2Trees.values(M.tree.testcluster, obs_id)),
            length(H2Trees.values(M.tree.trialcluster, src_id)),
        ),
        Bfs,
        sparse(near_rows, near_cols, near_vals, size(M.near_lookup)...),
        sparse(far_rows, far_cols, far_vals, size(M.far_lookup)...),
    )
end
