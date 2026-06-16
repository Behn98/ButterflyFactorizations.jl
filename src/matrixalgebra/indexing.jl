function (M::PetrovGalerkinBF{T})(obs_id::Int, src_id::Int) where {T}
    # =========================================================================
    # 1. DIRECT LOOKUPS
    # =========================================================================
    bf_idx = M.far_lookup[obs_id, src_id]
    if bf_idx != 0
        return FarBlockView{T}(obs_id, src_id, M.BFs[bf_idx])
    end

    near_idx = M.near_lookup[obs_id, src_id]
    if near_idx != 0
        mat = M.nearinteractions.blocks[near_idx]
        return NearBlockView{T}(obs_id, src_id, mat)
    end

    # =========================================================================
    # 2. ASYMMETRIC UPWARD TRAVERSAL (Find Level-Asymmetric Butterflies)
    # =========================================================================
    # Collect the entire ancestor lineage including the node itself
    obs_ancestors = Int[obs_id]
    curr_o = obs_id
    while curr_o > 1
        curr_o = H2trees.parent(M.tree.testcluster, curr_o)
        push!(obs_ancestors, curr_o)
    end

    src_ancestors = Int[src_id]
    curr_s = src_id
    while curr_s > 1
        curr_s = H2trees.parent(M.tree.trialcluster, curr_s)
        push!(src_ancestors, curr_s)
    end

    # Check the grid of all combinations of ancestors.
    # Skip (1,1) which corresponds to checking the exact query pair again.
    for p_obs in obs_ancestors
        for p_src in src_ancestors
            (p_obs == obs_id && p_src == src_id) && continue

            ancestor_bf_idx = M.far_lookup[p_obs, p_src]
            if ancestor_bf_idx != 0
                return FarBlockView{T}(p_obs, p_src, M.BFs[ancestor_bf_idx])
            end
        end
    end

    # =========================================================================
    # 3. DOWNWARD TRAVERSAL / COMPOSITE
    # =========================================================================

    return ZeroBlockView{T}(obs_id, src_id)
end
