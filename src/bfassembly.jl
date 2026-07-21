import H2Trees: values, center, halfsize, children, isleaf, trialtree, testtree

# ------------------------------------------------------------------
# Main Assembly Routine
# ------------------------------------------------------------------
function assemble_BF(
    kernelmatrix,
    H2Blocktree,
    NO::Int,
    NS::Int,
    k::Float64,
    τ::Float64;
    compressor=ButterflyFactorizations.PartialQR(),
    scheduler=OhMyThreads.SerialScheduler(),
)
    # --- Trees & Helpers ---
    trialT = trialtree(H2Blocktree)
    testT = testtree(H2Blocktree)
    treeS = traverseandpad(trialT, NS)
    treeO = traverseandpad(testT, NO)

    LS = length(treeS)
    LO = length(treeO)
    L = max(LS, LO)

    # Temporary workspace for skeletons: K[(Snode_id, Onode_id)] => Vector{Int}
    K = Dict{Tuple{Int,Int},Vector{Int}}()

    # ------------------------------------------------------------------
    # 1. Leaf-level Q
    # ------------------------------------------------------------------
    lS = length(treeS[end])
    Q = Vector{ButterflyBlock{ComplexF64}}(undef, lS)

    leaf_q_results = tmap(1:lS; scheduler=scheduler) do blockidx
        Sleaf = treeS[end][blockidx]
        srcindex = values(trialT, Sleaf)
        obsindex = values(testT, NO)

        n_otilde = estimate_rank_3d(k, trialT, testT, Sleaf, NO, τ)
        q_ks, k_l, _ = compressor(kernelmatrix, srcindex, obsindex, n_otilde, τ)

        return (blockidx, Sleaf, q_ks, k_l)
    end

    # Safely update Q and K sequentially after threading
    for (blockidx, Sleaf, q_ks, k_l) in leaf_q_results
        Q[blockidx] = ButterflyBlock(NO, Sleaf, NO, Sleaf, q_ks)
        K[(Sleaf, NO)] = k_l
    end

    # ------------------------------------------------------------------
    # 2. Level Traversal (R Blocks)
    # ------------------------------------------------------------------
    R = Vector{ButterflyLevel{ComplexF64}}(undef, L - 1)

    for l in 1:(L - 1)
        source_is_frozen = l >= LS
        obs_is_frozen = l >= LO

        if source_is_frozen && obs_is_frozen
            break
        end

        U = if source_is_frozen
            nothing
        else
            build_U_skeletons(treeS[LS - l], treeO[min(l, LO)], K, trialT, scheduler)
        end

        if !source_is_frozen && !obs_is_frozen
            level_R, k_updates = build_nonfrozen_R_blocks(
                treeO[l],
                treeS[LS - l],
                U,
                K,
                trialT,
                testT,
                kernelmatrix,
                compressor,
                k,
                τ,
                scheduler,
            )
        elseif source_is_frozen && !obs_is_frozen
            level_R, k_updates = build_sourcefrozen_R_blocks(
                treeO[l], NS, K, trialT, testT, kernelmatrix, compressor, k, τ, scheduler
            )
        elseif !source_is_frozen && obs_is_frozen
            level_R, k_updates = build_observerfrozen_R_blocks(
                treeO[LO],
                treeS[LS - l],
                U,
                K,
                trialT,
                testT,
                kernelmatrix,
                compressor,
                k,
                τ,
                scheduler,
            )
        end

        R[l] = level_R

        # Sequentially apply skeleton updates between levels
        for (s, o, k_l) in k_updates
            K[(s, o)] = k_l
        end
    end

    # ------------------------------------------------------------------
    # 3. Final P blocks
    # ------------------------------------------------------------------
    lO = length(treeO[end])
    P = Vector{ButterflyBlock{ComplexF64}}(undef, lO)

    tmap(1:lO; scheduler=scheduler) do idx
        Oleaf = treeO[end][idx]
        col = K[(NS, Oleaf)]
        row = values(testT, Oleaf)

        Z = zeros(ComplexF64, length(row), length(col))
        kernelmatrix(Z, row, col)

        return P[idx] = ButterflyBlock(Oleaf, NS, Oleaf, NS, Z)
    end

    return ButterflyFactorization(Q, R, P, H2Blocktree, k, τ)
end

# ------------------------------------------------------------------
# Builder Functions
# ------------------------------------------------------------------

function build_U_skeletons(treeS_level, treeO_level, K, trialT, scheduler)
    U = Dict{Tuple{Int,Int},Vector{Int}}()
    interactions_U = vec(collect(Iterators.product(treeS_level, treeO_level)))

    results_U = tmap(interactions_U; scheduler=scheduler) do (Svert, Overt)
        if !isleaf(trialT, Svert)
            temp_size = sum(
                length(K[(Schild, Overt)]) for Schild in children(trialT, Svert)
            )
            temp = sizehint!(Int[], temp_size)
            for Schild in children(trialT, Svert)
                append!(temp, K[(Schild, Overt)])
            end
        else
            temp = K[(Svert, Overt)]
        end
        return (Svert, Overt, temp)
    end

    for (s, o, temp_skel) in results_U
        U[(s, o)] = temp_skel
    end
    return U
end

function build_nonfrozen_R_blocks(
    treeO_level, treeS_level, U, K, trialT, testT, kernelmatrix, compressor, k, τ, scheduler
)
    interactions = vec(collect(Iterators.product(treeO_level, treeS_level)))

    results = tmap(interactions; scheduler=scheduler) do (Overt, Svert)
        local_blocks = ButterflyBlock{ComplexF64}[]
        local_k_updates = Tuple{Int,Int,Vector{Int}}[]

        if !isleaf(testT, Overt)
            for Ochild in children(testT, Overt)
                obsindex = values(testT, Ochild)
                srcindex = U[(Svert, Overt)]

                n_otilde = estimate_rank_3d(k, trialT, testT, Svert, Ochild, τ)
                q_ks, k_l, _ = compressor(kernelmatrix, srcindex, obsindex, n_otilde, τ)

                push!(local_k_updates, (Svert, Ochild, k_l))

                if !isleaf(trialT, Svert)
                    last_idx = 0
                    for Schild in children(trialT, Svert)
                        ks = length(K[(Schild, Overt)])
                        block_data = Matrix(view(q_ks, :, (last_idx + 1):(last_idx + ks)))
                        push!(
                            local_blocks,
                            ButterflyBlock(Ochild, Svert, Overt, Schild, block_data),
                        )
                        last_idx += ks
                    end
                else
                    push!(local_blocks, ButterflyBlock(Ochild, Svert, Overt, Svert, q_ks))
                end
            end
        else
            obsindex = values(testT, Overt)
            if !isleaf(trialT, Svert)
                srcindex = U[(Svert, Overt)]
                n_otilde = estimate_rank_3d(k, trialT, testT, Svert, Overt, τ)
                q_ks, k_l, _ = compressor(kernelmatrix, srcindex, obsindex, n_otilde, τ)

                last_idx = 0
                for Schild in children(trialT, Svert)
                    ks = length(K[(Schild, Overt)])
                    block_data = Matrix(view(q_ks, :, (last_idx + 1):(last_idx + ks)))
                    push!(
                        local_blocks,
                        ButterflyBlock(Overt, Svert, Overt, Schild, block_data),
                    )
                    last_idx += ks
                end
                push!(local_k_updates, (Svert, Overt, k_l))
            else
                push!(local_blocks, ButterflyBlock(Overt, Svert, Overt, Svert, I))
                push!(local_k_updates, (Svert, Overt, K[(Svert, Overt)]))
            end
        end
        return (local_blocks, local_k_updates)
    end

    all_blocks = ButterflyBlock{ComplexF64}[]
    all_k_updates = Tuple{Int,Int,Vector{Int}}[]
    for (blocks, k_updates) in results
        append!(all_blocks, blocks)
        append!(all_k_updates, k_updates)
    end
    sort!(all_blocks; by=block_key)

    return ButterflyLevel(all_blocks), all_k_updates
end

function build_sourcefrozen_R_blocks(
    treeO_level, NS, K, trialT, testT, kernelmatrix, compressor, k, τ, scheduler
)
    Svert = NS

    results = tmap(treeO_level; scheduler=scheduler) do Overt
        local_blocks = ButterflyBlock{ComplexF64}[]
        local_k_updates = Tuple{Int,Int,Vector{Int}}[]

        if !isleaf(testT, Overt)
            for Ochild in children(testT, Overt)
                obsindex = values(testT, Ochild)
                srcindex = K[(Svert, Overt)]

                n_otilde = estimate_rank_3d(k, trialT, testT, Svert, Ochild, τ)
                q_ks, k_l, _ = compressor(kernelmatrix, srcindex, obsindex, n_otilde, τ)

                push!(local_blocks, ButterflyBlock(Ochild, Svert, Overt, Svert, q_ks))
                push!(local_k_updates, (Svert, Ochild, k_l))
            end
        else
            push!(local_blocks, ButterflyBlock(Overt, Svert, Overt, Svert, I))
            push!(local_k_updates, (Svert, Overt, K[(Svert, Overt)]))
        end
        return (local_blocks, local_k_updates)
    end

    all_blocks = ButterflyBlock{ComplexF64}[]
    all_k_updates = Tuple{Int,Int,Vector{Int}}[]
    for (blocks, k_updates) in results
        append!(all_blocks, blocks)
        append!(all_k_updates, k_updates)
    end
    sort!(all_blocks; by=block_key)

    return ButterflyLevel(all_blocks), all_k_updates
end

function build_observerfrozen_R_blocks(
    treeO_LO, treeS_level, U, K, trialT, testT, kernelmatrix, compressor, k, τ, scheduler
)
    interactions = vec(collect(Iterators.product(treeO_LO, treeS_level)))

    results = tmap(interactions; scheduler=scheduler) do (Overt, Svert)
        local_blocks = ButterflyBlock{ComplexF64}[]
        local_k_updates = Tuple{Int,Int,Vector{Int}}[]

        obsindex = values(testT, Overt)

        if !isleaf(trialT, Svert)
            srcindex = U[(Svert, Overt)]
            n_otilde = estimate_rank_3d(k, trialT, testT, Svert, Overt, τ)
            q_ks, k_l, _ = compressor(kernelmatrix, srcindex, obsindex, n_otilde, τ)

            last_idx = 0
            for Schild in children(trialT, Svert)
                ks = length(K[(Schild, Overt)])
                block_data = Matrix(view(q_ks, :, (last_idx + 1):(last_idx + ks)))

                push!(local_blocks, ButterflyBlock(Overt, Svert, Overt, Schild, block_data))
                last_idx += ks
            end
            push!(local_k_updates, (Svert, Overt, k_l))
        else
            push!(local_blocks, ButterflyBlock(Overt, Svert, Overt, Svert, I))
            push!(local_k_updates, (Svert, Overt, K[(Svert, Overt)]))
        end
        return (local_blocks, local_k_updates)
    end

    all_blocks = ButterflyBlock{ComplexF64}[]
    all_k_updates = Tuple{Int,Int,Vector{Int}}[]
    for (blocks, k_updates) in results
        append!(all_blocks, blocks)
        append!(all_k_updates, k_updates)
    end
    sort!(all_blocks; by=block_key)

    return ButterflyLevel(all_blocks), all_k_updates
end
