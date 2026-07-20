import H2Trees: values, center, halfsize, children, isleaf, trialtree, testtree

# A single block in the butterfly factorization
struct ButterflyBlock{T}
    # CODOMAIN (Output / Row-equivalent)
    # The skeleton this block maps TO
    obs_out::Int  # e.g., Ochild
    src_out::Int  # e.g., Svert

    # DOMAIN (Input / Column-equivalent)
    # The skeleton this block maps FROM
    obs_in::Int   # e.g., Overt
    src_in::Int   # e.g., Schild

    data::Matrix{T}
end

# The sorting key now perfectly identifies the exact interaction pair
function block_key(b::ButterflyBlock)
    return (b.obs_out, b.src_out, b.obs_in, b.src_in)
end

# An entire level l of the R factors
struct ButterflyLevel{T}
    blocks::Vector{ButterflyBlock{T}}
end

# The complete factorization
struct ButterflyFactorization{T,M}
    Q::Vector{ButterflyBlock{T}}       # Leaf level
    R::Vector{ButterflyLevel{T}}       # Levels 1 to L-1
    P::Vector{ButterflyBlock{T}}       # Root/Observer leaf level
    tree::M
    k::Float64
    τ::Float64
end

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

    # --- trees & helpers ---
    trialT = trialtree(H2Blocktree)
    testT = testtree(H2Blocktree)
    treeS = traverseandpad(trialT, NS)
    treeO = traverseandpad(testT, NO)

    # --- containers ---
    lS = length(treeS[end])
    Q = Vector{ButterflyBlock{ComplexF64}}(undef, lS)
    K = Vector{Vector{Int}}(undef, lS)

    LS = length(treeS)
    LO = length(treeO)
    L = max(LS, LO)

    # ------------------------------------------------------------------
    # Leaf-level Q
    # ------------------------------------------------------------------
    tmap(1:lS; scheduler=scheduler) do blockidx
        Sleaf = treeS[end][blockidx]
        srcindex = values(trialT, Sleaf)
        obsindex = values(testT, NO)

        n_otilde = estimate_rank_3d(k, trialT, testT, Sleaf, NO, τ;)
        q_ks, k_l, r_l = compressor(kernelmatrix, srcindex, obsindex, n_otilde, τ)

        Q[blockidx] = q_ks
        return K[blockidx] = k_l
    end

    R = Vector{ButterflyLevel{ComplexF64}}(undef, L - 1)
    source_is_frozen = false
    obs_is_frozen = false

    # ------------------------------------------------------------------
    # Level traversal
    # ------------------------------------------------------------------
    for l in 1:(L - 1)
        l >= LS && (source_is_frozen = true)
        l >= LO && (obs_is_frozen = true)
        if source_is_frozen && obs_is_frozen
            break
        else
        end
        # --------------------------------------------------------------
        # Build U (union of child skeletons)
        # --------------------------------------------------------------
        if !source_is_frozen
            lu = length(treeS[LS - l]*treeO[min(l, LO)])
            U = Vector{Vector{Int}}(undef, lu)   #temporary unions
            tmap(1:lu; scheduler=scheduler) do idx
                Svert_idx = div(idx-1, length(treeO[min(l, LO)])) + 1
                Svert = treeS[LS - l][Svert_idx]
                Overt_idx = mod(idx-1, length(treeO[min(l, LO)])) + 1
                Overt = treeO[min(l, LO)][Overt_idx]
                if !isleaf(trialT, Svert)
                    temp_size = sum(
                        length(K[Schild][Overt]) for Schild in children(trialT, Svert)
                    )#<---- how to index K here....
                    temp = sizehint!(Int[], temp_size)
                    for Schild in children(trialT, Svert)
                        append!(temp, K[Schild][Overt])
                    end
                else#<---- how to treat the Svert is leaf case...
                    temp = K[Svert][Overt]
                end

                return U[idx] = temp
            end

            for (Svert, local_U_S) in results
                U[Svert] = local_U_S
            end

            # --------------------------------------------------------------
            # Compute R blocks
            # --------------------------------------------------------------
            if !obs_is_frozen
                lR = length(treeO[l + 1]) * length(treeS[LS - l + 1])
                R_level_l = Vector{ButterflyBlock{ComplexF64}}(undef, lR)
                K_new = Vector{Vector{Int}}(undef, lR)
                tmap(1:lR; scheduler=scheduler) do blkidx
                    #if Overt was a leaf in the current implementation we will find it in treeO[l+1] again as a ghost node due to traverseandpad....
                    Ochild_idx = div(blkidx-1, length(treeS[LS - l + 1])) + 1
                    Schild_idx = mod(blkidx-1, length(treeS[LS - l + 1])) + 1
                    Ochild = treeO[l + 1][Ochild_idx]
                    Schild = treeS[LS - l + 1][Svert_idx]
                    obsindex = values(testT, Ochild)
                    srcindex = U[someindex]     #we can find the parent of Schild.need to find a way to index U to find the accumulated skeleton...
                    n_otilde = estimate_rank_3d(k, trialT, testT, Svert, Ochild, τ;)
                    q_ks, k_l, r_l = compressor(
                        kernelmatrix, srcindex, obsindex, n_otilde, τ
                    )
                    if !haskey(local_R, (Ochild, Svert))
                        local_R[(Ochild, Svert)] = Dict{Tuple{Int,Int},Matrix{ComplexF64}}()
                    end
                    if !haskey(local_K, Svert)
                        local_K[Svert] = Dict{Int,Vector{Int}}()
                    end
                    if !isleaf(trialT, Svert)
                        last = 0
                        for Schild in children(trialT, Svert)
                            ks = length(K[Schild][Overt])
                            local_R[(Ochild, Svert)][(Overt, Schild)] = view(
                                q_ks, :, (last + 1):(last + ks)
                            )
                            #q_ks[:, (last + 1):(last + ks)]
                            last += ks
                        end
                    else
                        local_R[(Ochild, Svert)][(Overt, Svert)] = q_ks
                    end
                    local_K[Svert][Ochild] = k_l
                    #else
                    obsindex = values(testT, Overt)
                    for Svert in treeS[LS - l]
                        if !haskey(local_R, (Overt, Svert))
                            local_R[(Overt, Svert)] = Dict{
                                Tuple{Int,Int},Matrix{ComplexF64}
                            }()
                        end
                        if !haskey(local_K, Svert)
                            local_K[Svert] = Dict{Int,Vector{Int}}()
                        end
                        if !isleaf(trialT, Svert)
                            srcindex = U[Svert][Overt]
                            n_otilde = estimate_rank_3d(k, trialT, testT, Svert, Overt, τ;)
                            q_ks, k_l, r_l = compressor(
                                kernelmatrix, srcindex, obsindex, n_otilde, τ
                            )
                            last = 0
                            for Schild in children(trialT, Svert)
                                ks = length(K[Schild][Overt])
                                local_R[(Overt, Svert)][(Overt, Schild)] = view(
                                    q_ks, :, (last + 1):(last + ks)
                                )
                                #q_ks[ :, (last + 1):(last + ks)]
                                last += ks
                            end
                            local_K[Svert][Overt] = k_l
                        else
                            if l > 1
                                #=local_R[(Overt, Svert)][(Overt, Svert)] = I(
                                    size(
                                        R[l - 1][(Overt, Svert)][first(
                                            keys(R[l - 1][(Overt, Svert)])
                                        )],
                                        1,
                                    ),
                                )=#
                                #=n = size(
                                    R[l - 1][(Overt, Svert)][first(keys(R[l - 1][(Overt, Svert)]))],
                                    1,
                                )=#
                                local_R[(Overt, Svert)][(Overt, Svert)] = Matrix{ComplexF64}(
                                    I, 0, 0
                                )
                            else
                                #local_R[(Overt, Svert)][(Overt, Svert)] = I(size(Q[Svert], 1))

                                local_R[(Overt, Svert)][(Overt, Svert)] = Matrix{ComplexF64}(
                                    I, 0, 0
                                )
                            end
                            local_K[Svert][Overt] = K[Svert][Overt]
                        end
                    end
                    #end

                    # 2. Returnera båda de lokala strukturerna
                    return (local_R, local_K)
                end
                K = K_new
            else
                K_new = Dict{Int,Dict{Int,Vector{Int}}}()
                build_observerfrozen_R_blocks!(
                    R,
                    K,
                    K_new,
                    U,
                    Q,
                    treeS,
                    treeO,
                    l,
                    trialT,
                    testT,
                    kernelmatrix,
                    compressor,
                    k,
                    τ,
                    LO,
                    LS;
                    scheduler,
                )
                K = K_new
            end
        end
        K_new = Dict{Int,Dict{Int,Vector{Int}}}()
        build_sourcefrozen_R_blocks!(
            R,
            K,
            K_new,
            Q,
            NS,
            treeO,
            l,
            trialT,
            testT,
            kernelmatrix,
            compressor,
            k,
            τ;
            scheduler,
        )
        K = K_new
    end

    # ------------------------------------------------------------------
    # Final P blocks
    # ------------------------------------------------------------------
    lO = length(treeO[end])
    P = Vector{ButterflyBlock{ComplexF64}}(undef, lO)
    leaf_results = let K = K
        tmap(treeO[LO]; scheduler=scheduler) do Oleaf
            col = K[NS][Oleaf]
            row = values(testT, Oleaf)

            Z = zeros(ComplexF64, length(row), length(col))
            kernelmatrix(Z, row, col)
            return (Oleaf, Z)
        end
    end
    for (Oleaf, Z) in leaf_results
        P[Oleaf, NS] = Z
    end
    return BF(
        Q,
        R,
        P,
        (
            length(H2Trees.values(H2Blocktree.testcluster, NO)),
            length(H2Trees.values(H2Blocktree.trialcluster, NS)),
        ),
        NS,
        NO,
        k,
        τ,
        H2Blocktree.trialcluster,
        H2Blocktree.testcluster,
    )
end

function build_union_skeletons!(
    U::Dict{Int,Dict{Int,Vector{Int}}},
    K::Dict{Int,Dict{Int,Vector{Int}}},
    treeS,
    treeO,
    l,
    LS,
    LO,
    trialT;
    scheduler=OhMyThreads.SerialScheduler(),
)
    results = tmap(treeS[LS - l]; scheduler=scheduler) do Svert
        local_U_S = Dict{Int,Vector{Int}}()

        for Overt in treeO[min(l, LO)]
            if !isleaf(trialT, Svert)
                temp_size = sum(
                    length(K[Schild][Overt]) for Schild in children(trialT, Svert)
                )
                temp = sizehint!(Int[], temp_size)
                for Schild in children(trialT, Svert)
                    append!(temp, K[Schild][Overt])
                end
            else
                temp = K[Svert][Overt]
            end

            local_U_S[Overt] = temp
        end

        return (Svert, local_U_S)
    end

    for (Svert, local_U_S) in results
        U[Svert] = local_U_S
    end

    return nothing
end

function build_nonfrozen_R_blocks!(
    R::Vector{Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},Matrix{ComplexF64}}}},
    K::Dict{Int,Dict{Int,Vector{Int}}},
    K_new::Dict{Int,Dict{Int,Vector{Int}}},
    U::Dict{Int,Dict{Int,Vector{Int}}},
    Q::Dict{Tuple{Int,Int},Matrix{ComplexF64}},
    treeS,
    treeO,
    l,
    trialT,
    testT,
    kernelmatrix,
    compressor,
    k,
    τ,
    LS;
    scheduler=OhMyThreads.SerialScheduler(),
)

    # 3. Slå ihop all tråddata sekventiellt och säkert i huvudstrukturerna
    for (local_R, local_K) in results
        # Merga in i R[l]
        for (k1, v1) in local_R
            target_dict = getsubdict!(R[l], k1)
            for (k2, v2) in v1
                target_dict[k2] = v2
            end
        end

        # Merga in i globala K
        for (svert, o_dict) in local_K
            target_K_dict = getsubdict!(K_new, svert)
            for (obs_idx, k_l_val) in o_dict
                target_K_dict[obs_idx] = k_l_val
            end
        end
    end

    return nothing
end

function build_sourcefrozen_R_blocks!(
    R::Vector{Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},Matrix{ComplexF64}}}},
    K::Dict{Int,Dict{Int,Vector{Int}}},
    K_new::Dict{Int,Dict{Int,Vector{Int}}},
    Q::Dict{Tuple{Int,Int},Matrix{ComplexF64}},
    NS,
    treeO,
    l,
    trialT,
    testT,
    kernelmatrix,
    compressor,
    k,
    τ;
    scheduler=OhMyThreads.SerialScheduler(),
)
    results = tmap(treeO[l]; scheduler=scheduler) do Overt
        local_R = Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},Matrix{ComplexF64}}}()
        local_K = Dict{Int,Dict{Int,Vector{Int}}}()
        Svert = NS
        if !haskey(local_K, Svert)
            local_K[Svert] = Dict{Int,Vector{Int}}()
        end
        if !isleaf(testT, Overt)
            for Ochild in children(testT, Overt)
                obsindex = values(testT, Ochild)

                if !haskey(local_R, (Ochild, Svert))
                    local_R[(Ochild, Svert)] = Dict{Tuple{Int,Int},Matrix{ComplexF64}}()
                end

                srcindex = K[Svert][Overt]
                n_otilde = estimate_rank_3d(k, trialT, testT, Svert, Ochild, τ;)
                q_ks, k_l, r_l = compressor(kernelmatrix, srcindex, obsindex, n_otilde, τ)
                local_R[(Ochild, Svert)][(Overt, Svert)] = q_ks
                local_K[Svert][Ochild] = k_l
            end
        else
            if !haskey(local_R, (Overt, Svert))
                local_R[(Overt, Svert)] = Dict{Tuple{Int,Int},Matrix{ComplexF64}}()
            end
            if l > 1
                #=local_R[(Overt, Svert)][(Overt, Svert)] = I(
                    size(
                        R[l - 1][(Overt, Svert)][first(keys(R[l - 1][(Overt, Svert)]))],
                        1,
                    ),
                )=#
                #n = size(R[l - 1][(Overt, Svert)][first(keys(R[l - 1][(Overt, Svert)]))], 1)
                local_R[(Overt, Svert)][(Overt, Svert)] = Matrix{ComplexF64}(I, 0, 0)
            else
                #local_R[(Overt, Svert)][(Overt, Svert)] = I(size(Q[Svert], 1))
                #n = size(Q[Svert], 1)
                local_R[(Overt, Svert)][(Overt, Svert)] = Matrix{ComplexF64}(I, 0, 0)
            end
            local_K[Svert][Overt] = K[Svert][Overt]
        end
        return (local_R, local_K)
    end

    # Slå ihop datan säkert
    for (local_R, local_K) in results
        for (k1, v1) in local_R
            target_dict = getsubdict!(R[l], k1)
            for (k2, v2) in v1
                target_dict[k2] = v2
            end
        end
        for (svert, o_dict) in local_K
            target_K_dict = getsubdict!(K_new, svert)
            for (obs_idx, k_l_val) in o_dict
                target_K_dict[obs_idx] = k_l_val
            end
        end
    end

    return nothing
end

function build_observerfrozen_R_blocks!(
    R::Vector{Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},Matrix{ComplexF64}}}},
    K::Dict{Int,Dict{Int,Vector{Int}}},
    K_new::Dict{Int,Dict{Int,Vector{Int}}},
    U::Dict{Int,Dict{Int,Vector{Int}}},
    Q::Dict{Tuple{Int,Int},Matrix{ComplexF64}},
    treeS,
    treeO,
    l,
    trialT,
    testT,
    kernelmatrix,
    compressor,
    k,
    τ,
    LO,
    LS;
    scheduler=OhMyThreads.SerialScheduler(),
)
    LO = length(treeO)
    LS = length(treeS)

    results = tmap(treeO[LO]; scheduler=scheduler) do Overt
        local_R = Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},Matrix{ComplexF64}}}()
        local_K = Dict{Int,Dict{Int,Vector{Int}}}()
        obsindex = values(testT, Overt)
        for Svert in treeS[LS - l]
            if !haskey(local_K, Svert)
                local_K[Svert] = Dict{Int,Vector{Int}}()
            end
            if !haskey(local_R, (Overt, Svert))
                local_R[(Overt, Svert)] = Dict{Tuple{Int,Int},Matrix{ComplexF64}}()
            end
            if !isleaf(trialT, Svert)
                srcindex = U[Svert][Overt]
                n_otilde = estimate_rank_3d(k, trialT, testT, Svert, Overt, τ;)
                q_ks, k_l, r_l = compressor(kernelmatrix, srcindex, obsindex, n_otilde, τ)

                last = 0
                for Schild in children(trialT, Svert)
                    ks = length(K[Schild][Overt])
                    local_R[(Overt, Svert)][(Overt, Schild)] = view(
                        q_ks, :, (last + 1):(last + ks)
                    )
                    #q_ks[:, (last + 1):(last + ks)]
                    last += ks
                end
                local_K[Svert][Overt] = k_l
            else
                if l > 1
                    #=local_R[(Overt, Svert)][(Overt, Svert)] = I(
                        size(
                            R[l - 1][(Overt, Svert)][first(keys(R[l - 1][(Overt, Svert)]))],
                            1,
                        ),
                    )=#
                    #n = size(R[l - 1][(Overt, Svert)][first(keys(R[l - 1][(Overt, Svert)]))], 1)
                    local_R[(Overt, Svert)][(Overt, Svert)] = Matrix{ComplexF64}(I, 0, 0)
                else
                    #local_R[(Overt, Svert)][(Overt, Svert)] = I(size(Q[Svert], 1))
                    #n = size(Q[Svert], 1)
                    local_R[(Overt, Svert)][(Overt, Svert)] = Matrix{ComplexF64}(I, 0, 0)
                end
                local_K[Svert][Overt] = K[Svert][Overt]
            end
        end
        return (local_R, local_K)
    end

    # Slå ihop datan säkert
    for (local_R, local_K) in results
        for (k1, v1) in local_R
            target_dict = getsubdict!(R[l], k1)
            for (k2, v2) in v1
                target_dict[k2] = v2
            end
        end
        for (svert, o_dict) in local_K
            target_K_dict = getsubdict!(K_new, svert)
            for (obs_idx, k_l_val) in o_dict
                target_K_dict[obs_idx] = k_l_val
            end
        end
    end

    return nothing
end
