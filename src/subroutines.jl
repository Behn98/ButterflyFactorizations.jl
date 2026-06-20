import H2Trees: values, center, halfsize, children, isleaf, trialtree, testtree
"""
    subroutine_BF(kernelmatrix, H2Blocktree, NO, NS, k, τ; Compressor=PartialQR())

Constructs the Butterfly Factorization for a given block in a **dictionary format**.

This subroutine traverses the H2 tree structure from the leaf level moving up to the root of
the source tree, and to the leaves of the observer tree. It computes the low-rank
approximations to build the `Q`, `R`, and `P` factors as dictionaries. It also maps the
necessary index permutations to allow for correct, independent Matrix-Vector (MV) products
without permanently storing the tree.

**Arguments:**

  - `kernelmatrix`: Function computing matrix entries for specified row/column indices.
  - `H2Blocktree`: The paired source-observer tree structure.
  - `NO`, `NS`: The root IDs of the observer (test) and source (trial) spaces.
  - `k`, `τ`: Wavenumber (crucial for rank estimation) and precision tolerance.
  - `compressor`: Compression scheme for low-rank blocks (default: `PartialQR`).

**Why Dictionary format?** It is extremely memory efficient because it avoids the overhead
of saving nonzero entry indices required by sparse matrices. It is deeply intuitive, makes
algebraic manipulation (like butterfly multiplication or summation) flexible, and preserves
the semantic structure other than a matrix format. It is ideal when still working with the
tree structure is desired, such as in the case of recompression or algebraic operations on
the factors. However, it requires careful handling of index permutations to ensure correct
MV products, which is managed through the `PermQ` and `PermP` dictionaries.
"""

function subroutine_BF(
    kernelmatrix,
    H2Blocktree,
    NO::Int,
    NS::Int,
    k::Float64,
    τ::Float64;
    compressor=ButterflyFactorizations.PartialQR(),
    scheduler=OhMyThreads.SerialScheduler(),
)

    # --- containers ---
    Q = Dict{Tuple{Int,Int},Matrix{ComplexF64}}()
    K = Dict{Int,Dict{Int,Vector{Int}}}()
    P = Dict{Tuple{Int,Int},Matrix{ComplexF64}}()

    PermQ = Dict{Tuple{Int,Int},Vector{Int}}()
    PermP = Dict{Tuple{Int,Int},Vector{Int}}()
    # --- trees & helpers ---
    trialT = trialtree(H2Blocktree)
    testT = testtree(H2Blocktree)
    treeS = traverseandpad(trialT, NS)
    treeO = traverseandpad(testT, NO)

    LS = length(treeS)
    LO = length(treeO)
    L = max(LS, LO)
    R = Vector{Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},Matrix{ComplexF64}}}}(undef, L - 1)
    # 1. Get all global indices for this block's source/observer trees
    global_src_indices = values(trialT, NS) # All sources for this block
    global_obs_indices = values(testT, NO)  # All observers for this block

    # 2. Create a mapping from global to local (1 to N)
    src_map = Dict(g => l for (l, g) in enumerate(global_src_indices))
    obs_map = Dict(g => l for (l, g) in enumerate(global_obs_indices))

    # ------------------------------------------------------------------
    # Leaf-level Q
    # ------------------------------------------------------------------
    leaf_results = tmap(treeS[LS]; scheduler=scheduler) do Sleaf
        srcindex = values(trialT, Sleaf)
        obsindex = values(testT, NO)

        n_otilde = estimate_rank_3d(k, trialT, testT, Sleaf, NO, τ;)
        q_ks, k_l, r_l = compressor(kernelmatrix, srcindex, obsindex, n_otilde, τ)

        perm_q_val = [src_map[g] for g in srcindex]
        return (Sleaf, perm_q_val, q_ks, k_l)
    end

    for (Sleaf, perm_q_val, q_ks, k_l) in leaf_results
        PermQ[NO, Sleaf] = perm_q_val
        Q[NO, Sleaf] = q_ks
        getsubdict!(K, Sleaf)[NO] = k_l
    end

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
            R[l] = Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},Matrix{ComplexF64}}}()
        end
        # --------------------------------------------------------------
        # Build U (union of child skeletons)
        # --------------------------------------------------------------
        U = Dict{Int,Dict{Int,Vector{Int}}}()   #temporary unions
        if !source_is_frozen
            build_union_skeletons!(U, K, treeS, treeO, l, LS, LO, trialT)
        end

        # --------------------------------------------------------------
        # Compute R blocks
        # --------------------------------------------------------------
        if !source_is_frozen && !obs_is_frozen
            K_new = Dict{Int,Dict{Int,Vector{Int}}}()
            build_nonfrozen_R_blocks!(
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
                LS;
                scheduler,
            )
            K = K_new
        elseif source_is_frozen && !obs_is_frozen
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
        else #!source_is_frozen && obs_is_frozen
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

    # ------------------------------------------------------------------
    # Final P blocks
    # ------------------------------------------------------------------

    leaf_results = let K = K
        tmap(treeO[LO]; scheduler=scheduler) do Oleaf
            col = K[NS][Oleaf]
            row = values(testT, Oleaf)

            Z = zeros(ComplexF64, length(row), length(col))
            kernelmatrix(Z, row, col)
            perm_p_val = [obs_map[g] for g in row]
            return (Oleaf, perm_p_val, Z)
        end
    end
    for (Oleaf, perm_p_val, Z) in leaf_results
        PermP[Oleaf, NS] = perm_p_val
        P[Oleaf, NS] = Z
    end
    return BF(
        Q,
        R,
        P,
        PermQ,
        PermP,
        (length(global_obs_indices), length(global_src_indices)),
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
    results = tmap(treeO[l]; scheduler=scheduler) do Overt
        # 1. Skapa lokala ordböcker
        local_R = Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},Matrix{ComplexF64}}}()
        local_K = Dict{Int,Dict{Int,Vector{Int}}}()

        if !isleaf(testT, Overt)
            for Ochild in children(testT, Overt)
                obsindex = values(testT, Ochild)
                for Svert in treeS[LS - l]
                    srcindex = U[Svert][Overt]
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
                end
            end
        else
            obsindex = values(testT, Overt)
            for Svert in treeS[LS - l]
                if !haskey(local_R, (Overt, Svert))
                    local_R[(Overt, Svert)] = Dict{Tuple{Int,Int},Matrix{ComplexF64}}()
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
        end

        # 2. Returnera båda de lokala strukturerna
        return (local_R, local_K)
    end

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

"""
    subroutine_BF_mats(kernelmatrix, H2Blocktree, NO, NS, k, τ; compressor=PartialQR())

Constructs the Butterfly Factorization for a given block in a **sparse matrix format**.

Similar to `subroutine_BF`, it traverses the H2 tree structure to compute the `Q`, `R`,
and `P` factors, but specifically pieces them together into sparse block-diagonal matrices.
It also returns single continuous permutation vectors (`PermP`, `PermQ`) to map
interactions across the entire space.

**Why Matrix format?**
While slightly less memory efficient than the dictionary format (due to sparsity tracking
overhead), it allows for dramatically faster direct Matrix-Vector applications using
standard linear algebra methods. It also provides a clear visual and algebraic
representation of the overall block structure, which is invaluable for debugging.
"""
function subroutine_BF_mats(
    kernelmatrix,
    H2Blocktree,
    NO::Int,
    NS::Int,
    k::Float64,
    τ::Float64;
    compressor=ButterflyFactorizations.PartialQR(),
)

    # --- containers ---
    Q = Matrix{ComplexF64}(undef, 0, 0)
    R = Vector{AbstractMatrix{ComplexF64}}()
    P = Matrix{ComplexF64}(undef, 0, 0)
    K = Dict{Int,Dict{Int,Vector{Int}}}()
    U = Dict{Int,Dict{Int,Vector{Int}}}()   #temporary unions
    #AbstractMatrix for SparseArrays, BlockSparseMatrix for BlockSparseMatrices
    PermQ = Vector{Int}()
    PermP = Vector{Int}()

    # --- trees & helpers ---
    trialT = H2Trees.trialtree(H2Blocktree)
    testT = H2Trees.testtree(H2Blocktree)

    treeS = traverseandpad(trialT, NS)
    treeO = traverseandpad(testT, NO)

    LS = length(treeS)
    LO = length(treeO)
    L = LS + LO

    # ------------------------------------------------------------------
    # Leaf-level Q
    # ------------------------------------------------------------------
    for Sleaf in treeS[LS]
        srcindex = values(trialT, Sleaf)
        push!(PermQ, srcindex...)
        obsindex = values(testT, NO)
        n_otilde = estimate_rank_3d(k, trialT, testT, Sleaf, NO, τ;)
        q_ks, k_l, r_l = compressor(kernelmatrix, srcindex, obsindex, n_otilde, τ)
        Q = sparse_blockdiag(Q, q_ks)               #SPARSITY: sparse_ or blocksparse_
        getsubdict!(K, Sleaf)[NO] = k_l
    end
    source_is_frozen = false
    obs_is_frozen = false

    # ------------------------------------------------------------------
    # Level traversal
    # ------------------------------------------------------------------
    for l in 1:(L - 1)
        l >= LS && (source_is_frozen = true)
        l >= LO && (obs_is_frozen = true)

        # --------------------------------------------------------------
        # Build U (union of child skeletons)
        # --------------------------------------------------------------
        if !source_is_frozen
            for Svert in treeS[LS - l]
                U_S = getsubdict!(U, Svert)

                for Overt in treeO[min(l, LO)]
                    temp = Int[]

                    for Schild in children(trialT, Svert)
                        Ks = getsubdict!(K, Schild)
                        ks = get(Ks, Overt, nothing)
                        append!(temp, ks)
                    end

                    U_S[Overt] = temp
                end
            end
        end

        # --------------------------------------------------------------
        # Compute R blocks
        # --------------------------------------------------------------
        if !source_is_frozen && !obs_is_frozen
            R_temp1 = Matrix{ComplexF64}(undef, 0, 0)
            for Overt in treeO[l]
                R_temp2 = Vector{AbstractMatrix{ComplexF64}}()
                for Ochild in children(testT, Overt)
                    R_temp3 = Matrix{ComplexF64}(undef, 0, 0)
                    obsindex = values(testT, Ochild)
                    for Svert in treeS[LS - l]
                        srcindex = U[Svert][Overt]

                        n_otilde = estimate_rank_3d(k, trialT, testT, Svert, Ochild, τ;)
                        q_ks, k_l, r_l = compressor(
                            kernelmatrix, srcindex, obsindex, n_otilde, τ
                        )
                        R_temp3 = sparse_blockdiag(R_temp3, q_ks)
                        getsubdict!(K, Svert)[Ochild] = k_l
                    end
                    push!(R_temp2, R_temp3)
                    R_temp3 = Matrix{ComplexF64}(undef, 0, 0)
                end
                R_temp1 = sparse_blockdiag(R_temp1, sparse_vcat(R_temp2...))
                R_temp2 = Vector{AbstractMatrix{ComplexF64}}()
            end

            push!(R, R_temp1)

        elseif source_is_frozen && !obs_is_frozen
            R_temp1 = Matrix{ComplexF64}(undef, 0, 0)
            for Overt in treeO[l]
                R_temp2 = Vector{AbstractMatrix{ComplexF64}}()
                for Ochild in children(testT, Overt)
                    R_temp3 = Matrix{ComplexF64}(undef, 0, 0)
                    obsindex = values(testT, Ochild)
                    for Svert in treeS[1]
                        srcindex = K[Svert][Overt]
                        n_otilde = estimate_rank_3d(k, trialT, testT, Svert, Ochild, τ;)
                        q_ks, k_l, r_l = compressor(
                            kernelmatrix, srcindex, obsindex, n_otilde, τ
                        )
                        R_temp3 = sparse_blockdiag(R_temp3, q_ks)

                        getsubdict!(K, Svert)[Ochild] = k_l
                    end
                    push!(R_temp2, R_temp3)
                    R_temp3 = Matrix{ComplexF64}(undef, 0, 0)
                end
                R_temp1 = sparse_blockdiag(R_temp1, sparse_vcat(R_temp2...))
                R_temp2 = Vector{AbstractMatrix{ComplexF64}}()
            end
            push!(R, R_temp1)

        elseif !source_is_frozen && obs_is_frozen
            R_temp1 = Matrix{ComplexF64}(undef, 0, 0)
            for Overt in treeO[LO]
                obsindex = values(testT, Overt)
                R_temp2 = Matrix{ComplexF64}(undef, 0, 0)
                for Svert in treeS[LS - l]
                    srcindex = U[Svert][Overt]

                    n_otilde = estimate_rank_3d(k, trialT, testT, Svert, Overt, τ;)
                    q_ks, k_l, r_l = compressor(
                        kernelmatrix, srcindex, obsindex, n_otilde, τ
                    )
                    R_temp2 = sparse_blockdiag(R_temp2, q_ks)

                    getsubdict!(K, Svert)[Overt] = k_l
                end
                R_temp1 = sparse_blockdiag(R_temp1, R_temp2)
            end
            push!(R, R_temp1)
        else
            break
        end
    end

    # ------------------------------------------------------------------
    # Final P blocks
    # ------------------------------------------------------------------

    for Oleaf in treeO[LO]
        col = K[NS][Oleaf]
        row = values(testT, Oleaf)
        push!(PermP, row...)
        Z = zeros(ComplexF64, length(row), length(col))
        kernelmatrix(Z, row, col)
        P = sparse_blockdiag(P, Z)
    end
    return BF_Mats(Q, R, P, NS, NO, τ, k, PermP, PermQ)
end

#------------------------STATISTICS-------------------------------

function subroutine_BF(
    kernelmatrix,
    H2Blocktree,
    NO::Int,
    NS::Int,
    k::Float64,
    τ::Float64,
    dostat::Bool;
    compressor=ButterflyFactorizations.PartialQR(),
    scheduler=OhMyThreads.SerialScheduler(),
)

    # --- containers ---
    Q = Dict{Tuple{Int,Int},Matrix{ComplexF64}}()
    K = Dict{Int,Dict{Int,Vector{Int}}}()
    P = Dict{Tuple{Int,Int},Matrix{ComplexF64}}()

    PermQ = Dict{Tuple{Int,Int},Vector{Int}}()
    PermP = Dict{Tuple{Int,Int},Vector{Int}}()

    # --- trees & helpers ---
    trialT = trialtree(H2Blocktree)
    testT = testtree(H2Blocktree)
    treeS = traverseandpad(trialT, NS)
    treeO = traverseandpad(testT, NO)
    LS = length(treeS)
    LO = length(treeO)
    L = max(LS, LO)
    R = Vector{Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},Matrix{ComplexF64}}}}(undef, L - 1)

    # --- Statistics ---
    Q_ratios = Vector{Tuple{Int,Float64}}(undef, length(treeS[LS])) #stat
    R_ratios = Vector{Vector{Tuple{Int,Float64}}}(undef, L - 1) #stat

    # 1. Get all global indices for this block's source/observer trees
    global_src_indices = values(trialT, NS) # All sources for this block
    global_obs_indices = values(testT, NO)  # All observers for this block

    # 2. Create a mapping from global to local (1 to N)
    src_map = Dict(g => l for (l, g) in enumerate(global_src_indices))
    obs_map = Dict(g => l for (l, g) in enumerate(global_obs_indices))

    # ------------------------------------------------------------------
    # Leaf-level Q
    # ------------------------------------------------------------------
    leaf_results = tmap(treeS[LS]; scheduler=scheduler) do Sleaf
        srcindex = values(trialT, Sleaf)
        obsindex = values(testT, NO)

        n_otilde = estimate_rank_3d(k, trialT, testT, Sleaf, NO, τ;)
        q_ks, k_l, r_l = compressor(kernelmatrix, srcindex, obsindex, n_otilde, τ)

        perm_q_val = [src_map[g] for g in srcindex]
        return (Sleaf, perm_q_val, q_ks, k_l, (length(obsindex), r_l / n_otilde)) #stat
    end
    i = 1    #stat
    for (Sleaf, perm_q_val, q_ks, k_l, ratio) in leaf_results   #stat
        PermQ[NO, Sleaf] = perm_q_val
        Q[NO, Sleaf] = q_ks
        getsubdict!(K, Sleaf)[NO] = k_l
        Q_ratios[i] = ratio #stat
        i += 1  #stat
    end

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
            R_ratios[l] = Vector{Tuple{Int64,Float64}}() #stat
            R[l] = Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},Matrix{ComplexF64}}}()
        end
        # --------------------------------------------------------------
        # Build U (union of child skeletons)
        # --------------------------------------------------------------
        U = Dict{Int,Dict{Int,Vector{Int}}}()   #temporary unions
        if !source_is_frozen
            build_union_skeletons!(U, K, treeS, treeO, l, LS, LO, trialT)
        end

        # --------------------------------------------------------------
        # Compute R blocks
        # --------------------------------------------------------------
        if !source_is_frozen && !obs_is_frozen
            K_new = Dict{Int,Dict{Int,Vector{Int}}}()
            build_nonfrozen_R_blocks!(
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
                LS,
                R_ratios[l];
                scheduler,
            )
            K = K_new
        elseif source_is_frozen && !obs_is_frozen
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
                τ,
                R_ratios[l];
                scheduler,
            )
            K = K_new
        else #!source_is_frozen && obs_is_frozen
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
                LS,
                R_ratios[l];
                scheduler,
            )
            K = K_new
        end
    end

    # ------------------------------------------------------------------
    # Final P blocks
    # ------------------------------------------------------------------

    leaf_results = let K = K
        tmap(treeO[LO]; scheduler=scheduler) do Oleaf
            col = K[NS][Oleaf]
            row = values(testT, Oleaf)

            Z = zeros(ComplexF64, length(row), length(col))
            kernelmatrix(Z, row, col)
            perm_p_val = [obs_map[g] for g in row]
            return (Oleaf, perm_p_val, Z)
        end
    end
    for (Oleaf, perm_p_val, Z) in leaf_results
        PermP[Oleaf] = perm_p_val
        P[Oleaf] = Z
    end
    return BF(
        Q,
        R,
        P,
        PermQ,
        PermP,
        (length(global_obs_indices), length(global_src_indices)),
        NS,
        NO,
        k,
        τ,
        H2Blocktree.trialcluster,
        H2Blocktree.testcluster,
    ),
    BFSTAT(Q_ratios, R_ratios)  #stat
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
    LS,
    R_ratios;    #stat
    scheduler=OhMyThreads.SerialScheduler(),
)
    results = tmap(treeO[l]; scheduler=scheduler) do Overt
        # 1. Skapa lokala ordböcker
        local_R = Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},Matrix{ComplexF64}}}()
        local_K = Dict{Int,Dict{Int,Vector{Int}}}()
        local_ratios = Vector{Tuple{Int,Float64}}() #stat
        if !isleaf(testT, Overt)
            for Ochild in children(testT, Overt)
                obsindex = values(testT, Ochild)
                for Svert in treeS[LS - l]
                    srcindex = U[Svert][Overt]
                    n_otilde = estimate_rank_3d(k, trialT, testT, Svert, Ochild, τ;)
                    q_ks, k_l, r_l = compressor(
                        kernelmatrix, srcindex, obsindex, n_otilde, τ
                    )
                    push!(local_ratios, (length(obsindex), r_l / n_otilde)) #stat
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
                end
            end
        else
            obsindex = values(testT, Overt)
            for Svert in treeS[LS - l]
                if !haskey(local_R, (Overt, Svert))
                    local_R[(Overt, Svert)] = Dict{Tuple{Int,Int},Matrix{ComplexF64}}()
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
                    push!(local_ratios, (length(obsindex), r_l / n_otilde)) #stat
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
                        local_R[(Overt, Svert)][(Overt, Svert)] = Matrix{ComplexF64}(
                            I, 0, 0
                        )
                    else
                        local_R[(Overt, Svert)][(Overt, Svert)] = Matrix{ComplexF64}(
                            I, 0, 0
                        )
                    end
                    local_K[Svert][Overt] = K[Svert][Overt]
                end
            end
        end

        # 2. Returnera båda de lokala strukturerna
        return (local_R, local_K, local_ratios) #stat
    end

    # 3. Slå ihop all tråddata sekventiellt och säkert i huvudstrukturerna
    for (local_R, local_K, local_ratios) in results #stat
        append!(R_ratios, local_ratios) #stat
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
    τ,
    R_ratios;    #stat
    scheduler=OhMyThreads.SerialScheduler(),
)
    results = tmap(treeO[l]; scheduler=scheduler) do Overt
        local_R = Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},Matrix{ComplexF64}}}()
        local_K = Dict{Int,Dict{Int,Vector{Int}}}()
        local_ratios = Vector{Tuple{Int,Float64}}() #stat
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
                push!(local_ratios, (length(obsindex), r_l / n_otilde)) #stat
                local_R[(Ochild, Svert)][(Overt, Svert)] = q_ks
                local_K[Svert][Ochild] = k_l
            end
        else
            if !haskey(local_R, (Overt, Svert))
                local_R[(Overt, Svert)] = Dict{Tuple{Int,Int},Matrix{ComplexF64}}()
            end
            if l > 1
                local_R[(Overt, Svert)][(Overt, Svert)] = Matrix{ComplexF64}(I, 0, 0)
            else
                local_R[(Overt, Svert)][(Overt, Svert)] = Matrix{ComplexF64}(I, 0, 0)
            end
            local_K[Svert][Overt] = K[Svert][Overt]
        end
        return (local_R, local_K, local_ratios) #stat
    end

    # Slå ihop datan säkert
    for (local_R, local_K, local_ratios) in results
        append!(R_ratios, local_ratios) #stat
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
    LS,
    R_ratios;
    scheduler=OhMyThreads.SerialScheduler(),
)
    LO = length(treeO)
    LS = length(treeS)

    results = tmap(treeO[LO]; scheduler=scheduler) do Overt
        local_R = Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},Matrix{ComplexF64}}}()
        local_K = Dict{Int,Dict{Int,Vector{Int}}}()
        local_ratios = Vector{Tuple{Int,Float64}}() #stat
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
                push!(local_ratios, (length(obsindex), r_l / n_otilde)) #stat
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
        return (local_R, local_K, local_ratios) #stat
    end

    # Slå ihop datan säkert
    for (local_R, local_K, local_ratios) in results
        append!(R_ratios, local_ratios) #stat
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
