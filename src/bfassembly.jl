import H2Trees: values, center, halfsize, children, isleaf, trialtree, testtree

# ------------------------------------------------------------------
# Main Assembly Routine
# ------------------------------------------------------------------
"""
    assemble_BF(kernelmatrix, H2Blocktree, NO, NS, k, τ; Compressor=PartialQR())

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
  - `scheduler`: Threading scheduler for parallel execution (default: `SerialScheduler`).

**Why Dictionary format?** It is extremely memory efficient because it avoids the overhead
of saving nonzero entry indices required by sparse matrices. It is deeply intuitive, makes
algebraic manipulation (like butterfly multiplication or summation) flexible, and preserves
the semantic structure other than a matrix format. It is ideal when still working with the
tree structure is desired, such as in the case of recompression or algebraic operations on
the factors. However, it requires careful handling of index permutations to ensure correct
MV products, which is managed through the `PermQ` and `PermP` dictionaries.
"""
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
# ------------------------------------------------------------------
# Optimized Builder Functions
# ------------------------------------------------------------------

function build_nonfrozen_R_blocks(
    treeO_level, treeS_level, U, K, trialT, testT, kernelmatrix, compressor, k, τ, scheduler
)
    interactions = vec(collect(Iterators.product(treeO_level, treeS_level)))
    n_ints = length(interactions)

    # 1. Pre-calculate exact sizes using prefix sums (cumsum)
    block_counts = zeros(Int, n_ints)
    update_counts = zeros(Int, n_ints)
    for i in 1:n_ints
        Overt, Svert = interactions[i]
        num_O = isleaf(testT, Overt) ? 1 : length(collect(children(testT, Overt)))
        num_S = isleaf(trialT, Svert) ? 1 : length(collect(children(trialT, Svert)))
        block_counts[i] = num_O * num_S
        update_counts[i] = num_O
    end

    block_offsets = cumsum(vcat(1, block_counts[1:(end - 1)]))
    update_offsets = cumsum(vcat(1, update_counts[1:(end - 1)]))

    # 2. Allocate exactly once
    all_blocks = Vector{ButterflyBlock{ComplexF64}}(undef, sum(block_counts))
    all_k_updates = Vector{Tuple{Int,Int,Vector{Int}}}(undef, sum(update_counts))

    # 3. Threaded mapping directly into pre-allocated flat arrays
    tmap(1:n_ints; scheduler=scheduler) do i
        (Overt, Svert) = interactions[i]
        b_off = block_offsets[i] - 1
        u_off = update_offsets[i] - 1

        b_idx = 0
        u_idx = 0

        if !isleaf(testT, Overt)
            for Ochild in children(testT, Overt)
                obsindex = values(testT, Ochild)
                srcindex = U[(Svert, Overt)]
                n_otilde = estimate_rank_3d(k, trialT, testT, Svert, Ochild, τ)
                q_ks, k_l, _ = compressor(kernelmatrix, srcindex, obsindex, n_otilde, τ)

                u_idx += 1
                all_k_updates[u_off + u_idx] = (Svert, Ochild, k_l)

                if !isleaf(trialT, Svert)
                    last_idx = 0
                    for Schild in children(trialT, Svert)
                        ks = length(K[(Schild, Overt)])
                        block_data = Matrix(view(q_ks, :, (last_idx + 1):(last_idx + ks)))

                        b_idx += 1
                        all_blocks[b_off + b_idx] = ButterflyBlock(
                            Ochild, Svert, Overt, Schild, block_data
                        )
                        last_idx += ks
                    end
                else
                    b_idx += 1
                    all_blocks[b_off + b_idx] = ButterflyBlock(
                        Ochild, Svert, Overt, Svert, q_ks
                    )
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

                    b_idx += 1
                    all_blocks[b_off + b_idx] = ButterflyBlock(
                        Overt, Svert, Overt, Schild, block_data
                    )
                    last_idx += ks
                end

                u_idx += 1
                all_k_updates[u_off + u_idx] = (Svert, Overt, k_l)
            else
                b_idx += 1
                all_blocks[b_off + b_idx] = ButterflyBlock(Overt, Svert, Overt, Svert, I)

                u_idx += 1
                all_k_updates[u_off + u_idx] = (Svert, Overt, K[(Svert, Overt)])
            end
        end
    end

    sort!(all_blocks; by=block_key)
    return ButterflyLevel(all_blocks), all_k_updates
end

function build_sourcefrozen_R_blocks(
    treeO_level, NS, K, trialT, testT, kernelmatrix, compressor, k, τ, scheduler
)
    Svert = NS
    n_ints = length(treeO_level)

    block_counts = zeros(Int, n_ints)
    for i in 1:n_ints
        Overt = treeO_level[i]
        block_counts[i] = isleaf(testT, Overt) ? 1 : length(collect(children(testT, Overt)))
    end

    block_offsets = cumsum(vcat(1, block_counts[1:(end - 1)]))

    all_blocks = Vector{ButterflyBlock{ComplexF64}}(undef, sum(block_counts))
    all_k_updates = Vector{Tuple{Int,Int,Vector{Int}}}(undef, sum(block_counts))

    tmap(1:n_ints; scheduler=scheduler) do i
        Overt = treeO_level[i]
        b_off = block_offsets[i] - 1
        b_idx = 0

        if !isleaf(testT, Overt)
            for Ochild in children(testT, Overt)
                obsindex = values(testT, Ochild)
                srcindex = K[(Svert, Overt)]
                n_otilde = estimate_rank_3d(k, trialT, testT, Svert, Ochild, τ)
                q_ks, k_l, _ = compressor(kernelmatrix, srcindex, obsindex, n_otilde, τ)

                b_idx += 1
                all_blocks[b_off + b_idx] = ButterflyBlock(
                    Ochild, Svert, Overt, Svert, q_ks
                )
                all_k_updates[b_off + b_idx] = (Svert, Ochild, k_l)
            end
        else
            b_idx += 1
            all_blocks[b_off + b_idx] = ButterflyBlock(Overt, Svert, Overt, Svert, I)
            all_k_updates[b_off + b_idx] = (Svert, Overt, K[(Svert, Overt)])
        end
    end

    sort!(all_blocks; by=block_key)
    return ButterflyLevel(all_blocks), all_k_updates
end

function build_observerfrozen_R_blocks(
    treeO_LO, treeS_level, U, K, trialT, testT, kernelmatrix, compressor, k, τ, scheduler
)
    interactions = vec(collect(Iterators.product(treeO_LO, treeS_level)))
    n_ints = length(interactions)

    block_counts = zeros(Int, n_ints)
    for i in 1:n_ints
        Overt, Svert = interactions[i]
        block_counts[i] =
            isleaf(trialT, Svert) ? 1 : length(collect(children(trialT, Svert)))
    end

    block_offsets = cumsum(vcat(1, block_counts[1:(end - 1)]))

    all_blocks = Vector{ButterflyBlock{ComplexF64}}(undef, sum(block_counts))
    all_k_updates = Vector{Tuple{Int,Int,Vector{Int}}}(undef, n_ints)

    tmap(1:n_ints; scheduler=scheduler) do i
        (Overt, Svert) = interactions[i]
        b_off = block_offsets[i] - 1
        b_idx = 0
        obsindex = values(testT, Overt)

        if !isleaf(trialT, Svert)
            srcindex = U[(Svert, Overt)]
            n_otilde = estimate_rank_3d(k, trialT, testT, Svert, Overt, τ)
            q_ks, k_l, _ = compressor(kernelmatrix, srcindex, obsindex, n_otilde, τ)

            last_idx = 0
            for Schild in children(trialT, Svert)
                ks = length(K[(Schild, Overt)])
                block_data = Matrix(view(q_ks, :, (last_idx + 1):(last_idx + ks)))

                b_idx += 1
                all_blocks[b_off + b_idx] = ButterflyBlock(
                    Overt, Svert, Overt, Schild, block_data
                )
                last_idx += ks
            end
            all_k_updates[i] = (Svert, Overt, k_l)
        else
            b_idx += 1
            all_blocks[b_off + b_idx] = ButterflyBlock(Overt, Svert, Overt, Svert, I)
            all_k_updates[i] = (Svert, Overt, K[(Svert, Overt)])
        end
    end

    sort!(all_blocks; by=block_key)
    return ButterflyLevel(all_blocks), all_k_updates
end

"""
    assemble_ButterflyFactorization_Mat(kernelmatrix, H2Blocktree, NO, NS, k, τ; compressor=PartialQR())

Constructs the Butterfly Factorization for a given block in a **sparse matrix format**.

Similar to `assemble_BF`, it traverses the H2 tree structure to compute the `Q`, `R`,
and `P` factors, but specifically pieces them together into sparse block-diagonal matrices.
It also returns single continuous permutation vectors (`PermP`, `PermQ`) to map
interactions across the entire space.

**Why Matrix format?**
While slightly less memory efficient than the dictionary format (due to sparsity tracking
overhead), it allows for dramatically faster direct Matrix-Vector applications using
standard linear algebra methods. It also provides a clear visual and algebraic
representation of the overall block structure, which is invaluable for debugging.
"""
function assemble_ButterflyFactorization_Mat(
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
    return ButterflyFactorization_Mat(Q, R, P, NS, NO, τ, k, PermP, PermQ)
end
