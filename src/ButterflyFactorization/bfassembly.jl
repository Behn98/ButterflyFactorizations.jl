"""
    fastkey(s::Int, o::Int) -> Int

Generates a unique, highly efficient 64-bit integer key for a given source (`s`) and
observer (`o`) node pair using bitwise shifting. This avoids the allocation overhead
of using `Tuple{Int, Int}` as dictionary keys during assembly.
"""
@inline function fastkey(s::Int, o::Int)::Int
    return (s << 32) | (o & 0xFFFFFFFF)
end

# ------------------------------------------------------------------
# Main Assembly Routine
# ------------------------------------------------------------------
"""
    assemble_BF(kernelmatrix, blktree, no, ns, k, τ; kwargs...)

Constructs the Butterfly Factorization for a given block in a **dictionary format**.

This subroutine traverses the H2 tree structure from the leaf level moving up to the root of
the source tree, and to the leaves of the observer tree. It computes the low-rank
approximations to build the `Q`, `R`, and `P` factors.

To bound the global approximation error, the user-provided tolerance `τ` is internally
scaled by the maximum tree depth `L` (`τ / L`) to account for error accumulation across
the cascaded factor multiplications.

**Arguments:**

  - `kernelmatrix`: Function computing matrix entries for specified row/column indices.
  - `blktree`: The paired source-observer tree structure.
  - `no`, `ns`: The root IDs of the observer (test) and source (trial) spaces.
  - `k`, `τ`: Wavenumber and relative precision tolerance.

**Keyword Arguments:**

  - `compressor`: Compression scheme for low-rank blocks (default: `PartialQR()`).
  - `scheduler`: Threading scheduler for parallel execution.
  - `acctype`: The numeric type of the matrix elements (default: `ComplexF64`).
  - `admissibility`: The geometric admissibility functor (default: `isFarFunctor`).
  - `rankestimator`: The rank estimation strategy (default: `GeometricRankEstimator`).
  - `adaptive`: Whether to adaptively determine the rank during compression (default: `false`).

**Returns:**

  - A `ButterflyFactorization` object containing the optimized flat-array blocks.
"""
function assemble_BF(
    kernelmatrix,
    blktree,
    no::Int,
    ns::Int,
    k::Float64,
    τ::Float64;
    rankestimator::AbstractRankEstimator=ButterflyRankEstimator(
        tree_parameters(blktree).Cτ
    ),
    adaptive=true,
    compressor=ButterflyFactorizations.PartialQR(),
    scheduler=OhMyThreads.StaticScheduler(),
    acctype=ComplexF64,
)
    # --- Trees & Helpers ---
    trialT = cluster_trialtree(blktree)
    testT = cluster_testtree(blktree)
    treeS = traverseandpad(trialT, ns)
    treeO = traverseandpad(testT, no)

    LS = length(treeS)
    LO = length(treeO)
    L = max(LS, LO)

    # Scale error tolerance to account for accumulation over deep tree levels
    τ_scaled = τ / L

    # Temporary workspace for skeletons: K[(Snode_id, Onode_id)] => Vector{Int}
    K = Dict{Int,Vector{Int}}()

    # ------------------------------------------------------------------
    # 1. Leaf-level Q
    # ------------------------------------------------------------------
    lS = length(treeS[end])
    Q = Vector{ButterflyBlock{acctype}}(undef, lS)

    leaf_q_results = tmap(1:lS; scheduler=scheduler) do blockidx
        Sleaf = treeS[end][blockidx]
        srcindex = cluster_values(trialT, Sleaf)
        obsindex = cluster_values(testT, no)

        n_otilde = rankestimator(k, trialT, testT, Sleaf, no, τ_scaled)
        q_ks, k_l, _ = compressor(
            kernelmatrix, srcindex, obsindex, n_otilde, τ_scaled; adaptive=adaptive
        )

        return (blockidx, Sleaf, q_ks, k_l)
    end

    # Safely update Q and K sequentially after threading
    for (blockidx, Sleaf, q_ks, k_l) in leaf_q_results
        Q[blockidx] = ButterflyBlock(no, Sleaf, no, Sleaf, q_ks)
        K[fastkey(Sleaf, no)] = k_l
    end

    # ------------------------------------------------------------------
    # 2. Level Traversal (R Blocks)
    # ------------------------------------------------------------------
    R = Vector{ButterflyLevel{acctype}}(undef, L - 1)

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
                τ_scaled,
                scheduler,
                rankestimator,
                acctype;
                adaptive=adaptive,
            )
        elseif source_is_frozen && !obs_is_frozen
            level_R, k_updates = build_sourcefrozen_R_blocks(
                treeO[l],
                ns,
                K,
                trialT,
                testT,
                kernelmatrix,
                compressor,
                k,
                τ_scaled,
                scheduler,
                rankestimator,
                acctype;
                adaptive=adaptive,
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
                τ_scaled,
                scheduler,
                rankestimator,
                acctype;
                adaptive=adaptive,
            )
        end

        R[l] = level_R

        # Sequentially apply skeleton updates between levels
        for (s, o, k_l) in k_updates
            K[fastkey(s, o)] = k_l
        end
    end

    # ------------------------------------------------------------------
    # 3. Final P blocks
    # ------------------------------------------------------------------
    lO = length(treeO[end])
    P = Vector{ButterflyBlock{acctype}}(undef, lO)

    tmap(1:lO; scheduler=scheduler) do idx
        Oleaf = treeO[end][idx]
        col = K[fastkey(ns, Oleaf)]
        row = cluster_values(testT, Oleaf)
        Z = zeros(acctype, length(row), length(col))
        kernelmatrix(Z, row, col)

        return P[idx] = ButterflyBlock(Oleaf, ns, Oleaf, ns, Z)
    end

    return ButterflyFactorization(Q, R, P, blktree, k, τ)
end

# ------------------------------------------------------------------
# Builder Functions
# ------------------------------------------------------------------

function build_U_skeletons(treeS_level, treeO_level, K, trialT, scheduler)
    U = Dict{Int,Vector{Int}}()
    interactions_U = vec(collect(Iterators.product(treeS_level, treeO_level)))

    results_U = tmap(interactions_U; scheduler=scheduler) do (Svert, Overt)
        if !cluster_isleaf(trialT, Svert)
            temp_size = sum(
                length(K[fastkey(Schild, Overt)]) for
                Schild in cluster_children(trialT, Svert)
            )
            temp = sizehint!(Int[], temp_size)
            for Schild in cluster_children(trialT, Svert)
                append!(temp, K[fastkey(Schild, Overt)])
            end
        else
            temp = K[fastkey(Svert, Overt)]
        end
        return (Svert, Overt, temp)
    end

    for (s, o, temp_skel) in results_U
        U[fastkey(s, o)] = temp_skel
    end
    return U
end

function build_nonfrozen_R_blocks(
    treeO_level,
    treeS_level,
    U,
    K,
    trialT,
    testT,
    kernelmatrix,
    compressor,
    k,
    τ,
    scheduler,
    rankestimator,
    acctype;
    adaptive=false,
)
    interactions = vec(collect(Iterators.product(treeO_level, treeS_level)))
    n_ints = length(interactions)

    block_counts = zeros(Int, n_ints)
    update_counts = zeros(Int, n_ints)
    for i in 1:n_ints
        Overt, Svert = interactions[i]
        num_O = if cluster_isleaf(testT, Overt)
            1
        else
            length(collect(cluster_children(testT, Overt)))
        end
        num_S = if cluster_isleaf(trialT, Svert)
            1
        else
            length(collect(cluster_children(trialT, Svert)))
        end

        block_counts[i] = num_O * num_S
        update_counts[i] = num_O
    end

    block_offsets = cumsum(vcat(1, block_counts[1:(end - 1)]))
    update_offsets = cumsum(vcat(1, update_counts[1:(end - 1)]))

    all_blocks = Vector{ButterflyBlock{acctype}}(undef, sum(block_counts))
    all_k_updates = Vector{Tuple{Int,Int,Vector{Int}}}(undef, sum(update_counts))

    tmap(1:n_ints; scheduler=scheduler) do i
        (Overt, Svert) = interactions[i]
        b_off = block_offsets[i] - 1
        u_off = update_offsets[i] - 1

        b_idx = 0
        u_idx = 0

        if !cluster_isleaf(testT, Overt)
            for Ochild in cluster_children(testT, Overt)
                obsindex = cluster_values(testT, Ochild)
                srcindex = U[fastkey(Svert, Overt)]
                n_otilde = rankestimator(k, trialT, testT, Svert, Ochild, τ)
                q_ks, k_l, _ = compressor(
                    kernelmatrix, srcindex, obsindex, n_otilde, τ; adaptive=adaptive
                )

                u_idx += 1
                all_k_updates[u_off + u_idx] = (Svert, Ochild, k_l)

                if !cluster_isleaf(trialT, Svert)
                    last_idx = 0
                    for Schild in cluster_children(trialT, Svert)
                        ks = length(K[fastkey(Schild, Overt)])
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
            obsindex = cluster_values(testT, Overt)
            if !cluster_isleaf(trialT, Svert)
                srcindex = U[fastkey(Svert, Overt)]
                n_otilde = rankestimator(k, trialT, testT, Svert, Overt, τ)
                q_ks, k_l, _ = compressor(
                    kernelmatrix, srcindex, obsindex, n_otilde, τ; adaptive=adaptive
                )

                last_idx = 0
                for Schild in cluster_children(trialT, Svert)
                    ks = length(K[fastkey(Schild, Overt)])
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
                all_k_updates[u_off + u_idx] = (Svert, Overt, K[fastkey(Svert, Overt)])
            end
        end
    end

    return ButterflyLevel(all_blocks), all_k_updates
end

function build_sourcefrozen_R_blocks(
    treeO_level,
    ns,
    K,
    trialT,
    testT,
    kernelmatrix,
    compressor,
    k,
    τ,
    scheduler,
    rankestimator,
    acctype;
    adaptive=false,
)
    Svert = ns
    n_ints = length(treeO_level)

    block_counts = zeros(Int, n_ints)
    for i in 1:n_ints
        Overt = treeO_level[i]
        block_counts[i] = if cluster_isleaf(testT, Overt)
            1
        else
            length(collect(cluster_children(testT, Overt)))
        end
    end

    block_offsets = cumsum(vcat(1, block_counts[1:(end - 1)]))

    all_blocks = Vector{ButterflyBlock{acctype}}(undef, sum(block_counts))
    all_k_updates = Vector{Tuple{Int,Int,Vector{Int}}}(undef, sum(block_counts))

    tmap(1:n_ints; scheduler=scheduler) do i
        Overt = treeO_level[i]
        b_off = block_offsets[i] - 1
        b_idx = 0

        if !cluster_isleaf(testT, Overt)
            for Ochild in cluster_children(testT, Overt)
                obsindex = cluster_values(testT, Ochild)
                srcindex = K[fastkey(Svert, Overt)]
                n_otilde = rankestimator(k, trialT, testT, Svert, Ochild, τ)
                q_ks, k_l, _ = compressor(
                    kernelmatrix, srcindex, obsindex, n_otilde, τ; adaptive=adaptive
                )

                b_idx += 1
                all_blocks[b_off + b_idx] = ButterflyBlock(
                    Ochild, Svert, Overt, Svert, q_ks
                )
                all_k_updates[b_off + b_idx] = (Svert, Ochild, k_l)
            end
        else
            b_idx += 1
            all_blocks[b_off + b_idx] = ButterflyBlock(Overt, Svert, Overt, Svert, I)
            all_k_updates[b_off + b_idx] = (Svert, Overt, K[fastkey(Svert, Overt)])
        end
    end

    return ButterflyLevel(all_blocks), all_k_updates
end

function build_observerfrozen_R_blocks(
    treeO_LO,
    treeS_level,
    U,
    K,
    trialT,
    testT,
    kernelmatrix,
    compressor,
    k,
    τ,
    scheduler,
    rankestimator,
    acctype;
    adaptive=false,
)
    interactions = vec(collect(Iterators.product(treeO_LO, treeS_level)))
    n_ints = length(interactions)

    block_counts = zeros(Int, n_ints)
    for i in 1:n_ints
        Overt, Svert = interactions[i]
        block_counts[i] = if cluster_isleaf(trialT, Svert)
            1
        else
            length(collect(cluster_children(trialT, Svert)))
        end
    end

    block_offsets = cumsum(vcat(1, block_counts[1:(end - 1)]))

    all_blocks = Vector{ButterflyBlock{acctype}}(undef, sum(block_counts))
    all_k_updates = Vector{Tuple{Int,Int,Vector{Int}}}(undef, n_ints)

    tmap(1:n_ints; scheduler=scheduler) do i
        (Overt, Svert) = interactions[i]
        b_off = block_offsets[i] - 1
        b_idx = 0
        obsindex = cluster_values(testT, Overt)

        if !cluster_isleaf(trialT, Svert)
            srcindex = U[fastkey(Svert, Overt)]
            n_otilde = rankestimator(k, trialT, testT, Svert, Overt, τ)
            q_ks, k_l, _ = compressor(
                kernelmatrix, srcindex, obsindex, n_otilde, τ; adaptive=adaptive
            )

            last_idx = 0
            for Schild in cluster_children(trialT, Svert)
                ks = length(K[fastkey(Schild, Overt)])
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
            all_k_updates[i] = (Svert, Overt, K[fastkey(Svert, Overt)])
        end
    end

    return ButterflyLevel(all_blocks), all_k_updates
end

"""
    assemble_BF_Mat(kernelmatrix, blktree, no, ns, k, τ; kwargs...)

Constructs the Butterfly Factorization for a given block in a **sparse matrix format**.

Similar to `assemble_BF`, it traverses the tree structure to compute the `Q`, `R`,
and `P` factors, but specifically pieces them together into sparse block-diagonal matrices.
It also returns single continuous permutation vectors (`permP`, `permQ`) to map
interactions across the entire physical space.

**Why Matrix format?**
While slightly less memory efficient than the flat dictionary format (due to sparsity tracking
overhead), it allows for dramatically faster direct Matrix-Vector applications using
standard linear algebra methods, providing a clear visual and algebraic representation of the
overall block structure for debugging.
"""
function assemble_BF_Mat(
    kernelmatrix,
    blktree,
    no::Int,
    ns::Int,
    k::Float64,
    τ::Float64;
    adaptive=true,
    compressor=ButterflyFactorizations.PartialQR(),
    C=tree_parameters(cluster_testtree(blktree)).C,
    Cε=tree_parameters(cluster_testtree(blktree)).Cε,
    rankestimator::AbstractRankEstimator=ButterflyRankEstimator(
        tree_parameters(blktree).Cτ
    ),
    acctype=ComplexF64,
)
    Q = Matrix{acctype}(undef, 0, 0)
    R = Vector{AbstractMatrix{acctype}}()
    P = Matrix{acctype}(undef, 0, 0)
    K = Dict{Int,Dict{Int,Vector{Int}}}()
    U = Dict{Int,Dict{Int,Vector{Int}}}()
    permQ = Vector{Int}()
    permP = Vector{Int}()

    trialT = cluster_trialtree(blktree)
    testT = cluster_testtree(blktree)

    treeS = traverseandpad(trialT, ns)
    treeO = traverseandpad(testT, no)

    LS = length(treeS)
    LO = length(treeO)
    L = LS + LO
    τ_scaled = τ / max(LS, LO)

    # ------------------------------------------------------------------
    # Leaf-level Q
    # ------------------------------------------------------------------
    for Sleaf in treeS[LS]
        srcindex = cluster_values(trialT, Sleaf)
        push!(permQ, srcindex...)
        obsindex = cluster_values(testT, no)
        n_otilde = rankestimator(k, trialT, testT, Sleaf, no, τ_scaled)
        q_ks, k_l, r_l = compressor(
            kernelmatrix, srcindex, obsindex, n_otilde, τ_scaled; adaptive=adaptive
        )
        Q = sparse_blockdiag(Q, q_ks)
        getsubdict!(K, Sleaf)[no] = k_l
    end

    source_is_frozen = false
    obs_is_frozen = false

    # ------------------------------------------------------------------
    # Level traversal
    # ------------------------------------------------------------------
    for l in 1:(L - 1)
        l >= LS && (source_is_frozen = true)
        l >= LO && (obs_is_frozen = true)

        if !source_is_frozen
            for Svert in treeS[LS - l]
                U_S = getsubdict!(U, Svert)

                for Overt in treeO[min(l, LO)]
                    temp = Int[]
                    for Schild in cluster_children(trialT, Svert)
                        Ks = getsubdict!(K, Schild)
                        ks = get(Ks, Overt, nothing)
                        append!(temp, ks)
                    end
                    U_S[Overt] = temp
                end
            end
        end

        if !source_is_frozen && !obs_is_frozen
            R_temp1 = Matrix{acctype}(undef, 0, 0)
            for Overt in treeO[l]
                R_temp2 = Vector{AbstractMatrix{acctype}}()
                for Ochild in cluster_children(testT, Overt)
                    R_temp3 = Matrix{acctype}(undef, 0, 0)
                    obsindex = cluster_values(testT, Ochild)
                    for Svert in treeS[LS - l]
                        srcindex = U[Svert][Overt]

                        n_otilde = rankestimator(k, trialT, testT, Svert, Ochild, τ_scaled)
                        q_ks, k_l, r_l = compressor(
                            kernelmatrix,
                            srcindex,
                            obsindex,
                            n_otilde,
                            τ_scaled;
                            adaptive=adaptive,
                        )
                        R_temp3 = sparse_blockdiag(R_temp3, q_ks)
                        getsubdict!(K, Svert)[Ochild] = k_l
                    end
                    push!(R_temp2, R_temp3)
                end
                R_temp1 = sparse_blockdiag(R_temp1, sparse_vcat(R_temp2...))
            end
            push!(R, R_temp1)

        elseif source_is_frozen && !obs_is_frozen
            R_temp1 = Matrix{acctype}(undef, 0, 0)
            for Overt in treeO[l]
                R_temp2 = Vector{AbstractMatrix{acctype}}()
                for Ochild in cluster_children(testT, Overt)
                    R_temp3 = Matrix{acctype}(undef, 0, 0)
                    obsindex = cluster_values(testT, Ochild)
                    for Svert in treeS[1]
                        srcindex = K[Svert][Overt]
                        n_otilde = rankestimator(k, trialT, testT, Svert, Ochild, τ_scaled)
                        q_ks, k_l, r_l = compressor(
                            kernelmatrix,
                            srcindex,
                            obsindex,
                            n_otilde,
                            τ_scaled;
                            adaptive=adaptive,
                        )
                        R_temp3 = sparse_blockdiag(R_temp3, q_ks)
                        getsubdict!(K, Svert)[Ochild] = k_l
                    end
                    push!(R_temp2, R_temp3)
                end
                R_temp1 = sparse_blockdiag(R_temp1, sparse_vcat(R_temp2...))
            end
            push!(R, R_temp1)

        elseif !source_is_frozen && obs_is_frozen
            R_temp1 = Matrix{acctype}(undef, 0, 0)
            for Overt in treeO[LO]
                obsindex = cluster_values(testT, Overt)
                R_temp2 = Matrix{acctype}(undef, 0, 0)
                for Svert in treeS[LS - l]
                    srcindex = U[Svert][Overt]

                    n_otilde = rankestimator(k, trialT, testT, Svert, Overt, τ_scaled)
                    q_ks, k_l, r_l = compressor(
                        kernelmatrix,
                        srcindex,
                        obsindex,
                        n_otilde,
                        τ_scaled;
                        adaptive=adaptive,
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
        col = K[ns][Oleaf]
        row = cluster_values(testT, Oleaf)
        push!(permP, row...)
        Z = zeros(acctype, length(row), length(col))
        kernelmatrix(Z, row, col)
        P = sparse_blockdiag(P, Z)
    end

    return ButterflyFactorization_Mat(Q, R, P, ns, no, k, τ, permP, permQ)
end
