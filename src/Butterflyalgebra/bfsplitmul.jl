"""
    splitmulbf(butterflycluster_init::Matrix{BF}, higherkBF_init ::ButterflyFactorization, τ::Float64) -> BF

Compute the operator product of a hierarchically divided matrix cluster of Butterflies
(level \$k\$) and a single, larger Butterfly Factorization block (level \$k+1\$) using
Heldring's block-splitting algorithm.

This function addresses the architectural challenge of multiplying non-uniform hierarchical
levels by subdividing the structural components of the larger factor to match the sparse
data layout of the smaller cluster blocks. It separately processes individual
sub-multiplications before realigning and reducing them back into a single unified
representation.

# Arguments

  - `butterflycluster_init::Matrix{BF}`: A matrix layout of lower-level butterfly factors
    representing the hierarchically split operator components.
  - `higherkBF_init ::ButterflyFactorization`: A single, un-split butterfly factorization at a higher
    hierarchical tree tier.
  - `τ::Float64`: Accuracy tolerance threshold passed down to internal multiplications
    (`mulBFs`) and the final accumulation stages (`add_eqbfs`).

# Returns

  - `BF`: A single consolidated and recompressed butterfly factorization representing the
    entire operator product.

# Algorithmic Steps

 1. **Factor Subdivision (Step 1):** The structural `P` and `R` dictionaries of the
    higher-tier operator are sliced into `numchildren` distinct independent BFs matching the
    target output subtree nodes (`children`).
 2. **Intermediate Block Multiplications:** Executes element-wise sub-multiplications
    between the cluster elements and the newly generated lower-tier factors via `mulBFs`.
 3. **Supertree Alignment & Coordinate Remapping:** Builds a global supertree reference to
    map differing index spaces. It shifts spatial/frequency cluster coordinates up to common
    parent frames and merges data blocks through horizontal/vertical accumulations and
    diagonal padding where key configurations overlap.
 4. **Hierarchical Reduction:** Consolidates the array of realigned intermediate structures
    down into a single final operator via sequential `add_eqbfs` calls.

# Notes

  - The original structural `Q` factors of the larger operator are intentionally preserved
    and held back until the final phase to maintain dimension consistency across structural
    modifications.
"""
function splitmulbf(
    bfcluster::Matrix{ButterflyFactorization{T,M}},
    higherkBF::ButterflyFactorization{T,M};
    τ=higherkBF.τ,
) where {T,M}

    # Step 1: Split the higher-level BF into child-level BFs
    srcchildren = [getNSNO(childbf)[1] for childbf in bfcluster[1, :]]
    bfdiagonal = splitobsside(higherkBF; obschildren=srcchildren)
    @assert size(bfcluster, 1) == size(bfcluster, 2) "bfcluster must be square --> no unbalanced interactions allowed"
    N = size(bfcluster, 1)
    # Step 2: Multiply each child BF with the corresponding cluster BF
    intermediate_bfs = Vector{Matrix{ButterflyFactorization{T,M}}}(undef, N)
    for i in 1:N
        intermediate_bfs[i] = Matrix{ButterflyFactorization{T,M}}(undef, N, N)
        for (j, child_bf) in enumerate(bfdiagonal)
            intermediate_bfs[mod(j - i, N) + 1][i, j] = mulBFs(
                bfcluster[i, j], child_bf; τ=higherkBF.τ
            )
        end
    end

    # Step 3: Align and merge the intermediate BFs into a single BF
    merged_bf = intermediate_bfs[1]
    for i in 2:length(intermediate_bfs)
        merged_bf = add_eqbfs(
            merged_bf, intermediate_bfs[i], max(merged_bf.τ, intermediate_bfs[i].τ)
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

function partialcleanupidxs(BF::ButterflyFactorization{T,M}) where {T,M}
    tsttree = cluster_testtree(BF.tree)
    trialtree = cluster_trialtree(BF.tree)
    L = length(BF.R)

    # =========================================================================
    # PASS 1: Forward Traversal (Trace Source / Trial Tree Keys from Q)
    # =========================================================================
    src_translation = Dict{Tuple{Int,Int},Int}()
    # Preallocate storage for the computed source keys in R
    src_keys_R = [Vector{Tuple{Int,Int}}(undef, length(BF.R[l].blocks)) for l in 1:L]

    for l in 1:L
        next_src_translation = Dict{Tuple{Int,Int},Int}()
        for (i, b) in enumerate(BF.R[l].blocks)
            if haskey(src_translation, (b.obs_in, b.src_in))
                new_src_in = src_translation[(b.obs_in, b.src_in)]
            else
                new_src_in = b.src_in
            end
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

    # Rebuild P instantly and seed translation
    newP = Vector{ButterflyBlock{T}}(undef, length(BF.P))
    for (i, b) in enumerate(BF.P)
        true_obs_out = b.obs_out # P's output is the physical observer leaf
        true_obs_in = b.obs_in # Move 1 level up the observer tree
        root_src = src_translation[(b.obs_in, b.src_in)] # P's input is the root of the source tree
        newP[i] = ButterflyBlock(
            true_obs_out, root_src, true_obs_in, root_src, copy(b.data)
        )

        # Save P's input mapping to feed R_L
        obs_translation[(b.obs_in, b.src_in)] = true_obs_in
    end

    # Reconstruct R backwards, combining obs logic with stored src keys
    newR = Vector{ButterflyLevel{T}}(undef, L)
    for l in L:-1:1
        next_obs_translation = Dict{Tuple{Int,Int},Int}()
        new_blocks = Vector{ButterflyBlock{T}}(undef, length(BF.R[l].blocks))

        for (i, b) in enumerate(BF.R[l].blocks)
            # 1. Compute the new Observer keys
            new_obs_out = obs_translation[(b.obs_out, b.src_out)]
            new_obs_in = cluster_parent(tsttree, new_obs_out)

            # 2. Retrieve the new Source keys computed in Pass 1
            new_src_out, new_src_in = src_keys_R[l][i]

            # 3. Reconstruct block on the spot!
            new_blocks[i] = ButterflyBlock(
                new_obs_out, new_src_out, new_obs_in, new_src_in, copy(b.data)
            )

            # Propagate the obs_in upwards to R_{l-1}
            next_obs_translation[(b.obs_in, b.src_in)] = new_obs_in
        end
        newR[l] = ButterflyLevel(new_blocks)
        obs_translation = next_obs_translation
    end

    newQ = Vector{ButterflyBlock{T}}(undef, length(BF.Q))
    for (i, b) in enumerate(BF.Q)
        true_obs_out = obs_translation[(b.obs_out, b.src_out)]
        true_obs_in = true_obs_out

        newQ[i] = ButterflyBlock(
            true_obs_out, b.src_out, true_obs_in, b.src_in, copy(b.data)
        )
    end

    return ButterflyFactorization(newQ, newR, newP, BF.tree, BF.k, BF.τ)
end
