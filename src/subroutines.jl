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
  - `Compressor`: Compression scheme for low-rank blocks (default: `PartialQR`).

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
    Compressor=ButterflyFactorizations.PartialQR(),
)

    # --- containers ---
    Q = Dict{Int,Matrix{ComplexF64}}()
    K = Dict{Int,Dict{Int,Vector{Int}}}()
    PermQ = Dict{Int,Vector{Int}}()
    PermP = Dict{Int,Vector{Int}}()
    # --- trees & helpers ---
    trialT = H2Trees.trialtree(H2Blocktree)
    testT = H2Trees.testtree(H2Blocktree)

    values = H2Trees.values
    center = H2Trees.center
    halfsize = H2Trees.halfsize
    children = H2Trees.children

    treeS = traverseandpad(trialT, NS)
    treeO = traverseandpad(testT, NO)

    LS = length(treeS)
    LO = length(treeO)
    L = max(LS, LO)
    R = Vector{Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},AbstractMatrix{ComplexF64}}}}(
        undef, L - 1
    )
    # 1. Get all global indices for this block's source/observer trees
    global_src_indices = values(trialT, NS) # All sources for this block
    global_obs_indices = values(testT, NO)  # All observers for this block

    # 2. Create a mapping from global to local (1 to N)
    src_map = Dict(g => l for (l, g) in enumerate(global_src_indices))
    obs_map = Dict(g => l for (l, g) in enumerate(global_obs_indices))

    # ------------------------------------------------------------------
    # Leaf-level Q
    # ------------------------------------------------------------------
    for Sleaf in treeS[LS]  #--> watchout this does not take account of leaves being on
        #higher levels, but we assume the tree is balanced enough that this is not a problem
        srcindex = values(trialT, Sleaf)
        PermQ[Sleaf] = [src_map[g] for g in srcindex]
        obsindex = values(testT, NO)
        c_s = center(trialT, Sleaf)
        c_o = center(testT, NO)
        a_s = halfsize(trialT, Sleaf)
        a_o = halfsize(testT, NO)
        n_otilde = estimate_rank_3d(k, c_s, c_o, a_s, a_o, τ, ; C=1.0, Cε=3.0, Rmin=5)
        q_ks, k_l, r_l = Compressor(kernelmatrix, srcindex, obsindex, n_otilde, τ)
        Q[Sleaf] = q_ks
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
            R[l] = Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},AbstractMatrix{ComplexF64}}}()
        end
        # --------------------------------------------------------------
        # Build U (union of child skeletons)
        # --------------------------------------------------------------
        U = Dict{Int,Dict{Int,Vector{Int}}}()   #temporary unions
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
            rowsizeR = 0
            for Overt in treeO[l]
                for Ochild in children(testT, Overt)
                    obsindex = values(testT, Ochild)
                    c_o = center(testT, Ochild)
                    a_o = halfsize(testT, Ochild)
                    isempty(obsindex) && continue
                    for Svert in treeS[LS - l]
                        srcindex = U[Svert][Overt]
                        c_s = center(trialT, Svert)
                        a_s = halfsize(trialT, Svert)
                        n_otilde = estimate_rank_3d(
                            k, c_s, c_o, a_s, a_o, τ, ; C=1.0, Cε=3.0, Rmin=5
                        )
                        q_ks, k_l, r_l = Compressor(
                            kernelmatrix, srcindex, obsindex, n_otilde, τ
                        )
                        last = 0
                        for Schild in children(trialT, Svert)
                            ks = length(getsubdict!(K, Schild)[Overt])
                            getsubdict!(R[l], (Ochild, Svert))[(Overt, Schild)] = q_ks[
                                :, (last + 1):(last + ks)
                            ]
                            last += ks
                        end
                        getsubdict!(K, Svert)[Ochild] = k_l
                    end
                end
            end

        elseif source_is_frozen && !obs_is_frozen
            for Overt in treeO[l]
                for Ochild in children(testT, Overt)
                    obsindex = values(testT, Ochild)
                    c_o = center(testT, Ochild)
                    a_o = halfsize(testT, Ochild)
                    for Svert in treeS[1]
                        srcindex = K[Svert][Overt]
                        c_s = center(trialT, Svert)
                        a_s = halfsize(trialT, Svert)
                        n_otilde = estimate_rank_3d(
                            k, c_s, c_o, a_s, a_o, τ, ; C=1.0, Cε=3.0, Rmin=5
                        )
                        q_ks, k_l, r_l = Compressor(
                            kernelmatrix, srcindex, obsindex, n_otilde, τ
                        )
                        last = 0
                        getsubdict!(R[l], (Ochild, Svert))[(Overt, Svert)] = q_ks
                        getsubdict!(K, Svert)[Ochild] = k_l
                    end
                end
            end

        elseif !source_is_frozen && obs_is_frozen
            for Overt in treeO[LO]
                obsindex = values(testT, Overt)
                c_o = center(testT, Overt)
                a_o = halfsize(testT, Overt)
                for Svert in treeS[LS - l]
                    srcindex = U[Svert][Overt]
                    c_s = center(trialT, Svert)
                    a_s = halfsize(trialT, Svert)
                    n_otilde = estimate_rank_3d(
                        k, c_s, c_o, a_s, a_o, τ, ; C=1.0, Cε=3.0, Rmin=5
                    )
                    q_ks, k_l, r_l = Compressor(
                        kernelmatrix, srcindex, obsindex, n_otilde, τ
                    )

                    last = 0
                    for Schild in children(trialT, Svert)
                        ks = length(getsubdict!(K, Schild)[Overt])
                        getsubdict!(R[l], (Overt, Svert))[(Overt, Schild)] = q_ks[
                            :, (last + 1):(last + ks)
                        ]
                        last += ks
                    end
                    getsubdict!(K, Svert)[Overt] = k_l
                end
            end

        else
            break
        end
    end

    # ------------------------------------------------------------------
    # Final P blocks
    # ------------------------------------------------------------------
    P = Dict{Int,Matrix{ComplexF64}}()
    for Oleaf in treeO[LO]
        col = K[NS][Oleaf]
        row = values(testT, Oleaf)

        Z = zeros(ComplexF64, length(row), length(col))
        kernelmatrix(Z, row, col)
        #PermP[Oleaf] = row
        PermP[Oleaf] = [obs_map[g] for g in row]
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
    )
end

"""
While the former version works completely fine for balanced trees of arbitrary height which
can also vary between source and test tree as long as all leaves can be found on leaf level,
as of now this version of the subroutine is still a WIP to handle the case of unblanaced
treees. However it is basically inserting ghost nodes to make the tree balanced, and then
skipping the empty nodes during the traversal. Thus in the usual case one would want to
simply use a balanced tree with the desired aspect as discussed.

"""

function subroutine_BF_pruned(
    kernelmatrix,
    H2Blocktree,
    NO::Int,
    NS::Int,
    k::Float64,
    τ::Float64;
    Compressor=ButterflyFactorizations.PartialQR(),
)

    # --- containers ---
    Q = Dict{Int,Matrix{ComplexF64}}()
    K = Dict{Int,Dict{Int,Vector{Int}}}()

    PermQ = Dict{Int,Vector{Int}}()
    PermP = Dict{Int,Vector{Int}}()
    # --- trees & helpers ---
    trialT = H2Trees.trialtree(H2Blocktree)
    testT = H2Trees.testtree(H2Blocktree)

    values = H2Trees.values
    children = H2Trees.children
    isleaf = H2Trees.isleaf

    treeS = traverseandpad(trialT, NS)
    treeO = traverseandpad(testT, NO)

    LS = length(treeS)
    LO = length(treeO)
    L = max(LS, LO)
    R = Vector{Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},AbstractMatrix{ComplexF64}}}}(
        undef, L - 1
    )
    # 1. Get all global indices for this block's source/observer trees
    global_src_indices = values(trialT, NS) # All sources for this block
    global_obs_indices = values(testT, NO)  # All observers for this block

    # 2. Create a mapping from global to local (1 to N)
    src_map = Dict(g => l for (l, g) in enumerate(global_src_indices))
    obs_map = Dict(g => l for (l, g) in enumerate(global_obs_indices))

    # ------------------------------------------------------------------
    # Leaf-level Q
    # ------------------------------------------------------------------
    for Sleaf in treeS[LS]  #--> watchout this does not take account of leaves being on
        #higher levels, but we assume the tree is balanced enough that this is not a problem
        srcindex = values(trialT, Sleaf)
        obsindex = values(testT, NO)
        PermQ[Sleaf] = [src_map[g] for g in srcindex]
        n_otilde = estimate_rank_3d(k, trialT, testT, Sleaf, NO, τ; C=1.0, Cε=3.0, Rmin=3)
        q_ks, k_l, r_l = Compressor(kernelmatrix, srcindex, obsindex, n_otilde, τ)

        Q[Sleaf] = q_ks
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
            R[l] = Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},AbstractMatrix{ComplexF64}}}()
        end
        # --------------------------------------------------------------
        # Build U (union of child skeletons)
        # --------------------------------------------------------------
        U = Dict{Int,Dict{Int,Vector{Int}}}()   #temporary unions
        if !source_is_frozen
            for Svert in treeS[LS - l]
                U_S = getsubdict!(U, Svert)

                for Overt in treeO[min(l, LO)]
                    temp = Int[]
                    if !isleaf(trialT, Svert)
                        for Schild in children(trialT, Svert)
                            Ks = getsubdict!(K, Schild)
                            ks = get(Ks, Overt, nothing)
                            if ks === nothing
                                continue
                            end
                            append!(temp, ks)
                        end
                    else
                        temp = K[Svert][Overt]
                    end

                    U_S[Overt] = temp
                end
            end
        end

        # --------------------------------------------------------------
        # Compute R blocks
        # --------------------------------------------------------------
        if !source_is_frozen && !obs_is_frozen
            for Overt in treeO[l]
                if !isleaf(testT, Overt)
                    for Ochild in children(testT, Overt)
                        obsindex = values(testT, Ochild)
                        isempty(obsindex) && continue
                        for Svert in treeS[LS - l]
                            srcindex = U[Svert][Overt]
                            if isempty(srcindex)
                                @show "Warning: empty source index for Svert $Svert and
                                Ochild $Overt at level $l. This should not happen if
                                the tree is properly balanced."
                                continue
                            end
                            n_otilde = estimate_rank_3d(
                                k, trialT, testT, Svert, Ochild, τ; C=1.0, Cε=3.0, Rmin=3
                            )

                            q_ks, k_l, r_l = Compressor(
                                kernelmatrix, srcindex, obsindex, n_otilde, τ
                            )
                            if !isleaf(trialT, Svert)
                                last = 0
                                for Schild in children(trialT, Svert)
                                    if !haskey(K, Schild) || !haskey(K[Schild], Overt)
                                        @show "Warning: missing K entry for Schild $Schild
                                        and Overt $Overt at level $l. This should not happen
                                        if the tree is properly balanced."
                                        continue
                                    end
                                    ks = length(getsubdict!(K, Schild)[Overt])
                                    getsubdict!(R[l], (Ochild, Svert))[(Overt, Schild)] = q_ks[
                                        :, (last + 1):(last + ks)
                                    ]
                                    last += ks
                                end

                            else
                                getsubdict!(R[l], (Ochild, Svert))[(Overt, Svert)] = q_ks
                            end
                            getsubdict!(K, Svert)[Ochild] = k_l
                        end
                    end
                else
                    obsindex = values(testT, Overt)
                    for Svert in treeS[LS - l]
                        if !isleaf(trialT, Svert)
                            srcindex = U[Svert][Overt]
                            if isempty(srcindex)
                                @show "Warning: empty source index for Svert $Svert and
                                Ochild $Overt at level $l. This should not happen if the
                                tree is properly balanced."
                                continue
                            end
                            n_otilde = estimate_rank_3d(
                                k, trialT, testT, Svert, Overt, τ; C=1.0, Cε=3.0, Rmin=3
                            )
                            q_ks, k_l, r_l = Compressor(
                                kernelmatrix, srcindex, obsindex, n_otilde, τ
                            )
                            last = 0
                            for Schild in children(trialT, Svert)
                                if !haskey(K, Schild) || !haskey(K[Schild], Overt)
                                    @show "Warning: missing K entry for Schild $Schild and
                                    Overt $Overt at level $l. This should not happen if the
                                    tree is properly balanced."
                                    continue
                                end
                                ks = length(getsubdict!(K, Schild)[Overt])
                                getsubdict!(R[l], (Overt, Svert))[(Overt, Schild)] = q_ks[
                                    :, (last + 1):(last + ks)
                                ]
                                last += ks
                            end
                        else
                            getsubdict!(R[l], (Overt, Svert))[(Overt, Svert)] = q_ks
                        end
                        getsubdict!(K, Svert)[Overt] = k_l
                    end
                end
            end

        elseif source_is_frozen && !obs_is_frozen
            for Overt in treeO[l]
                if !isleaf(testT, Overt)
                    for Ochild in children(testT, Overt)
                        obsindex = values(testT, Ochild)
                        for Svert in treeS[1]
                            srcindex = K[Svert][Overt]
                            if isempty(srcindex)
                                @show "Warning: empty source index for Svert $Svert and
                                Ochild $Overt at level $l. This should not happen if the
                                tree is properly balanced."
                                continue
                            end
                            n_otilde = estimate_rank_3d(
                                k, trialT, testT, Svert, Ochild, τ; C=1.0, Cε=3.0, Rmin=3
                            )
                            q_ks, k_l, r_l = Compressor(
                                kernelmatrix, srcindex, obsindex, n_otilde, τ
                            )
                            last = 0
                            getsubdict!(R[l], (Ochild, Svert))[(Overt, Svert)] = q_ks
                            getsubdict!(K, Svert)[Ochild] = k_l
                        end
                    end
                else
                end
            end

        elseif !source_is_frozen && obs_is_frozen
            for Overt in treeO[LO]
                obsindex = values(testT, Overt)
                for Svert in treeS[LS - l]
                    srcindex = U[Svert][Overt]
                    if isempty(srcindex)
                        @show "Warning: empty source index for Svert $Svert and Ochild
                        $Overt at level $l. This should not happen if the tree is properly
                        balanced."
                        continue
                    end
                    n_otilde = estimate_rank_3d(
                        k, trialT, testT, Svert, Overt, τ; C=1.0, Cε=3.0, Rmin=3
                    )
                    q_ks, k_l, r_l = Compressor(
                        kernelmatrix, srcindex, obsindex, n_otilde, τ
                    )
                    if !isleaf(trialT, Svert)
                        last = 0
                        for Schild in children(trialT, Svert)
                            if !haskey(K, Schild) || !haskey(K[Schild], Overt)
                                @show "Warning: missing K entry for Schild $Schild and Overt
                                $Overt at level $l. This should not happen if the tree is
                                properly balanced."
                                continue
                            end
                            ks = length(getsubdict!(K, Schild)[Overt])
                            getsubdict!(R[l], (Overt, Svert))[(Overt, Schild)] = q_ks[
                                :, (last + 1):(last + ks)
                            ]
                            last += ks
                        end
                        getsubdict!(K, Svert)[Overt] = k_l
                    else
                    end
                end
            end

        else
            break
        end
    end

    # ------------------------------------------------------------------
    # Final P blocks
    # ------------------------------------------------------------------
    P = Dict{Int,Matrix{ComplexF64}}()
    for Oleaf in treeO[LO]
        if !haskey(K[NS], Oleaf)
            @show "Warning: missing K entry for NS $NS and Oleaf $Oleaf at final P block
            construction. This should not happen if the tree is properly balanced."
            continue
        end
        col = K[NS][Oleaf]
        row = values(testT, Oleaf)

        Z = zeros(ComplexF64, length(row), length(col))
        kernelmatrix(Z, row, col)
        PermP[Oleaf] = [obs_map[g] for g in row]
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
    )
end

"""
    subroutine_BF_mats(kernelmatrix, H2Blocktree, NO, NS, k, τ; Compressor=PartialQR())

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
    Compressor=ButterflyFactorizations.PartialQR(),
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

    values = H2Trees.values
    center = H2Trees.center
    halfsize = H2Trees.halfsize
    children = H2Trees.children

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
        n_otilde = estimate_rank_3d(k, trialT, testT, Sleaf, NO, τ; C=1.0, Cε=3.0, Rmin=3)
        q_ks, k_l, r_l = Compressor(kernelmatrix, srcindex, obsindex, n_otilde, τ)
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

                        n_otilde = estimate_rank_3d(
                            k, trialT, testT, Svert, Ochild, τ; C=1.0, Cε=3.0, Rmin=3
                        )
                        q_ks, k_l, r_l = Compressor(
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
                        n_otilde = estimate_rank_3d(
                            k, trialT, testT, Svert, Ochild, τ; C=1.0, Cε=3.0, Rmin=3
                        )
                        q_ks, k_l, r_l = Compressor(
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

                    n_otilde = estimate_rank_3d(
                        k, trialT, testT, Svert, Overt, τ; C=1.0, Cε=3.0, Rmin=3
                    )
                    q_ks, k_l, r_l = Compressor(
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
