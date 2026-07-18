import H2Trees: values, center, halfsize, children, isleaf, trialtree, testtree

"""
    mulBFs(BF_1_init::BF, BF_2_init::BF, τ::Float64) -> BF

Compute the operator product of two Butterfly Factorizations (`BF`) and compress the
resulting representation to a specified accuracy tolerance.

This function implements hierarchical butterfly-butterfly multiplication. It merges the
internal factors of both trees by initializing an intermediate structural "messenger"
matrix, and then sequentially alternates row-swapping (`browswap`) and low-rank truncation
(`recompress_BF`) to prevent rank explosion.

# Arguments

  - `BF_1_init::BF`: The left butterfly factorization operator.
  - `BF_2_init::BF`: The right butterfly factorization operator.
  - `τ::Float64`: The accuracy tolerance parameter used during internal row-swaps and factor
    recompressions.

# Constraints & Assumed Invariants

  - **Level Matching:** Both butterfly factorizations must possess the exact same number of
    hierarchical levels (`length`).
  - **Dimensional Compatibility:** The source dimension of the left operator
    (`BF_1_init.NS`) must match the observer dimension of the right operator
    (`BF_2_init.NO`).

# Returns

  - `BF`: A new, optimized, and recompressed `BF` object representing the combined operator
    product.

# Core Algorithm Steps

 1. **Messenger Initialization:** Creates an initial central block mapping by multiplying
    `BF_1.Q` and `BF_2.P`.
 2. **Layer Intertwining:** Absorbs the outermost structural remainder levels (`BF_1.R[1]`
    and `BF_2.R[end]`) into the messenger.
 3. **Iterative Row Swapping:** Loops through the internal tree layers, executing a series
    of butterfly row-swaps (`browswap`) to correctly align the hierarchical
    spatial/frequency boxes.
 4. **Trimming:** Truncates redundant rank dimensions via `recompress_BF` at each step to
    maintain the strict \$O(N \\log N)\$ butterfly complexity.
"""
function mulBFs(BF_1_init::BF, BF_2_init::BF, τ::Float64)
    @assert length(BF_1_init) == length(BF_2_init) "Both BFs must have the same number of levels"
    @assert BF_1_init.NS == BF_2_init.NO "Source and Observer dimensions must match"

    BF_1 = deepcopy(BF_1_init)
    BF_2 = deepcopy(BF_2_init)

    M_messenger = Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},Matrix{ComplexF64}}}()
    for (NO, leaf) in keys(BF_1.Q)
        M_messenger[NO, leaf] = Dict{Tuple{Int,Int},Matrix{ComplexF64}}()
        # Initialize as a nested dict to work with browswap
        M_messenger[BF_1.NO, leaf][leaf, BF_2.NS] = BF_1.Q[NO, leaf] * BF_2.P[leaf, BF_2.NS]
    end

    L = length(BF_1.R) # Number of R-levels
    BF_1_alg = AlgBF(BF_1)
    BF_2_alg = AlgBF(BF_2)
    M_messenger = mul_factors(BF_1.R[1], M_messenger)
    M_messenger = mul_factors(M_messenger, BF_2.R[L])
    M_messenger = R_factor(M_messenger)

    result = AlgBF(
        (size(BF_1_alg, 1), size(BF_2_alg, 2)),
        BF_2_alg.Q,
        vcat(BF_2_alg.R[1:(L - 1)], [M_messenger], BF_1_alg.R[2:L]),
        BF_1_alg.P,
        BF_1_alg.k,
        max(BF_1_alg.τ, BF_2_alg.τ),
    )
    for m in 1:(L - 1)
        for t in 1:m
            result = browswap(result, L + 2 - t)
            #print("swap done \n")
        end
        result = recompress_BF(mul_factors(result, L + 1 - m), τ)#
    end
    #@views result = recompress_BF(result, τ)
    return BF(
        result.Q.dict,         # Q_final = Q_2
        [r.dict for r in result.R],       # R_final[level][Snode][Onode]
        result.P.dict,          # Updated P
        (size(BF_1, 1), size(BF_2, 2)),
        BF_2.NS,
        BF_1.NO,
        BF_1.k,         # Or recalculated k
        τ,
        BF_2.stree,
        BF_1.otree,
    )
end

function Base.:*(
    Butterfly1::ButterflyFactorizations.BF, Butterfly2::ButterflyFactorizations.BF
)
    return mulBFs(Butterfly1, Butterfly2, max(Butterfly1.τ, Butterfly2.τ))
end
#=
function mul_factors(left::R_factor{M}, right::R_factor{M}) where {M}
    neweblocks = Vector{Matrix{M}}()
    new_rowspaces = Dict{RKey,Vector{CKey}}()
    new_colspaces = Dict{CKey,Vector{RKey}}()
    new_block_map = Dict{Tuple{RKey,CKey},Tuple{Int,Int,Int}}()
    new_invmap = Vector{Matrix{Tuple{RKey,CKey}}}()
    for (n, leftblock) in enumerate(left.elementblocks)
        newrowspace = [left.inverse_map[n][:, 1][1]]
        for i in 1:size(leftblock, 1)
            if !haskey(new_rowspaces, newrowspace[i])
                new_rowspaces[newrowspace[i]] = Vector{CKey}()
            end
            for j in 1:size(leftblock, 2)
                localcolspace = left.inverse_map[n][i, j][2]
                newcolspace = right.row_spaces[localcolspace]
                push!(new_rowspaces[newrowspace[i]], newcol for newcol in newcolspace)
                for col in colspaces
                    if !haskey(new_colspaces, col)
                        new_colspaces[col] = Vector{RKey}()
                    end
                    push!(new_colspaces[col], newrowspace[i])
                end
                rightblock = right.elementblocks[right.block_map[(
                    localcolspace, newcolspace[1]
                )][1]]
                if i == 1 && j == 1
                    neweblock = Matrix{M}(undef, size(newrowspace, 1), size(newcolspace, 1))
                    newinvblock = Matrix{Tuple{RKey,CKey}}(
                        undef, size(newrowspace, 1), size(newcolspace, 1)
                    )
                    push!(neweblocks, neweblock)
                    push!(new_invmap, newinvblock)
                    eidx = length(neweblocks)
                else
                    eidx = new_block_map[(newrowspace[1], newcolspace[1])][1]
                    neweblock = neweblocks[eidx]
                    newinvblock = new_invmap[eidx]
                end
                for k in 1:size(rightblock, 2)
                    new_block_map[(newrowspace[i], newcolspace[k])] = (eidx, i, k)
                    neweblock[i, k] = leftblock[i, j] * rightblock[j, k]
                    newinvblock[i, k] = (newrowspace[i], newcolspace[k])
                end
            end
        end
    end
    return R_factor{M}(new_rowspaces, new_colspaces, new_block_map, new_invmap, neweblocks)
end
=#

function mul_factors(left::R_factor{M}, right::R_factor{M}) where {M}
    neweblocks = Vector{Matrix{M}}()
    new_rowspaces = Dict{RKey,Vector{CKey}}()
    new_colspaces = Dict{CKey,Vector{RKey}}()
    new_block_map = Dict{Tuple{RKey,CKey},Tuple{Int,Int,Int}}()
    new_invmap = Vector{Matrix{Tuple{RKey,CKey}}}()
    for cols in keys(left.col_spaces)
        neweblock = Matrix{M}(
            undef, length(left.col_spaces[cols]), length(right.row_spaces[cols])
        )
        newinvblock = Matrix{Tuple{RKey,CKey}}(
            undef, length(left.col_spaces[cols]), length(right.row_spaces[cols])
        )
        for (i, row) in enumerate(left.col_spaces[cols])
            if !haskey(new_rowspaces, row)
                new_rowspaces[row] = Vector{CKey}()
            end
            for (j, col) in enumerate(right.row_spaces[cols])
                if !haskey(new_colspaces, col)
                    new_colspaces[col] = Vector{RKey}()
                end
                push!(new_rowspaces[row], col)
                push!(new_colspaces[col], row)
                neweblock[i, j] =
                    left.elementblocks[left.block_map[(row, cols)][1]][
                        left.block_map[(row, cols)][2], left.block_map[(row, cols)][3]
                    ] * right.elementblocks[right.block_map[(cols, col)][1]][
                        right.block_map[(cols, col)][2], right.block_map[(cols, col)][3]
                    ]
                newinvblock[i, j] = (row, col)
                new_block_map[(row, col)] = (length(neweblocks) + 1, i, j)
            end
        end
        push!(neweblocks, neweblock)
        push!(new_invmap, newinvblock)
    end
    return R_factor{M}(new_rowspaces, new_colspaces, new_block_map, new_invmap, neweblocks)
end

Base.:*(left::R_factor{M}, right::R_factor{M}) where {M} = mul_factors(left, right)

function mul_factors(bf::AlgBF, idx::Int)
    L = length(bf.R)
    if idx > 1 && idx < (L + 1)
        leftfactor = bf.R[L + 1 - (idx - 1)]
        rightfactor = bf.R[L + 1 - idx]
        product = mul_factors(leftfactor, rightfactor)
    elseif idx == 1
        error("Multiplying P and R[1]")
    else
        error("Multiplying R[end] and Q")
    end
    #product = mul_factors(leftfactor, rightfactor)
    return AlgBF(
        bf, bf.Q, vcat(bf.R[1:(L - idx)], [product], bf.R[(L - idx + 3):length(bf.R)]), bf.P
    )
end

function browswap(BF::AlgBF, idx::Int)
    L = length(BF.R)
    if idx > 1 && idx < (L + 1)
        leftfactor = BF.R[L + 1 - (idx - 1)]
        rightfactor = BF.R[L + 1 - idx]
    elseif idx == 1
        error("Multiplying P and R[L]")

    else
        error("Multiplying R[1] and Q")
    end
    nlfactor, nrfactor = browswap(leftfactor, rightfactor)

    return AlgBF(
        (size(BF, 1), size(BF, 2)),
        BF.Q,
        vcat(BF.R[1:(L - idx)], [nrfactor, nlfactor], BF.R[(L - idx + 3):length(BF.R)]),
        BF.P,
        BF.k,
        BF.τ,
    )
end

function browswap(LeftFactor::R_factor{M}, RightFactor::R_factor{M}) where {M}
    lf = (#newleftfactor with all necessary fields
        neweblocks    = Vector{Matrix{M}}(),
        new_rowspaces = Dict{RKey,Vector{CKey}}(),
        new_colspaces = Dict{CKey,Vector{RKey}}(),
        new_block_map = Dict{Tuple{RKey,CKey},Tuple{Int,Int,Int}}(),
        new_invmap    = Vector{Matrix{Tuple{RKey,CKey}}}(),
    )
    rf = (#newrightfactor with all necessary fields
        neweblocks    = Vector{Matrix{M}}(),
        new_rowspaces = Dict{RKey,Vector{CKey}}(),
        new_colspaces = Dict{CKey,Vector{RKey}}(),
        new_block_map = Dict{Tuple{RKey,CKey},Tuple{Int,Int,Int}}(),
        new_invmap    = Vector{Matrix{Tuple{RKey,CKey}}}(),
    )

    prod = mul_factors(LeftFactor, RightFactor)
    #we first compute the product of the two factors after absorbing the colspace of the
    #left factor and the rowspace of the right factor, we need to construct a new one in
    #between the new factors. Due to the permutations we are gonna perform inconsistently
    #across multiple rows, and the messenger matrix resulting from multiplication having no
    #semantic relationship between column and row space, we construct indices that are
    #completely unrelated to the tree. Note that for the butterfly to still fit into the
    #matrix world, only the outer rowspace of P and the colspace of Q are relevant!
    colcounter1 = 1
    colcounter2 = 1
    for col in keys(prod.col_spaces)
        #We first need to find the "4×4" Blocks that Heldring is talking about. He
        #immediately asserts that we can row and column permute the product we built earlier
        #into a block diagonal structure of 4×4 blocks. Thus any of those "4×4" blocks must
        #be occupying a unique column and row space of the product. They could look like
        #this, for example, with M being a matrix block and _ being a zero block:
        # [ M M _ _ M M _ _]
        # [ _ _ M M _ _ M M]
        # [ M M _ _ M M _ _]
        # [ _ _ M M _ _ M M]
        # [ M M _ _ M M _ _]
        # [ _ _ M M _ _ M M]
        # [ M M _ _ M M _ _]
        # [ _ _ M M _ _ M M]
        # In the arbitrary tree, case these will have different dimensions.
        #Sort so we dont mix up the order of the blocks M!
        localrowspace = sort!(prod.col_spaces[col])
        localcolspace = sort!(prod.row_spaces[localrowspace[1]])
        #Because due to the messenger matrix, we cant rely on regular semantic relationships
        #between row and col spaces of a "4×4 block" we cant query relationships between the
        #rows and columns of the to be found blocks M. That is why our data structure is in
        #fact saving all the "2×2" blocks, a "4×4" block is composed out of. These subblocks
        #are automatically implying the dimensions we need to concatenate over in the
        #colspace. on top of that the decoupling of the 2x2 blocks in the arbitrary tree
        #case is not restricted by checking on the rowidxs trying to infere tree
        #relationships, which anyways becomes more and more complicated, and doesnt work for
        #the Messenger either. However since the block map is addressing the internal blocks
        #directly, we need to traverse the unique subspaces and collect all unique 2x2
        #blocks to find our 4x4 block:
        # [E1 _ E2 _ ]
        # [__ _ __ _ ]
        # [E3 _ E4 _ ]
        # [__ _ __ _ ]
        #where E = [[M,M];[M,M]]
        #note that rows dont need to be contiguous in the 2x2 block, but the inside blocks
        #([M,M]) usually are.
        supeblock = Vector{Vector{Int}}()
        currenteidx = 0
        for (i, row) in enumerate(localrowspace)
            if currenteidx != prod.block_map[(row, localcolspace[end])][1]
                push!(supeblock, Vector{Int}())
            end
            for (j, col) in enumerate(localcolspace)
                eidx, ei, ej = prod.block_map[(row, col)]
                if currenteidx != eidx
                    currenteidx = eidx
                    push!(supeblock[end], eidx)
                end
            end
        end
        #We now have [[E1,E2],[E3,E4]] in terms of their ids
        #the goal:
        # [[E1 E2]  _____ ]
        # [ _____  [E3 E4]]
        firstn = true
        neweblocksright = Vector{Matrix{M}}()
        newinvblocksright = Vector{Matrix{Tuple{RKey,CKey}}}()
        for (n, rowblockids) in enumerate(supeblock)
            rowdim = size(prod.elementblocks[rowblockids[1]], 1)
            neweblock = Matrix{M}(undef, rowdim, length(rowblockids))
            newinvblock = Matrix{Tuple{RKey,CKey}}(undef, rowdim, length(rowblockids))
            colcounter2 = 1
            firsti = true
            if firstn == true
                neweblocksright = Vector{Matrix{M}}(undef, length(rowblockids))
                newinvblocksright = Vector{Matrix{Tuple{RKey,CKey}}}(
                    undef, length(rowblockids)
                )
            end
            for i in 1:rowdim
                colcounter2 = 1
                rowidx = prod.inverse_map[rowblockids[1]][i, 1][1]
                if !haskey(lf.new_rowspaces, rowidx)
                    lf.new_rowspaces[rowidx] = Vector{CKey}()
                end
                for (j, blockid) in enumerate(rowblockids)
                    newcolidx = (colcounter1, colcounter2)
                    if !haskey(lf.new_colspaces, newcolidx)
                        lf.new_colspaces[newcolidx] = Vector{RKey}()
                    end
                    push!(lf.new_rowspaces[rowidx], newcolidx)
                    push!(lf.new_colspaces[newcolidx], rowidx)
                    if firstn == true
                        neweblocksright[j] = Matrix{M}(
                            undef, length(rowblockids), size(prod.elementblocks[blockid], 2)
                        )
                        newinvblocksright[j] = Matrix{Tuple{RKey,CKey}}(
                            undef, length(rowblockids), size(prod.elementblocks[blockid], 2)
                        )
                    end
                    neweblock[i, j] = hcat(prod.elementblocks[blockid][i, :]...)
                    rowdimright = size(neweblock[i, j], 2)
                    if firsti == true
                        offset = 0
                        for h in eachindex(prod.elementblocks[blockid][i, :])
                            blksize = size(prod.elementblocks[blockid][i, h], 2)
                            colidxright = prod.inverse_map[blockid][i, h][2]
                            neweblocksright[j][i, h] = vcat(
                                zeros(ComplexF64, offset, blksize),
                                Matrix{ComplexF64}(I, blksize, blksize),
                                zeros(ComplexF64, rowdimright - blksize-offset, blksize),
                            )
                            newinvblocksright[j][i, h] = (newcolidx, colidxright)
                            rf.new_block_map[(newcolidx, colidxright)] = (
                                length(rf.neweblocks) + j, i, h
                            )
                            if !haskey(rf.new_rowspaces, newcolidx)
                                rf.new_rowspaces[newcolidx] = Vector{RKey}()
                            end
                            push!(rf.new_rowspaces[newcolidx], colidxright)
                            if !haskey(rf.new_colspaces, colidxright)
                                rf.new_colspaces[colidxright] = Vector{RKey}()
                            end
                            push!(rf.new_colspaces[colidxright], newcolidx)
                            offset += blksize
                        end
                        firsti = false
                    end
                    row = prod.inverse_map[blockid][i, 1][1]
                    newinvblock[i, j] = (row, newcolidx)
                    lf.new_block_map[(row, newcolidx)] = (length(lf.neweblocks) + 1, i, j)

                    colcounter2 += 1
                end
            end
            colcounter1 += 1
            push!(lf.neweblocks, neweblock)
            push!(lf.new_invmap, newinvblock)
            firstn = false
        end
        for i in eachindex(neweblocksright)
            push!(rf.neweblocks, neweblocksright[i])
            push!(rf.new_invmap, newinvblocksright[i])
        end
    end

    return R_factor(
        lf.new_rowspaces, lf.new_colspaces, lf.new_block_map, lf.new_invmap, lf.neweblocks
    ),
    R_factor(
        rf.new_rowspaces, rf.new_colspaces, rf.new_block_map, rf.new_invmap, rf.neweblocks
    )
end

function LinearAlgebra.mul!(
    C::ButterflyFactorizations.BF,
    A::ButterflyFactorizations.BF,
    B::ButterflyFactorizations.BF,
)
    LinearMaps.check_dim_mul(C, A, B)
    copyto!(C, mulBFs(A, B, max(A.τ, B.τ)))
    return C
end

function mul_factors(
    leftfactor::Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},Matrix{ComplexF64}}},
    rightfactor::Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},Matrix{ComplexF64}}},
)
    product = Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},Matrix{ComplexF64}}}()
    for row in keys(leftfactor)
        if !haskey(product, row)
            product[row] = Dict{Tuple{Int,Int},Matrix{ComplexF64}}()
        end
        for inner in keys(leftfactor[row])
            for col in keys(rightfactor[inner])
                if !haskey(product[row], col)
                    # First time seeing this block, allocate and multiply
                    product[row][col] = leftfactor[row][inner] * rightfactor[inner][col]
                else
                    # In-place accumulation: C = 1.0 * A * B + 1.0 * C
                    mul!(
                        product[row][col],
                        leftfactor[row][inner],
                        rightfactor[inner][col],
                        1.0,
                        1.0,
                    )
                end
            end
        end
    end

    return product
end

function mul_factors(
    leftfactor::Dict{Tuple{Int,Int},Matrix{ComplexF64}},
    rightfactor::Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},Matrix{ComplexF64}}},
)
    product = Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},Matrix{ComplexF64}}}()
    for row in keys(leftfactor)
        if !haskey(product, row)
            product[row] = Dict{Tuple{Int,Int},Matrix{ComplexF64}}()
        end
        for col in keys(rightfactor[inner])
            if !haskey(product[row], col)
                # First time seeing this block, allocate and multiply
                product[row][col] = leftfactor[row] * rightfactor[row][col]
            else
                # In-place accumulation: C = 1.0 * A * B + 1.0 * C
                mul!(product[row][col], leftfactor[row], rightfactor[row][col], 1.0, 1.0)
            end
        end
    end

    return product
end

function mul_factors(
    leftfactor::Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},Matrix{ComplexF64}}},
    rightfactor::Dict{Tuple{Int,Int},Matrix{ComplexF64}},
)
    product = Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},Matrix{ComplexF64}}}()
    for row in keys(leftfactor)
        if !haskey(product, row)
            product[row] = Dict{Tuple{Int,Int},Matrix{ComplexF64}}()
        end
        for inner in keys(leftfactor[row])
            if !haskey(product[row], inner)
                # First time seeing this block, allocate and multiply
                product[row][inner] = leftfactor[row][inner] * rightfactor[inner]
            else
                # In-place accumulation: C = 1.0 * A * B + 1.0 * C
                mul!(
                    product[row][inner],
                    leftfactor[row][inner],
                    rightfactor[inner],
                    1.0,
                    1.0,
                )
            end
        end
    end

    return product
end

function trivialmul(BF_1_init::BF, BF_2_init::BF)
    @assert length(BF_1_init) == length(BF_2_init) "Both BFs must have the same number of levels"
    @assert BF_1_init.NS == BF_2_init.NO "Source and Observer dimensions must match"
    BF_1 = deepcopy(BF_1_init)
    BF_2 = deepcopy(BF_2_init)
    M_messenger = Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},Matrix{ComplexF64}}}()
    for (NO, leaf) in keys(BF_1.Q)
        M_messenger[NO, leaf] = Dict{Tuple{Int,Int},Matrix{ComplexF64}}()
        M_messenger[BF_1.NO, leaf][leaf, BF_2.NS] = BF_1.Q[NO, leaf] * BF_2.P[leaf, BF_2.NS]
    end

    L = length(BF_1.R) # Number of R-levels
    BF_1_alg = AlgBF(BF_1)
    BF_2_alg = AlgBF(BF_2)
    M_messenger = mul_factors(BF_1.R[1], M_messenger)
    M_messenger = mul_factors(M_messenger, BF_2.R[L])
    M_messenger = R_factor(
        M_messenger,
        (BF_1_alg.R[L].slvl[1], BF_2_alg.R[1].slvl[2]),
        (BF_1_alg.R[L].olvl[1], BF_2_alg.R[1].olvl[2]),
        BF_1_alg.R[1].rowstree,
        BF_2_alg.R[1].rowotree,
        BF_1_alg.R[L].colstree,
        BF_2_alg.R[L].colotree,
    )

    result = AlgBF(
        (size(BF_1_alg, 1), size(BF_2_alg, 2)),
        BF_2_alg.Q,
        vcat(BF_2_alg.R[1:(L - 1)], [M_messenger], BF_1_alg.R[2:L]),
        BF_1_alg.P,
    )
    for m in 1:(L - 1)
        result = mul_factors(result, L + 1 - m)#recompress_BF(, τ)
    end
    #@views result = recompress_BF(result, τ, tree)
    return BF(
        result.Q.dict,         # Q_final = Q_2
        [r.dict for r in result.R],       # R_final[level][Snode][Onode]
        result.P.dict,          # Updated P
        (size(BF_1, 1), size(BF_2, 2)),
        BF_2.NS,
        BF_1.NO,
        BF_1.k,         # Or recalculated k
        max(BF_1.τ, BF_2.τ),
        BF_2.stree,
        BF_1.otree,
    )
end
