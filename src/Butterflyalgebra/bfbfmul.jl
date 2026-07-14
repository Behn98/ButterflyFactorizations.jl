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
        for t in 1:m
            result = browswap(result, L + 2 - t, τ)
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

function mul_factors(left::R_factor{T,M}, right::R_factor{T,M}) where {T,M}
    RowColKey = Tuple{Int,Int}
    ValueTuple = Tuple{Int,Int,Int}

    # =========================================================================
    # PHASE 1: Sparse Accumulation of Block Products
    # =========================================================================
    # Temporary storage mapping: row_key -> col_key -> accumulated_matrix
    computed_blocks = Dict{RowColKey,Dict{RowColKey,M}}()

    for (rowA, colsA_dict) in left.dict
        for (inner_key, (eidxA, iA, jA)) in colsA_dict
            # Check if there is a matching row in the right factor
            if haskey(right.dict, inner_key)
                matA = left.elementblocks[eidxA][iA, jA]

                for (colB, (eidxB, iB, jB)) in right.dict[inner_key]
                    matB = right.elementblocks[eidxB][iB, jB]

                    # Compute the submatrix multiplication
                    prod_mat = matA * matB

                    # Accumulate into the temporary lookup
                    inner_dict = get!(() -> Dict{RowColKey,M}(), computed_blocks, rowA)
                    if haskey(inner_dict, colB)
                        inner_dict[colB] += prod_mat
                    else
                        inner_dict[colB] = prod_mat
                    end
                end
            end
        end
    end

    # =========================================================================
    # PHASE 2: Re-grouping Rows by Column-Spaces
    # =========================================================================
    # Group row keys that share the exact same mathematical set of column keys
    groups = Dict{Set{RowColKey},Vector{RowColKey}}()
    for (row, col_dict) in computed_blocks
        col_space = Set{RowColKey}(keys(col_dict))
        push!(get!(() -> RowColKey[], groups, col_space), row)
    end

    # Pre-allocate containers for the new R_factor
    new_elementblocks = Vector{Matrix{M}}()
    new_inverse_map = Vector{Matrix{Union{Nothing,Tuple{RowColKey,RowColKey}}}}()
    new_dict = Dict{RowColKey,Dict{RowColKey,ValueTuple}}()

    # Build the structured grid matrices from our grouped sets
    for (col_space, rows_in_group) in groups
        # Lexicographical sorting guarantees alignment invariants
        sorted_rows = sort(rows_in_group)
        sorted_cols = sort(collect(col_space))

        n_rows = length(sorted_rows)
        n_cols = length(sorted_cols)

        # Allocate the grids for this specific partition block
        grid_matrix = Matrix{M}(undef, n_rows, n_cols)
        grid_imap = Matrix{Union{Nothing,Tuple{RowColKey,RowColKey}}}(
            nothing, n_rows, n_cols
        )

        push!(new_elementblocks, grid_matrix)
        push!(new_inverse_map, grid_imap)
        new_eidx = length(new_elementblocks) # Track current vector location

        # Populate the matrices and metadata maps
        for j in 1:n_cols
            col_key = sorted_cols[j]
            for i in 1:n_rows
                row_key = sorted_rows[i]

                # 1. Place the numerical data block
                grid_matrix[i, j] = computed_blocks[row_key][col_key]

                # 2. Update the O(1) inverse lookup layout
                grid_imap[i, j] = (row_key, col_key)

                # 3. Update the forward dictionary
                inner_dict = get!(() -> Dict{RowColKey,ValueTuple}(), new_dict, row_key)
                inner_dict[col_key] = (new_eidx, i, j)
            end
        end
    end

    # =========================================================================
    # PHASE 3: Inherit Structural Properties and Trees
    # =========================================================================
    return R_factor{T,M}(
        new_dict,
        new_inverse_map,
        new_elementblocks,
        (left.slvl[1], right.slvl[2]), # Row level context from left, Col from right
        (left.olvl[1], right.olvl[2]),
        left.rowstree,
        left.rowotree,
        right.colstree,
        right.colotree,
    )
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

function mul_factors(BF::AlgBF, idx::Int)
    L = length(BF.R)
    if idx > 1 && idx < (L + 1)
        leftfactor = BF.R[L + 1 - (idx - 1)].dict
        rightfactor = BF.R[L + 1 - idx].dict
        product = R_factor(
            mul_factors(leftfactor, rightfactor),
            (BF.R[L + 1 - idx].slvl[1], BF.R[L + 1 - (idx - 1)].slvl[2]),
            (BF.R[L + 1 - idx].olvl[1], BF.R[L + 1 - (idx - 1)].olvl[2]),
            BF.R[L + 1 - (idx - 1)].rowstree,
            BF.R[L + 1 - (idx - 1)].rowotree,
            BF.R[L + 1 - idx].colstree,
            BF.R[L + 1 - idx].colotree,
        )
    elseif idx == 1
        @show "Multiplying P and R[1]"
        leftfactor = BF.P.dict
        rightfactor = BF.R[L + 1 - idx].dict
        product = R_factor(
            mul_factors(leftfactor, rightfactor),
            (BF.R[L + 1 - idx].slvl[1], BF.R[L + 1 - idx].slvl[2]),
            (BF.R[L + 1 - idx].olvl[1], BF.R[L + 1 - idx].olvl[2]),
            BF.R[L + 1 - idx].rowstree,
            BF.R[L + 1 - idx].rowotree,
            BF.R[L + 1 - idx].colstree,
            BF.R[L + 1 - idx].colotree,
        )
        #should not occure since we only call this function for idx in 2:(L-1)
    else
        @show "Multiplying R[end] and Q"
        leftfactor = BF.R[L + 1 - idx].dict
        rightfactor = BF.Q.dict
        product = R_factor(
            mul_factors(leftfactor, rightfactor),
            (BF.R[L + 1 - idx].slvl[1], BF.R[L + 1 - idx].slvl[2]),
            (BF.R[L + 1 - idx].olvl[1], BF.R[L + 1 - idx].olvl[2]),
            BF.R[L + 1 - idx].rowstree,
            BF.R[L + 1 - idx].rowotree,
            BF.R[L + 1 - idx].colstree,
            BF.R[L + 1 - idx].colotree,
        )
        #should not occure since we only call this function for idx in 2:(L-1)
    end
    #product = mul_factors(leftfactor, rightfactor)
    return AlgBF(
        (size(BF, 1), size(BF, 2)),
        BF.Q,
        vcat(BF.R[1:(L - idx)], [product], BF.R[(L - idx + 3):length(BF.R)]),
        BF.P,
    )
end

function browswap(BF::AlgBF, idx::Int, τ)
    L = length(BF.R)
    if idx > 1 && idx < (L + 1)
        leftfactor = BF.R[L + 1 - (idx - 1)]
        rightfactor = BF.R[L + 1 - idx]
    elseif idx == 1
        @show "Multiplying P and R[L]"
        leftfactor = BF.P
        rightfactor = BF.R[L + 1 - idx]
        #should not happen!
    else
        @show "Multiplying R[1] and Q"
        leftfactor = BF.R[L + 1 - idx]
        rightfactor = BF.Q
        #should not happen!
    end
    nlfactor, nrfactor = browswap(leftfactor, rightfactor, τ)

    return AlgBF(
        (size(BF, 1), size(BF, 2)),
        BF.Q,
        vcat(BF.R[1:(L - idx)], [nrfactor, nlfactor], BF.R[(L - idx + 3):length(BF.R)]),
        BF.P,
    )
end

function browswap(LeftFactor::R_factor, RightFactor::R_factor, τ)
    NewLeftFactor = Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},Matrix{ComplexF64}}}()
    NewRightFactor = Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},Matrix{ComplexF64}}}()

    Intermediate = mul_factors(LeftFactor.dict, RightFactor.dict)
    col_tree = RightFactor.colstree
    parentkeyscols = Dict{Tuple{Int,Int},Vector{Tuple{Int,Int}}}()
    parentkeysrows = Dict{Tuple{Int,Int},Vector{Tuple{Int,Int}}}()
    for row in keys(Intermediate)
        parentgrps = group_by_parents(col_tree, keys(Intermediate[row]), 2)
        for (parentnodes, localcols) in parentgrps
            parentkey = (first(keys(LeftFactor.dict[row]))[1], parentnodes) #H2Trees.parent(row_tree, row[1])first(localcols)[1]parentnodeo
            if !haskey(parentkeysrows, parentkey)
                parentkeysrows[parentkey] = Vector{Tuple{Int,Int}}()
            end
            unique!(push!(parentkeysrows[parentkey], row))
            if !haskey(parentkeyscols, parentkey)
                parentkeyscols[parentkey] = Vector{Tuple{Int,Int}}()
            end
            for col in localcols
                unique!(push!(parentkeyscols[parentkey], col))
            end
            A_k = hcat([Intermediate[row][col] for col in localcols]...)
            if !haskey(NewLeftFactor, row)
                NewLeftFactor[row] = Dict{Tuple{Int,Int},Matrix{ComplexF64}}()
            end
            NewLeftFactor[row][parentkey] = A_k
        end
    end
    #=
    for parentkey in keys(parentkeyscols)
        sort!(parentkeyscols[parentkey])
    end
    =#
    for parentkey in keys(parentkeyscols)
        localrows = parentkeysrows[parentkey]
        localcols = parentkeyscols[parentkey]
        if !haskey(NewRightFactor, parentkey)
            NewRightFactor[parentkey] = Dict{Tuple{Int,Int},Matrix{ComplexF64}}()
        end
        coltracker = 0
        colsizeA_k = size(NewLeftFactor[first(localrows)][parentkey], 2)
        for col in localcols
            colcurent = size(Intermediate[first(localrows)][col], 2)
            NewRightFactor[parentkey][col] = vcat(
                zeros(ComplexF64, coltracker, colcurent),
                Matrix{ComplexF64}(I, colcurent, colcurent),
                zeros(ComplexF64, colsizeA_k - coltracker - colcurent, colcurent),
            )
            coltracker += colcurent
        end
    end

    return R_factor(
        NewLeftFactor,
        LeftFactor.slvl,
        LeftFactor.olvl,
        LeftFactor.rowstree,
        LeftFactor.rowotree,
        LeftFactor.colstree,
        LeftFactor.colotree,
    ),
    R_factor(
        NewRightFactor,
        RightFactor.slvl,
        RightFactor.olvl,
        RightFactor.rowstree,
        RightFactor.rowotree,
        RightFactor.colstree,
        RightFactor.colotree,
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
