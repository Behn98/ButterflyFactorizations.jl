import H2Trees: values, center, halfsize, children, isleaf, trialtree, testtree
"""
    mulBFs(BF_1::BF, BF_2::BF, τ::Float64)

Algebraically multiplies two Butterfly Factorization (BF) structures and recompresses the
result using the truncation tolerance `τ`.

Both factorizations must have the same number of levels, and the source dimensions of `BF_1`
must match the observer dimensions of `BF_2`. Additionally, the resulting structure is
purely algebraic and may lose its direct physical interpretation, similar to multiplying two
dense matrices directly. The function constructs an intermediate messenger structure to hold
the products of the factors, then iteratively multiplies and recompresses the factors level
by level, ultimately returning a new `BF` that represents the product of the two input
factorizations. The resulting `BF` maintains the same hierarchical structure but with
potentially reduced ranks in the `R` factors, leading to improved efficiency in storage and
matrix-vector products while preserving the overall accuracy within the specified tolerance.
In terms of storage future work has to include more aggressive recompression strategies to
prevent the intermediate factors from growing too large.
"""
function mulBFs(BF_1::BF, BF_2::BF, τ::Float64, tree::H2Trees.BlockTree)
    @assert length(BF_1) == length(BF_2) "Both BFs must have the same number of levels"
    @assert BF_1.NS == BF_2.NO "Source and Observer dimensions must match"
    M_messenger = Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},AbstractMatrix{ComplexF64}}}()
    for leaf in keys(BF_1.Q)
        M_messenger[BF_1.NO, leaf] = Dict{Tuple{Int,Int},AbstractMatrix{ComplexF64}}()
        # Initialize as a nested dict to work with swap_and_recompress
        M_messenger[BF_1.NO, leaf][leaf, BF_2.NS] = BF_1.Q[leaf] * BF_2.P[leaf]
    end

    L = length(BF_1.R) # Number of R-levels
    M_messenger = mul_factors(BF_1.R[1], M_messenger)
    M_messenger = mul_factors(M_messenger, BF_2.R[L])

    result = AlgBF(
        (size(BF_1, 1), size(BF_2, 2)),
        BF_2.Q,
        vcat(BF_2.R[1:(L - 1)], [M_messenger], BF_1.R[2:L]),
        BF_1.P,
    )
    for m in 1:(L - 1)
        for t in 1:m
            result = swap_and_recompress(result, L + 2 - t, τ, tree)
        end
        result = recompress_BF(mul_factors(result, L + 1 - m), τ, tree)
    end

    return BF(
        result.Q,         # Q_final = Q_2
        result.R,       # R_final[level][Snode][Onode]
        result.P,          # Updated P
        BF_2.PermQ,       # Q Permutations remain the same as BF_2
        BF_1.PermP,     # P Permutations remain the same as BF_1
        (size(BF_1, 1), size(BF_2, 2)),
        BF_2.NS,
        BF_1.NO,
        BF_1.k,         # Or recalculated k
        τ,
    )
end

function mul_factors(
    leftfactor::Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},AbstractMatrix{ComplexF64}}},
    rightfactor::Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},AbstractMatrix{ComplexF64}}},
)
    product = Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},AbstractMatrix{ComplexF64}}}()
    for row in keys(leftfactor)
        if !haskey(product, row)
            product[row] = Dict{Tuple{Int,Int},AbstractMatrix{ComplexF64}}()
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
                #=
                if !haskey(product[row], col)
                    product[row][col] = Matrix{ComplexF64}(
                        undef,
                        size(leftfactor[row][inner])[1],
                        size(rightfactor[inner][col])[2],
                    )
                    @views mul!(
                        product[row][col], leftfactor[row][inner], rightfactor[inner][col]
                    )
                else
                    temp = Matrix{ComplexF64}(
                        undef,
                        size(leftfactor[row][inner])[1],
                        size(rightfactor[inner][col])[2],
                    )
                    @views mul!(temp, leftfactor[row][inner], rightfactor[inner][col])
                    product[row][col] += temp
                end=#
            end
        end
    end

    return product
end

function mul_factors(BF::AlgBF, idx::Int)
    L = length(BF.R)
    if idx > 1 && idx < (L + 1)
        leftfactor = BF.R[L + 1 - (idx - 1)]
        rightfactor = BF.R[L + 1 - idx]
    elseif idx == 1
        @show "Multiplying P and R[1]"
        leftfactor = BF.P
        rightfactor = BF.R[L + 1 - idx]
    else
        @show "Multiplying R[end] and Q"
        leftfactor = BF.R[L + 1 - idx]
        rightfactor = BF.Q
    end
    product = mul_factors(leftfactor, rightfactor)

    return AlgBF(
        (size(BF, 1), size(BF, 2)),
        BF.Q,
        vcat(BF.R[1:(L - idx)], [product], BF.R[(L - idx + 3):length(BF.R)]),
        BF.P,
    )
end

function swap_and_recompress(BF::AlgBF, idx::Int, τ, tree::H2Trees.BlockTree)
    L = length(BF.R)
    if idx > 1 && idx < (L + 1)
        leftfactor = BF.R[L + 1 - (idx - 1)]
        rightfactor = BF.R[L + 1 - idx]
    elseif idx == 1
        @show "Multiplying P and R[L]"
        leftfactor = BF.P
        rightfactor = BF.R[L + 1 - idx]
    else
        @show "Multiplying R[1] and Q"
        leftfactor = BF.R[L + 1 - idx]
        rightfactor = BF.Q
    end
    nlfactor, nrfactor = swap_and_recompress(
        leftfactor, rightfactor, τ, tree::H2Trees.BlockTree
    )

    return AlgBF(
        (size(BF, 1), size(BF, 2)),
        BF.Q,
        vcat(BF.R[1:(L - idx)], [nrfactor, nlfactor], BF.R[(L - idx + 3):length(BF.R)]),
        BF.P,
    )
end

function swap_and_recompress(LeftFactor, RightFactor, τ, tree::H2Trees.BlockTree)
    NewLeftFactor = Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},AbstractMatrix{ComplexF64}}}()
    NewRightFactor = Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},AbstractMatrix{ComplexF64}}}()

    Intermediate = mul_factors(LeftFactor, RightFactor)
    row_tree = testtree(tree)

    # 1. Map target parent_skel to its constituent row and column components
    # parent_skel => (Set(row_skeletons), Set(col_skeletons))
    skel_groups = Dict{Tuple{Int,Int},Tuple{Set{Tuple{Int,Int}},Set{Tuple{Int,Int}}}}()

    for row_skel in keys(Intermediate)
        parent_node = H2Trees.parent(row_tree, row_skel[1])
        for col_skel in keys(Intermediate[row_skel])
            p_skel = (parent_node, col_skel[2])
            if !haskey(skel_groups, p_skel)
                skel_groups[p_skel] = (Set{Tuple{Int,Int}}(), Set{Tuple{Int,Int}}())
            end
            push!(skel_groups[p_skel][1], row_skel)
            push!(skel_groups[p_skel][2], col_skel)
        end
    end

    # 2. Process each unique intermediate space via joint tiling
    for (p_skel, (local_rows_set, local_cols_set)) in skel_groups
        local_rows = collect(local_rows_set)
        local_cols = collect(local_cols_set)

        # Determine individual block row sizes
        row_sizes = zeros(Int, length(local_rows))
        for (i, r) in enumerate(local_rows)
            for c in local_cols
                if haskey(Intermediate[r], c)
                    row_sizes[i] = size(Intermediate[r][c], 1)
                    break
                end
            end
        end

        # Determine individual block column sizes
        col_sizes = zeros(Int, length(local_cols))
        for (j, c) in enumerate(local_cols)
            for r in local_rows
                if haskey(Intermediate[r], c)
                    col_sizes[j] = size(Intermediate[r][c], 2)
                    break
                end
            end
        end

        # 3. Tile the active blocks into a single localized matrix grid
        A_local = zeros(ComplexF64, sum(row_sizes), sum(col_sizes))

        current_row = 0
        for (i, r) in enumerate(local_rows)
            current_col = 0
            for (j, c) in enumerate(local_cols)
                if haskey(Intermediate[r], c)
                    block = Intermediate[r][c]
                    A_local[
                        (current_row + 1):(current_row + row_sizes[i]),
                        (current_col + 1):(current_col + col_sizes[j]),
                    ] .= block
                end
                current_col += col_sizes[j]
            end
            current_row += row_sizes[i]
        end

        # 4. Joint Rank-Revealing Compression
        QRA = pqr(A_local; rtol=τ)
        Q_mat = QRA[1]
        R_mat = QRA[2][:, invperm(QRA[3])]

        # 5. Distribute Q_mat vertically to NewLeftFactor
        current_row = 0
        for (i, r) in enumerate(local_rows)
            if !haskey(NewLeftFactor, r)
                NewLeftFactor[r] = Dict{Tuple{Int,Int},AbstractMatrix{ComplexF64}}()
            end
            NewLeftFactor[r][p_skel] = Matrix(
                Q_mat[(current_row + 1):(current_row + row_sizes[i]), :]
            )
            current_row += row_sizes[i]
        end

        # 6. Distribute R_mat horizontally to NewRightFactor
        if !haskey(NewRightFactor, p_skel)
            NewRightFactor[p_skel] = Dict{Tuple{Int,Int},AbstractMatrix{ComplexF64}}()
        end

        current_col = 0
        for (j, c) in enumerate(local_cols)
            NewRightFactor[p_skel][c] = Matrix(
                R_mat[:, (current_col + 1):(current_col + col_sizes[j])]
            )
            current_col += col_sizes[j]
        end
    end

    return NewLeftFactor, NewRightFactor
end

function LinearAlgebra.mul!(
    C::ButterflyFactorizations.BF,
    A::ButterflyFactorizations.BF,
    B::ButterflyFactorizations.BF,
)
    copyto!(C, mulBFs(A, B, max(A.τ, B.τ)))
    return C
end
