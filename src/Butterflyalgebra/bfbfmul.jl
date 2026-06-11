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
            @show "swap done"
        end
        result = mul_factors(result, L + 1 - m)
    end
    @views result = recompress_BF(result, τ, tree)
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
    col_tree = trialtree(tree)

    for row in keys(Intermediate)
        parentgrps = group_by_parents(col_tree, keys(Intermediate[row]), 2)
        for (parentnode, localcols) in parentgrps
            parentkey = (first(localcols)[1], parentnode) #H2Trees.parent(row_tree, row[1])
            A_k = hcat([Intermediate[row][col] for col in localcols]...)
            if !haskey(NewLeftFactor, row)
                NewLeftFactor[row] = Dict{Tuple{Int,Int},AbstractMatrix{ComplexF64}}()
            end
            if haskey(NewLeftFactor[row], parentkey)
                @show "Warning: Overwriting existing block in NewLeftFactor at row $row and parentkey $parentkey"
            end
            NewLeftFactor[row][parentkey] = A_k
            if !haskey(NewRightFactor, parentkey)
                NewRightFactor[parentkey] = Dict{Tuple{Int,Int},AbstractMatrix{ComplexF64}}()
            end
            coltracker = 0
            colsizeA_k = size(A_k, 2)
            for col in localcols
                colcurent = size(Intermediate[row][col], 2)
                if haskey(NewRightFactor[parentkey], col)
                    #@show "Warning: Overwriting existing block in NewRightFactor at parentkey $parentkey and col $col"
                    er =
                        NewRightFactor[parentkey][col] == vcat(
                            zeros(ComplexF64, coltracker, colcurent),
                            Matrix{ComplexF64}(I, colcurent, colcurent),
                            zeros(
                                ComplexF64, colsizeA_k - coltracker - colcurent, colcurent
                            ),
                        )
                    @show er
                end
                NewRightFactor[parentkey][col] = vcat(
                    zeros(ComplexF64, coltracker, colcurent),
                    Matrix{ComplexF64}(I, colcurent, colcurent),
                    zeros(ComplexF64, colsizeA_k - coltracker - colcurent, colcurent),
                )
                coltracker += colcurent
            end
        end
    end
    mulfactor = mul_factors(NewLeftFactor, NewRightFactor)
    return NewLeftFactor, NewRightFactor
end

function swap_and_recompress2(LeftFactor, RightFactor, τ, tree::H2Trees.BlockTree)
    NewLeftFactor = Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},AbstractMatrix{ComplexF64}}}()
    NewRightFactor = Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},AbstractMatrix{ComplexF64}}}()

    Intermediate = mul_factors(LeftFactor, RightFactor)
    row_tree = testtree(tree)
    col_tree = trialtree(tree)
    parentgrps1 = group_by_parents(row_tree, keys(Intermediate), 1)
    for (parentnodeo, localrows) in parentgrps1
        for row in localrows
            parentgrps2 = group_by_parents(col_tree, keys(Intermediate[row]), 2)
            for (parentnodes, localcols) in parentgrps2
                parentkey = (parentnodeo, parentnodes) #H2Trees.parent(row_tree, row[1])
                A_k = hcat([Intermediate[row][col] for col in localcols]...)
                if !haskey(NewLeftFactor, row)
                    NewLeftFactor[row] = Dict{Tuple{Int,Int},AbstractMatrix{ComplexF64}}()
                end
                if haskey(NewLeftFactor[row], parentkey)
                    @show "Warning: Overwriting existing block in NewLeftFactor at row $row and parentkey $parentkey"
                end
                NewLeftFactor[row][parentkey] = A_k
                if !haskey(NewRightFactor, parentkey)
                    NewRightFactor[parentkey] = Dict{
                        Tuple{Int,Int},AbstractMatrix{ComplexF64}
                    }()
                end
                coltracker = 0
                colsizeA_k = size(A_k, 2)
                for col in localcols
                    colcurent = size(Intermediate[row][col], 2)
                    if haskey(NewRightFactor[parentkey], col)
                        #@show "Warning: Overwriting existing block in NewRightFactor at parentkey $parentkey and col $col"
                        @show NewRightFactor[parentkey][col] == vcat(
                            zeros(ComplexF64, coltracker, colcurent),
                            Matrix{ComplexF64}(I, colcurent, colcurent),
                            zeros(
                                ComplexF64, colsizeA_k - coltracker - colcurent, colcurent
                            ),
                        )
                    end
                    NewRightFactor[parentkey][col] = vcat(
                        zeros(ComplexF64, coltracker, colcurent),
                        Matrix{ComplexF64}(I, colcurent, colcurent),
                        zeros(ComplexF64, colsizeA_k - coltracker - colcurent, colcurent),
                    )
                    coltracker += colcurent
                end
            end
        end
    end
    mulfactor = mul_factors(NewLeftFactor, NewRightFactor)
    return NewLeftFactor, NewRightFactor
end

#= add tree to the struct...
function LinearAlgebra.mul!(
    C::ButterflyFactorizations.BF,
    A::ButterflyFactorizations.BF,
    B::ButterflyFactorizations.BF,
)
    copyto!(C, mulBFs(A, B, max(A.τ, B.τ)))
    return C
end
=#
