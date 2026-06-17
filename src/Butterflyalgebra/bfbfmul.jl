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
function mulBFs(BF_1::BF, BF_2::BF, τ::Float64)
    @assert length(BF_1) == length(BF_2) "Both BFs must have the same number of levels"
    @assert BF_1.NS == BF_2.NO "Source and Observer dimensions must match"
    M_messenger = Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},Matrix{ComplexF64}}}()
    for leaf in keys(BF_1.Q)
        M_messenger[BF_1.NO, leaf] = Dict{Tuple{Int,Int},Matrix{ComplexF64}}()
        # Initialize as a nested dict to work with swap_and_recompress
        M_messenger[BF_1.NO, leaf][leaf, BF_2.NS] = BF_1.Q[leaf] * BF_2.P[leaf]
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
            result = swap_and_recompress(result, L + 2 - t, τ)
            print("swap done \n")
        end
        result = recompress_BF(mul_factors(result, L + 1 - m), τ)#
    end
    #@views result = recompress_BF(result, τ)
    return BF(
        result.Q.Dict,         # Q_final = Q_2
        [r.Dict for r in result.R],       # R_final[level][Snode][Onode]
        result.P.Dict,          # Updated P
        BF_2.PermQ,       # Q Permutations remain the same as BF_2
        BF_1.PermP,     # P Permutations remain the same as BF_1
        (size(BF_1, 1), size(BF_2, 2)),
        BF_2.NS,
        BF_1.NO,
        BF_1.k,         # Or recalculated k
        τ,
        BF_2.stree,
        BF_1.otree,
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

function mul_factors(BF::AlgBF, idx::Int)
    L = length(BF.R)
    if idx > 1 && idx < (L + 1)
        leftfactor = BF.R[L + 1 - (idx - 1)].Dict
        rightfactor = BF.R[L + 1 - idx].Dict
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
        leftfactor = BF.P.Dict
        rightfactor = BF.R[L + 1 - idx].Dict
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
        leftfactor = BF.R[L + 1 - idx].Dict
        rightfactor = BF.Q.Dict
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

function swap_and_recompress(BF::AlgBF, idx::Int, τ)
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
    nlfactor, nrfactor = swap_and_recompress(leftfactor, rightfactor, τ)

    return AlgBF(
        (size(BF, 1), size(BF, 2)),
        BF.Q,
        vcat(BF.R[1:(L - idx)], [nrfactor, nlfactor], BF.R[(L - idx + 3):length(BF.R)]),
        BF.P,
    )
end

function swap_and_recompress(LeftFactor, RightFactor, τ)
    NewLeftFactor = Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},Matrix{ComplexF64}}}()
    NewRightFactor = Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},Matrix{ComplexF64}}}()

    Intermediate = mul_factors(LeftFactor.Dict, RightFactor.Dict)
    col_tree = RightFactor.colstree
    parentkeyscols = Dict{Tuple{Int,Int},Vector{Tuple{Int,Int}}}()
    parentkeysrows = Dict{Tuple{Int,Int},Vector{Tuple{Int,Int}}}()
    for row in keys(Intermediate)
        parentgrps = group_by_parents(col_tree, keys(Intermediate[row]), 2)
        for (parentnodes, localcols) in parentgrps
            parentkey = (first(keys(LeftFactor.Dict[row]))[1], parentnodes) #H2Trees.parent(row_tree, row[1])first(localcols)[1]parentnodeo
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

# add tree to the struct...
function LinearAlgebra.mul!(
    C::ButterflyFactorizations.BF,
    A::ButterflyFactorizations.BF,
    B::ButterflyFactorizations.BF,
)
    LinearMaps.check_dim_mul(C, A, B)
    copyto!(C, mulBFs(A, B, max(A.τ, B.τ)))
    return C
end

function trivialmul(BF_1::BF, BF_2::BF)
    @assert length(BF_1) == length(BF_2) "Both BFs must have the same number of levels"
    @assert BF_1.NS == BF_2.NO "Source and Observer dimensions must match"
    M_messenger = Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},Matrix{ComplexF64}}}()
    for leaf in keys(BF_1.Q)
        M_messenger[BF_1.NO, leaf] = Dict{Tuple{Int,Int},Matrix{ComplexF64}}()
        # Initialize as a nested dict to work with swap_and_recompress
        M_messenger[BF_1.NO, leaf][leaf, BF_2.NS] = BF_1.Q[leaf] * BF_2.P[leaf]
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
        result.Q.Dict,         # Q_final = Q_2
        [r.Dict for r in result.R],       # R_final[level][Snode][Onode]
        result.P.Dict,          # Updated P
        BF_2.PermQ,       # Q Permutations remain the same as BF_2
        BF_1.PermP,     # P Permutations remain the same as BF_1
        (size(BF_1, 1), size(BF_2, 2)),
        BF_2.NS,
        BF_1.NO,
        BF_1.k,         # Or recalculated k
        τ,
        BF_2.stree,
        BF_1.otree,
    )
end

function splitmulbf(butterflycluster::CompositeBlockView, higherkBF::BF)
    #forms matrix of BFs included in the CompositeBlockView
    Matrixr = Matrix{BF}(undef, 0, 0)   #to be changed to the right size
    return Matrixr
end

#When multiplying two Blocks in a BF, one possible case is that one of the factors turns out
#to be a hirachically divided block of Butterflies of some lvl k and the other factor is a
#single BF block of lvl k+1. In that case Heldring suggests a special algorithm, however he
#only provides a sketch of the algorithm and no details on how to implement it. The idea is
#to split the single BF block into multiple blocks that match the hirachical division of the
#other factor and then multiply them separately. Let us recall that the hirachical structure
#and the nature of multiplication do let us make some assumptions. The columnspace of the
#cluster block and the higher k BF's rowspace match and while we dont only treat binary
#trees, the tree structure will anyways remain consistent. Since the tree structure is
#implicitely realted to the Data architecture employed in the Dictionary architecture the
#main challenge is to find the right way to arrange the keys of the large Butterfly to
#perform the multiplications separately.
function splitmulbf(butterflycluster::Matrix{BF}, higherkBF::BF)
    #recall that for multiplication the source tree of the left factor and the observer tree
    #of the right factor must match.
    children = collect(children(higherkBF.testcluster, higherkBF.NO))
    numchildren = length(children)
    l = length(higherkBF.R) # Number of R-levels in the higher k BF
    #Step 1: subdividing the higher k BF into numchildren BFs of lvl k-1
    lowerkBFs = Vector{BF}(undef, numchildren)
    ssubtree = h2treelevels(higherkBF.trialcluster, higherkBF.NS)
    for i in 1:numchildren
        osubtree = h2treelevels(higherkBF.testcluster, children[i])
        new_P = Dict{Int,Matrix{ComplexF64}}()
        for leaf in osubtree[end]
            new_P[leaf] = higherkBF.P[leaf]
        end
        new_R = Vector{Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},Matrix{ComplexF64}}}}(
            undef, l - 1
        )
        for j in 1:(l - 1)
            new_R[j] = Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},Matrix{ComplexF64}}}()
            for onode in osubtree[j + 1]
                for snode in ssubtree[end - j - 1]
                    if haskey(higherkBF.R[j], (onode, snode))
                        new_R[j][(onode, snode)] = higherkBF.R[j][(onode, snode)]
                    end
                end
            end
        end
        new_Q = Dict{Int,Matrix{ComplexF64}}()
        lowerkBFs[i] = BF(
            newQ,
            newR,
            newP,
            higherkBF.PermQ,
            higherkBF.PermP,
            (size(higherkBF, 1), size(higherkBF, 2)),
            children[i],
            higherkBF.NS,
            higherkBF.k,
            higherkBF.τ,
            higherkBF.testcluster,
            higherkBF.trialcluster,
        )
    end
    #The new Butterflies will not have any Q factors, those are preserved together with the
    #last R factor in the higher k BF until we performed the multiplications between the
    #smaller BFs and the cluster of BFs.
    intermediate = Matrix{BF}(undef, size(butterflycluster, 1), numchildren)
    for i in 1:size(butterflycluster, 1)
        for j in 1:numchildren
            intermediate[i, j] = mulBFs(
                butterflycluster[i, j],
                lowerkBFs[j],
                max(butterflycluster[i, j].τ, lowerkBFs[j].τ),
            )
        end
    end
    rowsize = size(intermediate, 1)
    colsize = size(intermediate, 2)
    tobeadditioned = Vector{BF}(undef, rowsize)
    for i in 1:rowsize
        new_P = Dict{Int,Matrix{ComplexF64}}()
        new_R = Vector{Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},Matrix{ComplexF64}}}}(
            undef, l
        )
        for i in eachindex(new_R)
            new_R[i] = Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},Matrix{ComplexF64}}}()
        end
        for j in 1:size(intermediate, 2)
            new_P = merge(new_P, intermediate[(i + j) % rowsize + 1, j].P)
            for k in 1:(l - 1)
                new_R[k + 1] = merge(new_R[k], intermediate[(i + j) % rowsize + 1, j].R[k])
            end
        end
        new_R[1] = higherkBF.R[1]
        new_Q = higherkBF.Q
        tobeadditioned[i] = BF(
            newQ,
            newR,
            newP,
            higherkBF.PermQ,
            higherkBF.PermP,#needs to be formed anew....
            (
                H2Trees.values(
                    butterflycluster[1, 1].testcluster,
                    H2Trees.parent(
                        butterflycluster[1, 1].testcluster, butterflycluster[1, 1].NO
                    ),
                ),
                size(higherkBF, 2),
            ),
            H2Trees.parent(butterflycluster[1, 1].testcluster, butterflycluster[1, 1].NO),
            higherkBF.NS,
            higherkBF.k,
            higherkBF.τ,
            higherkBF.testcluster,
            higherkBF.trialcluster,
        )
    end
    result = tobeadditioned[1]
    for i in eachindex(tobeadditioned[2:end])
        result = addBFs(result, tobeadditioned[2:end][i])
    end
    return result
end
