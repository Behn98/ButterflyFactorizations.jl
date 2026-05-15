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
            result = swap_and_recompress(result, L + 2 - t, τ)
        end
        result = recompress_BF(mul_factors(result, L + 1 - m), τ)
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

function swap_and_recompress(BF::AlgBF, idx::Int, τ)
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
    nlfactor, nrfactor = swap_and_recompress(leftfactor, rightfactor, τ)

    return AlgBF(
        (size(BF, 1), size(BF, 2)),
        BF.Q,
        vcat(BF.R[1:(L - idx)], [nrfactor, nlfactor], BF.R[(L - idx + 3):length(BF.R)]),
        BF.P,
    )
end

# A generic factor is just Dict{RowKey, Dict{ColKey, Matrix}}
function swap_and_recompress(LeftFactor, RightFactor, τ)
    NewLeftFactor = Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},AbstractMatrix{ComplexF64}}}()
    NewRightFactor = Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},AbstractMatrix{ComplexF64}}}()
    #LeftFactor = reverse_dict_keys(LeftFactor)
    #RightFactor = reverse_dict_keys(RightFactor)
    NewLeftFactor = mul_factors(LeftFactor, RightFactor)
    col = Vector{Tuple{Int,Int}}(undef, 0)
    for row in keys(NewLeftFactor)
        col = unique(append!(col, keys(NewLeftFactor[row])))
    end
    for col_idx in col
        rows_with_col = [
            row for (row, inner_dict) in NewLeftFactor if haskey(inner_dict, col_idx)
        ]
        R_k = Vector{Matrix{ComplexF64}}()
        row_spc = Vector{Int}()
        i = 1
        for row in rows_with_col
            push!(R_k, NewLeftFactor[row][col_idx])
            push!(row_spc, size(R_k[i], 1))

            i += 1
        end
        A_k = vcat(R_k...)
        #@show size(A_k)
        QRA = pqr(A_k; rtol=τ)
        if !haskey(NewRightFactor, col_idx)
            NewRightFactor[col_idx] = Dict{Int,AbstractMatrix{ComplexF64}}()
        end
        #=
        if haskey(R_u[col_idx], col_idx)
            @show "Warning: Collision in R_u at column index $col_idx"
        end
        =#
        NewRightFactor[col_idx][col_idx] = QRA[2][:, invperm(QRA[3])]
        last = 0
        j = 1
        for row in rows_with_col
            NewLeftFactor[row][col_idx] = QRA[1][(last + 1):(last + row_spc[j]), :]
            last += row_spc[j]
            j += 1
        end
    end

    return NewLeftFactor, NewRightFactor
end

function swap_and_recompress2(LeftFactor, RightFactor, τ; kmax=100)

    # store low-rank triples per (row,col)
    Acc = Dict{
        Tuple{Tuple{Int,Int},Tuple{Int,Int}},
        Tuple{
            Union{Matrix{ComplexF64},Nothing},
            Union{Vector{Float64},Nothing},
            Union{Matrix{ComplexF64},Nothing},
        },
    }()

    for (row, innerL) in LeftFactor
        for (k, Lblock) in innerL
            if !haskey(RightFactor, k)
                continue
            end

            for (col, Rblock) in RightFactor[k]
                key = (row, col)

                A_new = Lblock * Rblock  # small

                if !haskey(Acc, key)
                    Acc[key] = (nothing, nothing, nothing)
                end

                U, S, V = Acc[key]

                U, S, V = lowrank_add!(U, S, V, A_new, τ, kmax)

                Acc[key] = (U, S, V)
            end
        end
    end

    # split into Left/Right factors
    NewLeft = Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},Matrix{ComplexF64}}}()
    NewRight = Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},Matrix{ComplexF64}}}()

    for ((row, col), (U, S, V)) in Acc
        if U === nothing
            continue
        end

        sqrtS = sqrt.(S)

        UL = U * Diagonal(sqrtS)
        VR = Diagonal(sqrtS) * V'

        if !haskey(NewLeft, row)
            NewLeft[row] = Dict{Tuple{Int,Int},Matrix{ComplexF64}}()
        end
        NewLeft[row][col] = UL

        if !haskey(NewRight, col)
            NewRight[col] = Dict{Tuple{Int,Int},Matrix{ComplexF64}}()
        end
        NewRight[col][col] = VR
    end

    return NewLeft, NewRight
end

function lowrank_add!(U, S, V, A_new, τ, kmax)
    # current approx: U * Diagonal(S) * V'
    # add: A_new

    if U === nothing
        # first contribution
        F = svd(A_new; full=false)
        r = min(sum(F.S .> τ), kmax)
        return F.U[:, 1:r], F.S[1:r], F.V[:, 1:r]
    end

    # form small augmented matrix
    A_old = U * Diagonal(S) * V'
    A = A_old + A_new   # small (k-sized), safe

    F = svd(A; full=false)
    r = min(sum(F.S .> τ), kmax)

    return F.U[:, 1:r], F.S[1:r], F.V[:, 1:r]
end

function LinearAlgebra.mul!(
    C::ButterflyFactorizations.BF,
    A::ButterflyFactorizations.BF,
    B::ButterflyFactorizations.BF,
)
    copyto!(C, mulBFs(A, B, max(A.τ, B.τ)))
    return C
end
