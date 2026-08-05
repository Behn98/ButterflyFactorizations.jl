"""
    add_eqbfs(BF_1_init::ButterflyFactorization, BF_2_init::ButterflyFactorization, τ) -> BF

Add two flat-array Butterfly Factorization (`BF`) representations together and recompress the result.

This function implements the addition of two hierarchical butterfly representations by combining
their respective structural components (`P`, `R`, and `Q` matrices) and subsequently truncating
the resulting operator to a desired accuracy tolerance.

# Arguments

  - `BF_1_init::ButterflyFactorization`: The first butterfly factorization operand.
  - `BF_2_init::ButterflyFactorization`: The second butterfly factorization operand. Must share the same number of source (`ns`) and output (`no`) root nodes as `BF_1_init`.
  - `τ`: The tolerance threshold used for the final recompression step.

# Returns

  - `BF`: A new, recompressed `BF` object representing the operator sum of the inputs.

# Implementation Details

  - **P Matrices:** Merged using horizontal concatenation (`hcat`).
  - **Q Matrices:** Merged using vertical concatenation (`vcat`).
  - **R Matrices:** Intersecting block keys across layers are combined via block-diagonalization (`blockdiag`). Unmatched keys are preserved.
  - **Recompression:** The final structural configuration is passed to `recompress_BF` with the parameter `τ` to optimize memory and rank efficiency.
    *(Note: The inner constructor of `ButterflyFactorization` will automatically regenerate the flat-array memory execution pointers `in_ptr` and `out_ptr` for the newly merged blocks).*
"""
function add_eqbfs(
    BF_1_init::ButterflyFactorization{T,M}, BF_2_init::ButterflyFactorization{T,M}, τ
) where {T,M}
    @assert length(BF_1_init.R) == length(BF_2_init.R) "BFs must have the same number of layers"
    @assert getnsno(BF_1_init) == getnsno(BF_2_init) "BFs must have the same number of source and output root nodes"

    BF1 = deepcopy(BF_1_init)
    BF2 = deepcopy(BF_2_init)

    # ------------------------------------------------------------------
    # 1. Merge Q Factors (Vertical Concatenation)
    # ------------------------------------------------------------------
    Q1idx = Dict{Int,Int}()
    for (i, b) in enumerate(BF1.Q)
        Q1idx[getrowidx(b)[2]] = i
    end

    newQ = Vector{ButterflyBlock{T}}(undef, length(BF1.Q))
    for b in BF2.Q
        idx = getrowidx(b)[2]
        if haskey(Q1idx, idx)
            newQ[Q1idx[idx]] = ButterflyBlock(
                b.obs_out,
                b.src_out,
                b.obs_in,
                b.src_in,
                vcat(BF1.Q[Q1idx[idx]].data, b.data),
            )
            delete!(Q1idx, idx)
        else
            push!(newQ, b)
            println("Adding new block to Q level")
        end
    end

    for (src, i) in Q1idx
        println("Preserving existing block in Q level")
        newQ[i] = BF1.Q[i]
    end

    # ------------------------------------------------------------------
    # 2. Merge R Factors (Block Diagonalization)
    # ------------------------------------------------------------------
    newR = Vector{ButterflyLevel{T}}(undef, length(BF1.R))
    for l in 1:length(BF1.R)
        R1idx = Dict{Tuple{Int,Int,Int,Int},Int}()
        for (i, b) in enumerate(BF1.R[l].blocks)
            R1idx[block_key(b)] = i
        end

        newRlvl = Vector{ButterflyBlock{T}}(undef, length(BF1.R[l].blocks))
        for b in BF2.R[l].blocks
            idx = block_key(b)
            if haskey(R1idx, idx)
                # Ensure custom blockdiag for ButterflyBlocks is called
                newRlvl[R1idx[idx]] = blockdiag(BF1.R[l].blocks[R1idx[idx]], b)
                delete!(R1idx, idx)
            else
                println("Adding new block to R level $l: ")
                push!(newRlvl, b)
            end
        end

        for (blkkey, i) in R1idx
            println("Preserving existing block in R level $l")
            newRlvl[i] = BF1.R[l].blocks[i]
        end
        newR[l] = ButterflyLevel(newRlvl)
    end

    # ------------------------------------------------------------------
    # 3. Merge P Factors (Horizontal Concatenation)
    # ------------------------------------------------------------------
    P1idx = Dict{Int,Int}()
    for (i, b) in enumerate(BF1.P)
        P1idx[getcolidx(b)[1]] = i
    end

    newP = Vector{ButterflyBlock{T}}(undef, length(BF1.P))
    for b in BF2.P
        idx = getcolidx(b)[1]
        if haskey(P1idx, idx)
            newP[P1idx[idx]] = ButterflyBlock(
                b.obs_out,
                b.src_out,
                b.obs_in,
                b.src_in,
                hcat(BF1.P[P1idx[idx]].data, b.data),
            )
            delete!(P1idx, idx)
        else
            push!(newP, b)
            println("Adding new block to P level")
        end
    end

    for (obs, i) in P1idx
        println("Preserving existing block in P level")
        newP[i] = BF1.P[i]
    end

    # 4. Recompress and Auto-Generate Memory Pointers via Inner Constructor
    return recompress_BF(ButterflyFactorization(newQ, newR, newP, BF1.tree, BF1.k, τ), τ)
end

"""
    +(BF_1::ButterflyFactorization, BF_2::ButterflyFactorization) -> BF

Overloads the `+` operator to safely add two hierarchical `ButterflyFactorization`s together.
Defaults to using the maximum target tolerance (`τ`) of the two input blocks for the
subsequent recompression.
"""
Base.:+(BF_1::ButterflyFactorization{T,M}, BF_2::ButterflyFactorization{T,M}) where {T,M} =
    add_eqbfs(BF_1, BF_2, max(BF_1.τ, BF_2.τ))
