"""
    add_eqbfs(BF_1_init::BF, BF_2_init::BF, τ) -> BF

Add two Butterfly Factorization (`BF`) representations together and recompress the result.

This function implements the addition of two hierarchical butterfly representations by combining
their respective structural components (`P`, `R`, and `Q` matrices) and subsequently truncating
the resulting operator to a desired accuracy tolerance.

# Arguments

  - `BF_1_init::BF`: The first butterfly factorization operand.
  - `BF_2_init::BF`: The second butterfly factorization operand. Must share the same number of source (`NS`) and output (`NO`) nodes as `BF_1_init`.
  - `τ`: The tolerance threshold used for the final recompression step.

# Returns

  - `BF`: A new, recompressed `BF` object representing the operator sum of the inputs.

# Implementation Details

  - **P Matrices:** Merged using horizontal concatenation (`hcat`).
  - **Q Matrices:** Merged using vertical concatenation (`vcat`).
  - **R Matrices:** Intersecting block keys across layers are combined via block-diagonalization (`blockdiag`). Unmatched keys are safely padded with zeros to preserve structural dimensions before block-diagonalization.
  - **Recompression:** The final structural configuration is passed to `recompress_BF` with the parameter `τ` to optimize memory and rank efficiency.
"""
function add_eqbfs(BF_1_init::BF, BF_2_init::BF, τ)
    @assert BF_1_init.NS == BF_2_init.NS && BF_1_init.NO == BF_2_init.NO "rootids must match for addition."
    BF_1 = deepcopy(BF_1_init)
    BF_2 = deepcopy(BF_2_init)

    P_new = Dict{Tuple{Int,Int},Matrix{ComplexF64}}()
    for k in keys(BF_1.P)
        if haskey(BF_2.P, k)
            P_new[k] = hcat(BF_1.P[k], BF_2.P[k])
        else
            P_new[k] = BF_1.P[k]
        end
    end
    #single = 0
    #comb = 0
    #=
    for k in keys(BF_2.P)
        if !haskey(BF_1.P, k)
            P_new[k] = BF_2.P[k]
        end
    end
    =#
    R_new = Vector{Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},Matrix{ComplexF64}}}}(
        undef, length(BF_1.R)
    )
    for l in eachindex(BF_1.R)
        R_new[l] = Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},Matrix{ComplexF64}}}()
        col_to_rows1 = Dict{Tuple{Int,Int},Vector{Tuple{Int,Int}}}()
        for row_skel in keys(BF_1.R[l])
            for col_idx in keys(BF_1.R[l][row_skel])
                if !haskey(col_to_rows1, col_idx)
                    col_to_rows1[col_idx] = Vector{Tuple{Int,Int}}()
                end
                push!(col_to_rows1[col_idx], row_skel)
            end
        end
        col_to_rows2 = Dict{Tuple{Int,Int},Vector{Tuple{Int,Int}}}()
        for row_skel in keys(BF_2.R[l])
            for col_idx in keys(BF_2.R[l][row_skel])
                if !haskey(col_to_rows2, col_idx)
                    col_to_rows2[col_idx] = Vector{Tuple{Int,Int}}()
                end
                push!(col_to_rows2[col_idx], row_skel)
            end
        end
        newrowspace = Dict{Tuple{Int,Int},Int}()
        for row in keys(BF_1.R[l])
            if haskey(BF_2.R[l], row)
                newrowspace[row] =
                    size(BF_1.R[l][row][first(keys(BF_1.R[l][row]))], 1) +
                    size(BF_2.R[l][row][first(keys(BF_2.R[l][row]))], 1)
            else
                newrowspace[row] = size(BF_1.R[l][row][first(keys(BF_1.R[l][row]))], 1)
            end
        end
        for row in keys(BF_2.R[l])
            if !haskey(newrowspace, row)
                newrowspace[row] = size(BF_2.R[l][row][first(keys(BF_2.R[l][row]))], 1)
            end
        end
        newcolspace = Dict{Tuple{Int,Int},Int}()
        for col in keys(col_to_rows1)
            if haskey(col_to_rows2, col)
                newcolspace[col] =
                    size(BF_1.R[l][col_to_rows1[col][1]][col], 2) +
                    size(BF_2.R[l][col_to_rows2[col][1]][col], 2)
            else
                newcolspace[col] = size(BF_1.R[l][col_to_rows1[col][1]][col], 2)
            end
        end
        for col in keys(col_to_rows2)
            if !haskey(newcolspace, col)
                newcolspace[col] = size(BF_2.R[l][col_to_rows2[col][1]][col], 2)
            end
        end
        for row in keys(newrowspace)
            R_new[l][row] = Dict{Tuple{Int,Int},Matrix{ComplexF64}}()
            for col in keys(newcolspace)
                if haskey(BF_1.R[l], row) &&
                    haskey(BF_1.R[l][row], col) &&
                    haskey(BF_2.R[l], row) &&
                    haskey(BF_2.R[l][row], col)
                    #comb += 1
                    R_new[l][row][col] = blockdiag(BF_1.R[l][row][col], BF_2.R[l][row][col])
                elseif haskey(BF_1.R[l], row) && haskey(BF_1.R[l][row], col)
                    #single += 1
                    R_new[l][row][col] = blockdiag(
                        BF_1.R[l][row][col],
                        zeros(
                            ComplexF64,
                            newrowspace[row]-size(BF_1.R[l][row][col], 1),
                            newcolspace[col]-size(BF_1.R[l][row][col], 2),
                        ),
                    )
                elseif haskey(BF_2.R[l], row) && haskey(BF_2.R[l][row], col)
                    #single += 1
                    R_new[l][row][col] = blockdiag(
                        zeros(
                            ComplexF64,
                            newrowspace[row]-size(BF_2.R[l][row][col], 1),
                            newcolspace[col]-size(BF_2.R[l][row][col], 2),
                        ),
                        BF_2.R[l][row][col],
                    )
                end
            end
        end
    end
    Q_new = Dict{Tuple{Int,Int},Matrix{ComplexF64}}()
    for k in keys(BF_1.Q)
        if haskey(BF_2.Q, k)
            Q_new[k] = vcat(BF_1.Q[k], BF_2.Q[k])
        else
            Q_new[k] = BF_1.Q[k]
        end
    end
    #=
    for k in keys(BF_2.Q)
        if !haskey(BF_1.Q, k)
            Q_new[k] = BF_2.Q[k]
        end
    end
    =#
    #@show single
    #@show comb
    return recompress_BF(
        BF(
            Q_new,
            R_new,
            P_new,
            BF_1.dim,
            BF_1.NS,
            BF_1.NO,
            BF_1.k,
            max(BF_1.τ, BF_2.τ),
            BF_1.stree,
            BF_1.otree,
        ),
        τ,
    )
end

function add_neqbfs(BF_1::BF, BF_2::BF)
    return (BF_1, BF_2)   #insert struct here if needed
end

function add_eqbfs(BF_1_init::AlgBF, BF_2_init::AlgBF, τ)
    BF_1 = BF(BF_1_init, 0, τ)
    BF_2 = BF(BF_2_init, 0, τ)
    BF_sum = add_eqbfs(BF_1, BF_2, τ)
    return AlgBF(BF_sum)
end
