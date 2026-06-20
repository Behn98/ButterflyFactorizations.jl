"""
There are exactly two cases to consider when adding Butterfly factorizations (BFs).
In both cases, the corresponding matrix blocks are of equal dimensions.

Case 1 (Same Source and Observer Clusters):
The BFs map between the same observer and source clusters. Because the underlying
compression scheme relies on a single tree, the physical degrees of freedom (DoF)
match. Thus, the `Q` and `P` matrices can be appropriately combined without
disturbing the underlying physics, applying the addition described in the literature.

Case 2 (Disjoint Source and Observer Clusters):
The BFs represent disjoint source and observer clusters. A direct spatial
concatenation cannot occur, so the BFs are simply joined into a new structure.
Note that this resulting structure is purely algebraic and loses its direct physical
interpretation, similar to adding the two dense matrices directly.
"""
function add_eqbfs(BF_1_init::BF, BF_2_init::BF, τ)
    @assert BF_1_init.NS == BF_2_init.NS && BF_1_init.NO == BF_2_init.NO "rootids must match for addition."
    BF_1 = deepcopy(BF_1_init)
    BF_2 = deepcopy(BF_2_init)
    R_new = Vector{Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},Matrix{ComplexF64}}}}(
        undef, length(BF_1.R)
    )
    for l in eachindex(BF_1.R)
        R_new[l] = BF_1.R[l]
        for row in keys(BF_2.R[l])
            if !haskey(R_new[l], row)
                R_new[l][row] = BF_2.R[l][row]
                continue
            end
            for col in keys(BF_2.R[l][row])
                if haskey(R_new[l][row], col)
                    R_new[l][row][col] = blockdiag(BF_1.R[l][row][col], BF_2.R[l][row][col])
                else
                    R_new[l][row][col] = vcat(
                        zeros(
                            ComplexF64,
                            size(BF_1_init.R[l][row][first(keys(BF_1_init.R[l][row]))], 1),
                            size(BF_2_init.R[l][row][col], 2),
                        ),
                        BF_2.R[l][row][col],
                    )
                end
            end
            for col in keys(BF_1.R[l][row])
                if haskey(BF_2.R[l][row], col)
                    continue
                else
                    R_new[l][row][col] = vcat(
                        BF_1.R[l][row][col],
                        zeros(
                            ComplexF64,
                            size(BF_2_init.R[l][row][first(keys(BF_2_init.R[l][row]))], 1),
                            size(BF_1_init.R[l][row][col], 2),
                        ),
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

    for k in keys(BF_2.Q)
        if !haskey(BF_1.Q, k)
            Q_new[k] = BF_2.Q[k]
        end
    end

    P_new = Dict{Tuple{Int,Int},Matrix{ComplexF64}}()
    for k in keys(BF_1.P)
        if haskey(BF_2.P, k)
            P_new[k] = hcat(BF_1.P[k], BF_2.P[k])
        else
            P_new[k] = BF_1.P[k]
        end
    end

    for k in keys(BF_2.P)
        if !haskey(BF_1.P, k)
            P_new[k] = BF_2.P[k]
        end
    end

    return recompress_BF(
        BF(
            Q_new,
            R_new,
            P_new,
            merge(BF_1.PermQ, BF_2.PermQ),
            merge(BF_1.PermP, BF_2.PermP),
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
