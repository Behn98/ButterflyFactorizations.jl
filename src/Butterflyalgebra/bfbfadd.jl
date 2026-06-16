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
function add_eqbfs(BF1::BF, BF_2::BF, τ)
    @assert BF1.NS == BF_2.NS && BF1.NO == BF_2.NO "rootids must match for addition."
    # --- Case 1: Same source and observer clusters ---
    R_new = Vector{Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},Matrix{ComplexF64}}}}(
        undef, length(BF1.R)
    )
    for l in eachindex(BF1.R)
        R_new[l] = Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},Matrix{ComplexF64}}}()
        for nodeS in keys(BF1.R[l])
            for nodeO in keys(BF1.R[l][nodeS])
                if !haskey(R_new[l], nodeS)
                    R_new[l][nodeS] = Dict{Tuple{Int,Int},Matrix{ComplexF64}}()
                end
                R_new[l][nodeS][nodeO] = blockdiag(
                    BF1.R[l][nodeS][nodeO], BF_2.R[l][nodeS][nodeO]
                )
            end
        end
    end
    Q_new = Dict{Int,Matrix{ComplexF64}}()
    for k in keys(BF1.Q)
        Q_new[k] = vcat(BF1.Q[k], BF_2.Q[k])
    end

    P_new = Dict{Int,Matrix{ComplexF64}}()
    for k in keys(BF1.P)
        P_new[k] = hcat(BF1.P[k], BF_2.P[k])
    end

    return recompress_BF(
        BF(
            Q_new,
            R_new,
            P_new,
            BF1.PermQ,
            BF1.PermP,
            BF1.dim,
            BF1.NS,
            BF1.NO,
            BF1.k,
            max(BF1.τ, BF_2.τ),
            BF1.stree,
            BF1.otree,
        ),
        τ,
    )
end

function add_neqbfs(BF1::BF, BF_2::BF)
    return (BF1, BF_2)   #insert struct here if needed
end
