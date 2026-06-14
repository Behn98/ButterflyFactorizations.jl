function Base.adjoint(B::ButterflyFactorizations.BF)
    R_adj = Vector{Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},Matrix{ComplexF64}}}}(
        undef, length(B.R)
    )

    for l in eachindex(B.R)
        newl = length(B.R) - l + 1
        R_adj[newl] = Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},Matrix{ComplexF64}}}()
        for nodeS in keys(B.R[l])
            for nodeO in keys(B.R[l][nodeS])
                if !haskey(R_adj[newl], reverse(nodeO))
                    R_adj[newl][reverse(nodeO)] = Dict{Tuple{Int,Int},Matrix{ComplexF64}}()
                end
                R_adj[newl][reverse(nodeO)][reverse(nodeS)] = adjoint(B.R[l][nodeS][nodeO])
            end
        end
    end

    Q_adj = Dict{Int,Matrix{ComplexF64}}()
    for k in keys(B.Q)
        Q_adj[k] = adjoint(B.Q[k])
    end

    P_adj = Dict{Int,Matrix{ComplexF64}}()
    for k in keys(B.P)
        P_adj[k] = adjoint(B.P[k])
    end

    return BF(
        P_adj,
        R_adj,
        Q_adj,
        B.PermP,
        B.PermQ,
        (B.dim[2], B.dim[1]),
        B.NO,
        B.NS,
        B.k,
        B.τ,
        B.otree,
        B.stree,
    )
end

function Base.transpose(B::ButterflyFactorizations.BF)
    R_tr = Vector{Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},Matrix{ComplexF64}}}}(
        undef, length(B.R)
    )

    for l in eachindex(B.R)
        newl = length(B.R) - l + 1
        R_tr[newl] = Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},Matrix{ComplexF64}}}()
        for nodeS in keys(B.R[l])
            for nodeO in keys(B.R[l][nodeS])
                if !haskey(R_tr[newl], reverse(nodeO))
                    R_tr[newl][reverse(nodeO)] = Dict{Tuple{Int,Int},Matrix{ComplexF64}}()
                end
                R_tr[newl][reverse(nodeO)][reverse(nodeS)] = transpose(B.R[l][nodeS][nodeO])
            end
        end
    end

    Q_tr = Dict{Int,Matrix{ComplexF64}}()
    for k in keys(B.Q)
        Q_tr[k] = transpose(B.Q[k])
    end

    P_tr = Dict{Int,Matrix{ComplexF64}}()
    for k in keys(B.P)
        P_tr[k] = transpose(B.P[k])
    end

    return BF(
        P_tr,
        R_tr,
        Q_tr,
        B.PermP,
        B.PermQ,
        (B.dim[2], B.dim[1]),
        B.NO,
        B.NS,
        B.k,
        B.τ,
        B.otree,
        B.stree,
    )
end

function Base.adjoint(t::ButterflyFactorizations.BF_Mats)
    return BF_Mats(
        t.P',                                                      # Q becomes P'
        AbstractMatrix{ComplexF64}[r' for r in Iterators.reverse(t.R)], # Reverse and map R
        t.Q',                                                      # P becomes Q'
        t.NO,                                                      # NS and NO swap roles
        t.NS,
        t.k,
        t.τ,
        t.PermQ,                                                   # Permutations swap roles
        t.PermP,
    )
end

function Base.transpose(t::ButterflyFactorizations.BF_Mats)
    return BF_Mats(
        transpose(t.P),
        AbstractMatrix{ComplexF64}[transpose(r) for r in Iterators.reverse(t.R)],
        transpose(t.Q),
        t.NO,
        t.NS,
        t.k,
        t.τ,
        t.PermQ,
        t.PermP,
    )
end

function Base.adjoint(B::ButterflyFactorizations.AlgBF{T}) where {T} # 1. Added {T} here
    lr = length(B.R)
    R_adj = Vector{Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},Matrix{ComplexF64}}}}(undef, lr)

    # 2. Fix: Parameterize this vector with {T} so it is not a UnionAll vector
    Rf_adj = Vector{R_factor{T}}(undef, lr)

    for l in eachindex(B.R)
        newl = lr - l + 1
        R_adj[newl] = Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},Matrix{ComplexF64}}}()
        for skel1 in keys(B.R[l].Dict)
            for skel2 in keys(B.R[l].Dict[skel1])
                if !haskey(R_adj[newl], reverse(skel2))
                    R_adj[newl][reverse(skel2)] = Dict{Tuple{Int,Int},Matrix{ComplexF64}}()
                end
                R_adj[newl][reverse(skel2)][reverse(skel1)] = adjoint(
                    B.R[l].Dict[skel1][skel2]
                )
            end
        end

        # 3. Explicitly construct R_factor with {T}
        Rf_adj[newl] = R_factor{T}(
            R_adj[newl],
            reverse(B.R[l].olvl),
            reverse(B.R[l].slvl),
            B.R[l].colotree,
            B.R[l].colstree,
            B.R[l].rowstree,
            B.R[l].rowotree,
        )
    end

    Q_adj = Dict{Int,Matrix{ComplexF64}}()
    for k in keys(B.Q.Dict)
        Q_adj[k] = adjoint(B.Q.Dict[k])
    end
    # 4. Explicitly construct P_factor with {T}
    Qf_adj = P_factor{T}(Q_adj, B.Q.stree)

    P_adj = Dict{Int,Matrix{ComplexF64}}()
    for k in keys(B.P.Dict)
        P_adj[k] = adjoint(B.P.Dict[k])
    end
    # 5. Explicitly construct Q_factor with {T}
    Pf_adj = Q_factor{T}(P_adj, B.P.otree)

    return AlgBF(reverse(B.dim), Pf_adj, Rf_adj, Qf_adj)
end

function Base.transpose(B::ButterflyFactorizations.AlgBF{T}) where {T}
    lr = length(B.R)
    R_adj = Vector{Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},Matrix{ComplexF64}}}}(undef, lr)
    Rf_adj = Vector{R_factor{T}}(undef, lr)
    for l in eachindex(B.R)
        newl = lr - l + 1
        R_adj[newl] = Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},Matrix{ComplexF64}}}()
        for skel1 in keys(B.R[l].Dict)
            for skel2 in keys(B.R[l].Dict[skel1])
                if !haskey(R_adj[newl], reverse(skel2))
                    R_adj[newl][reverse(skel2)] = Dict{Tuple{Int,Int},Matrix{ComplexF64}}()
                end
                R_adj[newl][reverse(skel2)][reverse(skel1)] = transpose(
                    B.R[l].Dict[skel1][skel2]
                )
            end
        end
        Rf_adj[newl] = R_factor{T}(
            R_adj[newl],
            reverse(B.R[l].olvl),
            reverse(B.R[l].slvl),
            B.R[l].colotree,
            B.R[l].colstree,
            B.R[l].rowotree,
            B.R[l].rowstree,
        )
    end

    Q_adj = Dict{Int,Matrix{ComplexF64}}()
    for k in keys(B.Q.Dict)
        Q_adj[k] = transpose(B.Q.Dict[k])
    end
    Qf_adj = P_factor{T}(Q_adj, B.Q.stree)

    P_adj = Dict{Int,Matrix{ComplexF64}}()
    for k in keys(B.P.Dict)
        P_adj[k] = transpose(B.P.Dict[k])
    end
    Pf_adj = Q_factor{T}(P_adj, B.P.otree)
    return AlgBF(reverse(B.dim), Pf_adj, Rf_adj, Qf_adj)
end

"""
    adjoint(bf::FlatBF)

Returnerar en ny `FlatBF` som representerar det konjugerade transponatet (Aᴴ).
"""
Base.adjoint(bf::FlatBF) = transform_bf(bf, true)

"""
    transpose(bf::FlatBF)

Returnerar en ny `FlatBF` som representerar transponatet (Aᵀ).
"""
Base.transpose(bf::FlatBF) = transform_bf(bf, false)
