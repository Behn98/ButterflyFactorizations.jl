function Base.adjoint(B::ButterflyFactorizations.BF)
    R_adj = Vector{Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},AbstractMatrix{ComplexF64}}}}(
        undef, length(B.R)
    )

    for l in eachindex(B.R)
        newl = length(B.R) - l + 1
        R_adj[newl] = Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},AbstractMatrix{ComplexF64}}}()
        for nodeS in keys(B.R[l])
            for nodeO in keys(B.R[l][nodeS])
                if !haskey(R_adj[newl], reverse(nodeO))
                    R_adj[newl][reverse(nodeO)] = Dict{
                        Tuple{Int,Int},AbstractMatrix{ComplexF64}
                    }()
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
        P_adj, R_adj, Q_adj, B.PermP, B.PermQ, (B.dim[2], B.dim[1]), B.NO, B.NS, B.k, B.τ
    )
end

function Base.transpose(B::ButterflyFactorizations.BF)
    R_tr = Vector{Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},AbstractMatrix{ComplexF64}}}}(
        undef, length(B.R)
    )

    for l in eachindex(B.R)
        newl = length(B.R) - l + 1
        R_tr[newl] = Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},AbstractMatrix{ComplexF64}}}()
        for nodeS in keys(B.R[l])
            for nodeO in keys(B.R[l][nodeS])
                if !haskey(R_tr[newl], reverse(nodeO))
                    R_tr[newl][reverse(nodeO)] = Dict{
                        Tuple{Int,Int},AbstractMatrix{ComplexF64}
                    }()
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
        P_tr, R_tr, Q_tr, B.PermP, B.PermQ, (B.dim[2], B.dim[1]), B.NO, B.NS, B.k, B.τ
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

function Base.adjoint(B::ButterflyFactorizations.AlgBF)
    lr = length(B.R)
    R_adj = Vector{Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},AbstractMatrix{ComplexF64}}}}(
        undef, lr
    )
    for l in eachindex(B.R)
        newl = lr - l + 1
        R_adj[newl] = Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},AbstractMatrix{ComplexF64}}}()
        for nodeS in keys(B.R[l])
            for nodeO in keys(B.R[l][nodeS])
                if !haskey(R_adj[newl], reverse(nodeO))
                    R_adj[newl][reverse(nodeO)] = Dict{
                        Tuple{Int,Int},AbstractMatrix{ComplexF64}
                    }()
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
    return AlgBF(reverse(B.dim), P_adj, R_adj, Q_adj)
end

function Base.transpose(B::ButterflyFactorizations.AlgBF)
    lr = length(B.R)
    R_adj = Vector{Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},AbstractMatrix{ComplexF64}}}}(
        undef, lr
    )
    for l in eachindex(B.R)
        newl = lr - l + 1
        R_adj[newl] = Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},AbstractMatrix{ComplexF64}}}()
        for nodeS in keys(B.R[l])
            for nodeO in keys(B.R[l][nodeS])
                if !haskey(R_adj[newl], reverse(nodeO))
                    R_adj[newl][reverse(nodeO)] = Dict{
                        Tuple{Int,Int},AbstractMatrix{ComplexF64}
                    }()
                end
                R_adj[newl][reverse(nodeO)][reverse(nodeS)] = transpose(
                    B.R[l][nodeS][nodeO]
                )
            end
        end
    end

    Q_adj = Dict{Int,Matrix{ComplexF64}}()
    for k in keys(B.Q)
        Q_adj[k] = transpose(B.Q[k])
    end

    P_adj = Dict{Int,Matrix{ComplexF64}}()
    for k in keys(B.P)
        P_adj[k] = transpose(B.P[k])
    end
    return AlgBF(reverse(B.dim), P_adj, R_adj, Q_adj)
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
