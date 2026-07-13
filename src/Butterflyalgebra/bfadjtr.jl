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

    Q_adj = Dict{Tuple{Int,Int},Matrix{ComplexF64}}()
    for k in keys(B.Q)
        Q_adj[reverse(k)] = adjoint(B.Q[k])
    end

    P_adj = Dict{Tuple{Int,Int},Matrix{ComplexF64}}()
    for k in keys(B.P)
        P_adj[reverse(k)] = adjoint(B.P[k])
    end

    return BF(
        P_adj, R_adj, Q_adj, (B.dim[2], B.dim[1]), B.NO, B.NS, B.k, B.τ, B.otree, B.stree
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

    Q_tr = Dict{Tuple{Int,Int},Matrix{ComplexF64}}()
    for k in keys(B.Q)
        Q_tr[reverse(k)] = transpose(B.Q[k])
    end

    P_tr = Dict{Tuple{Int,Int},Matrix{ComplexF64}}()
    for k in keys(B.P)
        P_tr[reverse(k)] = transpose(B.P[k])
    end

    return BF(
        P_tr, R_tr, Q_tr, (B.dim[2], B.dim[1]), B.NO, B.NS, B.k, B.τ, B.otree, B.stree
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

function Base.adjoint(R::R_factor{T,M}) where {T,M}
    # 1. Transform the element blocks (Dual-Layer Transposition)
    new_elementblocks = map(R.elementblocks) do grid
        old_rows, old_cols = size(grid)

        # Allocate a new grid matrix with flipped dimensions
        new_grid = Matrix{eltype(grid)}(undef, old_cols, old_rows)

        for j in 1:old_cols
            for i in 1:old_rows
                # Move element from (i,j) to (j,i) and eagerly compute its adjoint
                # collect() unwraps the lazy Adjoint view back into a standard Matrix{T}
                new_grid[j, i] = collect(adjoint(grid[i, j]))
            end
        end
        return new_grid
    end

    # 2. Transform the inverse map (Transpose grids + swap internal coordinate pairs)
    new_inverse_map = map(R.inverse_map) do imap
        old_rows, old_cols = size(imap)
        new_imap = Matrix{eltype(imap)}(undef, old_cols, old_rows)
        for j in 1:old_cols
            for i in 1:old_rows
                val = imap[i, j]
                if val === nothing
                    new_imap[j, i] = nothing
                else
                    row_key, col_key = val
                    # Swap the semantic keys because rows and columns have flipped roles
                    new_imap[j, i] = (reverse(col_key), reverse(row_key))
                end
            end
        end
        return new_imap
    end

    # 3. Transform the semantic lookup dictionary
    # We dynamically infer the key types based on your row/col tuples
    new_lookup = Dict{Any,Dict{Any,Tuple{Int,Int,Int}}}()

    for (row_key, col_dict) in R.dict
        for (col_key, (grid_idx, r_idx, c_idx)) in col_dict
            # Look up or initialize the new outer key (which was the old column key)
            inner_dict = get!(
                () -> Dict{Any,Tuple{Int,Int,Int}}(), new_lookup, reverse(col_key)
            )

            # CRUCIAL: Swap r_idx and c_idx because the underlying grid matrix was transposed!
            inner_dict[reverse(row_key)] = (grid_idx, c_idx, r_idx)
        end
    end

    # Return a brand new, fully valid instance of your struct
    return R_factor{T,M}(
        new_lookup,
        new_inverse_map,
        new_elementblocks,
        reverse(R.slvl),
        reverse(R.olvl),
        R.colotree,
        R.colstree,
        R.rowotree,
        R.rowstree,
    )
end

function Base.transpose(R::R_factor{T,M}) where {T,M}
    # 1. Transform the element blocks (Dual-Layer Transposition)
    new_elementblocks = map(R.elementblocks) do grid
        old_rows, old_cols = size(grid)

        # Allocate a new grid matrix with flipped dimensions
        new_grid = Matrix{eltype(grid)}(undef, old_cols, old_rows)

        for j in 1:old_cols
            for i in 1:old_rows
                new_grid[j, i] = collect(transpose(grid[i, j]))
            end
        end
        return new_grid
    end

    # 2. Transform the inverse map (Transpose grids + swap internal coordinate pairs)
    new_inverse_map = map(R.inverse_map) do imap
        old_rows, old_cols = size(imap)
        new_imap = Matrix{eltype(imap)}(undef, old_cols, old_rows)
        for j in 1:old_cols
            for i in 1:old_rows
                val = imap[i, j]
                if val === nothing
                    new_imap[j, i] = nothing
                else
                    row_key, col_key = val
                    # Swap the semantic keys because rows and columns have flipped roles
                    new_imap[j, i] = (reverse(col_key), reverse(row_key))
                end
            end
        end
        return new_imap
    end

    # 3. Transform the semantic lookup dictionary
    # We dynamically infer the key types based on your row/col tuples
    new_lookup = Dict{Any,Dict{Any,Tuple{Int,Int,Int}}}()

    for (row_key, col_dict) in R.dict
        for (col_key, (grid_idx, r_idx, c_idx)) in col_dict
            # Look up or initialize the new outer key (which was the old column key)
            inner_dict = get!(
                () -> Dict{Any,Tuple{Int,Int,Int}}(), new_lookup, reverse(col_key)
            )

            # CRUCIAL: Swap r_idx and c_idx because the underlying grid matrix was transposed!
            inner_dict[reverse(row_key)] = (grid_idx, c_idx, r_idx)
        end
    end

    # Return a brand new, fully valid instance of your struct
    return R_factor{T,M}(
        new_lookup,
        new_inverse_map,
        new_elementblocks,
        reverse(R.slvl),
        reverse(R.olvl),
        R.colotree,
        R.colstree,
        R.rowotree,
        R.rowstree,
    )
end

function Base.adjoint(B::ButterflyFactorizations.AlgBF{T,M}) where {T,M} # 1. Added {T,M} here
    lr = length(B.R)

    Rf_adj = Vector{R_factor{T,M}}(undef, lr)

    for l in eachindex(B.R)
        newl = lr - l + 1
        # 3. Explicitly construct R_factor with {T}
        Rf_adj[newl] = B.R[l]'
    end

    Q_adj = Dict{Tuple{Int,Int},Matrix{ComplexF64}}()
    for k in keys(B.Q.dict)
        Q_adj[reverse(k)] = adjoint(B.Q.dict[k])
    end
    # 4. Explicitly construct P_factor with {T}
    Qf_adj = P_factor{T,M}(Q_adj, B.Q.stree, B.Q.otree)

    P_adj = Dict{Tuple{Int,Int},Matrix{ComplexF64}}()
    for k in keys(B.P.dict)
        P_adj[reverse(k)] = adjoint(B.P.dict[k])
    end
    # 5. Explicitly construct Q_factor with {T}
    Pf_adj = Q_factor{T,M}(P_adj, B.P.otree, B.P.stree)

    return AlgBF(reverse(B.dim), Pf_adj, Rf_adj, Qf_adj)
end

function Base.transpose(B::ButterflyFactorizations.AlgBF{T,M}) where {T,M}
    lr = length(B.R)
    R_adj = Vector{Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},Matrix{ComplexF64}}}}(undef, lr)
    Rf_adj = Vector{R_factor{T,M}}(undef, lr)
    for l in eachindex(B.R)
        newl = lr - l + 1
        # 3. Explicitly construct R_factor with {T}
        Rf_adj[newl] = transpose(B.R[l])
    end

    Q_adj = Dict{Tuple{Int,Int},Matrix{ComplexF64}}()
    for k in keys(B.Q.dict)
        Q_adj[reverse(k)] = transpose(B.Q.dict[k])
    end
    Qf_adj = P_factor{T,M}(Q_adj, B.Q.stree, B.Q.otree)

    P_adj = Dict{Tuple{Int,Int},Matrix{ComplexF64}}()
    for k in keys(B.P.dict)
        P_adj[reverse(k)] = transpose(B.P.dict[k])
    end
    Pf_adj = Q_factor{T,M}(P_adj, B.P.otree, B.P.stree)
    return AlgBF{T,M}(reverse(B.dim), Pf_adj, Rf_adj, Qf_adj)
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
