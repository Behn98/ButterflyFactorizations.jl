function flatten_bf(bf::BF)
    L = length(bf.R)
    NO = bf.NO
    NS = bf.NS

    # -- NYTT: Om BF bara består av Q och P (nivå 0 för R) --
    if L == 0
        q_blocks = Matrix{ComplexF64}[]
        q_col_offsets = Int[]
        q_perms = Vector{Int}[]
        curr_offset_q = 1
        for (Sleaf, block) in bf.Q
            push!(q_blocks, block)
            push!(q_perms, bf.PermQ[Sleaf])
            push!(q_col_offsets, curr_offset_q)
            curr_offset_q += size(block, 2) # Ut-dimension för Q
        end
        flat_Q = FlatQLayer(q_blocks, q_col_offsets, q_perms)

        p_blocks = Matrix{ComplexF64}[]
        p_row_offsets = Int[]
        p_perms = Vector{Int}[]
        curr_offset_p = 1
        for (Oleaf, block) in bf.P
            push!(p_blocks, block)
            push!(p_perms, bf.PermP[Oleaf])
            push!(p_row_offsets, curr_offset_p)
            curr_offset_p += size(block, 2) # In-dimension för P
        end
        flat_P = FlatPLayer(p_blocks, p_row_offsets, p_perms)

        # Returnera direkt med en tom array av FlatLinearLayer
        return FlatBF(flat_Q, FlatLinearLayer[], flat_P, bf.dim, NS, NO)
    end
    # -------------------------------------------------------

    # 1. Bygg synkroniserade ID-mappningar för alla R-nivåer
    R_col_maps = Vector{Dict{Tuple{Int,Int},Int32}}(undef, L)
    R_row_maps = Vector{Dict{Tuple{Int,Int},Int32}}(undef, L)

    # Nivå 1: Samla alla unika kolumn-nycklar
    cols_1 = Set{Tuple{Int,Int}}()
    for (r_key, cols_dict) in bf.R[1], c_key in keys(cols_dict)
        push!(cols_1, c_key)
    end
    R_col_maps[1] = Dict(k => Int32(i) for (i, k) in enumerate(sort!(collect(cols_1))))

    # Kedjekoppla resten av nivåerna: rows[l] blir cols[l+1]
    for l in 1:L
        rows_l = sort!(collect(keys(bf.R[l])))
        R_row_maps[l] = Dict(k => Int32(i) for (i, k) in enumerate(rows_l))
        if l < L
            R_col_maps[l + 1] = R_row_maps[l]
        end
    end

    # 2. Platta till R-nivåerna till FlatLinearLayer
    flat_R = Vector{FlatLinearLayer}(undef, L)
    for l in 1:L
        R_dict = bf.R[l]
        row_map = R_row_maps[l]
        col_map = R_col_maps[l]

        # Räkna ut skalära storlekar/offsets för kolumner
        col_sizes = zeros(Int, length(col_map))
        for (r_key, cols_dict) in R_dict, (c_key, block) in cols_dict
            col_sizes[col_map[c_key]] = size(block, 2)
        end
        col_offsets = Vector{Int}(undef, length(col_map))
        curr = 1
        for i in 1:length(col_map)
            col_offsets[i] = curr
            curr += col_sizes[i]
        end
        in_size = curr - 1

        # Räkna ut skalära storlekar/offsets för rader
        row_sizes = zeros(Int, length(row_map))
        for (r_key, cols_dict) in R_dict
            row_sizes[row_map[r_key]] = size(first(Base.values(cols_dict)), 1)
        end
        row_offsets = Vector{Int}(undef, length(row_map))
        curr = 1
        for i in 1:length(row_map)
            row_offsets[i] = curr
            curr += row_sizes[i]
        end
        out_size = curr - 1

        # Bygg CSR-struktur
        row_ptr = Vector{Int32}(undef, length(row_map) + 1)
        col_idx = Int32[]
        blocks = Matrix{ComplexF64}[]

        block_counter = 1
        for (i, r_key) in enumerate(sort!(collect(keys(row_map))))
            row_ptr[i] = block_counter
            cols_dict = R_dict[r_key]

            # Sortera kolumner för maximal cache-prestanda i CSR
            sorted_c_keys = sort!(collect(keys(cols_dict)); by=k -> col_map[k])
            for c_key in sorted_c_keys
                push!(col_idx, col_map[c_key])
                push!(blocks, cols_dict[c_key])
                block_counter += 1
            end
        end
        row_ptr[end] = block_counter

        flat_R[l] = FlatLinearLayer(
            row_ptr, col_idx, blocks, row_offsets, col_offsets, in_size, out_size
        )
    end

    # 3. Platta till Q (Kopplas till R[1]:s kolumn-offsets)
    q_blocks = Matrix{ComplexF64}[]
    q_col_offsets = Int[]
    q_perms = Vector{Int}[]
    for (Sleaf, block) in bf.Q
        push!(q_blocks, block)
        push!(q_perms, bf.PermQ[Sleaf])
        # Q-blocket matar direkt in i R[1] på rätt par-nyckel
        r1_c_id = R_col_maps[1][(NO, Sleaf)]
        push!(q_col_offsets, flat_R[1].col_offsets[r1_c_id])
    end
    flat_Q = FlatQLayer(q_blocks, q_col_offsets, q_perms)

    # 4. Platta till P (Kopplas från R[end]:s rad-offsets)
    p_blocks = Matrix{ComplexF64}[]
    p_row_offsets = Int[]
    p_perms = Vector{Int}[]
    for (Oleaf, block) in bf.P
        push!(p_blocks, block)
        push!(p_perms, bf.PermP[Oleaf])
        # P-blocket läser direkt från R[end] baserat på par-nyckeln
        rend_r_id = R_row_maps[end][(Oleaf, NS)]
        push!(p_row_offsets, flat_R[end].row_offsets[rend_r_id])
    end
    flat_P = FlatPLayer(p_blocks, p_row_offsets, p_perms)

    return FlatBF(flat_Q, flat_R, flat_P, bf.dim, NS, NO)
end

# Intern hjälpfunktion som vänder på strukturen och transponerar blocken
function transform_bf(bf::FlatBF, conjugate::Bool)
    L = length(bf.R)

    # 1. Transformera R-lagren och vänd på ordningen (l -> L - l + 1)
    new_R = Vector{FlatLinearLayer}(undef, L)
    for l in 1:L
        old_layer = bf.R[l]

        num_old_rows = length(old_layer.row_ptr) - 1
        num_old_cols = length(old_layer.col_offsets)

        # Samla block för de nya raderna (vilka är de gamla kolumnerna)
        new_row_blocks = [Int[] for _ in 1:num_old_cols]
        new_row_cols = [Int32[] for _ in 1:num_old_cols]

        for i in 1:num_old_rows
            for b in old_layer.row_ptr[i]:(old_layer.row_ptr[i + 1] - 1)
                j = old_layer.col_idx[b]
                push!(new_row_blocks[j], b)
                push!(new_row_cols[j], Int32(i))
            end
        end

        # Bygg den nya CSR-strukturen för detta transponerade lager
        new_row_ptr = Vector{Int32}(undef, num_old_cols + 1)
        new_col_idx = Int32[]
        new_blocks = Matrix{ComplexF64}[]

        counter = 1
        for j in 1:num_old_cols
            new_row_ptr[j] = counter
            # Rad-indexen 'old_i' är redan sorterade eftersom vi loopade i från 1:num_old_rows
            for (k, b) in enumerate(new_row_blocks[j])
                old_i = new_row_cols[j][k]
                push!(new_col_idx, old_i)
                B = old_layer.blocks[b]

                # Transponera eller adjungera det enskilda matrisblocket
                push!(new_blocks, conjugate ? collect(B') : collect(transpose(B)))
                counter += 1
            end
        end
        new_row_ptr[end] = counter

        # Rad- och kolumn-offsets byter plats
        new_row_offsets = old_layer.col_offsets
        new_col_offsets = old_layer.row_offsets
        new_in_size     = old_layer.out_size
        new_out_size    = old_layer.in_size

        # Placera lagret i omvänd ordning i den nya vektorn
        new_R[L - l + 1] = FlatLinearLayer(
            new_row_ptr,
            new_col_idx,
            new_blocks,
            new_row_offsets,
            new_col_offsets,
            new_in_size,
            new_out_size,
        )
    end

    # 2. Transformera Q (som blir nya P eftersom den ligger sist i kedjan nu)
    new_P_blocks = Matrix{ComplexF64}[]
    for B in bf.Q.blocks
        push!(new_P_blocks, conjugate ? collect(B') : collect(transpose(B)))
    end
    new_P = FlatPLayer(new_P_blocks, bf.Q.col_offsets, bf.Q.perm)

    # 3. Transformera P (som blir nya Q eftersom den ligger först i kedjan nu)
    new_Q_blocks = Matrix{ComplexF64}[]
    for B in bf.P.blocks
        push!(new_Q_blocks, conjugate ? collect(B') : collect(transpose(B)))
    end
    new_Q = FlatQLayer(new_Q_blocks, bf.P.row_offsets, bf.P.perm)

    # Vänd på den globala matrisdimensionen
    new_shape = (bf.out_shape[2], bf.out_shape[1])

    return FlatBF(new_Q, new_R, new_P, new_shape, bf.NO, bf.NS)
end
