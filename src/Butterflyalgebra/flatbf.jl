function FlatBF(bf::BF)
    L = length(bf.R)
    NO = bf.NO
    NS = bf.NS
    dim = (length(H2Trees.values(bf.otree, 1)), length(H2Trees.values(bf.stree, 1)))

    # -- Om BF bara består av Q och P (nivå 0 för R) --
    if L == 0
        q_blocks = Matrix{ComplexF64}[]
        q_col_offsets = Int[]
        q_perms = Vector{Int}[]
        curr_offset_q = 1
        for (qkey, block) in bf.Q
            push!(q_blocks, block)
            push!(q_perms, H2Trees.values(bf.stree, qkey[2]))
            push!(q_col_offsets, curr_offset_q)
            curr_offset_q += size(block, 2)
        end
        flat_Q = FlatQLayer(q_blocks, q_col_offsets, q_perms)

        p_blocks = Matrix{ComplexF64}[]
        p_row_offsets = Int[]
        p_perms = Vector{Int}[]
        curr_offset_p = 1
        for (pkey, block) in bf.P
            push!(p_blocks, block)
            push!(p_perms, H2Trees.values(bf.otree, pkey[1]))
            push!(p_row_offsets, curr_offset_p)
            curr_offset_p += size(block, 2)
        end
        flat_P = FlatPLayer(p_blocks, p_row_offsets, p_perms)

        flat_R = FlatLinearLayer[]

        # Pre-allocate workspace for L=0
        int_size = if isempty(flat_Q.blocks)
            0
        else
            (flat_Q.col_offsets[end] + size(flat_Q.blocks[end], 1) - 1)
        end
        layer_vectors = [zeros(ComplexF64, int_size)]

        return FlatBF(flat_Q, flat_R, flat_P, dim, NS, NO, layer_vectors)
    end
    # -------------------------------------------------------

    # 1. Bygg synkroniserade ID-mappningar för alla R-nivåer
    R_col_maps = Vector{Dict{Tuple{Int,Int},Int32}}(undef, L)
    R_row_maps = Vector{Dict{Tuple{Int,Int},Int32}}(undef, L)

    cols_1 = Set{Tuple{Int,Int}}()
    for (r_key, cols_dict) in bf.R[1], c_key in keys(cols_dict)
        push!(cols_1, c_key)
    end
    R_col_maps[1] = Dict(k => Int32(i) for (i, k) in enumerate(sort!(collect(cols_1))))

    for l in 1:L
        rows_l = sort!(collect(keys(bf.R[l])))
        R_row_maps[l] = Dict(k => Int32(i) for (i, k) in enumerate(rows_l))
        if l < L
            R_col_maps[l + 1] = R_row_maps[l]
        end
    end

    # 2. Platta till R-nivåerna till FlatLinearLayer
    flat_R = Vector{FlatLinearLayer}(undef, L)
    old_row_sizes = zeros(Int, length(R_row_maps[1]))

    for l in 1:L
        R_dict = bf.R[l]
        row_map = R_row_maps[l]
        col_map = R_col_maps[l]

        col_sizes = zeros(Int, length(col_map))
        row_sizes = zeros(Int, length(row_map))

        for (r_key, cols_dict) in R_dict
            for (c_key, block) in cols_dict
                if l == 1
                    q_block = bf.Q[c_key]
                    col_sizes[col_map[c_key]] = size(q_block, 1)
                else
                    col_sizes[col_map[c_key]] = old_row_sizes[R_row_maps[l - 1][c_key]]
                end
            end
            if isempty(first(Base.values(cols_dict)))
                row_sizes[row_map[r_key]] = col_sizes[col_map[first(keys(cols_dict))]]
                continue
            end
            row_sizes[row_map[r_key]] = size(first(Base.values(cols_dict)), 1)
        end

        col_offsets = Vector{Int}(undef, length(col_map))
        curr = 1
        for i in 1:length(col_map)
            col_offsets[i] = curr
            curr += col_sizes[i]
        end
        in_size = curr - 1

        row_offsets = Vector{Int}(undef, length(row_map))
        curr = 1
        for i in 1:length(row_map)
            row_offsets[i] = curr
            curr += row_sizes[i]
        end
        out_size = curr - 1

        row_ptr = Vector{Int32}(undef, length(row_map) + 1)
        col_idx = Int32[]
        blocks = Matrix{ComplexF64}[]

        block_counter = 1
        for (i, r_key) in enumerate(sort!(collect(keys(row_map))))
            row_ptr[i] = block_counter
            cols_dict = R_dict[r_key]

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
        old_row_sizes = row_sizes
    end

    # 3. Platta till Q
    q_blocks = Matrix{ComplexF64}[]
    q_col_offsets = Int[]
    q_perms = Vector{Int}[]
    for (qkey, block) in bf.Q
        push!(q_blocks, block)
        push!(q_perms, H2Trees.values(bf.stree, qkey[2]))
        r1_c_id = R_col_maps[1][qkey]
        push!(q_col_offsets, flat_R[1].col_offsets[r1_c_id])
    end
    flat_Q = FlatQLayer(q_blocks, q_col_offsets, q_perms)

    # 4. Platta till P
    p_blocks = Matrix{ComplexF64}[]
    p_row_offsets = Int[]
    p_perms = Vector{Int}[]
    for (pkey, block) in bf.P
        push!(p_blocks, block)
        push!(p_perms, H2Trees.values(bf.otree, pkey[1]))
        rend_r_id = R_row_maps[end][pkey]
        push!(p_row_offsets, flat_R[end].row_offsets[rend_r_id])
    end
    flat_P = FlatPLayer(p_blocks, p_row_offsets, p_perms)

    # 5. Pre-allocate workspace based on R-layer sizes
    layer_vectors = Vector{Vector{ComplexF64}}(undef, L + 1)
    layer_vectors[1] = zeros(ComplexF64, flat_R[1].in_size)
    for l in 1:L
        layer_vectors[l + 1] = zeros(ComplexF64, flat_R[l].out_size)
    end

    return FlatBF(flat_Q, flat_R, flat_P, dim, NS, NO, layer_vectors)
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
