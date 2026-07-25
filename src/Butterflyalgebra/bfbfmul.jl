"""
    mulBFs(BF_1::ButterflyFactorization, BF_2::ButterflyFactorization, τ::Float64) -> BF

Compute the operator product of two Butterfly Factorizations (`BF`) and compress the
resulting representation to a specified accuracy tolerance.
"""
function mulBFs(
    BF_1::ButterflyFactorization{T,M}, BF_2::ButterflyFactorization{T,M}, τ::Float64
) where {T,M}
    @assert length(BF_1.R) == length(BF_2.R) "Both BFs must have the same number of levels"

    L = length(BF_1.R)

    # 1. Messenger Initialization
    # We natively multiply the leaf factor mappings assuming Q and P are ButterflyLevels
    M_leaf = multiply_levels(BF_1.Q, BF_2.P)

    # 2. Layer Intertwining
    # M = R_1[1] * Q_1 * P_2 * R_2[L]
    M_mid = multiply_levels(BF_1.R[1], multiply_levels(M_leaf, BF_2.R[L]))

    # Assemble the active levels flat array: [R2_1 ... R2_L-1, M, R1_2 ... R1_L]
    active_levels = Vector{ButterflyLevel{T}}()
    append!(active_levels, BF_2.R[1:(L - 1)])
    push!(active_levels, M_mid)
    append!(active_levels, BF_1.R[2:L])

    # 3. Iterative Row Swapping and Trimming
    for m in 1:(L - 1)
        # Push the messenger level through the tree via row-swaps
        for t in 1:m
            idx = L + 1 - t
            swap_adjacent!(active_levels, idx, BF_1.tree)
        end

        # Multiply the aligned levels
        idx_mul = L - m
        multiply_adjacent!(active_levels, idx_mul)

        # 4. Trimming / Recompression
        # Construct a temporary BF to utilize your existing recompress_BF function
        temp_BF = ButterflyFactorization{T,M}(
            BF_2.Q, copy(active_levels), BF_1.P, BF_1.tree, BF_1.k, τ
        )
        temp_BF = recompress_BF(temp_BF, τ)
        active_levels = temp_BF.R
    end

    # Return the newly assembled, fully compressed BF
    return ButterflyFactorization{T,M}(BF_2.Q, active_levels, BF_1.P, BF_1.tree, BF_1.k, τ)
end

function trivialmul(
    BF_1::ButterflyFactorization{T,M}, BF_2::ButterflyFactorization{T,M}
) where {T,M}
    @assert length(BF_1.R) == length(BF_2.R) "Both BFs must have the same number of levels"

    L = length(BF_1.R)
    M_leaf = multiply_levels(BF_1.Q, BF_2.P)
    M_mid = multiply_levels(BF_1.R[1], multiply_levels(M_leaf, BF_2.R[L]))

    active_levels = Vector{ButterflyLevel{T}}()
    append!(active_levels, BF_2.R[1:(L - 1)])
    push!(active_levels, M_mid)
    append!(active_levels, BF_1.R[2:L])

    for m in 1:(L - 1)
        multiply_adjacent!(active_levels, L - m)
    end

    return ButterflyFactorization{T,M}(
        BF_2.Q, active_levels, BF_1.P, BF_1.tree, BF_1.k, max(BF_1.τ, BF_2.τ)
    )
end

# --- Array Manipulators ---

function multiply_adjacent!(levels::Vector{ButterflyLevel{T}}, idx::Int) where {T}
    merged = multiply_levels(levels[idx], levels[idx + 1])
    splice!(levels, idx:(idx + 1), [merged])
    return levels
end

function swap_adjacent!(levels::Vector{ButterflyLevel{T}}, idx::Int, tree) where {T}
    merged = multiply_levels(levels[idx], levels[idx + 1])
    L_new, R_new = browswap_split(merged, tree)
    splice!(levels, idx:(idx + 1), [L_new, R_new])
    return levels
end

# --- Core Matrix Operations ---

# Overloads to allow native multiplication of Q and P vectors
function multiply_levels(A::Vector{<:ButterflyBlock}, B::Vector{<:ButterflyBlock})
    return multiply_levels(ButterflyLevel(A), ButterflyLevel(B))
end

function multiply_levels(A::ButterflyLevel, B::Vector{<:ButterflyBlock})
    return multiply_levels(A, ButterflyLevel(B))
end

function multiply_levels(A::Vector{<:ButterflyBlock}, B::ButterflyLevel)
    return multiply_levels(ButterflyLevel(A), B)
end

function multiply_levels(A::ButterflyLevel{T}, B::ButterflyLevel{T}) where {T}
    B_grouped = Dict{Tuple{Int,Int},Vector{ButterflyBlock{T}}}()
    for b in B.blocks
        r_key = (b.obs_out, b.src_out)
        if !haskey(B_grouped, r_key)
            B_grouped[r_key] = ButterflyBlock{T}[]
        end
        push!(B_grouped[r_key], b)
    end

    P_dict = Dict{Tuple{Tuple{Int,Int},Tuple{Int,Int}},Matrix{T}}()

    for a in A.blocks
        a_col_key = (a.obs_in, a.src_in)
        if haskey(B_grouped, a_col_key)
            for b in B_grouped[a_col_key]
                product_key = ((a.obs_out, a.src_out), (b.obs_in, b.src_in))
                mat_prod = a.data * b.data
                if haskey(P_dict, product_key)
                    P_dict[product_key] .+= mat_prod
                else
                    P_dict[product_key] = mat_prod
                end
            end
        end
    end

    return ButterflyLevel([
        ButterflyBlock(row[1], row[2], col[1], col[2], data) for
        ((row, col), data) in P_dict
    ])
end

function browswap_split(P::ButterflyLevel{T}, tree) where {T}
    parent_id(node) = cluster_parent(tree, node)

    # Helper: returns 1 if node is the first child, 2 if second child
    function child_parity(node, parent)
        parent == 0 && return 1 # Failsafe for root
        ch = tree(parent).children
        isempty(ch) && return 1 # Failsafe for leaves
        return node == ch[1] ? 1 : 2
    end

    # 1. Group by Parent Spaces
    clusters = Dict{Tuple{Tuple{Int,Int},Tuple{Int,Int}},Vector{ButterflyBlock{T}}}()
    for b in P.blocks
        cluster_key = (
            (parent_id(b.obs_out), parent_id(b.src_out)),
            (parent_id(b.obs_in), parent_id(b.src_in)),
        )
        if !haskey(clusters, cluster_key)
            clusters[cluster_key] = ButterflyBlock{T}[]
        end
        push!(clusters[cluster_key], b)
    end

    B_blocks = ButterflyBlock{T}[]
    C_blocks = ButterflyBlock{T}[]

    # 2. Process clusters using exact tree parity
    for (cluster_key, blocks) in clusters
        parent_row, _ = cluster_key

        # We only need to route the column spaces that actually survived compression
        unique_cols = unique([(b.obs_in, b.src_in) for b in blocks])

        for b in blocks
            # Determine exactly which of the 4 row quadrants this block belongs to
            r1 = child_parity(b.obs_out, parent_row[1])
            r2 = child_parity(b.src_out, parent_row[2])
            r_quadrant = (r1 - 1) * 2 + r2  # Maps to 1, 2, 3, or 4

            c_idx = findfirst(==((b.obs_in, b.src_in)), unique_cols)
            new_inner_col = unique_cols[c_idx]

            if r_quadrant <= 2
                # Top-left block diagonal (Rows 1 & 2)
                push!(
                    B_blocks,
                    ButterflyBlock(
                        b.obs_out, b.src_out, new_inner_col[1], new_inner_col[2], b.data
                    ),
                )
            else
                # Bottom-right block diagonal (Rows 3 & 4)
                push!(
                    B_blocks,
                    ButterflyBlock(
                        b.obs_out, b.src_out, new_inner_col[1], new_inner_col[2], b.data
                    ),
                )
            end
        end

        # Factor C (Right) - Identity routing for the surviving inner columns
        for c_val in unique_cols
            # Extract size dynamically from the first block that references this column
            col_size = size(
                blocks[findfirst(b -> (b.obs_in, b.src_in) == c_val, blocks)].data, 2
            )

            push!(
                C_blocks,
                ButterflyBlock(
                    c_val[1], c_val[2], c_val[1], c_val[2], Matrix{T}(I, col_size, col_size)
                ),
            )
        end
    end

    return ButterflyLevel(B_blocks), ButterflyLevel(C_blocks)
end

# --- Overloads ---

function Base.:*(BF_1::ButterflyFactorization, BF_2::ButterflyFactorization)
    return mulBFs(BF_1, BF_2, max(BF_1.τ, BF_2.τ))
end

function LinearAlgebra.mul!(
    C::ButterflyFactorization, A::ButterflyFactorization, B::ButterflyFactorization
)
    # Replaces the internal fields of C with the newly computed product
    res = mulBFs(A, B, max(A.τ, B.τ))
    C.Q = res.Q
    C.R = res.R
    C.P = res.P
    C.τ = res.τ
    return C
end
