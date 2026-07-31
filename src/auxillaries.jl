"""
    blockdiag(blocks::AbstractMatrix...)

Constructs a block diagonal dense matrix from the given `blocks`.

Each input matrix is placed on the diagonal, and the off-diagonal blocks are filled with
zeros. The resulting matrix has dimensions equal to the sum of the dimensions of the input
matrices.

**Arguments:**

  - `blocks`: A variable number of dense matrices (`AbstractMatrix`).

**Returns:**

  - A single dense matrix `M` containing the block diagonal combination.
"""
function blockdiag(blocks::AbstractMatrix...)
    isempty(blocks) && return zeros(0, 0)

    T = promote_type(map(eltype, blocks)...)

    rows = sum(size(b, 1) for b in blocks)
    cols = sum(size(b, 2) for b in blocks)

    M = zeros(T, rows, cols)

    r = 1
    c = 1
    for B in blocks
        nr, nc = size(B)
        M[r:(r + nr - 1), c:(c + nc - 1)] .= B
        r += nr
        c += nc
    end

    return M
end

"""
    sparse_blockdiag(blocks::AbstractMatrix...)

Constructs a block diagonal sparse matrix from the given `blocks`.

Each input matrix is converted to a sparse matrix and placed on the diagonal,
with off-diagonal blocks remaining empty (structural zeros).

**Arguments:**

  - `blocks`: A variable number of matrices (`AbstractMatrix`).

**Returns:**

  - A `SparseMatrixCSC` encompassing all input blocks diagonally.
"""
function sparse_blockdiag(blocks::AbstractMatrix...)
    isempty(blocks) && return spzeros(ComplexF64, 0, 0)

    # Convert blocks to sparse to utilize SparseArrays.blockdiag
    sparse_blocks = map(sparse, blocks)
    return SparseArrays.blockdiag(sparse_blocks...)
end

"""
    sparse_vcat(blocks::AbstractMatrix...)

Vertically concatenates the given `blocks` into a sparse matrix.

Each input matrix is stacked on top of the next. All blocks must have the same number of
columns.

**Arguments:**

  - `blocks`: A variable number of matrices (`AbstractMatrix`).

**Returns:**

  - A `SparseMatrixCSC` resulting from the vertical stack.
"""
function sparse_vcat(blocks::AbstractMatrix...)
    isempty(blocks) && return spzeros(ComplexF64, 0, 0)

    # Convert blocks to sparse and vertically concatenate
    sparse_blocks = map(sparse, blocks)
    return vcat(sparse_blocks...)
end

"""
    blocksparse_blockdiag(blocks...)

Constructs a block diagonal matrix specifically handling `BlockSparseMatrix` instances.

This function extends the standard block diagonal logic to keep track of the internal
row and column indices essential for the `BlockSparseMatrix` custom type, allowing
seamless combination of both regular matrices and block-sparse matrices.

**Arguments:**

  - `blocks`: Variables number of regular `Matrix` or `BlockSparseMatrix` objects.

**Returns:**

  - A well-formed `BlockSparseMatrix` representing the block diagonal.
"""
function blocksparse_blockdiag(blocks...)
    isempty(blocks) && return BlockSparseMatrix(
        Matrix{ComplexF64}[], UnitRange{Int}[], UnitRange{Int}[], (0, 0)
    )

    # Helper to get indices whether it's a BlockSparseMatrix or a regular Matrix
    get_rowidx(b) = hasproperty(b, :rowindices) ? b.rowindices : [1:size(b, 1)]
    get_colidx(b) = hasproperty(b, :colindices) ? b.colindices : [1:size(b, 2)]
    get_blocks(b) = hasproperty(b, :blocks) ? b.blocks : [b]

    if length(blocks) == 1
        b = blocks[1]
        return BlockSparseMatrix(get_blocks(b), get_rowidx(b), get_colidx(b), size(b))
    elseif length(blocks) > 2
        return blocksparse_blockdiag(
            blocksparse_blockdiag(blocks[1], blocks[2]), blocks[3:end]...
        )
    end

    s1 = size(blocks[1])
    s2 = size(blocks[2])

    rowindices = vcat(get_rowidx(blocks[1]), [vs .+ s1[1] for vs in get_rowidx(blocks[2])])
    colindices = vcat(get_colidx(blocks[1]), [vs .+ s1[2] for vs in get_colidx(blocks[2])])

    combined_blocks = vcat(get_blocks(blocks[1]), get_blocks(blocks[2]))

    return BlockSparseMatrix(
        combined_blocks, rowindices, colindices, (s1[1] + s2[1], s1[2] + s2[2])
    )
end

"""
    blocksparse_vcat(blocks...)

Vertically concatenates regular matrices and `BlockSparseMatrix` instances.

It ensures that the resulting `BlockSparseMatrix` maintains the correct structure
by appropriately combining the row indices while keeping the column indices
consistent across the blocks.

**Arguments:**

  - `blocks`: Variables number of regular `Matrix` or `BlockSparseMatrix` objects.

**Returns:**

  - A vertically concatenated `BlockSparseMatrix`.
"""
function blocksparse_vcat(blocks...)
    isempty(blocks) && return BlockSparseMatrix(
        Matrix{ComplexF64}[], UnitRange{Int}[], UnitRange{Int}[], (0, 0)
    )

    get_rowidx(b) = hasproperty(b, :rowindices) ? b.rowindices : [1:size(b, 1)]
    get_colidx(b) = hasproperty(b, :colindices) ? b.colindices : [1:size(b, 2)]
    get_blocks(b) = hasproperty(b, :blocks) ? b.blocks : [b]

    if length(blocks) == 1
        b = blocks[1]
        return BlockSparseMatrix(get_blocks(b), get_rowidx(b), get_colidx(b), size(b))
    elseif length(blocks) > 2
        return blocksparse_vcat(blocksparse_vcat(blocks[1], blocks[2]), blocks[3:end]...)
    end

    s1 = size(blocks[1])
    s2 = size(blocks[2])
    @assert s1[2] == s2[2] "All blocks must have the same number of columns."

    rowindices = vcat(get_rowidx(blocks[1]), [vs .+ s1[1] for vs in get_rowidx(blocks[2])])
    colindices = get_colidx(blocks[1])

    combined_blocks = vcat(get_blocks(blocks[1]), get_blocks(blocks[2]))

    return BlockSparseMatrix(
        combined_blocks, rowindices, colindices, (s1[1] + s2[1], s1[2])
    )
end

"""
    getsubdict!(D, k)

Retrieves a sub-dictionary from a nested dictionary `D` at key `k`.

If the specified key does not exist in the outer dictionary, it initializes a new empty
inner dictionary at that key and returns it. This allows for safe, on-the-fly construction
of nested dictionaries like those used for `Q`, `R`, and `P` factors.
"""
@inline function getsubdict!(D::Dict{Int,Dict{Int,T}}, k::Int) where {T}
    get!(D, k) do
        return Dict{Int,T}()
    end
end

@inline function getsubdict!(
    D::Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},T}}, k::Tuple{Int,Int}
) where {T}
    get!(D, k) do
        return Dict{Tuple{Int,Int},T}()
    end
end

@inline function getsubdict!(D::Dict{Int,Dict{Tuple{Int,Int},T}}, k::Int) where {T}
    get!(D, k) do
        return Dict{Tuple{Int,Int},T}()
    end
end

"""
    find_rows_for_column(R, col_idx)

Finds all the row keys in a nested dictionary `R` where the inner dictionary contains a
specific `col_idx`.

This is highly useful for operations that need to invert the relationships in the
observer/source tree indices stored within the ButterflyFactorizations's hierarchical `R`
factors.

**Arguments:**

  - `R`: The nested dictionary representing the `R` factor block.
  - `col_idx`: The column target key to search for.

**Returns:**

  - A `Vector` of row keys that map to the given column index.
"""
function find_rows_for_column(R::Dict{T,Dict{T,U}}, col_idx::T) where {T,U}
    rows = Vector{T}()
    for (row, inner_dict) in R
        if haskey(inner_dict, col_idx)
            push!(rows, row)
        end
    end
    return rows
end

"""
    treelevels(tree, root)

Computes the breath-first hierarchical levels of an tree.

Starting from a specified `root` node, it traverses the tree level by level,
collecting the nodes at each depth.

**Returns:**

  - A `Vector{Vector{Int}}` where each inner vector represents the node IDs at that level.
"""
function treelevels(tree::T, root::Int64) where {T}
    levels = Vector{Vector{Int}}()
    current = [root]

    while !isempty(current)
        push!(levels, current)
        next = Int[]

        for node in current
            if !cluster_isleaf(tree, node)
                append!(next, cluster_children(tree, node))
            end
        end

        current = next
    end

    return levels
end

function emptynodes(tree::T, root::Int64) where {T}

    # Stores the empty nodes per level
    empty_levels = Vector{Vector{Int}}()

    current = [root]

    while !isempty(current)
        # Hitta alla tomma noder på den aktuella nivån
        empty_current = filter(node -> isempty(cluster_values(tree, node)), current)
        push!(empty_levels, empty_current)

        next = Int[]
        # Bygg upp nästa nivå genom att samla alla barn från den nuvarande nivån
        for node in current
            if !cluster_isleaf(tree, node)
                append!(next, cluster_children(tree, node))
            end
        end

        current = next
    end

    return empty_levels
end

"""
    traverseandpad(tree, root)

Extracts the hierarchical levels of a tree and generates "virtual" ghost nodes.

The Butterfly logic enforces that physical Degrees of Freedom (DoFs) belong solely to the
`Q` and `P` factors, while `R` factors handle hierarchical rank-transfers. To maintain
a perfectly balanced structure in unbalanced trees, this function artificially pushes
shallow leaf nodes down to the maximum depth of the tree block.

**Arguments:**

  - `tree`: The hierarchical block tree.
  - `root`: The ID of the root node to traverse from.

**Returns:**

  - A `Vector{Vector{Int}}` representing the padded tree nodes per level.
"""
function traverseandpad(tree::T, root::Int64) where {T}
    treelvls = treelevels(tree, root)
    for l in 2:(length(treelvls) - 1)
        for node in treelvls[l]
            if cluster_isleaf(tree, node)
                push!(treelvls[l + 1], node)
            end
        end
    end
    return treelvls
end

function group_by_parents(tree, childkeys, s_o::Int) #s_o = 1 for observer, 2 for source
    parentedgrps = Dict{Int,Vector{Tuple{Int,Int}}}()
    for childkey in childkeys
        parentnode = cluster_parent(tree, childkey[s_o])
        if !haskey(parentedgrps, parentnode)
            parentedgrps[parentnode] = Vector{Tuple{Int,Int}}()
        end
        push!(parentedgrps[parentnode], childkey)
    end
    for (parent, keys) in parentedgrps
        sort!(keys)
    end
    return parentedgrps
end

function checkequality(trees, treeo)
    trialt = ButterflyFactorizations.treelevels(trees, 1)
    tstt = ButterflyFactorizations.treelevels(treeo, 1)
    if trialt != tstt
        return false
    end
    commont = trialt
    for lvl in commont
        for node in lvl
            trialcluster = cluster_values(trees, node)
            tstcluster = cluster_values(treeo, node)
            if trialcluster != tstcluster
                return false
            end
        end
    end

    return true
end

function retrievecolspace(
    rmat::Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},Matrix{ComplexF64}}}
)
    colspace = Dict{Tuple{Int,Int},Int}()
    for (row, inner_dict) in rmat
        for (col, mat) in inner_dict
            if !haskey(colspace, col)
                colspace[col] = size(mat, 2)
            end
        end
    end

    return colspace
end

function compute_interaction_percentages(nearints, farints, test_tree, trial_tree)
    near_elements = 0
    for (snode, onode) in nearints
        s_dofs = length(cluster_values(trial_tree, snode))
        o_dofs = length(cluster_values(test_tree, onode))
        near_elements += s_dofs * o_dofs
    end

    far_elements = 0
    for (snode, onode) in farints
        s_dofs = length(cluster_values(trial_tree, snode))
        o_dofs = length(cluster_values(test_tree, onode))
        far_elements += s_dofs * o_dofs
    end

    total_elements = near_elements + far_elements
    println(
        "True Near-field fraction: ",
        round(100 * near_elements / total_elements; digits=2),
        "% of matrix entries",
    )
    return println(
        "True Far-field fraction:  ",
        round(100 * far_elements / total_elements; digits=2),
        "% of matrix entries",
    )
end
