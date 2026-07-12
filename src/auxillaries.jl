import H2Trees: values, center, halfsize, children, isleaf, trialtree, testtree
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
    h2treelevels(tree, root)

Computes the breath-first hierarchical levels of an `H2Trees.TwoNTree`.

Starting from a specified `root` node, it traverses the tree level by level,
collecting the nodes at each depth.

**Returns:**

  - A `Vector{Vector{Int}}` where each inner vector represents the node IDs at that level.
"""
function h2treelevels(tree::T, root::Int64) where {T}
    isleaf = H2Trees.isleaf
    getchildren = H2Trees.children

    levels = Vector{Vector{Int}}()
    current = [root]

    while !isempty(current)
        push!(levels, current)
        next = Int[]

        for node in current
            if !isleaf(tree, node)
                append!(next, getchildren(tree, node))
            end
        end

        current = next
    end

    return levels
end

function h2emptynodes(tree::T, root::Int64) where {T}
    isleaf = H2Trees.isleaf
    getchildren = H2Trees.children
    getvalues = H2Trees.values

    # Stores the empty nodes per level
    empty_levels = Vector{Vector{Int}}()

    current = [root]

    while !isempty(current)
        # Hitta alla tomma noder på den aktuella nivån
        empty_current = filter(node -> isempty(getvalues(tree, node)), current)
        push!(empty_levels, empty_current)

        next = Int[]
        # Bygg upp nästa nivå genom att samla alla barn från den nuvarande nivån
        for node in current
            if !isleaf(tree, node)
                append!(next, getchildren(tree, node))
            end
        end

        current = next
    end

    return empty_levels
end

"""
    traverseandpad(H2tree, root)

Extracts the hierarchical levels of a tree and generates "virtual" ghost nodes.

The Butterfly logic enforces that physical Degrees of Freedom (DoFs) belong solely to the
`Q` and `P` factors, while `R` factors handle hierarchical rank-transfers. To maintain
a perfectly balanced structure in unbalanced trees, this function artificially pushes
shallow leaf nodes down to the maximum depth of the tree block.

**Arguments:**

  - `H2tree`: The hierarchical block tree.
  - `root`: The ID of the root node to traverse from.

**Returns:**

  - A `Vector{Vector{Int}}` representing the padded tree nodes per level.
"""
function traverseandpad(H2tree::T, root::Int64) where {T}
    isleaf = H2Trees.isleaf
    tree = h2treelevels(H2tree, root)
    for l in 2:(length(tree) - 1)
        for node in tree[l]
            if isleaf(H2tree, node)
                push!(tree[l + 1], node)
            end
        end
    end
    return tree
end

"""
Abstract base type defining how physical spaces inside the `H2Trees` are ordered.
"""
permute(space, perm) = permute!(copy(space), perm)

abstract type SpaceOrderingStyle end

"""
    PermuteSpaceInPlace()

A `SpaceOrderingStyle` that permutes the test and trial spaces in place
according to the permutation derived from the tree leaf structure.
"""
struct PermuteSpaceInPlace <: SpaceOrderingStyle end
function (::PermuteSpaceInPlace)(tree, testspace, trialspace)
    testperm = permutation(testtree(tree))
    permute!(testspace, testperm)

    if testspace === trialspace && testtree(tree) === trialtree(tree)
        return nothing
    elseif !(testspace === trialspace) && !(testtree(tree) === trialtree(tree))
        trialperm = permutation(trialtree(tree))
        permute!(trialspace, trialperm)
        return nothing
    else
        @warn "Risky territory: Permuting trialtree not trialspace."
        trialperm = permutation(trialtree(tree))
        return nothing
    end
end
struct PreserveSpaceOrder <: SpaceOrderingStyle end
function (::PreserveSpaceOrder)(tree, testspace, trialspace)
    return nothing
end

function permutation(tree::H2Trees.H2ClusterTree)
    perm = zeros(Int, H2Trees.numberofvalues(tree))
    n = 1
    for leaf in H2Trees.leaves(tree)
        perm[n:(n + length(H2Trees.values(tree, leaf)) - 1)] = H2Trees.values(tree, leaf)
        tree.nodes[leaf].data.values .= n:(n + length(H2Trees.values(tree, leaf)) - 1)
        n += length(H2Trees.values(tree, leaf))
    end
    return perm
end

function group_by_parents(tree, childkeys, s_o::Int) #s_o = 1 for observer, 2 for source
    parentedgrps = Dict{Int,Vector{Tuple{Int,Int}}}()
    for childkey in childkeys
        parentnode = H2Trees.parent(tree, childkey[s_o])
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
    trialt = ButterflyFactorizations.h2treelevels(trees, 1)
    tstt = ButterflyFactorizations.h2treelevels(treeo, 1)
    if trialt != tstt
        return false
    end
    commont = trialt
    for lvl in commont
        for node in lvl
            trialcluster = H2Trees.values(trees, node)
            tstcluster = H2Trees.values(treeo, node)
            if trialcluster != tstcluster
                return false
            end
        end
    end

    return true
end

function deep_accumulate_P!(
    dest::Dict{Tuple{Int,Int},Matrix{ComplexF64}},
    src::Dict{Tuple{Int,Int},Matrix{ComplexF64}},
)
    for (key, mat_src) in src
        if !haskey(dest, key)
            dest[key] = copy(mat_src)
        else
            println("Overlapping block detected at key: $key")
            #may not happen --> preserve the existing block and append the new one
            dest[key] = hcat(dest[key], mat_src)
        end
    end
    return dest
end

function deep_accumulate_Q!(
    dest::Dict{Tuple{Int,Int},Matrix{ComplexF64}},
    src::Dict{Tuple{Int,Int},Matrix{ComplexF64}},
)
    for (key, mat_src) in src
        if !haskey(dest, key)
            dest[key] = copy(mat_src)
        else
            println("Overlapping block detected at key: $key")
            #may not happen --> preserve the existing block and append the new one
            dest[key] = vcat(dest[key], mat_src)
        end
    end
    return dest
end

function deep_accumulate_R!(
    dest::Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},Matrix{ComplexF64}}},
    src::Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},Matrix{ComplexF64}}};
    otmap::Dict{Int,Int} = Dict{Int,Int}(),
    stmap::Dict{Int,Int} = Dict{Int,Int}(),
)
    for (node_key, inner_dict_src) in src
        if haskey(otmap, node_key[1]) && haskey(stmap, node_key[2])
            new_node_key = (otmap[node_key[1]], stmap[node_key[2]])
        elseif haskey(otmap, node_key[1])
            new_node_key = (otmap[node_key[1]], node_key[2])
        elseif haskey(stmap, node_key[2])
            new_node_key = (node_key[1], stmap[node_key[2]])
        else
            new_node_key = node_key
        end
        if !haskey(dest, new_node_key)
            dest[new_node_key] = Dict{Tuple{Int,Int},Matrix{ComplexF64}}()
            for (sub_key, mat_src) in inner_dict_src
                if haskey(otmap, sub_key[1]) && haskey(stmap, sub_key[2])
                    new_sub_key = (otmap[sub_key[1]], stmap[sub_key[2]])
                elseif haskey(otmap, sub_key[1])
                    new_sub_key = (otmap[sub_key[1]], sub_key[2])
                elseif haskey(stmap, sub_key[2])
                    new_sub_key = (sub_key[1], stmap[sub_key[2]])
                else
                    new_sub_key = sub_key
                end
                dest[new_node_key][new_sub_key] = copy(mat_src)
            end
        else
            # Node key exists, merge the inner mapping level
            inner_dict_dest = dest[new_node_key]
            for (sub_key, mat_src) in inner_dict_src
                if haskey(otmap, sub_key[1]) && haskey(stmap, sub_key[2])
                    new_sub_key = (otmap[sub_key[1]], stmap[sub_key[2]])
                elseif haskey(otmap, sub_key[1])
                    new_sub_key = (otmap[sub_key[1]], sub_key[2])
                elseif haskey(stmap, sub_key[2])
                    new_sub_key = (sub_key[1], stmap[sub_key[2]])
                else
                    new_sub_key = sub_key
                end
                if !haskey(inner_dict_dest, new_sub_key)
                    inner_dict_dest[new_sub_key] = copy(mat_src)
                else
                    println(
                        "Overlapping block detected at node_key: $new_node_key, sub_key: $new_sub_key",
                    )
                    #may not happen --> preserve the existing block and append the new one
                    blockdiag(inner_dict_dest[new_sub_key], mat_src)
                end
            end
        end
    end
    return dest
end

function treemapping(a::Int, b::Int, tree)
    mapping = Dict{Int,Int}()

    # Initialize a stack with our starting pair of node IDs
    stack = [(a, b)]

    while !isempty(stack)
        curr_a, curr_b = pop!(stack)

        mapping[curr_a] = curr_b

        # Push all paired children onto the stack
        for (child_a, child_b) in zip(children(tree, curr_a), children(tree, curr_b))
            push!(stack, (child_a, child_b))
        end
    end

    return mapping
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

function col_to_row_map(rmat::Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},Matrix{ComplexF64}}})
    colspace = Dict{Tuple{Int,Int},Vector{Tuple{Int,Int}}}()
    for (row, inner_dict) in rmat
        for (col, mat) in inner_dict
            if !haskey(colspace, col)
                colspace[col] = Vector{Tuple{Int,Int}}()
            end
            push!(colspace[col], row)
        end
    end

    return colspace
end

function group_identical_colspaces(
    data::Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},M}}
) where {M}
    RowType = Tuple{Int,Int}
    ColType = Tuple{Int,Int}

    # Map a Set of column keys to a Vector of row keys
    groups = Dict{Set{ColType},Vector{RowType}}()

    for (row, col_dict) in data
        # Abstract Dict keys into a Set. Sets have order-independent hashing.
        col_space = Set{ColType}(keys(col_dict))

        # Group the row by its column space signature
        vec = get!(() -> RowType[], groups, col_space)
        push!(vec, row)
    end

    # If you only want groups that actually have duplicates (more than 1 row):
    #duplicates_only = filter(p -> length(p.second) > 1, groups)

    return groups
end

function build_matrix_from_group(data, rows, cols)
    # 1. Sort keys (Julia automatically orders them exactly as you requested)
    sorted_rows = sort(collect(rows))
    sorted_cols = sort(collect(cols))

    # 2. Get dimensions
    n_rows = length(sorted_rows)
    n_cols = length(sorted_cols)

    # 3. Grab the concrete matrix type M from the first element
    M = typeof(data[first(sorted_rows)][first(sorted_cols)])

    # 4. Allocate the 2D grid
    grid_matrix = Matrix{M}(undef, n_rows, n_cols)

    # 5. Populate the grid (looping column-first for Julia's memory layout)
    for j in 1:n_cols
        col_key = sorted_cols[j]
        for i in 1:n_rows
            row_key = sorted_rows[i]
            grid_matrix[i, j] = data[row_key][col_key]
        end
    end

    return grid_matrix, sorted_rows, sorted_cols
end

function build_supertree(roots::Vector{Int}, tree, init_super_id::Int)
    N = length(roots)

    # 1. The new Super-Tree structure
    # Maps a Super-Node ID -> Array of child Super-Node IDs
    supertree = Dict{Int,Vector{Int}}()

    # 2. The Lifts (Mappings)
    # An array of N dictionaries.
    # mappings[i] maps: Original Node ID in Tree i -> Super-Node ID
    mappings = [Dict{Int,Int}() for _ in 1:N]

    # Counter to generate fresh, unique IDs for the Super-Tree
    next_super_id = init_super_id

    # Recursive traversal function
    function traverse(curr_nodes::Vector{Union{Int,Nothing}})
        # Allocate a new ID for this position in the Super-Tree
        super_id = next_super_id
        next_super_id += 1

        # Initialize empty children array for this Super-Node
        supertree[super_id] = Int[]

        # Record the mapping for any tree that has a real node here
        for i in 1:N
            if !isnothing(curr_nodes[i])
                mappings[i][curr_nodes[i]] = super_id
            end
        end

        # Gather all child iterators. If a tree has no node here (nothing),
        # it provides an empty array of children.
        all_children = [
            isnothing(n) ? Int[] : collect(children(tree, n)) for n in curr_nodes
        ]

        # The union must account for the widest branch at this level across all N trees
        max_children = maximum(length.(all_children))

        # Traverse downwards for each child index
        for child_idx in 1:max_children
            # Build the next layer of nodes to evaluate
            next_nodes = Vector{Union{Int,Nothing}}(undef, N)

            for i in 1:N
                # If tree 'i' has a child at this index, grab it. Otherwise, it's a ghost.
                if child_idx <= length(all_children[i])
                    next_nodes[i] = all_children[i][child_idx]
                else
                    next_nodes[i] = nothing
                end
            end

            # Recurse and link the resulting child Super-Node to the current one
            child_super_id = traverse(next_nodes)
            push!(supertree[super_id], child_super_id)
        end

        return super_id
    end

    # Initialize the traversal with our N roots
    initial_nodes = Union{Int,Nothing}[roots[i] for i in 1:N]
    root_super_id = traverse(initial_nodes)
    nodaloffset = next_super_id - 1
    return supertree, mappings, root_super_id, nodaloffset
end
