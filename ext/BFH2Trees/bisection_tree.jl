struct BisectionTree{N,D,T} <: H2Trees.H2ClusterTree
    nodes::Vector{H2Trees.Node{D}}
    root::Int
    center::SVector{N,T}
    halfsize::T
    nodesatlevel::Vector{Vector{Int}}
    permutation::Vector{Int}
end

# Enable standard H2Trees flat-array indexing
function (tree::BisectionTree)(node::Int)
    # Allows `tree(nodeid)` to fetch the Node struct
    return tree.nodes[node - tree.root + 1]
end

H2Trees.treetrait(::Type{BisectionTree{N,D,T}}) where {N,D,T} = isBisectionTree()

"""
    BisectionTree(positions::AbstractVector{SVector{N,T}}; max_points=-1, maxhalfsize =
    -1.0)

    Builds a bisection tree from a set of points in N-dimensional space. The tree is
    constructed by recursively splitting the set of points along the longest axis of the
    bounding box until either the maximum number of points per leaf node (`max_points`) or
    the maximum half-size of the bounding box (`maxhalfsize`) is reached.
"""
function BisectionTree(
    positions::AbstractVector{SVector{N,T}}; max_points=-1, maxhalfsize=-1.0
) where {N,T}
    if max_points <= 0 && maxhalfsize <= 0.0
        error("You must provide either a valid max_points (>0) or maxhalfsize (>0.0)")
    end

    N_points = length(positions)
    indices = collect(1:N_points)
    max_depth = if max_points > 0
        max(1, ceil(Int, log2(N_points / max_points)+1))
    else
        -1
    end

    # Compute Global Bounding Box from vector of SVectors
    # We can find min/max component-wise or via broadcast reduction
    bbox_min = reduce((acc, pt) -> min.(acc, pt), positions)
    bbox_max = reduce((acc, pt) -> max.(acc, pt), positions)
    center = SVector{N,T}((bbox_max .+ bbox_min) ./ 2)
    halfsize = maximum(bbox_max .- bbox_min) / 2

    D = H2Trees.BoxData{N,T}
    nodes = Vector{H2Trees.Node{D}}()
    nodesatlevel = [Int[]]

    _build_bisection_node!(
        nodes, nodesatlevel, indices, positions, 1, N_points, 1, max_depth, maxhalfsize, 0
    )

    return BisectionTree{N,D,T}(nodes, 1, center, halfsize, nodesatlevel, indices)
end

function _build_bisection_node!(
    nodes,
    nodesatlevel,
    indices,
    points,
    start_idx,
    stop_idx,
    current_depth,
    max_depth,
    maxhalfsize,
    parent_idx,
)
    # 1. Local node bounds via vector of SVectors
    sub_indices = @view indices[start_idx:stop_idx]
    sub_points = @view points[sub_indices]

    bbox_min = reduce((acc, pt) -> min.(acc, pt), sub_points)
    bbox_max = reduce((acc, pt) -> max.(acc, pt), sub_points)
    center = SVector((bbox_max .+ bbox_min) ./ 2)
    halfsize = maximum(bbox_max .- bbox_min) / 2

    if current_depth > length(nodesatlevel)
        push!(nodesatlevel, Int[])
    end

    # 2. Original Mesh Indices for BEAST
    node_values = collect(sub_indices) # copy subset of indices
    boxdata = H2Trees.BoxData(0, node_values, center, halfsize, current_depth)

    my_idx = length(nodes) + 1
    push!(nodes, H2Trees.Node(boxdata, 0, parent_idx, 0))
    push!(nodesatlevel[current_depth], my_idx)

    # 3. Evaluate Stopping Criteria
    stop_by_depth = (max_depth > 0) && (current_depth == max_depth)
    stop_by_size  = (maxhalfsize > 0.0) && (halfsize <= maxhalfsize)

    # 4. Split and Recurse if neither stopping condition is met
    if !stop_by_depth && !stop_by_size && (stop_idx > start_idx)
        # Find longest axis of current bounding box dimension (1, 2, or 3)
        axis = argmax(bbox_max .- bbox_min)

        # IN-PLACE SORT of indices based on the specific coordinate axis of the SVector
        sort!(sub_indices; by=i -> points[i][axis])

        num_points = stop_idx - start_idx + 1
        mid_idx = start_idx + div(num_points, 2) - 1

        # Left Child
        left_idx = _build_bisection_node!(
            nodes,
            nodesatlevel,
            indices,
            points,
            start_idx,
            mid_idx,
            current_depth + 1,
            max_depth,
            maxhalfsize,
            my_idx,
        )

        # Right Child
        right_idx = _build_bisection_node!(
            nodes,
            nodesatlevel,
            indices,
            points,
            mid_idx + 1,
            stop_idx,
            current_depth + 1,
            max_depth,
            maxhalfsize,
            my_idx,
        )

        # Patch the tree relationships
        nodes[my_idx] = H2Trees.Node(boxdata, 0, parent_idx, left_idx)
        left_node = nodes[left_idx]
        nodes[left_idx] = H2Trees.Node(
            left_node.data, right_idx, left_node.parent, left_node.firstchild
        )
    end

    return my_idx
end

struct isBisectionTree <: H2Trees.AbstractTreeTrait end

function H2Trees.printtree(io::IO, tree, ::isBisectionTree)
    println(io, typeof(tree))
    for level in H2Trees.levels(tree)
        avgnpoints = round(H2Trees.averagenumberofpoints(tree, level); digits=2)
        avgnchildren = round(H2Trees.averagenumberofchildrens(tree, level); digits=2)
        hs = H2Trees.halfsize(tree, H2Trees.LevelIterator(tree, level)[begin])
        numnodes = length(collect(H2Trees.LevelIterator(tree, level)))
        print(io, "-"^(level - H2Trees.minimumlevel(tree)))
        println(
            io,
            " level: $level with $numnodes node(s) with on average $avgnpoints points and $avgnchildren children and halfsize: $hs",
        )
    end
end

function ButterflyFactorizations.build_bisection_tree(
    positions::AbstractVector; max_points=-1, maxhalfsize=-1.0
)
    # This calls the BisectionTree constructor inside bisection_tree.jl
    return BisectionTree(positions; max_points=max_points, maxhalfsize=maxhalfsize)
end
