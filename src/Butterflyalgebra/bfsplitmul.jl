
"""
    splitmulbf(butterflycluster_init::Matrix{BF}, higherkBF_init ::ButterflyFactorization, τ::Float64) -> BF

Compute the operator product of a hierarchically divided matrix cluster of Butterflies
(level \$k\$) and a single, larger Butterfly Factorization block (level \$k+1\$) using
Heldring's block-splitting algorithm.

This function addresses the architectural challenge of multiplying non-uniform hierarchical
levels by subdividing the structural components of the larger factor to match the sparse
data layout of the smaller cluster blocks. It separately processes individual
sub-multiplications before realigning and reducing them back into a single unified
representation.

# Arguments

  - `butterflycluster_init::Matrix{BF}`: A matrix layout of lower-level butterfly factors
    representing the hierarchically split operator components.
  - `higherkBF_init ::ButterflyFactorization`: A single, un-split butterfly factorization at a higher
    hierarchical tree tier.
  - `τ::Float64`: Accuracy tolerance threshold passed down to internal multiplications
    (`mulBFs`) and the final accumulation stages (`add_eqbfs`).

# Returns

  - `BF`: A single consolidated and recompressed butterfly factorization representing the
    entire operator product.

# Algorithmic Steps

 1. **Factor Subdivision (Step 1):** The structural `P` and `R` dictionaries of the
    higher-tier operator are sliced into `numchildren` distinct independent BFs matching the
    target output subtree nodes (`children`).
 2. **Intermediate Block Multiplications:** Executes element-wise sub-multiplications
    between the cluster elements and the newly generated lower-tier factors via `mulBFs`.
 3. **Supertree Alignment & Coordinate Remapping:** Builds a global supertree reference to
    map differing index spaces. It shifts spatial/frequency cluster coordinates up to common
    parent frames and merges data blocks through horizontal/vertical accumulations and
    diagonal padding where key configurations overlap.
 4. **Hierarchical Reduction:** Consolidates the array of realigned intermediate structures
    down into a single final operator via sequential `add_eqbfs` calls.

# Notes

  - The original structural `Q` factors of the larger operator are intentionally preserved
    and held back until the final phase to maintain dimension consistency across structural
    modifications.
"""
function splitmulbf(
    butterflycluster_init::Matrix{ButterflyFactorization},
    higherkBF_init::ButterflyFactorization,
    τ::Float64,
)
    butterflycluster = deepcopy(butterflycluster_init)
    higherkBF = deepcopy(higherkBF_init)
    #recall that for multiplication the source tree of the left factor and the observer tree
    #of the right factor must match.
    children = [bf.NS for bf in butterflycluster[1, :]]
    numchildren = length(children)
    l = length(higherkBF.R) # Number of R-levels in the higher k BF
    #Step 1: subdividing the higher k BF into numchildren BFs of lvl k-1
    lowerkBFs = Vector{ButterflyFactorization}(undef, numchildren)
    ssubtree = h2treelevels(higherkBF.stree, higherkBF.NS)
    for i in 1:numchildren
        osubtree = h2treelevels(higherkBF.otree, children[i])
        new_P = Dict{Tuple{Int,Int},Matrix{ComplexF64}}()
        for leaf in osubtree[end]
            new_P[leaf, higherkBF.NS] = copy(higherkBF.P[leaf, higherkBF.NS])
        end
        new_R = Vector{Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},Matrix{ComplexF64}}}}(
            undef, l - 1
        )
        for j in 1:(l - 1)
            new_R[j] = Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},Matrix{ComplexF64}}}()
            for onode in osubtree[j + 1]
                for snode in ssubtree[end - j - 1]
                    #if haskey(higherkBF.R[j], (onode, snode))
                    new_R[j][(onode, snode)] = copy(higherkBF.R[j + 1][(onode, snode)])
                    #end
                end
            end
        end
        new_Q = Dict{Tuple{Int,Int},Matrix{ComplexF64}}()
        lowerkBFs[i] = ButterflyFactorization(
            new_Q,
            new_R,
            new_P,
            (size(higherkBF, 1), size(higherkBF, 2)),
            higherkBF.NS,
            children[i],
            higherkBF.k,
            higherkBF.τ,
            higherkBF.stree,
            higherkBF.otree,
        )
    end
    #The new Butterflies will not have any Q factors, those are preserved together with the
    #last R factor in the higher k BF until we performed the multiplications between the
    #smaller BFs and the cluster of BFs.
    intermediate = Matrix{ButterflyFactorization}(
        undef, size(butterflycluster, 1), numchildren
    )
    for i in 1:size(butterflycluster, 1)
        for j in 1:numchildren
            intermediate[i, j] = mulBFs(#trivialmul
                butterflycluster[i, j],
                lowerkBFs[j],
                τ,
            )
        end
    end
    rowsize = size(intermediate, 1)
    colsize = size(intermediate, 2)
    tobeadditioned = Vector{ButterflyFactorization}(undef, rowsize)
    nodaloffset = 1
    otmapv = Vector{Vector{Dict{Int,Int}}}(undef, colsize)
    for i in 1:rowsize
        supertree, mappings, root_super_id, nodaloffset = build_supertree(
            children, higherkBF.otree, nodaloffset
        )
        otmapv[i] = mappings
    end
    for i in 1:rowsize
        new_P = Dict{Tuple{Int,Int},Matrix{ComplexF64}}()
        new_R = Vector{Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},Matrix{ComplexF64}}}}(
            undef, l
        )
        for s in eachindex(new_R)
            new_R[s] = Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},Matrix{ComplexF64}}}()
        end

        #otmapv = Vector{Dict{Int,Int}}(undef, colsize)
        colspaces = Vector{Dict{Tuple{Int,Int},Int}}(undef, colsize)
        for j in 1:colsize
            # Target index based on your wrap-around logic
            target_idx = ((i + j-2) % rowsize) + 1
            target_bf = intermediate[target_idx, j]
            #otmapv[j] = treemapping(children[j], children[target_idx], higherkBF.otree)
            otmap = otmapv[target_idx][j]
            colspaces[j] = retrievecolspace(target_bf.R[1])
            deep_accumulate_P!(new_P, target_bf.P)
            for k in 1:(l - 2)
                deep_accumulate_R!(
                    new_R[k + 1], target_bf.R[k]; otmap=otmap, stmap=Dict{Int,Int}()
                )
            end
            for (node_key, inner_dict_src) in target_bf.R[l - 1]
                if !haskey(new_R[l], node_key)
                    new_R[l][node_key] = Dict{Tuple{Int,Int},Matrix{ComplexF64}}()
                    for (sub_key, mat_src) in inner_dict_src
                        if haskey(otmap, sub_key[1])
                            new_sub_key = (otmap[sub_key[1]], sub_key[2])
                        else
                            new_sub_key = sub_key
                        end
                        new_R[l][node_key][new_sub_key] = copy(mat_src)
                    end
                else
                    # Node key exists, merge the inner mapping level
                    inner_dict_dest = dest[node_key]
                    for (sub_key, mat_src) in inner_dict_src
                        if haskey(otmap, sub_key[1])
                            new_sub_key = (otmap[sub_key[1]], sub_key[2])
                        else
                            new_sub_key = sub_key
                        end
                        if !haskey(inner_dict_dest, new_sub_key)
                            inner_dict_dest[new_sub_key] = copy(mat_src)
                        else
                            println(
                                "Overlapping block detected at node_key: $new_node_key, sub_key: $new_sub_key",
                            )
                            blockdiag(inner_dict_dest[new_sub_key], mat_src)
                        end
                    end
                end
            end
        end
        localkeys = collect(keys(new_P))
        for key in localkeys
            nk = (key[1], H2Trees.parent(butterflycluster[1, 1].otree, key[2]))
            new_P[nk] = copy(new_P[key])
            new_R[end][nk] = copy(new_R[end][key])
            delete!(new_P, key)
            delete!(new_R[end], key)
        end
        new_R[1] = Dict{Tuple{Int,Int},Dict{Tuple{Int,Int},Matrix{ComplexF64}}}()
        for j in eachindex(colspaces)
            target_idx = ((i + j-2) % rowsize) + 1
            otmap = otmapv[target_idx][j]
            for key in keys(colspaces[j])
                if haskey(otmap, key[1])
                    new_key = (otmap[key[1]], key[2])
                    new_R[1][new_key] = copy(higherkBF.R[1][key])
                end
            end
        end
        new_Q = higherkBF.Q
        tobeadditioned[i] = #recompress_BF(
        ButterflyFactorization(
            new_Q,
            new_R,
            new_P,
            (
                length(
                    H2Trees.values(
                        butterflycluster[1, 1].otree,
                        H2Trees.parent(
                            butterflycluster[1, 1].otree, butterflycluster[1, 1].NO
                        ),
                    ),
                ),
                size(higherkBF, 2),
            ),
            higherkBF.NS,
            H2Trees.parent(butterflycluster[1, 1].otree, butterflycluster[1, 1].NO),
            higherkBF.k,
            higherkBF.τ,
            higherkBF.stree,
            butterflycluster[1, 1].otree,
        )#,
        #τ,
        #)
    end
    #return tobeadditioned
    l = length(tobeadditioned)-1
    result = add_eqbfs(tobeadditioned[1], tobeadditioned[2], τ)

    #println("addition 1 of $l done \n")
    for i in eachindex(tobeadditioned[3:end])
        result = add_eqbfs(result, tobeadditioned[3:end][i], τ)
        h = i+1
        #println("addition $h of $l done \n")
    end
    return result
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
