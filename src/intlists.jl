import H2Trees: isleaf, testtree, trialtree, root, children

"""
    isFarFunctor

A functor (callable struct) used to determine if two bounding boxes (clusters)
in a hierarchical tree are well-separated ("far") enough to be compressed.

**Fields:**

  - `α::Float64`: The separation parameter. A larger `α` forces clusters to be
    further apart before they are considered admissible for low-rank approximation.
"""
struct isFarFunctor
    α::Float64
    isFarFunctor(α) = new(α)
end

"""
    (t::isFarFunctor)(srctree, tsttree, snode, onode)

Evaluates the admissibility condition between a source node and an observer node.

The rule checks if the distance between the bounding box centers is sufficiently
larger than the sum of their physical sizes, scaled by the separation parameter `α`
and the maximum half-size `W` of the two clusters.

**Arguments:**

  - `srctree`: The tree structure containing source (trial) clusters.
  - `tsttree`: The tree structure containing observer (test) clusters.
  - `snode`: The ID of the source node to evaluate.
  - `onode`: The ID of the observer node to evaluate.

**Returns:**

  - `true` if the nodes are well-separated (admissible for far-field compression).
  - `false` if they are too close (must be treated as near-field or split further).
"""
function (t::isFarFunctor)(
    srctree::H2Trees.TwoNTree, tsttree::H2Trees.TwoNTree, snode, onode
)
    ocenter = H2Trees.center(tsttree, onode)
    olength = H2Trees.halfsize(tsttree, onode)
    scenter = H2Trees.center(srctree, snode)
    slength = H2Trees.halfsize(srctree, snode)
    W = max(H2Trees.halfsize(srctree, snode), H2Trees.halfsize(tsttree, onode))
    ro = (sqrt(3) / 2) * olength
    rs = (sqrt(3) / 2) * slength
    if norm(scenter - ocenter) - (ro + rs) > t.α * W
        return true
    else
        mind = 0.0
        length = (slength + olength) / 2
        for i in 1:3
            mind += max(0.0, abs(ocenter[i] - scenter[i]) - length)^2
        end
        mind = sqrt(mind)
        if mind > t.α * W
            return true
        else
            return false
        end
    end
end

function (t::isFarFunctor)(
    srctree::H2Trees.BoundingBallTree, tsttree::H2Trees.BoundingBallTree, snode, onode
)
    ocenter = H2Trees.center(tsttree, onode)
    scenter = H2Trees.center(srctree, snode)

    # För BoundingBalls är halfsize exakt sfärens radie
    olength = H2Trees.radius(tsttree, onode)
    slength = H2Trees.radius(srctree, snode)

    W = max(slength, olength)

    # Avståndet mellan mittpunkterna
    dist = norm(scenter - ocenter)

    # Kolla om avståndet minus BÅDA radierna är tillräckligt stort
    # (Dvs. avståndet mellan sfärernas yttersta kanter)
    if dist - (olength + slength) > t.α * W
        return true
    else
        return false
    end
    #=
    cs = H2Trees.center(srctree, snode)
    co = H2Trees.center(tsttree, onode)
    rs = H2Trees.radius(srctree, snode)
    ro = H2Trees.radius(tsttree, onode)

    d = norm(cs .- co)

    # GARANTERA att sfärerna har ett tryggt gap mellan varandra!
    # α bestämmer hur stort gapet ska vara i relation till radierna.
    return d > t.α * (rs + ro)
    =#
end

"""
    nearandfar(tree::H2Trees.BlockTree, α)

Traverses a block-tree and categorizes interactions between source and observer
clusters into "near-field" and "far-field" lists.

This function acts as the entry point for the dual-tree traversal algorithm.
It uses the admissibility condition (`isFarFunctor`) to separate interactions.

**Arguments:**

  - `tree`: A strictly coupled `BlockTree` containing both the test and trial trees.
  - `α`: The geometric separation parameter.

**Returns:**

  - `nearov`: A `Vector` containing the global observer indices for near-field blocks.
  - `nearsv`: A `Vector` containing the global source indices for near-field blocks.
  - `farinteractions`: A `Dict` mapping an observer node ID to a list of source node
    IDs that are well-separated from it.
"""
function nearandfar(tree::H2Trees.BlockTree, α)
    admissible = isFarFunctor(α)
    srctree = trialtree(tree)
    tsttree = testtree(tree)
    node_o = root(tsttree)
    node_s = root(srctree)
    nearsv = Vector{Int}[]
    nearov = Vector{Int}[]
    #nearinteractions = Dict{Int64,Vector{Int64}}()          #observernodeid --> sourcenodeid
    farinteractions = Dict{Int64,Vector{Int64}}()           #observernodeid --> sourcenodeid
    process_nodes!(
        srctree, tsttree, node_o, node_s, admissible, farinteractions, nearsv, nearov
    )
    return nearov, nearsv, farinteractions
end

"""
    process_nodes!(srctree, tsttree, node_o, node_s, admissible, far, nearsv, nearov)

Recursively analyzes the interaction between a source node and an observer node.

  - If the nodes are `admissible` (well-separated), they are added to the `farinteractions` list.
  - If they are not admissible but both are leaf nodes, their direct mesh indices are
    extracted and appended to the near-field lists (`nearsv` and `nearov`).
  - If they are not admissible and can be split, the functionally larger node is subdivided
    into its children, and the process repeats.

**Arguments:**

  - `srctree`, `tsttree`: The tree hierarchies.
  - `node_o`, `node_s`: Current observer and source node IDs.
  - `admissible`: The `isFarFunctor` used to check geometric separation.
  - `farinteractions`: Accumulator dictionary for far-field node pairs.
  - `nearsv`, `nearov`: Accumulator vectors for near-field global indices.
"""
function process_nodes!(
    srctree::H2Trees.TwoNTree,
    tsttree::H2Trees.TwoNTree,
    node_o,
    node_s,
    admissible::isFarFunctor,
    farinteractions,
    nearsv,
    nearov,
)
    if admissible(srctree, tsttree, node_s, node_o) #&&
        #!(isleaf(tsttree, node_o) && isleaf(srctree, node_s))
        push!(get!(farinteractions, node_o, Int64[]), node_s)
        return nothing
    elseif isleaf(tsttree, node_o) && isleaf(srctree, node_s)
        push!(nearsv, H2Trees.values(srctree, node_s))
        push!(nearov, H2Trees.values(tsttree, node_o))
        return nothing
    end
    # split the larger node

    if H2Trees.halfsize(tsttree, node_o) >= H2Trees.halfsize(srctree, node_s)
        for child_o in collect(children(tsttree, node_o))
            process_nodes!(
                srctree,
                tsttree,
                child_o,
                node_s,
                admissible,
                farinteractions,
                nearsv,
                nearov,
            )
        end
    else
        for child_s in collect(children(srctree, node_s))
            process_nodes!(
                srctree,
                tsttree,
                node_o,
                child_s,
                admissible,
                farinteractions,
                nearsv,
                nearov,
            )
        end
    end
    #=
    for child_o in collect(children(tsttree, node_o))
        for child_s in collect(children(srctree, node_s))
            process_nodes!(
                srctree,
                tsttree,
                child_o,
                child_s,
                admissible,
                farinteractions,
                nearsv,
                nearov,
            )
        end
    end
    =#
end

function process_nodes!(
    srctree::H2Trees.BoundingBallTree,
    tsttree::H2Trees.BoundingBallTree,
    node_o,
    node_s,
    admissible::isFarFunctor,
    farinteractions,
    nearsv,
    nearov,
)
    if admissible(srctree, tsttree, node_s, node_o) #&&
        #!(isleaf(tsttree, node_o) && isleaf(srctree, node_s))
        push!(get!(farinteractions, node_o, Int64[]), node_s)
        return nothing
    elseif isleaf(tsttree, node_o) && isleaf(srctree, node_s)
        push!(nearsv, H2Trees.values(srctree, node_s))
        push!(nearov, H2Trees.values(tsttree, node_o))
        return nothing
    end
    # split the larger node

    if H2Trees.radius(tsttree, node_o) >= H2Trees.radius(srctree, node_s)
        for child_o in collect(children(tsttree, node_o))
            process_nodes!(
                srctree,
                tsttree,
                child_o,
                node_s,
                admissible,
                farinteractions,
                nearsv,
                nearov,
            )
        end
    else
        for child_s in collect(children(srctree, node_s))
            process_nodes!(
                srctree,
                tsttree,
                node_o,
                child_s,
                admissible,
                farinteractions,
                nearsv,
                nearov,
            )
        end
    end
    #=
    for child_o in collect(children(tsttree, node_o))
        for child_s in collect(children(srctree, node_s))
            process_nodes!(
                srctree,
                tsttree,
                child_o,
                child_s,
                admissible,
                farinteractions,
                nearsv,
                nearov,
            )
        end
    end
    =#
end
