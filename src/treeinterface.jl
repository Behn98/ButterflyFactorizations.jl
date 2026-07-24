#------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------#
#---------MUST BE IMPLEMENTED FOR ANY TREE TYPE TO BE USED WITH THIS PACKAGE!!!!!----------#
#------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------#

"""
    cluster_center(tree, node)

Return the geometric center of a cluster.
"""
function cluster_center(tree, node)
    throw(MethodError(cluster_center, (tree, node)))
end

"""
    cluster_radius(tree, node)

Return the radius (or half-width) of a cluster.
"""
function cluster_radius(tree, node)
    throw(MethodError(cluster_radius, (tree, node)))
end

"""
    cluster_children(tree, node)

Return the children of a cluster.
"""
function cluster_children(tree, node)
    throw(MethodError(cluster_children, (tree, node)))
end

"""
    cluster_isleaf(tree, node)
"""
function cluster_isleaf(tree, node)
    throw(MethodError(cluster_isleaf, (tree, node)))
end

"""
    root_node(tree)
"""
function cluster_root(tree)
    throw(MethodError(cluster_root, (tree,)))
end

function cluster_testtree(tree)
    throw(MethodError(cluster_testtree, (tree,)))
end

function cluster_trialtree(tree)
    throw(MethodError(cluster_trialtree, (tree,)))
end

function cluster_parent(tree, node)
    throw(MethodError(cluster_parent, (tree, node)))
end

function cluster_values(tree, node)
    throw(MethodError(cluster_values, (tree, node)))
end

largernode(tree1, tree2, node1, node2) =
    cluster_radius(tree1, node1) >= cluster_radius(tree2, node2)
