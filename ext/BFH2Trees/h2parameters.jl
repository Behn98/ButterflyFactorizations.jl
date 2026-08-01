#-----------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------#
#------------ Optimize and declare tree parameters for ButterflyFactorizations -----------#
#-----------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------#
"""
use admissibility and rank logging files in the evaluation folder to compute the parameters
α, C, and Cϵ for the given tree. The parameters are then stored in a TreeParameters struct,
which is used by the ButterflyFactorizations module to optimize the admissibility condition
and rank estimation for the given tree structure. The parameters are computed based on the
geometry and structure of the tree, and are used to determine whether two clusters are
well-separated (admissible for far-field compression) or too close (must be treated as
near-field """

function ButterflyFactorizations.tree_parameters(tree::H2Trees.BlockTree)
    return ButterflyFactorizations.tree_parameters(tree.testcluster)
end
#=
struct TreeParameters{T} <: AbstractTreeParameters
    α::T
    C::T
    Cε::T
    β::T
end
=#
function ButterflyFactorizations.tree_parameters(tree::H2Trees.TwoNTree)
    return ButterflyFactorizations.TreeParameters(0.1, 2.3858, 1.8072, 1.0)
end

function ButterflyFactorizations.tree_parameters(tree::H2Trees.BisectionTree)
    return ButterflyFactorizations.TreeParameters(0.1, 0.5197, 2.6569, 1.0)
end

function ButterflyFactorizations.tree_parameters(tree::H2Trees.BoundingBallTree)
    return ButterflyFactorizations.TreeParameters(0.1, 0.9828, 3.0000, 1.0)
end
