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

# struct TreeParameters{T} <: AbstractTreeParameters α::T C::T Cε::T end

function ButterflyFactorizations.tree_parameters(tree::H2Trees.TwoNTree)
    return ButterflyFactorizations.TreeParameters(1.0, 48.1385, 0.8303)
end

function ButterflyFactorizations.tree_parameters(tree::H2Trees.BisectionTree)
    return ButterflyFactorizations.TreeParameters(1.0, 6.6665, 1.7462)
end

function ButterflyFactorizations.tree_parameters(tree::H2Trees.BoundingBallTree)
    return ButterflyFactorizations.TreeParameters(1.0, 1.5738, 1.6357)
end
