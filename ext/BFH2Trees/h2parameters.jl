#-----------------------------------------------------------------------------------------#
#------------ Optimize and declare tree parameters for ButterflyFactorizations -----------#
#-----------------------------------------------------------------------------------------#

#-----------------------------------------------------------------------------------------#
#------------------------------ Single Block compression ---------------------------------#
#-----------------------------------------------------------------------------------------#

"""
    tree_parameters(tree, admissiblity) -> TreeParameters

Retrieves the calibrated parameters (`α`, `C`, `Cε`, `β`) for a given tree structure.

These constants are derived from empirical rank-logging runs and are used by the
`ButterflyFactorizations` module to optimize admissibility checks and guide the
`PartialQR` rank estimator.
"""
function ButterflyFactorizations.tree_parameters(tree::H2Trees.BlockTree)
    return ButterflyFactorizations.tree_parameters(tree.testcluster)
end

function ButterflyFactorizations.tree_parameters(tree::H2Trees.TwoNTree)
    return ButterflyFactorizations.TreeParameters(0.2, 2.3858, 1.8072, 1.0, 4.0)
end

function ButterflyFactorizations.tree_parameters(tree::BisectionTree)
    return ButterflyFactorizations.TreeParameters(0.2, 0.5197, 2.6569, 1.0, 4.0)
end

function ButterflyFactorizations.tree_parameters(tree::H2Trees.BoundingBallTree)
    return ButterflyFactorizations.TreeParameters(0.6, 3.2, 2.4, 1.5, 4.0)
end

#-----------------------------------------------------------------------------------------#
#----------------------------------- isFarFunctor ----------------------------------------#
#-----------------------------------------------------------------------------------------#

function ButterflyFactorizations.tree_parameters(
    tree::H2Trees.BlockTree, admissiblity::ButterflyFactorizations.isFarFunctor
)
    return ButterflyFactorizations.tree_parameters(tree.testcluster, admissiblity)
end

function ButterflyFactorizations.tree_parameters(
    tree::H2Trees.TwoNTree, admissiblity::ButterflyFactorizations.isFarFunctor
)
    return ButterflyFactorizations.TreeParameters(0.2, 2.3858, 1.8072, 1.0, 4.0)
end

function ButterflyFactorizations.tree_parameters(
    tree::BisectionTree, admissiblity::ButterflyFactorizations.isFarFunctor
)
    return ButterflyFactorizations.TreeParameters(0.2, 0.5197, 2.6569, 1.0, 4.0)
end

function ButterflyFactorizations.tree_parameters(
    tree::H2Trees.BoundingBallTree, admissiblity::ButterflyFactorizations.isFarFunctor
)
    return ButterflyFactorizations.TreeParameters(0.6, 3.2, 2.4, 1.5, 4.0)
end

#-----------------------------------------------------------------------------------------#
#------------------------------ CenterDistanceAdmissibility ------------------------------#
#-----------------------------------------------------------------------------------------#

function ButterflyFactorizations.tree_parameters(
    tree::H2Trees.BlockTree,
    admissiblity::ButterflyFactorizations.CenterDistanceAdmissibility,
)
    return ButterflyFactorizations.tree_parameters(tree.testcluster, admissiblity)
end

function ButterflyFactorizations.tree_parameters(
    tree::H2Trees.TwoNTree,
    admissiblity::ButterflyFactorizations.CenterDistanceAdmissibility,
)
    return ButterflyFactorizations.TreeParameters(0.2, 2.3858, 1.8072, 1.0, 4.0)
end

function ButterflyFactorizations.tree_parameters(
    tree::BisectionTree, admissiblity::ButterflyFactorizations.CenterDistanceAdmissibility
)
    return ButterflyFactorizations.TreeParameters(0.2, 0.5197, 2.6569, 1.0, 4.0)
end

function ButterflyFactorizations.tree_parameters(
    tree::H2Trees.BoundingBallTree,
    admissiblity::ButterflyFactorizations.CenterDistanceAdmissibility,
)
    return ButterflyFactorizations.TreeParameters(0.6, 3.2, 2.4, 1.5, 4.0)
end
