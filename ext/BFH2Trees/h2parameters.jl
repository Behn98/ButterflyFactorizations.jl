#-----------------------------------------------------------------------------------------#
#------------ Optimize and declare tree parameters for ButterflyFactorizations -----------#
#-----------------------------------------------------------------------------------------#

#-----------------------------------------------------------------------------------------#
#------------------------------ Single Block compression ---------------------------------#
#-----------------------------------------------------------------------------------------#

"""
    tree_parameters(tree, admissiblity) -> TreeParameters

Retrieves the calibrated parameters (`α`, `C`, `Cε`, `β`, `Cτ`) for a given tree structure.

These constants are derived from empirical rank-logging runs and are used by the
`ButterflyFactorizations` module to optimize admissibility checks and guide the
`PartialQR` rank estimator.
"""
function ButterflyFactorizations.tree_parameters(tree::H2Trees.BlockTree)
    return ButterflyFactorizations.tree_parameters(tree.testcluster)
end

function ButterflyFactorizations.tree_parameters(tree::H2Trees.TwoNTree)
    return ButterflyFactorizations.TreeParameters(1.0, 2.0, 2.0, 1.0, 4.0)
end

function ButterflyFactorizations.tree_parameters(tree::BisectionTree)
    return ButterflyFactorizations.TreeParameters(0.0, 2.0, 2.0, 1.0, 4.0)
end

function ButterflyFactorizations.tree_parameters(tree::H2Trees.BoundingBallTree)
    return ButterflyFactorizations.TreeParameters(0.4, 2.0, 2.0, 1.2, 4.0)
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
    return ButterflyFactorizations.TreeParameters(1.0, 2.0, 2.0, 1.0, 2.9592)
end

function ButterflyFactorizations.tree_parameters(
    tree::BisectionTree, admissiblity::ButterflyFactorizations.isFarFunctor
)
    return ButterflyFactorizations.TreeParameters(0.0, 0.8426, 2.7788, 1.0, 2.4114)
end

function ButterflyFactorizations.tree_parameters(
    tree::H2Trees.BoundingBallTree, admissiblity::ButterflyFactorizations.isFarFunctor
)
    return ButterflyFactorizations.TreeParameters(0.4, 6.3404, 2.4151, 1.2, 2.7128)
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
    return ButterflyFactorizations.TreeParameters(1.0, 2.0, 2.0, 1.0, 2.9352)
end

function ButterflyFactorizations.tree_parameters(
    tree::BisectionTree, admissiblity::ButterflyFactorizations.CenterDistanceAdmissibility
)
    return ButterflyFactorizations.TreeParameters(0.0, 2.4163, 2.7163, 1.0, 3.3048)
end

function ButterflyFactorizations.tree_parameters(
    tree::H2Trees.BoundingBallTree,
    admissiblity::ButterflyFactorizations.CenterDistanceAdmissibility,
)
    return ButterflyFactorizations.TreeParameters(0.4, 6.1545, 2.4274, 1.2, 2.5334)
end
