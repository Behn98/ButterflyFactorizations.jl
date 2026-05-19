@testitem "Testing Blockassembly subroutines for pruned trees" begin
    using BEAST
    using CompScienceMeshes
    using ParallelKMeans
    using H2Trees
    using ButterflyFactorizations
    using StaticArrays
    using LinearAlgebra

    h = 0.05
    lambda = 1
    k = 2 * pi / lambda

    op = Maxwell3D.singlelayer(; wavenumber=k)

    #Geometry
    m1 = meshsphere(1.0, h)
    X1 = raviartthomas(m1)
    m2 = translate(m1, SVector(5.0, 0.0, 0.0))
    X2 = raviartthomas(m2)

    Ttree = H2Trees.KMeansTree(X1.pos, 2; minvalues=100)
    Stree = H2Trees.KMeansTree(X2.pos, 2; minvalues=100)

    blktree1 = H2Trees.BlockTree(Ttree, Stree)
    go1 = H2Trees.values(blktree1.testcluster, H2Trees.root(blktree1.testcluster))

    gs1 = H2Trees.values(blktree1.trialcluster, H2Trees.root(blktree1.trialcluster))

    kernelmatrix = ButterflyFactorizations.AbstractKernelMatrix(op, X1, X2)

    Bfmat = ButterflyFactorizations.subroutine_BF(
        kernelmatrix,
        blktree1,
        1,
        1,
        k,
        10^(-4);
        Compressor=ButterflyFactorizations.PartialQR(),
    )

    A = assemble(op, X1, X2)
    x = randn(ComplexF64, size(A, 2))
    y_exact = A * x
    y_approx = zeros(ComplexF64, size(A, 1))
    y_approx[go1] = Bfmat * x[gs1]
    @test (norm(y_exact - y_approx) / norm(y_exact)) < 10^-3
end
