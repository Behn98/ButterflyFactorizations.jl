
@testitem "Testing Blockassembly subroutines for pruned trees" begin
    using Test
    using BEAST
    using OhMyThreads
    using CompScienceMeshes
    using ParallelKMeans
    using H2Trees
    using ButterflyFactorizations
    using StaticArrays
    using LinearAlgebra

    h = 0.1
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

    Ttree2 = H2Trees.TwoNTree(X1, h; minvalues=100)
    Stree2 = H2Trees.TwoNTree(X2, h; minvalues=100)

    blktree2 = BlockTree(Ttree2, Stree2)
    go2 = H2Trees.values(blktree2.testcluster, H2Trees.root(blktree2.testcluster))

    gs2 = H2Trees.values(blktree2.trialcluster, H2Trees.root(blktree2.trialcluster))
    kernelmatrix = ButterflyFactorizations.AbstractKernelMatrix(op, X1, X2)

    Bfmat = ButterflyFactorizations.PetrovGalerkinBF(
        op,
        X1,
        X2,
        blktree1,
        k;
        compressor=ButterflyFactorizations.PartialQR(),
        tol=1e-4,
        α=1.5,
        scheduler=OhMyThreads.DynamicScheduler(),
    )

    Bfmat2 = ButterflyFactorizations.PetrovGalerkinBF(
        op,
        X1,
        X2,
        blktree2,
        k;
        compressor=ButterflyFactorizations.PartialQR(),
        tol=1e-4,
        α=1.5,
        scheduler=OhMyThreads.DynamicScheduler(),
    )

    A = assemble(op, X1, X2)
    x = randn(ComplexF64, size(A, 2))
    y_exact = A * x

    y_approx1 = zeros(ComplexF64, size(A, 1))
    y_approx1 = Bfmat * x#[go1][gs1]
    (norm(y_exact - y_approx1) / norm(y_exact))

    @test ((norm(y_exact - y_approx1) / norm(y_exact)) < 10^-3)

    y_approx = zeros(ComplexF64, size(A, 1))
    y_approx = Bfmat2 * x
    (norm(y_exact - y_approx) / norm(y_exact))
    @test ((norm(y_exact - y_approx) / norm(y_exact)) < 10^-3)
end
