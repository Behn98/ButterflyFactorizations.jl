@testitem "Testing Full Matrix Assembly" begin
    using Test
    using H2Trees
    using OhMyThreads
    using CompScienceMeshes
    using BEAST
    using ButterflyFactorizations
    using StaticArrays
    using LinearAlgebra
    using LinearMaps
    using ParallelKMeans
    #Power iteration to estimate the spectral norm of a matrix, used for estimating the
    #relative difference between the butterfly approximation and the fully assembled matrix

    function estimate_norm(mat; tol=1e-4, itmax=1000)
        v = rand(size(mat, 2))

        v = v / norm(v)
        itermin = 3
        i = 1
        σold = 1
        σnew = 1
        @info "Estimate norm"
        while (norm(sqrt(σold) - sqrt(σnew)) / norm(sqrt(σold)) > tol || i < itermin) &&
            i < itmax
            @info i, norm(sqrt(σold) - sqrt(σnew)) / norm(sqrt(σold))
            σold = σnew
            w = mat * v
            x = adjoint(mat) * w
            σnew = norm(x)
            v = x / norm(x)
            i += 1
        end
        return sqrt(σnew)
    end

    function estimate_reldifference(
        hmat::H, refmat; tol=1e-4
    ) where {F,H<:LinearMaps.LinearMap{F}}  #hmat - BFApprox, refmat - fully assembled A
        #if size(hmat) != size(refmat)
        #    error("Dimensions of matrices do not match")
        #end
        #@show F
        v = rand(ComplexF64, size(hmat, 2))

        v = v / norm(v)
        itermin = 3
        i = 1
        σold = 1
        σnew = 1
        @info "Estimate norm of reference matrix"
        while norm(sqrt(σold) - sqrt(σnew)) / norm(sqrt(σold)) > tol || i < itermin
            @info i, norm(sqrt(σold) - sqrt(σnew)) / norm(sqrt(σold))
            σold = σnew
            w = (hmat * v) - (refmat * v)
            x = (adjoint(hmat) * w) - (adjoint(refmat) * w)
            σnew = norm(x)
            v = x / σnew
            i += 1
        end
        @info "Estimate norm of reference matrix"
        norm_refmat = estimate_norm(refmat; tol=tol)

        return sqrt(σnew) / norm_refmat
    end

    #========================================================================
    =========================================================================
                            Geometry and Operators
    =========================================================================
    =========================================================================#
    lambda = 1.0
    k = 2 * pi / lambda
    x = meshsphere(0.25, lambda / 10)
    op = Maxwell3D.singlelayer(; wavenumber=k)
    T = raviartthomas(x)
    length(T)

    ##
    #========================================================================
    =========================================================================
                    Tree construction  and Kernelmatrix assembly
    =========================================================================
    =========================================================================#

    tree1 = TwoNTree(T, T, lambda / 10)     #testspace, trialspace

    #========================================================================
    =========================================================================
                        Assembly of Matrices and Vectors
    =========================================================================
    =========================================================================#

    A1 = assemble(op, T, T)

    #========================================================================
    =========================================================================
                            Buttefly routines calling
    =========================================================================
    =========================================================================#

    Bfly1 = ButterflyFactorizations.PetrovGalerkinBF(op, T, T, tree1, k; tol=1e-3, α=2)
    Bfly2 = ButterflyFactorizations.PetrovGalerkinBF_mats(op, T, T, tree1, k; tol=1e-3, α=2)

    @test estimate_reldifference(Bfly1, A1; tol=1e-4) < 1e-2
    @test estimate_reldifference(Bfly2, A1; tol=1e-4) < 1e-2
end
