@testitem "Testing Full Matrix Assembly (Global Operator)" begin
    using Test
    using H2Trees
    using OhMyThreads
    using CompScienceMeshes
    using BEAST
    using ButterflyFactorizations
    using StaticArrays
    using LinearAlgebra
    using LinearMaps

    # Prevent BLAS from oversubscribing threads on GitHub Actions CI runners
    BLAS.set_num_threads(1)

    # =========================================================================
    # 1. Power Iteration Utility Functions
    # =========================================================================

    """
        estimate_norm(mat; tol=1e-4, itmax=100)

    Estimates the spectral norm (||A||_2) of a matrix or linear operator using
    Power Iteration on the normal matrix (A' * A).

    This is highly memory-efficient as it only requires matrix-vector products,
    avoiding the need to ever instantiate `mat` as a dense array.
    """
    function estimate_norm(mat; tol=1e-4, itmax=100)
        v = rand(ComplexF64, size(mat, 2))
        v ./= norm(v)

        itermin = 3
        i = 1
        σold = 1.0
        σnew = 1.0

        while (norm(sqrt(σold) - sqrt(σnew)) / norm(sqrt(σold)) > tol || i < itermin) &&
            i < itmax
            σold = σnew
            w = mat * v
            x = adjoint(mat) * w
            σnew = norm(x)
            v = x ./ σnew
            i += 1
        end
        @info "Power iteration (Norm) converged in $i steps: ||A||_2 ≈ $(sqrt(σnew))"
        return sqrt(σnew)
    end

    """
        estimate_reldifference(hmat, refmat; tol=1e-4)

    Estimates the relative spectral error ||H - A||_2 / ||A||_2 between the
    compressed operator `hmat` and the dense reference matrix `refmat`.
    """
    function estimate_reldifference(
        hmat::H, refmat; tol=1e-4, itmax=100
    ) where {F,H<:LinearMaps.LinearMap{F}}
        @assert size(hmat) == size(refmat) "Dimensions of matrices do not match"

        v = rand(ComplexF64, size(hmat, 2))
        v ./= norm(v)

        itermin = 3
        i = 1
        σold = 1.0
        σnew = 1.0

        while (norm(sqrt(σold) - sqrt(σnew)) / norm(sqrt(σold)) > tol || i < itermin) &&
            i < itmax
            σold = σnew
            # Matrix-free application of the difference operator (H - A)
            w = (hmat * v) .- (refmat * v)
            x = (adjoint(hmat) * w) .- (adjoint(refmat) * w)
            σnew = norm(x)

            # 🚀 THE FIX: Prevent division by zero if the error drops to machine zero
            if σnew < 1e-14
                v .= zero(eltype(v))
                break
            end

            v = x ./ σnew
            i += 1
        end

        @info "Power iteration (Error) converged in $i steps: ||H - A||_2 ≈ $(sqrt(σnew))"

        norm_refmat = estimate_norm(refmat; tol=tol)
        return sqrt(σnew) / norm_refmat
    end

    # =========================================================================
    # 2. Geometry and Operators
    # =========================================================================
    @info "Setting up BEAST Geometry and Operators..."
    lambda = 1.0
    k = 2 * pi / lambda

    # Generate a sphere mesh. lambda/10 provides a good balance between
    # testing hierarchy depth and executing quickly on CI servers.
    x = meshsphere(0.25, lambda / 10)
    op = Maxwell3D.singlelayer(; wavenumber=k)
    T = raviartthomas(x)

    @info "Global Degrees of Freedom: $(length(T))"

    # =========================================================================
    # 3. Tree Construction and Ground Truth Assembly
    # =========================================================================
    @info "Constructing TwoNTree and Dense Ground Truth..."
    tree1 = TwoNTree(T, T, lambda / 10) # testspace, trialspace, box size

    # Assemble the full dense matrix for baseline error comparison
    @time A1 = assemble(op, T, T)

    # =========================================================================
    # 4. Global Butterfly Operator Assembly
    # =========================================================================
    target_tol = 1e-3

    @info "Assembling Flat-Array PetrovGalerkinBF..."
    @time Bfly1 = ButterflyFactorizations.PetrovGalerkinBF(
        op, T, T, tree1, k; tol=target_tol, scheduler=OhMyThreads.StaticScheduler()
    )

    @info "Assembling Sparse-Matrix PetrovGalerkinBF_Mat..."
    @time Bfly2 = ButterflyFactorizations.PetrovGalerkinBF_Mat(
        op, T, T, tree1, k; tol=target_tol, scheduler=OhMyThreads.StaticScheduler()
    )

    # =========================================================================
    # 5. Verification & Memory Profiling
    # =========================================================================
    @info """
    --- Global Operator Memory Footprint ---
    Dense Matrix       : $(round(Base.summarysize(A1) / 1024^2, digits=3)) MB
    Flat-Array BF      : $(round(Base.summarysize(Bfly1) / 1024^2, digits=3)) MB
    Sparse-Matrix BF   : $(round(Base.summarysize(Bfly2) / 1024^2, digits=3)) MB
    """

    # Check relative spectral norm differences
    err_flat = estimate_reldifference(Bfly1, A1; tol=1e-4)
    err_sparse = estimate_reldifference(Bfly2, A1; tol=1e-4)

    @info """
    --- Accuracy Results ---
    Target Tolerance   : $target_tol
    Flat-Array Error   : $(round(err_flat, sigdigits=4))
    Sparse-Matrix Error: $(round(err_sparse, sigdigits=4))
    """

    # Allow a factor of 10 for compression error accumulation
    # across the hierarchical approximations.
    @test err_flat < target_tol * 10
    @test err_sparse < target_tol * 10
end
