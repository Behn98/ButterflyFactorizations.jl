"""
    AbstractKernelMatrix(operator::BEAST.IntegralOperator, testspace::BEAST.Space, trialspace::BEAST.Space; kwargs...)

Constructs a BEAST-compatible kernel matrix evaluator for the Butterfly Factorization.

This extension automatically assigns the appropriate quadrature strategy based on whether
the block is meant for near-field (dense, high accuracy) or far-field (compressed, lower quadrature order)
evaluation.

**Arguments:**

  - `operator`: The BEAST boundary integral operator (e.g., `Maxwell3D.singlelayer`).
  - `testspace`: The observer (test) basis function space.
  - `trialspace`: The source (trial) basis function space.

**Keyword Arguments:**

  - `type`: Symbol indicating the interaction type. Valid options are `:near` (uses BEAST's default
    quadrature) and `:far` (uses a lightweight `DoubleNumQStrat(2, 3)`). Default is `:near`.
  - `quadstrat`: Optional manual override for the quadrature strategy.
"""
function ButterflyFactorizations.AbstractKernelMatrix(
    operator::BEAST.IntegralOperator,
    testspace::BEAST.Space,
    trialspace::BEAST.Space;
    type=:near,
    quadstrat=nothing,
)
    # Select quadrature strategy based on the type flag unless overridden by the user
    actual_quadstrat = if quadstrat !== nothing
        quadstrat
    elseif type == :far
        BEAST.DoubleNumQStrat(2, 3)
    else
        BEAST.defaultquadstrat(operator, testspace, trialspace)
    end

    return ButterflyFactorizations.BEASTKernelMatrix{BEAST.scalartype(operator)}(
        BEAST.blockassembler(operator, testspace, trialspace; quadstrat=actual_quadstrat)
    )
end

"""
    BlockStoreFunctor{M}

An internal helper functor used to accumulate integral operator evaluations directly
into a pre-allocated dense matrix block.
"""
struct BlockStoreFunctor{M}
    matrix::M
end

function (f::BlockStoreFunctor)(v, m, n)
    @views f.matrix[m, n] += v
    return nothing
end

"""
    (blk::BEASTKernelMatrix)(matrixblock, tdata, sdata)

Evaluates the integral operator for a specific subset of test and trial indices and
stores the result in the provided `matrixblock`.

If either the test data (`tdata`) or source data (`sdata`) is empty, the `matrixblock`
is explicitly filled with zeros to maintain structural integrity without triggering
BEAST assembly errors.

**Arguments:**

  - `matrixblock`: A pre-allocated dense matrix buffer to store the evaluated entries.
  - `tdata`: The vector of global test (observer) space indices.
  - `sdata`: The vector of global trial (source) space indices.
"""
function (blk::ButterflyFactorizations.BEASTKernelMatrix)(matrixblock, tdata, sdata)
    if isempty(tdata) || isempty(sdata)
        fill!(matrixblock, zero(eltype(matrixblock)))
        return nothing
    end

    blk.nearassembler(tdata, sdata, BlockStoreFunctor(matrixblock))
    return nothing
end
