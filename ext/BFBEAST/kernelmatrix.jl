function ButterflyFactorizations.AbstractKernelMatrix(
    operator::BEAST.IntegralOperator,
    testspace::BEAST.Space,
    trialspace::BEAST.Space;
    type=:near, # Används för att skilja mellan near/far
    quadstrat=nothing, # Frivillig överskrivning från testskript
)
    # Välj strategi baserat på type-flaggan om användaren inte skickat in en egen
    actual_quadstrat = if quadstrat !== nothing
        quadstrat
    elseif type == :far
        BEAST.DoubleNumQStrat(2, 3)
    else
        BEAST.defaultquadstrat(operator, testspace, trialspace)
    end

    return ButterflyFactorizations.BEASTKernelMatrix{scalartype(operator)}(
        BEAST.blockassembler(operator, testspace, trialspace; quadstrat=actual_quadstrat)
    )
end
struct BlockStoreFunctor{M}
    matrix::M
end

function (f::BlockStoreFunctor)(v, m, n)
    @views f.matrix[m, n] += v
    return nothing
end

function (blk::ButterflyFactorizations.BEASTKernelMatrix)(matrixblock, tdata, sdata)
    blk.nearassembler(tdata, sdata, BlockStoreFunctor(matrixblock))
    return nothing
end
