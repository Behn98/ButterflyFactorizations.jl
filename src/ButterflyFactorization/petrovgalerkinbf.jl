"""
    PetrovGalerkinBF(operator, testspace, trialspace, tree, k; kwargs...)

Constructs the complete Butterfly Factorization for a Petrov-Galerkin boundary element
problem using the **dictionary-based** format.

This function separates the interactions based on the admissibility condition `α`. The
near-field interactions are evaluated directly and stored as a `BlockSparseMatrix`. The
far-field interactions are compressed block-by-block using the dictionary-based
parameterized `subroutine_BF`, resulting in a collection of highly memory-efficient
Butterfly blocks. Keep in mind that as off right now this work is still restricted to
balanced trees in the sense that all leaves need to be on leaf level.

**Arguments:**
- `operator`: The integral operator (e.g., Maxwell single-layer) to be evaluated.
- `testspace`: The observer (test) function space.
- `trialspace`: The source (trial) function space.
- `tree`: A coupled `BlockTree` representing the hierarchical clustering.
- `k`: The physical wavenumber of the problem.

**Keyword Arguments:**
- `Compressor`: The low-rank approximation strategy (default: `PartialQR()`).
- `tol`: The relative precision tolerance for compression (default: `1e-3`).
- `ntasks`: Number of threads to use for parallel near-field assembly.
- `α`: The geometric admissibility parameter (default: `2.0`).

**Returns:**
- A `PetrovGalerkinBF` struct containing the near-field matrix and the array of far-field
  `BF` blocks.
"""

function PetrovGalerkinBF(
    operator,
    testspace,
    trialspace,
    tree::BlockTree,
    k::Float64;
    Compressor=ButterflyFactorizations.PartialQR(),
    tol=1e-3,
    ntasks=Threads.nthreads(),
    α=2,
)
    nearmatrix = AbstractKernelMatrix(operator, testspace, trialspace;)
    #quadstrat=nearquadstrat
    values, nearvalues, farints = nearandfar(tree, α)

    blocks = Vector{Matrix{eltype(nearmatrix)}}(undef, length(values))
    @tasks for i in eachindex(values)
        @set ntasks = 1 #DynamicScheduler()
        blk = zeros(ComplexF64, length(values[i]), length(nearvalues[i]))
        #eltype(nearmatrix)
        nearmatrix(blk, values[i], nearvalues[i])
        blocks[i] = blk
    end
    nears = BlockSparseMatrix(blocks, values, nearvalues, size(nearmatrix))
    #size(nearmatrix)
    fly = Vector{BF}()
    for (NO, source_nodes) in farints
        for NS in source_nodes
            push!(
                fly, subroutine_BF(nearmatrix, tree, NO, NS, k, tol; Compressor=Compressor)
            )
        end
    end

    return PetrovGalerkinBF{eltype(operator)}(  #BEAST.scalartype(operator)
        nears,
        tree,
        farints,
        fly,        # Here come all other fields needed for the ButterflyFactorizations
        size(nearmatrix),
    )
end
"""
    PetrovGalerkinBF_mats(operator, testspace, trialspace, tree, k; kwargs...)

Constructs the complete Butterfly Factorization for a Petrov-Galerkin boundary element
problem using the **sparse matrix-based** format.

Similar to `PetrovGalerkinBF`, this isolates the near-field into a `BlockSparseMatrix`.
However, the far-field interactions are compressed using `subroutine_BF_mats`, which
assembles the Butterfly factors (`Q`, `R`, `P`) explicitly as sparse block-diagonal
matrices. This format requires more memory but provides significantly faster Matrix-Vector
multiplication.

**Arguments:**
- `operator`: The integral operator to be evaluated.
- `testspace`: The observer (test) function space.
- `trialspace`: The source (trial) function space.
- `tree`: A coupled `BlockTree` representing the hierarchical clustering.
- `k`: The physical wavenumber of the problem.

**Keyword Arguments:**
- `Compressor`: The low-rank approximation strategy (default: `PartialQR()`).
- `tol`: The relative precision tolerance for compression (default: `1e-3`).
- `ntasks`: Number of threads to use for parallel near-field assembly.
- `α`: The geometric admissibility parameter (default: `2.0`).

**Returns:**
- A `PetrovGalerkinBF_mats` struct containing the near-field matrix and the array of
  far-field `BF_Mats` blocks.
"""

function PetrovGalerkinBF_mats(
    operator,
    testspace,
    trialspace,
    tree::BlockTree,
    k::Float64;
    Compressor=ButterflyFactorizations.PartialQR(),
    tol=1e-3,
    ntasks=Threads.nthreads(),
    α=2,
)
    nearmatrix = AbstractKernelMatrix(operator, testspace, trialspace;)
    #quadstrat=nearquadstrat
    #values, nearvalues = nearinteractions(tree; isnear=isnear)
    values, nearvalues, farints = nearandfar(tree, α)

    blocks = Vector{Matrix{eltype(nearmatrix)}}(undef, length(values))
    @tasks for i in eachindex(values)
        @set ntasks = 1#ntasks
        blk = zeros(ComplexF64, length(values[i]), length(nearvalues[i]))
        #eltype(nearmatrix)
        nearmatrix(blk, values[i], nearvalues[i])
        blocks[i] = blk
    end
    nears = BlockSparseMatrix(blocks, values, nearvalues, size(nearmatrix))
    fly = Vector{BF_Mats}()
    for (NO, source_nodes) in farints
        for NS in source_nodes
            push!(
                fly,
                subroutine_BF_mats(nearmatrix, tree, NO, NS, k, tol; Compressor=Compressor),
            )
            #end
        end
    end

    return PetrovGalerkinBF_mats{eltype(operator)}(  #BEAST.scalartype(operator)
        nears,
        #tree,
        farints,
        fly,        # Here come all other fields needed for the ButterflyFactorizations
        size(nearmatrix),
    )
end
