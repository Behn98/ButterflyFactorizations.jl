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
    α=2,
    scheduler=OhMyThreads.DynamicScheduler(),
    acctype=ComplexF64,
)
    # 0. Spara gammal BLAS-inställning och sätt till 1 för att undvika överbelastning
    old_blas_threads = BLAS.get_num_threads()
    BLAS.set_num_threads(1)

    #println("\n--- Profiling PetrovGalerkinBF ---")
    # 1. Mät tid för uppsättning och träd-sökning
    #t_nearfar = @elapsed begin
    nearmatrix = AbstractKernelMatrix(operator, testspace, trialspace; type=:near)
    #quadstrat=nearquadstrat
    values, nearvalues, farints = nearandfar(tree, α)
    #end
    #println("1. nearandfar & initial setup : ", round(t_nearfar; digits=4), " s")

    # 2. Mät tid för Near-field assembly (Tät matrisuträkning)
    #t_near = @elapsed begin
    blocks = Vector{Matrix{acctype}}(undef, length(values))
    let nearmatrix = nearmatrix
        @tasks for i in eachindex(values)
            @set scheduler = scheduler #DynamicScheduler() #SerialScheduler
            blk = zeros(ComplexF64, length(values[i]), length(nearvalues[i]))
            nearmatrix(blk, values[i], nearvalues[i])
            blocks[i] = blk
        end
    end
    nears = BlockSparseMatrix(blocks, values, nearvalues, size(nearmatrix))
    #end
    #println("2. Near-field matrix assembly : ", round(t_near; digits=4), " s")

    # 3. Mät tid för list-allokering (detta borde vara nära 0 sekunder)
    #t_list = @elapsed begin
    far_tasks = Tuple{Int,Int}[]
    for (NO, source_nodes) in farints
        for NS in source_nodes
            push!(far_tasks, (NO, NS))
        end
    end
    #end
    #println(
    #    "3. Far-field task generation  : ",
    #    round(t_list; digits=4),
    #    " s (Antal block: ",
    #   length(far_tasks),
    #    ")",
    #)
    nearmatrix = AbstractKernelMatrix(operator, testspace, trialspace; type=:far)
    # 4. Mät tid för uppbyggnad av Butterfly Factorizations
    #t_far = @elapsed begin
    fly = Vector{BF}(undef, length(far_tasks))
    let nearmatrix = nearmatrix
        @tasks for i in eachindex(far_tasks)
            @set scheduler = scheduler
            (NO, NS) = far_tasks[i]
            fly[i] = subroutine_BF(nearmatrix, tree, NO, NS, k, tol; Compressor=Compressor)
        end
    end
    #end
    #println("4. Far-field Butterfly Factor.: ", round(t_far; digits=4), " s")
    #println("----------------------------------\n")

    # Återställ BLAS-trådarna innan vi returnerar
    BLAS.set_num_threads(old_blas_threads)

    return PetrovGalerkinBF{acctype}(  #BEAST.scalartype(operator)
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
    acctype=ComplexF64,
)
    nearmatrix = AbstractKernelMatrix(operator, testspace, trialspace;)
    #quadstrat=nearquadstrat
    #values, nearvalues = nearinteractions(tree; isnear=isnear)
    values, nearvalues, farints = nearandfar(tree, α)

    blocks = Vector{Matrix{acctype}}(undef, length(values))
    @tasks for i in eachindex(values)
        @set scheduler = DynamicScheduler() #SerialScheduler()
        blk = zeros(ComplexF64, length(values[i]), length(nearvalues[i]))
        nearmatrix(blk, values[i], nearvalues[i])
        blocks[i] = blk
    end
    nears = BlockSparseMatrix(blocks, values, nearvalues, size(nearmatrix))
    far_tasks = Tuple{Int,Int}[]
    for (NO, source_nodes) in farints
        for NS in source_nodes
            push!(far_tasks, (NO, NS))
        end
    end

    # 2. Använd tmap för att bygga alla Butterfly-matrisblocken parallellt
    fly = Vector{BF_Mats}(undef, length(far_tasks))
    @tasks for i in eachindex(far_tasks)
        @set scheduler = DynamicScheduler()
        (NO, NS) = far_tasks[i]
        fly[i] = subroutine_BF_mats(nearmatrix, tree, NO, NS, k, tol; Compressor=Compressor)
    end

    return PetrovGalerkinBF_mats{acctype}(  #BEAST.scalartype(operator)
        nears,
        #tree,
        farints,
        fly,        # Here come all other fields needed for the ButterflyFactorizations
        size(nearmatrix),
    )
end
