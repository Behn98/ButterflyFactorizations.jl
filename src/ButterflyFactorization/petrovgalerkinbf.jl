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

  - `compressor`: The low-rank approximation strategy (default: `PartialQR()`).
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
    compressor=ButterflyFactorizations.PartialQR(),
    tol=1e-3,
    α=2,
    scheduler=OhMyThreads.DynamicScheduler(),
    acctype=ComplexF64,
)
    # --- NEAR INTERACTIONS ---
    nearmatrix = AbstractKernelMatrix(operator, testspace, trialspace; type=:near)
    farints, nearints = nearandfar(tree, α)

    blocks = Vector{Matrix{acctype}}(undef, length(nearints))
    values = Vector{Vector{Int64}}(undef, length(nearints))
    nearvalues = Vector{Vector{Int64}}(undef, length(nearints))

    # Vectors to harvest coordinates for the near lookup matrix
    near_rows = Vector{Int}(undef, length(nearints))
    near_cols = Vector{Int}(undef, length(nearints))
    near_vals = Vector{Int}(undef, length(nearints))

    let nearmatrix = nearmatrix
        @tasks for i in eachindex(nearints)
            (node_o, node_s) = nearints[i]
            @set scheduler = scheduler

            # Capture coordinates and map to loop index 'i'
            near_rows[i] = node_o
            near_cols[i] = node_s
            near_vals[i] = i

            values[i] = H2Trees.values(tree.testcluster, node_o)
            nearvalues[i] = H2Trees.values(tree.trialcluster, node_s)
            blk = zeros(acctype, length(values[i]), length(nearvalues[i]))
            nearmatrix(blk, values[i], nearvalues[i])
            blocks[i] = blk
        end
    end
    nears = BlockSparseMatrix(
        blocks, values, nearvalues, size(nearmatrix); scheduler=scheduler
    )

    # --- FAR INTERACTIONS (BUTTERFLIES) ---
    nearmatrix = AbstractKernelMatrix(operator, testspace, trialspace; type=:far)
    fly = Vector{BF}(undef, length(farints))

    # Vectors to harvest coordinates for the far lookup matrix
    far_rows = Vector{Int}(undef, length(farints))
    far_cols = Vector{Int}(undef, length(farints))
    far_vals = Vector{Int}(undef, length(farints))

    let nearmatrix = nearmatrix
        @tasks for i in eachindex(farints)
            @set scheduler = scheduler
            (NO, NS) = farints[i]

            # Capture coordinates and map to loop index 'i'
            far_rows[i] = NO
            far_cols[i] = NS
            far_vals[i] = i

            fly[i] = subroutine_BF(nearmatrix, tree, NO, NS, k, tol; compressor=compressor)
        end
    end

    # --- BUILD CSC LOOKUP MATRICES ---
    # The total number of nodes can be found from the number of clusters in your trees
    num_test_nodes  = sum(length(lvl) for lvl in h2treelevels(tree.testcluster, 1))
    num_trial_nodes = sum(length(lvl) for lvl in h2treelevels(tree.trialcluster, 1))

    # sparse(rows, cols, vals, m, n) compiles them into clean SparseMatrixCSC layout
    near_lookup = sparse(near_rows, near_cols, near_vals, num_test_nodes, num_trial_nodes)
    far_lookup  = sparse(far_rows, far_cols, far_vals, num_test_nodes, num_trial_nodes)

    return PetrovGalerkinBF{acctype}(
        nears, tree, fly, size(nearmatrix), near_lookup, far_lookup
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

  - `compressor`: The low-rank approximation strategy (default: `PartialQR()`).
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
    compressor=ButterflyFactorizations.PartialQR(),
    tol=1e-3,
    scheduler=OhMyThreads.DynamicScheduler(),
    α=2,
    acctype=ComplexF64,
)
    nearmatrix = AbstractKernelMatrix(operator, testspace, trialspace;)
    farints, nearints = nearandfar(tree, α)
    blocks = Vector{Matrix{acctype}}(undef, length(nearints))
    values = Vector{Vector{Int64}}(undef, length(nearints))
    nearvalues = Vector{Vector{Int64}}(undef, length(nearints))
    let nearmatrix = nearmatrix
        @tasks for i in eachindex(nearints)
            (node_o, node_s) = nearints[i]
            @set scheduler = scheduler #DynamicScheduler() #SerialScheduler
            values[i] = H2Trees.values(tree.testcluster, node_o)
            nearvalues[i] = H2Trees.values(tree.trialcluster, node_s)
            blk = zeros(acctype, length(values[i]), length(nearvalues[i]))
            nearmatrix(blk, values[i], nearvalues[i])
            blocks[i] = blk
        end
    end
    nears = BlockSparseMatrix(
        blocks, values, nearvalues, size(nearmatrix); scheduler=scheduler
    )
    nearmatrix = AbstractKernelMatrix(operator, testspace, trialspace; type=:far)
    fly = Vector{BF_Mats}(undef, length(farints))
    let nearmatrix = nearmatrix
        @tasks for i in eachindex(farints)
            @set scheduler = DynamicScheduler()
            (NO, NS) = farints[i]
            fly[i] = subroutine_BF_mats(
                nearmatrix, tree, NO, NS, k, tol; compressor=compressor
            )
        end
    end

    return PetrovGalerkinBF_mats{acctype}(nears, farints, fly, size(nearmatrix))
end

#--------------------------STATISTICS--------------------------

function PetrovGalerkinBF(
    operator,
    testspace,
    trialspace,
    tree::BlockTree,
    k::Float64,
    dostat::Bool;
    compressor=ButterflyFactorizations.PartialQR(),
    tol=1e-3,
    α=2,
    scheduler=OhMyThreads.DynamicScheduler(),
    acctype=ComplexF64,
)
    # --- NEAR INTERACTIONS ---
    nearmatrix = AbstractKernelMatrix(operator, testspace, trialspace; type=:near)
    farints, nearints = nearandfar(tree, α)

    blocks = Vector{Matrix{acctype}}(undef, length(nearints))
    values = Vector{Vector{Int64}}(undef, length(nearints))
    nearvalues = Vector{Vector{Int64}}(undef, length(nearints))

    # Vectors to harvest coordinates for the near lookup matrix
    near_rows = Vector{Int}(undef, length(nearints))
    near_cols = Vector{Int}(undef, length(nearints))
    near_vals = Vector{Int}(undef, length(nearints))

    let nearmatrix = nearmatrix
        @tasks for i in eachindex(nearints)
            (node_o, node_s) = nearints[i]
            @set scheduler = scheduler

            # Capture coordinates and map to loop index 'i'
            near_rows[i] = node_o
            near_cols[i] = node_s
            near_vals[i] = i

            values[i] = H2Trees.values(tree.testcluster, node_o)
            nearvalues[i] = H2Trees.values(tree.trialcluster, node_s)
            blk = zeros(acctype, length(values[i]), length(nearvalues[i]))
            nearmatrix(blk, values[i], nearvalues[i])
            blocks[i] = blk
        end
    end
    nears = BlockSparseMatrix(
        blocks, values, nearvalues, size(nearmatrix); scheduler=scheduler
    )

    # --- FAR INTERACTIONS (BUTTERFLIES) ---
    nearmatrix = AbstractKernelMatrix(operator, testspace, trialspace; type=:far)
    fly = Vector{BF}(undef, length(farints))

    # Vectors to harvest coordinates for the far lookup matrix
    far_rows = Vector{Int}(undef, length(farints))
    far_cols = Vector{Int}(undef, length(farints))
    far_vals = Vector{Int}(undef, length(farints))

    let nearmatrix = nearmatrix
        @tasks for i in eachindex(farints)
            @set scheduler = scheduler
            (NO, NS) = farints[i]

            # Capture coordinates and map to loop index 'i'
            far_rows[i] = NO
            far_cols[i] = NS
            far_vals[i] = i

            fly[i] = subroutine_BF(
                nearmatrix, tree, NO, NS, k, tol, true; compressor=compressor
            )
        end
    end

    # --- BUILD CSC LOOKUP MATRICES ---
    # The total number of nodes can be found from the number of clusters in your trees
    num_test_nodes  = length(tree.testcluster)
    num_trial_nodes = length(tree.trialcluster)

    # sparse(rows, cols, vals, m, n) compiles them into clean SparseMatrixCSC layout
    near_lookup = sparse(near_rows, near_cols, near_vals, num_test_nodes, num_trial_nodes)
    far_lookup  = sparse(far_rows, far_cols, far_vals, num_test_nodes, num_trial_nodes)

    return PetrovGalerkinBF{acctype}(
        nears, tree, fly, size(nearmatrix), near_lookup, far_lookup
    )
end
