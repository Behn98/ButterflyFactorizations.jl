"""
    PetrovGalerkinBF(operator, testspace, trialspace, tree, k; kwargs...)

Constructs the complete Butterfly Factorization for a Petrov-Galerkin boundary element
problem now using flat butterfly formats.

This function separates the interactions based on the admissibility condition `α`. The
near-field interactions are evaluated directly and stored as a `BlockSparseMatrix`. The
far-field interactions are compressed block-by-block using the dictionary-based
parameterized `assemble_BF`, resulting in a collection of highly memory-efficient
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
    α=2.0,
    scheduler=OhMyThreads.DynamicScheduler(),
    acctype=ComplexF64,
    unbalancedints=true,
    leafcom=true,
)
    # --- NEAR INTERACTIONS ---
    nearmatrix_near = AbstractKernelMatrix(operator, testspace, trialspace; type=:near)
    farints, nearints = nearandfar(tree, α; unbalancedints=unbalancedints, leafcom=leafcom)

    blocks = Vector{Matrix{acctype}}(undef, length(nearints))
    test_indices = Vector{Vector{Int64}}(undef, length(nearints))
    trial_indices = Vector{Vector{Int64}}(undef, length(nearints))

    near_rows = Vector{Int}(undef, length(nearints))
    near_cols = Vector{Int}(undef, length(nearints))
    near_vals = Vector{Int}(undef, length(nearints))

    let nearmatrix_near = nearmatrix_near
        @tasks for i in eachindex(nearints)
            @set scheduler = scheduler
            (node_o, node_s) = nearints[i]

            near_rows[i] = node_o
            near_cols[i] = node_s
            near_vals[i] = i

            test_indices[i] = H2Trees.values(tree.testcluster, node_o)
            trial_indices[i] = H2Trees.values(tree.trialcluster, node_s)

            blk = zeros(acctype, length(test_indices[i]), length(trial_indices[i]))
            nearmatrix_near(blk, test_indices[i], trial_indices[i])
            blocks[i] = blk
        end
    end

    nears = if !isempty(nearints)
        BlockSparseMatrix(
            blocks,
            test_indices,
            trial_indices,
            size(nearmatrix_near);
            scheduler=OhMyThreads.DynamicScheduler(),#SerialScheduler()
        )
    else
        sparse(zeros(acctype, size(nearmatrix_near)...))
    end

    # --- FAR INTERACTIONS (FLAT BUTTERFLIES) ---
    nearmatrix_far = AbstractKernelMatrix(operator, testspace, trialspace; type=:far)
    fly = Vector{ButterflyFactorization{acctype,typeof(tree)}}(undef, length(farints))
    far_rows = Vector{Int}(undef, length(farints))
    far_cols = Vector{Int}(undef, length(farints))
    far_vals = Vector{Int}(undef, length(farints))

    let nearmatrix_far = nearmatrix_far
        @tasks for i in eachindex(farints)
            @set scheduler = scheduler
            (NO, NS) = farints[i]

            far_rows[i] = NO
            far_cols[i] = NS
            far_vals[i] = i

            fly[i] = assemble_BF(
                nearmatrix_far,
                tree,
                NO,
                NS,
                k,
                tol;
                compressor=compressor,
                scheduler=OhMyThreads.DynamicScheduler(),#SerialScheduler()
            )
        end
    end

    num_test_nodes  = sum(length(lvl) for lvl in h2treelevels(tree.testcluster, 1))
    num_trial_nodes = sum(length(lvl) for lvl in h2treelevels(tree.trialcluster, 1))

    near_lookup = sparse(near_rows, near_cols, near_vals, num_test_nodes, num_trial_nodes)
    far_lookup  = sparse(far_rows, far_cols, far_vals, num_test_nodes, num_trial_nodes)
    # 4. Initialize standard workspaces
    workspaces = map(ButterflyWorkspace, fly)

    return PetrovGalerkinBF{acctype}(
        nears, tree, fly, workspaces, size(nearmatrix_far), near_lookup, far_lookup
    )
end

"""
    PetrovGalerkinBF_Mat(operator, testspace, trialspace, tree, k; kwargs...)

Constructs the complete Butterfly Factorization for a Petrov-Galerkin boundary element
problem using the **sparse matrix-based** format.

Similar to `PetrovGalerkinBF`, this isolates the near-field into a `BlockSparseMatrix`.
However, the far-field interactions are compressed using `assemble_ButterflyFactorization_Mat`, which
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

  - A `PetrovGalerkinBF_Mat` struct containing the near-field matrix and the array of
    far-field `ButterflyFactorization_Mat` blocks.
"""
function PetrovGalerkinBF_Mat(
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
    fly = Vector{ButterflyFactorization_Mat}(undef, length(farints))
    let nearmatrix = nearmatrix
        @tasks for i in eachindex(farints)
            @set scheduler = DynamicScheduler()
            (NO, NS) = farints[i]
            fly[i] = assemble_ButterflyFactorization_Mat(
                nearmatrix, tree, NO, NS, k, tol; compressor=compressor
            )
        end
    end

    return PetrovGalerkinBF_Mat{acctype}(nears, farints, fly, size(nearmatrix))
end
