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
  - `unbalancedints`: Whether to allow unbalanced interactions (default: `true`).
  - `leafcom`: Whether to allow leaf-level compression (default: `true`).
  - `acctype`: The complex type to use for the factorization (default: `ComplexF64`).
  - `scheduler`: The threading scheduler to use for parallel assembly (default:
    `OhMyThreads.DynamicScheduler()`).

**Returns:**

  - A `PetrovGalerkinBF` struct containing the near-field matrix and the array of far-field
    `BF` blocks.
"""
function PetrovGalerkinBF(
    operator,
    testspace,
    trialspace,
    tree,
    k::Float64;
    compressor=ButterflyFactorizations.PartialQR(),
    tol=1e-3,
    admissibility=isFarFunctor(tree_parameters(cluster_testtree(tree)).α),
    C=tree_parameters(cluster_testtree(tree)).C,
    Cε=tree_parameters(cluster_testtree(tree)).Cε,
    scheduler=OhMyThreads.DynamicScheduler(),
    acctype=ComplexF64,
    unbalancedints=false,
    leafcomp=true,
    leafimbalance=true,
)
    # --- NEAR INTERACTIONS ---
    nearmatrix_near = AbstractKernelMatrix(operator, testspace, trialspace; type=:near)

    # 🚀 Pass the functor directly instead of α
    farints, nearints = nearandfar(
        tree,
        admissibility;
        unbalancedints=unbalancedints,
        leafcomp=leafcomp,
        leafimbalance=leafimbalance,
    )
    n_ints = length(nearints)
    blocks = Vector{Matrix{acctype}}(undef, n_ints)
    test_indices = Vector{Vector{Int64}}(undef, n_ints)
    trial_indices = Vector{Vector{Int64}}(undef, n_ints)

    near_rows = Vector{Int}(undef, n_ints)
    near_cols = Vector{Int}(undef, n_ints)
    near_vals = Vector{Int}(undef, n_ints)

    # ---------------------------------------------------------
    # PASS 1: Sequential Pre-allocation & Index Fetching
    # (Keeps all dynamic allocations and array pushes on the main thread)
    # ---------------------------------------------------------

    for i in 1:n_ints
        (node_o, node_s) = nearints[i]

        near_rows[i] = node_o
        near_cols[i] = node_s
        near_vals[i] = i

        test_indices[i] = cluster_values(tree.testcluster, node_o)
        trial_indices[i] = cluster_values(tree.trialcluster, node_s)

        # Allocate uninitialized memory outside the threaded loop
        blocks[i] = Matrix{acctype}(
            undef, length(test_indices[i]), length(trial_indices[i])
        )
    end

    # ---------------------------------------------------------
    # PASS 2: Parallel Block Evaluation
    # (Threads only execute pure compute and memory overwrites)
    # ---------------------------------------------------------
    let nearmatrix_near = nearmatrix_near
        @tasks for i in 1:n_ints
            @set scheduler = scheduler

            blk = blocks[i]

            # Explicitly zero out the pre-allocated block memory
            # (matches the behavior of `zeros()`)
            fill!(blk, zero(acctype))

            # Pure compute
            nearmatrix_near(blk, test_indices[i], trial_indices[i])
        end
    end
    nears = if n_ints > 0
        BlockSparseMatrix(
            blocks,
            test_indices,
            trial_indices,
            size(nearmatrix_near);
            scheduler=scheduler,
        )
    else
        BlockSparseMatrix(
            Matrix{acctype}[], # Changed from ComplexF64 to acctype for consistency
            Int[],
            Int[],
            size(nearmatrix_near);
            scheduler=SerialScheduler(),
        )
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
                C=C,
                Cε=Cε,
                scheduler=OhMyThreads.SerialScheduler(),#SerialScheduler()DynamicScheduler()
            )
        end
    end

    num_test_nodes  = sum(length(lvl) for lvl in treelevels(tree.testcluster, 1))
    num_trial_nodes = sum(length(lvl) for lvl in treelevels(tree.trialcluster, 1))
    near_lookup     = sparse(near_rows, near_cols, near_vals, num_test_nodes, num_trial_nodes)
    far_lookup      = sparse(far_rows, far_cols, far_vals, num_test_nodes, num_trial_nodes)

    return PetrovGalerkinBF{acctype}(
        nears, tree, fly, size(nearmatrix_far), near_lookup, far_lookup
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
    tree,
    k::Float64;
    compressor=ButterflyFactorizations.PartialQR(),
    α=tree_parameters(cluster_testtree(tree)).α,
    C=tree_parameters(cluster_testtree(tree)).C,
    Cε=tree_parameters(cluster_testtree(tree)).Cε,
    tol=1e-3,
    scheduler=OhMyThreads.DynamicScheduler(),
    acctype=ComplexF64,
)
    # --- NEAR INTERACTIONS ---
    nearmatrix = AbstractKernelMatrix(operator, testspace, trialspace;)
    farints, nearints = nearandfar(tree, α)
    n_ints = length(nearints)

    blocks = Vector{Matrix{acctype}}(undef, n_ints)
    values = Vector{Vector{Int64}}(undef, n_ints)
    nearvalues = Vector{Vector{Int64}}(undef, n_ints)

    # ---------------------------------------------------------
    # PASS 1: Sequential Pre-allocation
    # (Fast, lock-free, keeps the GC happy)
    # ---------------------------------------------------------
    for i in 1:n_ints
        (node_o, node_s) = nearints[i]
        values[i] = cluster_values(tree.testcluster, node_o)
        nearvalues[i] = cluster_values(tree.trialcluster, node_s)

        # Allocate uninitialized memory OUTSIDE the threaded loop
        blocks[i] = Matrix{acctype}(undef, length(values[i]), length(nearvalues[i]))
    end

    # ---------------------------------------------------------
    # PASS 2: Parallel Evaluation
    # (Zero allocations = maximum parallel scaling)
    # ---------------------------------------------------------
    let nearmatrix = nearmatrix
        @tasks for i in 1:n_ints
            @set scheduler = scheduler

            blk = blocks[i]

            # If your `nearmatrix` function purely over-writes the matrix,
            # you DO NOT need this fill! line (saving even more time).
            # If it accumulates (+=), then keep it.
            fill!(blk, zero(acctype))

            nearmatrix(blk, values[i], nearvalues[i])
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
            fly[i] = assemble_BF_Mat(
                nearmatrix, tree, NO, NS, k, tol; compressor=compressor, C=C, Cε=Cε
            )
        end
    end

    return PetrovGalerkinBF_Mat{acctype}(nears, farints, fly, size(nearmatrix))
end
