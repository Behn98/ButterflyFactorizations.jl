
"""
    validate_farfield_accuracy(operator, testspace, trialspace, tree, farints, fly; n_samples=20)

Randomly samples far-field blocks, computes the exact BEAST dense block, and compares
the matrix-vector product against the Butterfly factorization.
"""
function validate_farfield_accuracy(
    operator, testspace, trialspace, BF::PetrovGalerkinBF; n_samples=20
)
    # We only need the far-field kernel evaluator
    kernelmatrix = ButterflyFactorizations.AbstractKernelMatrix(
        operator, testspace, trialspace; type=:far
    )

    # Pick a random subset of blocks to test so we don't wait forever
    n_total = length(BF.BFs)
    sample_indices = shuffle(1:n_total)[1:min(n_samples, n_total)]

    max_err = 0.0
    avg_err = 0.0

    println("--- Far-Field Accuracy Validation ($n_samples blocks) ---")

    for i in sample_indices
        (node_s, node_o) = ButterflyFactorizations.getNSNO(BF.BFs[i])

        # 1. Get the exact global indices for this block
        test_idx = H2Trees.values(BF.tree.testcluster, node_o)
        trial_idx = H2Trees.values(BF.tree.trialcluster, node_s)

        # 2. Assemble the EXACT dense block using BEAST
        Z_exact = zeros(ComplexF64, length(test_idx), length(trial_idx))
        kernelmatrix(Z_exact, test_idx, trial_idx)

        # 3. Generate a random input vector
        x = randn(ComplexF64, length(trial_idx))

        # 4. Compute exact matvec
        y_exact = Z_exact * x

        # 5. Compute Butterfly matvec
        # Assuming `fly[i]` supports multiplication: `mul!(y_bf, fly[i], x)`
        # If your Butterfly uses an allocated `*`:
        x_tmp = zeros(ComplexF64, length(trialspace))
        x_tmp[trial_idx] .= x
        y_bf = zeros(ComplexF64, length(testspace))
        mul!(y_bf, BF.BFs[i], x_tmp)

        # 6. Calculate relative error
        rel_err = norm(y_exact - y_bf[test_idx]) / norm(y_exact)

        max_err = max(max_err, rel_err)
        avg_err += rel_err

        println(
            "Block $i (Size: $(length(test_idx))x$(length(trial_idx))) -> Rel Error: $(round(rel_err, sigdigits=3))",
        )
    end

    avg_err /= length(sample_indices)

    println("-------------------------------------------------")
    println("Maximum Relative Error: $(round(max_err, sigdigits=4))")
    println("Average Relative Error: $(round(avg_err, sigdigits=4))")

    return avg_err, max_err
end
