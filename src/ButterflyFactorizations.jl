"""
    ButterflyFactorizations

A high-performance Julia module for constructing, manipulating, and applying Butterfly Factorizations.

This package provides hierarchical low-rank approximation techniques designed for highly
oscillatory integral operators. It leverages tree-based domain partitioning to achieve
``O(N \\log N)`` complexity for both storage and matrix-vector evaluations.

# Core Architecture

The package features a dual-format architecture to balance performance and flexibility:

 1. **Flat-Array Format (Production):** Highly optimized, dictionary-free data structures
    (`ButterflyFactorization`) coupled with zero-allocation memory pools (`ThreadButterflyWorkspace`).
    This format is designed for maximum cache-efficiency and multi-threaded scaling.
 2. **Sparse Matrix Format (Debugging & Direct Linear Algebra):** Explicit block-diagonal
    sparse matrices (`ButterflyFactorization_Mat`). While slightly more memory-intensive,
    this format allows for seamless integration with standard Julia sparse linear algebra routines.

# Key Features

  - **Zero-Allocation Mat-Vec:** Highly optimized `mul!` overloads utilizing pre-allocated thread workspaces.
  - **Algebraic Composition:** Native support for adding (`+`), multiplying (`*`), and recompressing (`recompress_BF`) entire hierarchical butterfly operators without inflating them to dense matrices.
  - **Structural Transformations:** Native `adjoint` and `transpose` overloads that physically reverse the tree routing mappings.
  - **Petrov-Galerkin Assembly:** `PetrovGalerkinBF` constructs global linear operators by splitting near-field (dense evaluations) and far-field (butterfly-compressed) interactions using customizable admissibility rules.
  - **Extensibility:** Agnostic tree interfaces allowing integration with packages like `H2Trees.jl`, and abstract kernel interfaces for evaluating custom integral operators (e.g., `BEAST.jl`).

# Main API

**Global Operators:**

  - [`PetrovGalerkinBF`](@ref), [`PetrovGalerkinBF_Mat`](@ref)

**Local Factorizations:**

  - [`ButterflyFactorization`](@ref), [`ButterflyFactorization_Mat`](@ref)
  - [`assemble_BF`](@ref), [`assemble_BF_Mat`](@ref)

**Workspaces (Zero-Allocation):**

  - [`ThreadButterflyWorkspace`](@ref), [`ButterflyWorkspace`](@ref)

**Compressors:**

  - [`PartialQR`](@ref)

**Algebraic Operations:**

  - [`mulBFs`](@ref), [`add_eqbfs`](@ref), [`recompress_BF`](@ref)

# Example

Check the `test/` folder or `README.md` for example usage, including how to construct a
`PetrovGalerkinBF` from a BEAST operator, apply it to a vector, and perform algebraic recompression.
"""
module ButterflyFactorizations

using BlockSparseMatrices
using LinearAlgebra
using LinearMaps
using StaticArrays
using Random
using OhMyThreads
using LowRankApprox
using SparseArrays
using H2Trees

# Export Global Operators & Assembly
export PetrovGalerkinBF, PetrovGalerkinBF_Mat
export assemble_BF, assemble_BF_Mat

# Export Core Data Structures
export ButterflyFactorization, ButterflyFactorization_Mat
export ButterflyBlock, ButterflyLevel
export ThreadButterflyWorkspace, ButterflyWorkspace

# Export Compressors
export PartialQR

# Export Algebra
export mulBFs, add_eqbfs, recompress_BF

# ------------------------------------------------------------------
# Includes
# ------------------------------------------------------------------

include("treeinterface.jl")
include("auxillaries.jl")

# Kernel Matrix Import
include("kernelmatrix/abstractkernelmatrix.jl")
include("kernelmatrix/beastkernelmatrix.jl")

# Butterfly Factorization Assembly
include("ButterflyFactorization/bfstructs.jl")
include("ButterflyFactorization/bfstructfcts.jl")
include("ButterflyFactorization/compressors.jl")
include("ButterflyFactorization/bfassembly.jl")

# Butterfly Algebra
include("Butterflyalgebra/bfadjtr.jl")
include("Butterflyalgebra/bfdims.jl")
include("Butterflyalgebra/bfvector.jl")
include("Butterflyalgebra/bfmatrix.jl")
include("Butterflyalgebra/algrecomp.jl")
include("Butterflyalgebra/bfaddbf.jl")
include("Butterflyalgebra/bfmulbf.jl")
include("Butterflyalgebra/bfsplitmul.jl")
include("Butterflyalgebra/bfsplit.jl")

# Tree Traversal & Butterfly Construction
include("GlobalMatrixAssembly/farfieldcrit.jl")
include("GlobalMatrixAssembly/intlists.jl")

# Full Matrix Assembly
include("GlobalMatrixAssembly/matrixstructs.jl")
include("GlobalMatrixAssembly/petrovgalerkinbf.jl")

# Matrix Algebra Overloads
include("matrixalgebra/matrix_adj_tr.jl")
include("matrixalgebra/dims.jl")
include("matrixalgebra/matrixvector.jl")
include("matrixalgebra/matrixmatrix.jl")

# I/O and Display
include("showfcts.jl")

end # module ButterflyFactorizations
